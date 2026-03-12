#!/usr/bin/env python
"""
DeepFLAIR* inference & evaluation script.

Runs sliding-window inference on the held-out test split and reports:
  - MSE, SSIM, PSNR  (computed on foreground voxels only)

Outputs:
  - Per-subject metrics printed to console
  - outputs/eval/<model_type>_metrics.csv
  - vis/test_<subject_id>_<model_type>.png  (EPI / GT / Pred × Axial/Sagittal/Coronal)
  - (optional) outputs/eval/<subject_id>_<model_type>_pred.nii.gz

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \\
      --ckpt_path outputs/DF_unet_BS16/checkpoints/last.ckpt \\
      --model_type unet

  # use best checkpoint instead of last:
  CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \\
      --ckpt_path outputs/DF_unet_BS16/checkpoints/deepflair-epoch=066-val_loss=0.0812.ckpt \\
      --model_type unet --save_nii
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Allow running from repo root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PyTorch 2.6 fix: allowlist MONAI's TraceKeys before any checkpoint loading
try:
    import torch.serialization
    from monai.utils.enums import TraceKeys
    torch.serialization.add_safe_globals([TraceKeys])
except Exception:
    pass

from monai.inferers import sliding_window_inference
from monai.losses import SSIMLoss

from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, gt):
    """
    pred, gt: CPU float tensors [1, 1, D, H, W], values in [0, 1].

    Metrics are computed on foreground voxels only (gt > 0.03) to avoid
    inflating scores with the large zero-padded background region.

    Returns dict: mse, ssim, psnr
    """
    mask = (gt > 0.03).squeeze()          # [D, H, W] bool
    p_flat = pred.squeeze()[mask]         # [N]
    g_flat = gt.squeeze()[mask]           # [N]

    if p_flat.numel() == 0:               # fallback: full volume
        p_flat = pred.flatten()
        g_flat = gt.flatten()

    mse  = F.mse_loss(p_flat, g_flat).item()
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))

    # SSIM via MONAI on the full volume (window-based, needs spatial context)
    ssim_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
    with torch.no_grad():
        ssim_loss = ssim_fn(pred.float(), gt.float()).item()
    ssim = 1.0 - ssim_loss

    return {"mse": mse, "ssim": ssim, "psnr": psnr}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_figure(epi, gt, pred, subject_id, model_type, out_dir):
    """
    3-row (Axial / Sagittal / Coronal) × 3-col (Input EPI / GT FLAIR* / Pred) figure.
    Slice positions chosen at foreground centroid of GT to avoid zero-padded regions.
    """
    nonzero = np.argwhere(gt > 0.03)
    c = nonzero.mean(axis=0).astype(int).tolist() if len(nonzero) > 0 else [s // 2 for s in gt.shape]

    slices = [
        (epi[c[0], :, :],  gt[c[0], :, :],  pred[c[0], :, :],  "Axial"),
        (epi[:, c[1], :],  gt[:, c[1], :],  pred[:, c[1], :],  "Sagittal"),
        (epi[:, :, c[2]],  gt[:, :, c[2]],  pred[:, :, c[2]],  "Coronal"),
    ]

    col_titles = ["Input EPI", "GT FLAIR*", f"Pred ({model_type})"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row, (e_sl, g_sl, p_sl, view) in enumerate(slices):
        for col, (img, col_title) in enumerate(zip([e_sl, g_sl, p_sl], col_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(view, fontsize=11)

    plt.suptitle(f"Subject: {subject_id}", fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_{subject_id}_{model_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model from checkpoint (hparams saved automatically by PL) ---
    print(f"Loading: {args.ckpt_path}")
    model = DeepFLAIRLightningModule.load_from_checkpoint(
        args.ckpt_path,
        map_location=device,
    )
    model.eval()
    model.to(device)

    patch_size = tuple(model.hparams.patch_size)
    model_type = model.hparams.model_type
    print(f"Model: {model_type}  patch_size: {patch_size}")

    # --- Data: test split (same random_state=42 as training) ---
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=args.num_workers,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    print(f"Test subjects: {len(test_loader)}")

    results = []

    for i, batch in enumerate(test_loader):
        # subject_id is a plain string key passed through MONAI transforms untouched
        subj_raw = batch.get("subject_id", [f"subj_{i:03d}"])
        subject_id = subj_raw[0] if isinstance(subj_raw, (list, tuple)) else subj_raw

        x = batch["image"].to(device)   # [1, 1, D, H, W]
        y = batch["label"].to(device)

        with torch.no_grad():
            y_hat = sliding_window_inference(
                inputs=x,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=model.model,
                overlap=0.5,
                mode="gaussian",
            )

        metrics = compute_metrics(y_hat.cpu(), y.cpu())
        metrics["subject_id"] = subject_id
        results.append(metrics)

        print(f"  [{i+1:02d}] {subject_id}  "
              f"MSE={metrics['mse']:.5f}  "
              f"SSIM={metrics['ssim']:.4f}  "
              f"PSNR={metrics['psnr']:.2f} dB")

        # Visualization figure
        epi_np  = x[0, 0].cpu().numpy()
        gt_np   = y[0, 0].cpu().numpy()
        pred_np = y_hat[0, 0].cpu().numpy()

        fig_path = save_figure(epi_np, gt_np, pred_np, subject_id, model_type, "vis")
        print(f"         → {fig_path}")

        # Optional .nii.gz save
        if args.save_nii:
            import nibabel as nib
            os.makedirs(args.output_dir, exist_ok=True)
            nii = nib.Nifti1Image(pred_np.astype(np.float32), affine=np.eye(4))
            nii_path = os.path.join(args.output_dir, f"{subject_id}_{model_type}_pred.nii.gz")
            nib.save(nii, nii_path)
            print(f"         → {nii_path}")

    # --- Summary ---
    mse_arr  = np.array([r["mse"]  for r in results])
    ssim_arr = np.array([r["ssim"] for r in results])
    psnr_arr = np.array([r["psnr"] for r in results])

    print()
    print("=" * 60)
    print(f"  Model : {model_type}")
    print(f"  Ckpt  : {args.ckpt_path}")
    print(f"  N     : {len(results)} test subjects")
    print(f"  MSE   : {mse_arr.mean():.5f} ± {mse_arr.std():.5f}")
    print(f"  SSIM  : {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}")
    print(f"  PSNR  : {psnr_arr.mean():.2f} ± {psnr_arr.std():.2f} dB")
    print("=" * 60)

    # --- CSV ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{model_type}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "mse", "ssim", "psnr"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["subject_id", "mse", "ssim", "psnr"]})
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFLAIR* inference & evaluation")
    parser.add_argument("--ckpt_path",   required=True,  help="Path to .ckpt file")
    parser.add_argument("--data_dir",    default="data", help="Root data directory")
    parser.add_argument("--output_dir",  default="outputs/eval")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_nii",    action="store_true", help="Save predictions as .nii.gz")
    args = parser.parse_args()
    main(args)
