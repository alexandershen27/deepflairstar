#!/usr/bin/env python
"""
Multi-architecture comparison figure for DeepFLAIR*.

Generates a 2-row × 5-col figure for a chosen test subject:
  Row 0: EPI        | UNet (ReLU) | Attention U-Net | Swin-UNETR | UNETR++ (VFA)
  Row 1: GT FLAIR*  | diff        | diff            | diff        | diff

Difference maps: Pred − GT, displayed with a diverging colormap.
Absolute difference is also saved as a standalone image.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=2 python scripts/compare.py --data_dir data --subject_idx 0

    # Scan first 5 subjects and save a comparison figure for each
    CUDA_VISIBLE_DEVICES=2 python scripts/compare.py --data_dir data --n_subjects 5
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch.serialization
    from monai.utils.enums import TraceKeys
    torch.serialization.add_safe_globals([TraceKeys])
except Exception:
    pass

from monai.inferers import sliding_window_inference

from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule

# ---------------------------------------------------------------------------
# Model registry — ReLU activations only for architecture comparison
# ---------------------------------------------------------------------------
MODELS = [
    dict(
        name="U-Net (ReLU)",
        model_type="unet",
        base_channels=16,
        activation="relu",
        ckpt="outputs/DF_unet_v2/checkpoints/deepflair-epoch=172-val_loss=0.0737.ckpt",
    ),
    dict(
        name="Attention U-Net",
        model_type="attention_unet",
        base_channels=16,
        activation="relu",
        ckpt="outputs/DF_attention_unet_v2/checkpoints/deepflair-epoch=198-val_loss=0.0750.ckpt",
    ),
    dict(
        name="Swin-UNETR",
        model_type="swin",
        base_channels=24,
        activation="relu",
        ckpt="outputs/DF_swin_v2/checkpoints/deepflair-epoch=229-val_loss=0.0637.ckpt",
    ),
    dict(
        name="UNETR++ (VFA)",
        model_type="unetrpp",
        base_channels=32,
        activation="relu",
        ckpt="outputs/DF_unetrpp_BS32/checkpoints/deepflair-epoch=050-val_loss=0.1018.ckpt",
    ),
]


def load_model(cfg, device):
    ckpt = cfg["ckpt"]
    try:
        m = DeepFLAIRLightningModule.load_from_checkpoint(
            ckpt, strict=False, map_location=device
        )
    except Exception:
        m = DeepFLAIRLightningModule.load_from_checkpoint(
            ckpt,
            model_type=cfg["model_type"],
            base_channels=cfg["base_channels"],
            activation=cfg["activation"],
            strict=False,
            map_location=device,
        )
    m.eval()
    m.to(device)
    return m


@torch.no_grad()
def infer(model, epi_tensor, device):
    epi_tensor = epi_tensor.to(device)
    pred = sliding_window_inference(
        inputs=epi_tensor,
        roi_size=(64, 64, 64),
        sw_batch_size=4,
        predictor=model.model,
        overlap=0.5,
        mode="gaussian",
    )
    return pred.cpu().numpy()[0, 0]  # (D, H, W)


def foreground_centroid(vol, thresh=0.03):
    nz = np.argwhere(vol > thresh)
    if len(nz) > 0:
        return nz.mean(axis=0).astype(int).tolist()
    return [s // 2 for s in vol.shape]


def make_comparison_figure(subject_id, epi_np, gt_np, preds, out_path):
    """
    2-row × (1 + n_models)-col figure.
    Row 0: EPI | pred0 | pred1 | ...
    Row 1: GT  | diff0 | diff1 | ...
    """
    n = len(preds)
    ncols = n + 1
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 7.5),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    c = foreground_centroid(gt_np)
    ax_sl = c[0]

    # --- Column 0: EPI (top) and GT (bottom) ---
    for row, (arr, title) in enumerate([(epi_np, "Input EPI"), (gt_np, "GT FLAIR*")]):
        axes[row, 0].imshow(arr[ax_sl], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(title, fontsize=11, fontweight="bold", pad=4)
        axes[row, 0].axis("off")

    # --- Columns 1–n: predictions and difference maps ---
    diff_vmax = 0.25  # symmetric colorbar limit
    last_diff_im = None

    for i, (name, pred_np) in enumerate(preds):
        col = i + 1
        pred_sl = pred_np[ax_sl]
        diff_sl = pred_np[ax_sl] - gt_np[ax_sl]

        axes[0, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(name, fontsize=10, fontweight="bold", pad=4)
        axes[0, col].axis("off")

        last_diff_im = axes[1, col].imshow(
            diff_sl, cmap="bwr", vmin=-diff_vmax, vmax=diff_vmax
        )
        axes[1, col].set_title("Difference (Pred − GT)", fontsize=9, pad=4)
        axes[1, col].axis("off")

    # Colorbar attached to bottom-right axes
    if last_diff_im is not None:
        cbar = fig.colorbar(last_diff_im, ax=axes[1, 1:], shrink=0.75,
                            pad=0.01, label="Pred − GT")
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Architecture Comparison — {subject_id}  (axial slice {ax_sl})",
        fontsize=12, fontweight="bold", y=1.00,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison → {out_path}")
    return ax_sl


def make_individual_panels(subject_id, ax_sl, epi_np, gt_np, preds, out_dir):
    """Save each panel as a standalone high-res PNG for paper inclusion."""
    for label, arr in [("EPI", epi_np), ("GT", gt_np)]:
        p = os.path.join(out_dir, f"{subject_id}_{label}.png")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr[ax_sl], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    for name, pred_np in preds:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        # Prediction
        p = os.path.join(out_dir, f"{subject_id}_pred_{safe}.png")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(pred_np[ax_sl], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Absolute difference
        diff_sl = np.abs(pred_np[ax_sl] - gt_np[ax_sl])
        p = os.path.join(out_dir, f"{subject_id}_absdiff_{safe}.png")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(diff_sl, cmap="hot", vmin=0, vmax=0.25)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|Pred − GT|")
        plt.tight_layout(pad=0)
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Signed difference
        diff_sl_signed = pred_np[ax_sl] - gt_np[ax_sl]
        p = os.path.join(out_dir, f"{subject_id}_diff_{safe}.png")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(diff_sl_signed, cmap="bwr", vmin=-0.25, vmax=0.25)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pred − GT")
        plt.tight_layout(pad=0)
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {2 + 3 * len(preds)} individual panels → {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--subject_idx", type=int, default=None,
                        help="Single subject index (0-based in test split)")
    parser.add_argument("--n_subjects", type=int, default=1,
                        help="Run on first N test subjects (ignored if --subject_idx set)")
    parser.add_argument("--out_dir", default="vis/compare")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load test dataset
    dm = DeepFLAIRDataModule(data_dir=args.data_dir, batch_size=1, num_workers=0)
    dm.setup("test")
    test_ds = dm.test_ds
    print(f"Test subjects available: {len(test_ds)}\n")

    indices = ([args.subject_idx] if args.subject_idx is not None
               else list(range(min(args.n_subjects, len(test_ds)))))

    # Load all models once
    print("Loading models...")
    loaded = []
    for cfg in MODELS:
        if not os.path.exists(cfg["ckpt"]):
            print(f"  SKIP (ckpt not found): {cfg['name']}")
            continue
        print(f"  {cfg['name']} ← {cfg['ckpt']}")
        m = load_model(cfg, device)
        loaded.append((cfg["name"], m))
    print()

    for idx in indices:
        subject = test_ds[idx]
        epi = subject["image"].unsqueeze(0)   # (1,1,D,H,W)
        gt  = subject["label"].unsqueeze(0)
        subject_id = subject.get("subject_id", f"subject_{idx:03d}")

        epi_np = epi[0, 0].numpy()
        gt_np  = gt[0, 0].numpy()

        print(f"[{idx}] {subject_id}")

        preds = []
        for name, model in loaded:
            print(f"  Inferring {name}...")
            pred_np = infer(model, epi, device)
            preds.append((name, pred_np))
            # Save .npy for reuse
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
            np.save(os.path.join(args.out_dir, f"{subject_id}_pred_{safe}.npy"), pred_np)

        # Comparison figure
        comp_path = os.path.join(args.out_dir, f"{subject_id}_comparison.png")
        ax_sl = make_comparison_figure(subject_id, epi_np, gt_np, preds, comp_path)

        # Individual panels
        make_individual_panels(subject_id, ax_sl, epi_np, gt_np, preds, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
