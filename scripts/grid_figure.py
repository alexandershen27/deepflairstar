#!/usr/bin/env python
"""
Generate a 3-row × 4-col publication figure from saved .npy predictions.

Layout:
  Row 0 (Input EPI):  col0  col1  col2  col3
  Row 1 (GT FLAIR*):  col0  col1  col2  col3
  Row 2 (Prediction): UNet  Attn  Swin  UNETR++

All from the same subject and axial slice.

Usage:
    python scripts/grid_figure.py --subject_id 05_004 --npy_dir vis/compare --data_dir data
    python scripts/grid_figure.py --subject_id 05_004 --npy_dir vis/compare --data_dir data --all
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DeepFLAIRDataModule

MODELS = [
    ("U-Net",           "U-Net_ReLU"),
    ("Attention U-Net", "Attention_U-Net"),
    ("Swin-UNETR",      "Swin-UNETR"),
    ("UNETR++ (VFA)",   "UNETRpp_VFA"),
]


def foreground_centroid(vol, thresh=0.03):
    nz = np.argwhere(vol > thresh)
    return nz.mean(axis=0).astype(int).tolist() if len(nz) > 0 else [s // 2 for s in vol.shape]


def get_slice(vol, c, view):
    if view == "axial":
        return vol[c[0], :, :]
    elif view == "coronal":
        return vol[:, c[1], :]
    else:  # sagittal
        return vol[:, :, c[2]]


def make_grid(subject_id, epi_np, gt_np, preds, out_path, view="coronal"):
    """
    3-row × 4-col figure.
    Row 0: EPI repeated across columns (with architecture label as col title)
    Row 1: GT repeated across columns
    Row 2: one prediction per column
    """
    n = len(preds)
    c = foreground_centroid(gt_np)
    sl_idx = {"axial": c[0], "coronal": c[1], "sagittal": c[2]}[view]

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})

    row_labels = ["Input EPI", "GT FLAIR*", "Prediction"]
    sources = [epi_np, gt_np, None]

    for col, (name, pred_np) in enumerate(preds):
        for row in range(3):
            ax = axes[row, col]
            arr = sources[row] if sources[row] is not None else pred_np
            ax.imshow(get_slice(arr, c, view), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=11, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10, labelpad=6)

    fig.suptitle(f"Architecture Comparison — {subject_id}  ({view} slice {sl_idx})",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def run(subject_id, npy_dir, data_dir, out_dir, args):
    # Load GT and EPI from dataset
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=1, num_workers=0)
    dm.setup("test")
    test_ds = dm.test_ds

    subject = None
    for i in range(len(test_ds)):
        s = test_ds[i]
        sid = s.get("subject_id", f"subject_{i:03d}")
        if sid == subject_id:
            subject = s
            break
    if subject is None:
        raise ValueError(f"Subject {subject_id} not found in test split")

    epi_np = subject["image"].numpy()[0]
    gt_np  = subject["label"].numpy()[0]

    preds = []
    for name, npy_key in MODELS:
        npy_path = os.path.join(npy_dir, f"{subject_id}_pred_{npy_key}.npy")
        if not os.path.exists(npy_path):
            print(f"  MISSING: {npy_path}")
            continue
        pred_np = np.load(npy_path)
        preds.append((name, pred_np))

    if not preds:
        print(f"No predictions found for {subject_id}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject_id}_grid_{args.view}.png")
    make_grid(subject_id, epi_np, gt_np, preds, out_path, view=args.view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str, default=None,
                        help="Subject ID (e.g. 05_004)")
    parser.add_argument("--npy_dir",   default="vis/compare")
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--out_dir",   default="vis/figures")
    parser.add_argument("--view",      default="coronal", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--all",       action="store_true",
                        help="Generate figures for all subjects with saved predictions")
    args = parser.parse_args()

    if args.all:
        # Find all unique subject IDs from .npy files
        npy_files = [f for f in os.listdir(args.npy_dir) if f.endswith(".npy")]
        subject_ids = sorted(set(f.split("_pred_")[0] for f in npy_files))
        print(f"Found {len(subject_ids)} subjects: {subject_ids}")
        for sid in subject_ids:
            run(sid, args.npy_dir, args.data_dir, args.out_dir, args)
    elif args.subject_id:
        run(args.subject_id, args.npy_dir, args.data_dir, args.out_dir, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
