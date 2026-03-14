#!/usr/bin/env python
"""
Generate publication figures from saved .npy predictions.

Figures generated per subject per view:
  1. gt_pred_diff  — 3 rows (GT / Pred / Diff) × 4 cols (one per architecture)
  2. gt_pred       — 2 rows (GT / Pred)         × 4 cols  [cleaner for paper]

Views: axial (top-down) or coronal (front-back).
Difference: signed Pred−GT, bwr colormap (red=over, blue=under).

Usage:
    python scripts/grid_figure.py --npy_dir vis/compare --data_dir data --all
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

DIFF_CLIM = 0.25   # symmetric colorbar limit for difference maps


def foreground_centroid(vol, thresh=0.03):
    nz = np.argwhere(vol > thresh)
    return nz.mean(axis=0).astype(int).tolist() if len(nz) > 0 else [s // 2 for s in vol.shape]


def get_slice(vol, c, view):
    if view == "axial":
        return vol[c[0], :, :]
    elif view == "coronal":
        return vol[:, c[1], :]
    else:
        return vol[:, :, c[2]]


def make_gt_pred_diff(subject_id, gt_np, preds, out_path, view):
    """3-row × 4-col: GT / Pred / Diff(bwr)"""
    n = len(preds)
    c = foreground_centroid(gt_np)
    sl_idx = {"axial": c[0], "coronal": c[1], "sagittal": c[2]}[view]
    gt_sl = get_slice(gt_np, c, view)

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})

    diff_im = None
    for col, (name, pred_np) in enumerate(preds):
        pred_sl = get_slice(pred_np, c, view)
        diff_sl = pred_sl - gt_sl

        axes[0, col].imshow(gt_sl,   cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
        diff_im = axes[2, col].imshow(diff_sl, cmap="bwr",
                                      vmin=-DIFF_CLIM, vmax=DIFF_CLIM)

        axes[0, col].set_title(name, fontsize=11, fontweight="bold", pad=5)
        for row in range(3):
            axes[row, col].axis("off")

    for row, label in enumerate(["GT FLAIR*", "Prediction", "Pred − GT"]):
        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=6)

    if diff_im is not None:
        fig.colorbar(diff_im, ax=axes[2, :], shrink=0.6, pad=0.01,
                     orientation="horizontal", label="Pred − GT  (red=over, blue=under)")

    fig.suptitle(f"{subject_id}  ·  {view}  slice {sl_idx}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def make_gt_pred(subject_id, gt_np, preds, out_path, view):
    """2-row × 4-col: GT / Pred  (clean, no diff)"""
    n = len(preds)
    c = foreground_centroid(gt_np)
    sl_idx = {"axial": c[0], "coronal": c[1], "sagittal": c[2]}[view]
    gt_sl = get_slice(gt_np, c, view)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})

    for col, (name, pred_np) in enumerate(preds):
        pred_sl = get_slice(pred_np, c, view)
        axes[0, col].imshow(gt_sl,   cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(name, fontsize=11, fontweight="bold", pad=5)
        for row in range(2):
            axes[row, col].axis("off")

    for row, label in enumerate(["GT FLAIR*", "Prediction"]):
        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=6)

    fig.suptitle(f"{subject_id}  ·  {view}  slice {sl_idx}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def make_diff_only(subject_id, gt_np, preds, out_path, view):
    """1-row × 4-col: difference maps only (absolute, hot colormap)"""
    n = len(preds)
    c = foreground_centroid(gt_np)
    sl_idx = {"axial": c[0], "coronal": c[1], "sagittal": c[2]}[view]
    gt_sl = get_slice(gt_np, c, view)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5),
                             gridspec_kw={"wspace": 0.04})

    last_im = None
    for col, (name, pred_np) in enumerate(preds):
        diff_sl = np.abs(get_slice(pred_np, c, view) - gt_sl)
        last_im = axes[col].imshow(diff_sl, cmap="hot", vmin=0, vmax=DIFF_CLIM)
        axes[col].set_title(name, fontsize=11, fontweight="bold", pad=5)
        axes[col].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, pad=0.01,
                     orientation="vertical", label="|Pred − GT|")

    fig.suptitle(f"{subject_id}  ·  {view}  slice {sl_idx}  ·  Absolute Error",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run(subject_id, npy_dir, data_dir, out_dir):
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=1, num_workers=0)
    dm.setup("test")
    test_ds = dm.test_ds

    subject = None
    for i in range(len(test_ds)):
        s = test_ds[i]
        if s.get("subject_id", f"subject_{i:03d}") == subject_id:
            subject = s
            break
    if subject is None:
        raise ValueError(f"Subject {subject_id} not found in test split")

    gt_np = subject["label"].numpy()[0]

    preds = []
    for name, npy_key in MODELS:
        npy_path = os.path.join(npy_dir, f"{subject_id}_pred_{npy_key}.npy")
        if not os.path.exists(npy_path):
            print(f"  MISSING: {npy_path}")
            continue
        preds.append((name, np.load(npy_path)))

    if not preds:
        print(f"No predictions found for {subject_id}")
        return

    os.makedirs(out_dir, exist_ok=True)
    sid = subject_id

    for view in ["axial", "coronal", "sagittal"]:
        make_gt_pred_diff(sid, gt_np, preds,
                          os.path.join(out_dir, f"{sid}_{view}_gt_pred_diff.png"), view)
        make_gt_pred(sid, gt_np, preds,
                     os.path.join(out_dir, f"{sid}_{view}_gt_pred.png"), view)
        make_diff_only(sid, gt_np, preds,
                       os.path.join(out_dir, f"{sid}_{view}_abserr.png"), view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str, default=None)
    parser.add_argument("--npy_dir",    default="vis/compare")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--out_dir",    default="vis/figures")
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    if args.all:
        npy_files = [f for f in os.listdir(args.npy_dir) if f.endswith(".npy")]
        subject_ids = sorted(set(f.split("_pred_")[0] for f in npy_files))
        print(f"Found {len(subject_ids)} subjects: {subject_ids}")
        for sid in subject_ids:
            print(f"\n[{sid}]")
            run(sid, args.npy_dir, args.data_dir, args.out_dir)
    elif args.subject_id:
        run(args.subject_id, args.npy_dir, args.data_dir, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
