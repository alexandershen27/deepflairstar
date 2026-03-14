#!/usr/bin/env python
"""
Plot val_loss training curves for all v2 models.

Usage (from repo root):
    python scripts/training_curves.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow

EXPERIMENTS = [
    ("U-Net (ReLU)",     "DF_unet_v2"),
    ("U-Net (Sigmoid)",  "DF_unet_sigmoid_v2"),
    ("Attention U-Net",  "DF_attention_unet_v2"),
    ("Swin-UNETR",       "DF_swin_v2"),
    ("UNETR++ (VFA)",    "DF_unetrpp_BS32"),
]

STEPS_PER_EPOCH = 32
COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]


def main():
    mlflow.set_tracking_uri("logs/mlflow")
    client = mlflow.tracking.MlflowClient()

    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, exp_name), color in zip(EXPERIMENTS, COLORS):
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"  Not found: {exp_name}")
            continue
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            continue
        history = client.get_metric_history(runs[0].info.run_id, "val_loss")
        if not history:
            continue
        history.sort(key=lambda x: x.step)
        epochs = [h.step / STEPS_PER_EPOCH for h in history]
        values = [h.value for h in history]
        best_val = min(values)
        ax.plot(epochs, values, label=f"{label}  (best={best_val:.4f})", color=color, linewidth=1.5)
        best_ep = epochs[values.index(best_val)]
        ax.axvline(best_ep, color=color, linestyle="--", linewidth=0.7, alpha=0.5)
        print(f"  {label}: {len(epochs)} epochs, best={best_val:.4f} @ ep {best_ep:.0f}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Training Curves — Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs("vis/figures", exist_ok=True)
    out = "vis/figures/training_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
