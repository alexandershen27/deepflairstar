#!/bin/bash
# Evaluate all 4 models on the test set.
# Run from repo root after training is complete.
#
# Usage: bash scripts/eval_all.sh [last|best]
#
# "last" uses last.ckpt (default)
# "best" picks the checkpoint with the lowest val_loss in the filename

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python
MODE=${1:-last}

pick_ckpt() {
    local dir=$1
    if [ "$MODE" = "best" ]; then
        # Sort by val_loss float in filename, pick the lowest
        ls "$dir"/deepflair-epoch=*.ckpt 2>/dev/null \
            | grep -oP 'val_loss=\K[0-9.]+' \
            | paste - <(ls "$dir"/deepflair-epoch=*.ckpt 2>/dev/null) \
            | sort -n | head -1 | awk '{print $2}'
    else
        echo "$dir/last.ckpt"
    fi
}

UNET_CKPT=$(pick_ckpt "$REPO_DIR/outputs/DF_unet_BS16/checkpoints")
SWIN_CKPT=$(pick_ckpt "$REPO_DIR/outputs/DF_swin_BS2/checkpoints")
ATTN_CKPT=$(pick_ckpt "$REPO_DIR/outputs/DF_attention_unet_BS16/checkpoints")
UNETRPP_CKPT=$(pick_ckpt "$REPO_DIR/outputs/DF_unetrpp_BS32/checkpoints")

echo "=== Evaluating with mode: $MODE ==="
echo "  unet        → $UNET_CKPT"
echo "  swin        → $SWIN_CKPT"
echo "  attn_unet   → $ATTN_CKPT"
echo "  unetrpp     → $UNETRPP_CKPT"
echo ""

cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/infer.py \
    --ckpt_path "$UNET_CKPT" \
    --output_dir outputs/eval &

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/infer.py \
    --ckpt_path "$SWIN_CKPT" \
    --output_dir outputs/eval &

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/infer.py \
    --ckpt_path "$ATTN_CKPT" \
    --output_dir outputs/eval &

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/infer.py \
    --ckpt_path "$UNETRPP_CKPT" \
    --output_dir outputs/eval &

wait
echo ""
echo "=== All evaluations complete. Results in outputs/eval/ ==="
echo ""

# Print combined summary table
$PYTHON - <<'EOF'
import csv, os, numpy as np

models = ["unet", "swin", "attention_unet", "unetrpp"]
eval_dir = "outputs/eval"

print(f"{'Model':<16}  {'MSE':>10}  {'SSIM':>10}  {'PSNR (dB)':>12}")
print("-" * 55)
for m in models:
    path = os.path.join(eval_dir, f"{m}_metrics.csv")
    if not os.path.exists(path):
        print(f"{m:<16}  (no results)")
        continue
    rows = list(csv.DictReader(open(path)))
    mse  = np.array([float(r["mse"])  for r in rows])
    ssim = np.array([float(r["ssim"]) for r in rows])
    psnr = np.array([float(r["psnr"]) for r in rows])
    print(f"{m:<16}  {mse.mean():.5f}±{mse.std():.5f}  "
          f"{ssim.mean():.4f}±{ssim.std():.4f}  "
          f"{psnr.mean():.2f}±{psnr.std():.2f}")
EOF
