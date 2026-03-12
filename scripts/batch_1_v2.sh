#!/bin/bash
# =============================================================================
# BATCH 1 v2 — clean run with fixed preprocessing (no CropForeground bug)
#
# GPU 0: UNet       — ReLU final,    BS=16, equal loss weights, baseline
# GPU 1: Swin       — ReLU final,    BS=2,  equal loss weights
# GPU 2: AttnUNet   — ReLU final,    BS=16, equal loss weights
# GPU 3: UNet       — Sigmoid final, BS=16, equal loss weights (activation ablation)
#
# All runs:
#   lr=1e-4, beta1=0.9, beta2=0.999
#   mse=1.0, ssim=1.0, grad=1.0
#   patch_size=64, padding=(320,384,320)
#   aug: 50% flip (axes 0,1), 33%/33%/33% no/weak/medium Gaussian blur
#   data split: random_state=42, test=0.2, val=0.1 of remainder
#   steps/epoch: all normalized to 128 (see repeat_dataset)
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python
MLFLOW=/home/shena2/miniconda3/envs/swiflair/bin/mlflow

# Kill any leftover sessions from previous batches
for s in unet_eq swin_eq attn_eq swin_opt unet_opt attn_optw unet_optw swin_optw mlflow; do
    tmux kill-session -t $s 2>/dev/null || true
done
sleep 2

# MLflow UI
tmux new-session -d -s mlflow
tmux send-keys -t mlflow \
    "$MLFLOW ui --backend-store-uri $REPO_DIR/logs/mlflow --host 0.0.0.0 --port 5000" Enter

# GPU 0 — UNet, ReLU (baseline)
# 64 subjects × 8 repeat × 4 patches / BS 16 = 128 steps/epoch
tmux new-session -d -s unet_relu
tmux send-keys -t unet_relu "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.train \
    --model_type unet \
    --base_channels 16 \
    --batch_size 16 \
    --repeat_dataset 8 \
    --max_epochs 300 \
    --num_workers 4 \
    --experiment_name DF_unet_v2" Enter

# GPU 1 — Swin, ReLU, instance norm
# 64 subjects × 1 repeat × 4 patches / BS 2 = 128 steps/epoch
tmux new-session -d -s swin_relu
tmux send-keys -t swin_relu "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=1 $PYTHON -m src.train \
    --model_type swin \
    --base_channels 24 \
    --batch_size 2 \
    --repeat_dataset 1 \
    --max_epochs 300 \
    --num_workers 4 \
    --experiment_name DF_swin_v2" Enter

# GPU 2 — Attention UNet, ReLU
# 64 subjects × 8 repeat × 4 patches / BS 16 = 128 steps/epoch
tmux new-session -d -s attn_relu
tmux send-keys -t attn_relu "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=2 $PYTHON -m src.train \
    --model_type attention_unet \
    --base_channels 16 \
    --batch_size 16 \
    --repeat_dataset 8 \
    --max_epochs 300 \
    --num_workers 4 \
    --experiment_name DF_attention_unet_v2" Enter

# GPU 3 — UNet, Sigmoid (activation ablation vs GPU 0)
# same config as GPU 0, only --activation sigmoid differs
tmux new-session -d -s unet_sig
tmux send-keys -t unet_sig "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=3 $PYTHON -m src.train \
    --model_type unet \
    --base_channels 16 \
    --batch_size 16 \
    --repeat_dataset 8 \
    --activation sigmoid \
    --max_epochs 300 \
    --num_workers 4 \
    --experiment_name DF_unet_sigmoid_v2" Enter

echo ""
echo "=== Batch 1 v2 launched ==="
echo "  GPU 0: unet_relu   (tmux attach -t unet_relu)"
echo "  GPU 1: swin_relu   (tmux attach -t swin_relu)"
echo "  GPU 2: attn_relu   (tmux attach -t attn_relu)"
echo "  GPU 3: unet_sig    (tmux attach -t unet_sig)"
echo ""
echo "MLflow: tmux attach -t mlflow  (port 5000)"
