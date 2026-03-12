#!/bin/bash
# =============================================================================
# BATCH 1 — Run this first.
# Kills any current training sessions and starts 4 new ones.
#
# GPU 0: Swin       — equal weights,  resume epoch 151 (val_loss=0.0674)
# GPU 1: UNet       — equal weights,  resume epoch 88  (val_loss=0.0787)
# GPU 2: AttnUNet   — equal weights,  resume epoch 105 (val_loss=0.0958)
# GPU 3: Swin       — full optuna,    resume epoch 38  (val_loss=0.0188)
#
# When all 4 finish, run: bash scripts/batch_2.sh
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python
MLFLOW=/home/shena2/miniconda3/envs/swiflair/bin/mlflow

# Kill any existing training sessions
for s in unet swin attention_unet unetrpp mlflow \
          unet_eq swin_eq attn_eq swin_opt \
          unet_opt attn_optw unet_optw swin_optw; do
    tmux kill-session -t $s 2>/dev/null || true
done
sleep 2

# MLflow UI
tmux new-session -d -s mlflow
tmux send-keys -t mlflow \
    "$MLFLOW ui --backend-store-uri $REPO_DIR/logs/mlflow --host 0.0.0.0 --port 5000" Enter

# GPU 0 — Swin, equal weights (lr=1e-4, mse=ssim=grad=1)
# Resume from best equal-weights checkpoint (epoch 151)
tmux new-session -d -s swin_eq
tmux send-keys -t swin_eq "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.train \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.0 --ssim_weight 1.0 --grad_weight 1.0 \
    --num_workers 4 \
    --experiment_name DF_swin_equal \
    --ckpt_path $REPO_DIR/outputs/DF_swin_BS2/checkpoints/deepflair-epoch=151-val_loss=0.0674.ckpt" Enter

# GPU 1 — UNet, equal weights (lr=1e-4, mse=ssim=grad=1)
# Resume from best equal-weights checkpoint (epoch 88)
tmux new-session -d -s unet_eq
tmux send-keys -t unet_eq "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=1 $PYTHON -m src.train \
    --model_type unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.0 --ssim_weight 1.0 --grad_weight 1.0 \
    --repeat_dataset 8 \
    --num_workers 4 \
    --experiment_name DF_unet_equal \
    --ckpt_path $REPO_DIR/outputs/DF_unet_BS16/checkpoints/deepflair-epoch=088-val_loss=0.0787.ckpt" Enter

# GPU 2 — Attention U-Net, equal weights (lr=1e-4, mse=ssim=grad=1)
# Resume from best equal-weights checkpoint (epoch 105, was still improving)
tmux new-session -d -s attn_eq
tmux send-keys -t attn_eq "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=2 $PYTHON -m src.train \
    --model_type attention_unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.0 --ssim_weight 1.0 --grad_weight 1.0 \
    --repeat_dataset 8 \
    --num_workers 4 \
    --experiment_name DF_attention_unet_equal \
    --ckpt_path $REPO_DIR/outputs/DF_attention_unet_BS16/checkpoints/deepflair-epoch=105-val_loss=0.0958.ckpt" Enter

# GPU 3 — Swin, full optuna (lr=0.00028, beta1=0.468, beta2=0.94, mse=1.5, ssim=0.2, grad=0.15)
# Resume from best optuna-run checkpoint (epoch 38)
tmux new-session -d -s swin_opt
tmux send-keys -t swin_opt "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=3 $PYTHON -m src.train \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 300 \
    --preset optuna \
    --num_workers 4 \
    --experiment_name DF_swin_optuna \
    --ckpt_path $REPO_DIR/outputs/DF_swin_BS2/checkpoints/deepflair-epoch=038-val_loss=0.0188.ckpt" Enter

echo ""
echo "=== Batch 1 launched ==="
echo "  GPU 0: swin_eq       (tmux attach -t swin_eq)"
echo "  GPU 1: unet_eq       (tmux attach -t unet_eq)"
echo "  GPU 2: attn_eq       (tmux attach -t attn_eq)"
echo "  GPU 3: swin_opt      (tmux attach -t swin_opt)"
echo ""
echo "When all 4 finish, run: bash scripts/batch_2.sh"
