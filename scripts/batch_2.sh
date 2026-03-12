#!/bin/bash
# =============================================================================
# BATCH 2 — Run this after batch_1.sh finishes.
# Kills batch 1 sessions and starts 4 new ones.
#
# GPU 0: UNet       — full optuna,         resume epoch 24  (val_loss=0.0180)
# GPU 1: AttnUNet   — optuna weights only, fresh start      (safe lr=1e-4)
# GPU 2: UNet       — optuna weights only, resume epoch 88  (val_loss=0.0787)
# GPU 3: Swin       — optuna weights only, resume epoch 151 (val_loss=0.0674)
#
# Ablation design:
#   UNet_equal     vs  UNet_optweights  →  effect of loss rebalancing alone
#   UNet_optweights vs  UNet_optuna     →  effect of lr/beta change on top
#   Same story for Swin.
#   AttnUNet_optweights: does heavy MSE weighting help without the risky lr?
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python
MLFLOW=/home/shena2/miniconda3/envs/swiflair/bin/mlflow

# Kill batch 1 sessions
for s in unet_eq swin_eq attn_eq swin_opt mlflow; do
    tmux kill-session -t $s 2>/dev/null || true
done
sleep 2

# MLflow UI
tmux new-session -d -s mlflow
tmux send-keys -t mlflow \
    "$MLFLOW ui --backend-store-uri $REPO_DIR/logs/mlflow --host 0.0.0.0 --port 5000" Enter

# GPU 0 — UNet, full optuna (lr=0.00028, beta1=0.468, beta2=0.94, mse=1.5, ssim=0.2, grad=0.15)
# Resume from best optuna-run checkpoint (epoch 24)
tmux new-session -d -s unet_opt
tmux send-keys -t unet_opt "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.train \
    --model_type unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --preset optuna \
    --repeat_dataset 8 \
    --num_workers 4 \
    --experiment_name DF_unet_optuna \
    --ckpt_path $REPO_DIR/outputs/DF_unet_BS16/checkpoints/deepflair-epoch=024-val_loss=0.0180.ckpt" Enter

# GPU 1 — Attention U-Net, optuna weights only (lr=1e-4 — conservative, weights=1.5/0.2/0.15)
# Fresh start: no good stable checkpoint exists for attn_unet + optuna settings
tmux new-session -d -s attn_optw
tmux send-keys -t attn_optw "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=1 $PYTHON -m src.train \
    --model_type attention_unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.5 --ssim_weight 0.2 --grad_weight 0.15 \
    --repeat_dataset 8 \
    --num_workers 4 \
    --experiment_name DF_attention_unet_optweights" Enter

# GPU 2 — UNet, optuna weights only (lr=1e-4, mse=1.5, ssim=0.2, grad=0.15)
# Resume from best equal-weights checkpoint (epoch 88).
# Ablation: same model, same lr — only the loss rebalancing changes.
tmux new-session -d -s unet_optw
tmux send-keys -t unet_optw "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=2 $PYTHON -m src.train \
    --model_type unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.5 --ssim_weight 0.2 --grad_weight 0.15 \
    --repeat_dataset 8 \
    --num_workers 4 \
    --experiment_name DF_unet_optweights \
    --ckpt_path $REPO_DIR/outputs/DF_unet_BS16/checkpoints/deepflair-epoch=088-val_loss=0.0787.ckpt" Enter

# GPU 3 — Swin, optuna weights only (lr=1e-4, mse=1.5, ssim=0.2, grad=0.15)
# Resume from best equal-weights checkpoint (epoch 151).
# Ablation: same model, same lr — only the loss rebalancing changes.
tmux new-session -d -s swin_optw
tmux send-keys -t swin_optw "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=3 $PYTHON -m src.train \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 300 \
    --lr 1e-4 --beta1 0.5 --beta2 0.999 \
    --mse_weight 1.5 --ssim_weight 0.2 --grad_weight 0.15 \
    --num_workers 4 \
    --experiment_name DF_swin_optweights \
    --ckpt_path $REPO_DIR/outputs/DF_swin_BS2/checkpoints/deepflair-epoch=151-val_loss=0.0674.ckpt" Enter

echo ""
echo "=== Batch 2 launched ==="
echo "  GPU 0: unet_opt      (tmux attach -t unet_opt)"
echo "  GPU 1: attn_optw     (tmux attach -t attn_optw)"
echo "  GPU 2: unet_optw     (tmux attach -t unet_optw)"
echo "  GPU 3: swin_optw     (tmux attach -t swin_optw)"
