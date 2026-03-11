#!/bin/bash
# Launch all 4 architectures on separate GPUs via tmux.
# Can be run from any directory.
#
# Usage: bash scripts/launch_all_gpus.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python
MLFLOW=/home/shena2/miniconda3/envs/swiflair/bin/mlflow

SESSION_UNET="unet"
SESSION_SWIN="swin"
SESSION_ATTN="attention_unet"
SESSION_UNETRPP="unetrpp"
SESSION_MLFLOW="mlflow"

# Kill any zombie training processes first
pkill -f "python train.py" 2>/dev/null || true
sleep 2

# Kill any existing tmux sessions with these names
for s in $SESSION_UNET $SESSION_SWIN $SESSION_ATTN $SESSION_UNETRPP $SESSION_MLFLOW; do
    tmux kill-session -t $s 2>/dev/null || true
done

# MLflow UI
tmux new-session -d -s $SESSION_MLFLOW
tmux send-keys -t $SESSION_MLFLOW \
    "$MLFLOW ui --backend-store-uri $REPO_DIR/logs/mlflow --host 0.0.0.0 --port 5000" Enter

# GPU 0 — CNN U-Net (BS=16, repeat=8 → 32 iters/epoch)
tmux new-session -d -s $SESSION_UNET
tmux send-keys -t $SESSION_UNET \
    "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    --model_type unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --repeat_dataset 8 \
    --experiment_name DF_unet_BS16 \
    --num_workers 4" Enter

# GPU 1 — Swin-UNETR (BS=2, repeat=1 → 32 iters/epoch naturally)
tmux new-session -d -s $SESSION_SWIN
tmux send-keys -t $SESSION_SWIN \
    "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=1 $PYTHON train.py \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 300 \
    --experiment_name DF_swin_BS2 \
    --num_workers 4" Enter

# GPU 2 — Attention U-Net (BS=16, repeat=8 → 32 iters/epoch)
tmux new-session -d -s $SESSION_ATTN
tmux send-keys -t $SESSION_ATTN \
    "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=2 $PYTHON train.py \
    --model_type attention_unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 300 \
    --repeat_dataset 8 \
    --experiment_name DF_attention_unet_BS16 \
    --num_workers 4" Enter

# GPU 3 — UNETR++ with VFA (BS=32, repeat=16 → 32 iters/epoch, ~1.2M params)
tmux new-session -d -s $SESSION_UNETRPP
tmux send-keys -t $SESSION_UNETRPP \
    "cd $REPO_DIR && CUDA_VISIBLE_DEVICES=3 $PYTHON train.py \
    --model_type unetrpp \
    --batch_size 32 \
    --base_channels 32 \
    --max_epochs 300 \
    --repeat_dataset 16 \
    --experiment_name DF_unetrpp_BS32 \
    --num_workers 4" Enter

echo "All 4 sessions launched. Monitor with:"
echo "  tmux attach -t mlflow"
echo "  tmux attach -t unet"
echo "  tmux attach -t swin"
echo "  tmux attach -t attention_unet"
echo "  tmux attach -t unetrpp"
