#!/bin/bash
# Launch 3 architectures on separate GPUs via tmux.
# Run from the repo root on titan2 after: git pull && conda activate <env>
#
# Usage: bash scripts/launch_all_gpus.sh

PYTHON=/home/shena2/miniconda3/envs/swiflair/bin/python

SESSION_UNET="unet"
SESSION_SWIN="swin"
SESSION_ATTN="attention_unet"

# Kill any existing sessions with these names
for s in $SESSION_UNET $SESSION_SWIN $SESSION_ATTN; do
    tmux kill-session -t $s 2>/dev/null || true
done

# GPU 0 — CNN U-Net (batch 16, repeat x8 → ~32 iters/epoch)
tmux new-session -d -s $SESSION_UNET
tmux send-keys -t $SESSION_UNET \
    "CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    --model_type unet \
    --batch_size 16 \
    --base_channels 16 \
    --max_epochs 75 \
    --repeat_dataset 8 \
    --experiment_name DF_unet_BS16 \
    --num_workers 4" Enter

# GPU 1 — Swin-UNETR (batch 2, already ~32 iters/epoch naturally)
tmux new-session -d -s $SESSION_SWIN
tmux send-keys -t $SESSION_SWIN \
    "CUDA_VISIBLE_DEVICES=1 $PYTHON train.py \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 75 \
    --experiment_name DF_swin_BS2 \
    --num_workers 4" Enter

# GPU 2 — Attention U-Net (batch 8, repeat x4 → ~32 iters/epoch)
tmux new-session -d -s $SESSION_ATTN
tmux send-keys -t $SESSION_ATTN \
    "CUDA_VISIBLE_DEVICES=2 $PYTHON train.py \
    --model_type attention_unet \
    --batch_size 8 \
    --base_channels 16 \
    --max_epochs 75 \
    --repeat_dataset 4 \
    --experiment_name DF_attention_unet_BS8 \
    --num_workers 4" Enter

echo "3 sessions launched. Monitor with:"
echo "  tmux attach -t unet"
echo "  tmux attach -t swin"
echo "  tmux attach -t attention_unet"
echo ""
echo "GPU 3 is free for UNETR++ when ready."
