#!/bin/bash
# Launch all 4 architectures on separate GPUs via tmux.
# Run from the repo root on titan2 after: git pull && conda activate <env>
#
# Usage: bash scripts/launch_all_gpus.sh

SESSION_UNET="unet"
SESSION_SWIN="swin"
SESSION_ATTN="attention_unet"
SESSION_UNETR="unetr"

# Kill any existing sessions with these names
for s in $SESSION_UNET $SESSION_SWIN $SESSION_ATTN $SESSION_UNETR; do
    tmux kill-session -t $s 2>/dev/null || true
done

# GPU 0 — CNN U-Net (batch 8, base_channels=16 per paper)
tmux new-session -d -s $SESSION_UNET
tmux send-keys -t $SESSION_UNET \
    "CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type unet \
    --batch_size 8 \
    --base_channels 16 \
    --max_epochs 75 \
    --experiment_name DF_unet_BS8 \
    --num_workers 4" Enter

# GPU 1 — Swin-UNETR (batch 2, memory constrained)
tmux new-session -d -s $SESSION_SWIN
tmux send-keys -t $SESSION_SWIN \
    "CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_type swin \
    --batch_size 2 \
    --base_channels 24 \
    --max_epochs 75 \
    --experiment_name DF_swin_BS2 \
    --num_workers 4" Enter

# GPU 2 — Attention U-Net (batch 4, attention maps add some memory)
tmux new-session -d -s $SESSION_ATTN
tmux send-keys -t $SESSION_ATTN \
    "CUDA_VISIBLE_DEVICES=2 python train.py \
    --model_type attention_unet \
    --batch_size 4 \
    --base_channels 16 \
    --max_epochs 75 \
    --experiment_name DF_attention_unet_BS4 \
    --num_workers 4" Enter

# GPU 3 — UNETR (batch 2, ViT encoder is memory-heavy)
tmux new-session -d -s $SESSION_UNETR
tmux send-keys -t $SESSION_UNETR \
    "CUDA_VISIBLE_DEVICES=3 python train.py \
    --model_type unetr \
    --batch_size 2 \
    --base_channels 16 \
    --max_epochs 75 \
    --experiment_name DF_unetr_BS2 \
    --num_workers 4" Enter

echo "All 4 sessions launched. Monitor with:"
echo "  tmux attach -t unet"
echo "  tmux attach -t swin"
echo "  tmux attach -t attention_unet"
echo "  tmux attach -t unetr"
