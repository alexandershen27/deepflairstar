#!/bin/bash

# Clear previous logs and cache to ensure a clean start
rm -rf logs/* outputs/* vis/*

echo "Starting DeepFLAIR* Production Training..."
echo "All output is being redirected to: train_output.log"
echo "You can monitor it with: tail -f train_output.log"

python3 train.py \
    --data_dir "data" \
    --batch_size 8 \
    --num_workers 16 \
    --devices auto \
    --max_epochs 300 \
    --cache_rate 1.0 \
    --patch_size 64 \
    --base_channels 32 \
    --lr 1e-4 > train_output.log 2>&1
