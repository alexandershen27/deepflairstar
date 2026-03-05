#!/bin/bash

echo "=========================================================="
echo "Running DeepFLAIR* Short Signal Run (Max 1 Hour)..."
echo "=========================================================="
echo "This run will process 5 epochs with limited batches."
echo "Goal: Confirm that loss is decreasing and images are logging."
echo ""

# We limit train batches to 20 and val batches to 2 to get fast feedback.
# We also use base_channels=8 to make it faster for the test.

python3 train.py \
    --data_dir "data" \
    --batch_size 4 \
    --patch_size 64 \
    --base_channels 8 \
    --num_workers 8 \
    --max_epochs 5 \
    --limit_train_batches 20 \
    --limit_val_batches 2 \
    --cache_rate 0.0

echo ""
echo "=========================================================="
echo "Short run complete. Check 'logs/deepflair_tb/' in TensorBoard."
echo "If the trend is good, start the full run with:"
echo "python3 train.py --batch_size 4 --num_workers 8 --devices auto"
echo "=========================================================="
