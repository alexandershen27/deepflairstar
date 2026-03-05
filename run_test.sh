#!/bin/bash

echo "=========================================================="
echo "Running DeepFLAIR* Dry-Run Test..."
echo "=========================================================="
echo "This will test the entire pipeline (data loading, model forward/backward,"
echo "loss calculation, sliding window inference) for exactly 1 batch."
echo "If this passes, your environment and code are ready for the full run."
echo ""

# Run train.py with fast_dev_run which executes exactly 1 step of train, val, and test.
# We also turn down the cache_rate for the test so it doesn't take long to load.
# Replace python3 with your server's python command if using a specific conda env.

python3 train.py \
    --data_dir "data" \
    --batch_size 2 \
    --patch_size 64 \
    --base_channels 16 \
    --num_workers 4 \
    --cache_rate 0.0 \
    --fast_dev_run

echo "=========================================================="
echo "If you saw 'fast_dev_run mode was successfully enabled', you are good to go!"
echo ""
echo "To start the REAL training run on your server, use:"
echo "python3 train.py --batch_size 4 --num_workers 8 --devices auto"
echo "(If you have less than 100GB of RAM, add: --cache_rate 0.0)"
echo "=========================================================="
