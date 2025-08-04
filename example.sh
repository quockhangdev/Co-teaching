# === Bash Run Script ===
# Save this as run_cxr.sh and make executable (chmod +x run_cxr.sh)
#!/bin/bash

# Enable memory optimization for PyTorch (recommended for large models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set visible GPU (optional: only if using multi-GPU setups)
export CUDA_VISIBLE_DEVICES=0

# Define arguments
DATASET="cxr"
# NOISE_TYPE="symmetric"
NOISE_TYPE="pairflip"
NOISE_RATE=0.2
BATCH_SIZE=16
LR=1e-4
EPOCHS=200
WORKERS=16
RESULT_DIR="results"
NUM_ITER_PER_EPOCH=1000

# Run the training script with AMP
python main.py \
    --noise_type $NOISE_TYPE \
    --noise_rate $NOISE_RATE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --n_epoch $EPOCHS \
    --num_workers $WORKERS \
    --result_dir $RESULT_DIR \
    --num_iter_per_epoch $NUM_ITER_PER_EPOCH