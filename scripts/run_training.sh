#!/bin/bash

# Optimized training script for one year of All of Us data
# Run this from the project root directory

# Configuration
dataset="all_of_us"
data_path="ethos_data/train"

# Check if tokenized data exists
if [[ ! -d $data_path ]]; then
    echo "Dataset directory not found: $data_path"
    echo "Please run tokenization first: bash scripts/run_tokenization.sh"
    exit 1
fi

# Optimized hyperparameters for one year of All of Us data
BATCH_SIZE=64          # Increased from 32 for better GPU utilization
N_POSITIONS=1024       # Reduced from 2048 for one year of data
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.1            # Reduced from 0.3 for smaller dataset
LR=0.001               # Increased from 0.0006 for faster convergence
MIN_LR=0.00001
WARMUP_ITERS=1000      # Reduced from 5000 for smaller dataset
MAX_ITERS=50000        # Reduced from 200000 for one year of data
LR_DECAY_ITERS=25000   # Reduced from 100000

model_name="all_of_us_1year_layer_${N_LAYER}_do_${DROPOUT}"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ $NUM_GPUS -eq 0 ]]; then
    echo "No GPUs detected, using CPU"
    DEVICE="cpu"
    NUM_GPUS=1
    # Reduce batch size for CPU
    BATCH_SIZE=16
else
    echo "Detected $NUM_GPUS GPU(s)"
    DEVICE="cuda"
fi

echo "Starting optimized training for one year of All of Us data:"
echo "  Dataset: $dataset"
echo "  Data path: $data_path"
echo "  Model: $model_name"
echo "  GPUs: $NUM_GPUS"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Max iterations: $MAX_ITERS"
echo "  Learning rate: $LR"
echo ""

# Run training
if [[ $DEVICE == "cuda" ]]; then
    # Multi-GPU training with optimized settings
    torchrun --no_python --standalone --nproc_per_node=$NUM_GPUS ethos_train \
        data_fp=$data_path \
        val_size=10 \
        batch_size=$BATCH_SIZE \
        n_positions=$N_POSITIONS \
        n_layer=$N_LAYER \
        n_head=$N_HEAD \
        n_embd=$N_EMBD \
        dropout=$DROPOUT \
        lr=$LR \
        min_lr=$MIN_LR \
        log_interval=50 \
        eval_interval=500 \
        gradient_accumulation_steps=8 \
        warmup_iters=$WARMUP_ITERS \
        max_iters=$MAX_ITERS \
        lr_decay_iters=$LR_DECAY_ITERS \
        wandb_log=true \
        wandb_project="ethos-all-of-us-1year" \
        wandb_run_name=$model_name \
        $* \
        out_dir="${data_path}/models/${model_name}"
else
    # CPU training with reduced settings
    ethos_train \
        data_fp=$data_path \
        val_size=10 \
        batch_size=$BATCH_SIZE \
        n_positions=$N_POSITIONS \
        n_layer=$N_LAYER \
        n_head=$N_HEAD \
        n_embd=$N_EMBD \
        dropout=$DROPOUT \
        lr=$LR \
        min_lr=$MIN_LR \
        log_interval=50 \
        eval_interval=500 \
        gradient_accumulation_steps=8 \
        warmup_iters=$WARMUP_ITERS \
        max_iters=$MAX_ITERS \
        lr_decay_iters=$LR_DECAY_ITERS \
        wandb_log=true \
        wandb_project="ethos-all-of-us-1year" \
        wandb_run_name=$model_name \
        device=cpu \
        $* \
        out_dir="${data_path}/models/${model_name}"
fi
