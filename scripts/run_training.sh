#!/bin/bash

# Optimized training script for 4x NVIDIA T4 GPUs with All of Us data
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

# T4-optimized hyperparameters (16GB VRAM per GPU)
BATCH_SIZE=32          # Conservative for T4 memory
N_POSITIONS=1024       # Good for one year of data
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.1
LR=0.001
MIN_LR=0.00001
WARMUP_ITERS=1000
MAX_ITERS=50000
LR_DECAY_ITERS=25000

model_name="all_of_us_1year_t4_layer_${N_LAYER}_do_${DROPOUT}"

# T4-specific configuration
NUM_GPUS=4
DEVICE="cuda"

echo "Starting T4-optimized training for All of Us data:"
echo "  Dataset: $dataset"
echo "  Data path: $data_path"
echo "  Model: $model_name"
echo "  GPUs: $NUM_GPUS x NVIDIA T4 (16GB each)"
echo "  Total VRAM: 64 GB"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Max iterations: $MAX_ITERS"
echo "  Learning rate: $LR"
echo ""

# T4-optimized training with memory management
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
    gradient_accumulation_steps=4 \
    warmup_iters=$WARMUP_ITERS \
    max_iters=$MAX_ITERS \
    lr_decay_iters=$LR_DECAY_ITERS \
    wandb_log=true \
    wandb_project="ethos-all-of-us-t4" \
    wandb_run_name=$model_name \
    $* \
    out_dir="${data_path}/models/${model_name}"
