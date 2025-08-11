#!/bin/bash

# Run this from the project root directory

# Configuration
dataset="all_of_us"  # Updated to match your dataset
data_path="ethos_data/train"  # Updated to match your output directory

# Check if tokenized data exists
if [[ ! -d $data_path ]]; then
    echo "Dataset directory not found: $data_path"
    echo "Please run tokenization first: bash scripts/run_tokenization.sh"
    exit 1
fi

# Model hyperparameters
BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001

model_name="layer_${N_LAYER}_do_${DROPOUT}"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ $NUM_GPUS -eq 0 ]]; then
    echo "No GPUs detected, using CPU"
    DEVICE="cpu"
    NUM_GPUS=1
else
    echo "Detected $NUM_GPUS GPU(s)"
    DEVICE="cuda"
fi

echo "Starting training with:"
echo "  Dataset: $dataset"
echo "  Data path: $data_path"
echo "  Model: $model_name"
echo "  GPUs: $NUM_GPUS"
echo "  Device: $DEVICE"
echo ""

# Run training
if [[ $DEVICE == "cuda" ]]; then
    # Multi-GPU training
    torchrun --no_python --standalone --nproc_per_node=$NUM_GPUS ethos_train \
        data_fp=$data_path \
        val_size=6 \
        batch_size=$BATCH_SIZE \
        n_positions=$N_POSITIONS \
        n_layer=$N_LAYER \
        n_head=$N_HEAD \
        n_embd=$N_EMBD \
        dropout=$DROPOUT \
        lr=$LR \
        min_lr=$MIN_LR \
        log_interval=10 \
        eval_interval=1500 \
        gradient_accumulation_steps=16 \
        warmup_iters=5000 \
        max_iters=200000 \
        lr_decay_iters=100000 \
        wandb_log=true \
        wandb_project="ethos-meds-$dataset" \
        wandb_run_name=$model_name \
        $* \
        out_dir="${data_path}/models/${model_name}"
else
    # CPU training
    ethos_train \
        data_fp=$data_path \
        val_size=6 \
        batch_size=$BATCH_SIZE \
        n_positions=$N_POSITIONS \
        n_layer=$N_LAYER \
        n_head=$N_HEAD \
        n_embd=$N_EMBD \
        dropout=$DROPOUT \
        lr=$LR \
        min_lr=$MIN_LR \
        log_interval=10 \
        eval_interval=1500 \
        gradient_accumulation_steps=16 \
        warmup_iters=5000 \
        max_iters=200000 \
        lr_decay_iters=100000 \
        wandb_log=true \
        wandb_project="ethos-meds-$dataset" \
        wandb_run_name=$model_name \
        device=cpu \
        $* \
        out_dir="${data_path}/models/${model_name}"
fi
