#!/bin/bash

# Clean up function to remove stale lock files
cleanup_stale_locks() {
    echo "Cleaning up stale lock files..."
    find . -name "*.json" -path "*/.data_*.parquet_cache/locks/*" -mtime +1 -delete 2>/dev/null || true
    echo "Cleanup completed."
}

# Set directories
input_dir="/home/jupyter/workspaces/ehrtransformerbaseline/meds_data/data"
output_dir="/home/jupyter/workspaces/ehrtransformerbaseline/ethos_data"

# Clean up stale locks before starting
cleanup_stale_locks

# Run tokenization
ethos_tokenize -m worker='range(0,8)' \
    input_dir=$input_dir \
    output_dir=$output_dir \
    out_fn=train \
    resume=false \
    max_workers=8 \
    memory_limit_gb=10 \
    chunk_size=4 \
    use_memory_optimized=false
