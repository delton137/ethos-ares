#!/bin/bash -l
#SBATCH --job-name=ethos_tokenize_hp::proj=IRB2023P002279,
#SBATCH --time=6:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=ethos_tokenize_hp.log

# High-performance tokenization script optimized for 16 cores with 200GB RAM
# Usage: ./scripts/run_tokenization_high_performance.sh [suffix] [resume] [root_dir]

# this script is intended to be run from the project root
suffix=$1
resume=${2:-true}  # Default to true for high-performance runs
root_dir=${3:-/home/jupyter/workspaces/ehrtransformerbaseline}  # Default root directory

if [[ -n "$suffix" && ! "$suffix" =~ ^[-_] ]]; then
    suffix="-$suffix"
fi

# Change to the root directory
cd "$root_dir"

input_dir="/home/jupyter/workspaces/ehrtransformerbaseline/meds_data/data"
output_dir="/home/jupyter/workspaces/ehrtransformerbaseline/ethos_data"

echo "=== HIGH-PERFORMANCE ETHOS TOKENIZATION ==="
echo "Root directory: $root_dir"
echo "System specifications:"
echo "  CPU cores: 16"
echo "  Memory: 200GB"
echo "  Input directory: $input_dir"
echo "  Output directory: $output_dir"
echo "  Resume: $resume"
echo

# Check if input directory exists
if [[ ! -d "$input_dir" ]]; then
    echo "ERROR: Input directory not found: $input_dir"
    echo "Please check the path and ensure the MEDS data is available."
    exit 1
fi

# Performance tuning for high-performance system
export OMP_NUM_THREADS=16
export POLARS_MAX_THREADS=16
export PYARROW_IGNORE_TIMEZONE=1
export PYTHONUNBUFFERED=1

# Check if output directory exists and has checkpoint
if [[ "$resume" == "true" && -d "$output_dir" ]]; then
    checkpoint_file="$output_dir/tokenization_checkpoint.json"
    if [[ -f "$checkpoint_file" ]]; then
        echo "Found existing checkpoint at: $checkpoint_file"
        echo "Will resume from the last completed stage."
    else
        echo "No checkpoint found. Starting fresh tokenization."
    fi
else
    echo "Starting fresh tokenization."
fi
echo

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH

# Performance optimizations
export OMP_NUM_THREADS=16
export POLARS_MAX_THREADS=16
export PYARROW_IGNORE_TIMEZONE=1
export PYTHONUNBUFFERED=1

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null
"

script_body="
set -e

clear
echo 'Starting high-performance train data tokenization...'
ethos_tokenize -m worker='range(0,16)' \
    input_dir=$input_dir/train \
    output_dir=$output_dir \
    out_fn=train \
    resume=$resume \
    max_workers=16 \
    memory_limit_gb=12 \
    chunk_size=4 \
    use_memory_optimized=false

echo 'Starting high-performance test data tokenization...'
ethos_tokenize -m worker='range(0,8)' \
    input_dir=$input_dir/test \
    vocab=$output_dir/train \
    output_dir=$output_dir \
    out_fn=test \
    resume=$resume \
    max_workers=8 \
    memory_limit_gb=12 \
    chunk_size=4 \
    use_memory_optimized=false

echo 'High-performance tokenization completed successfully!'
"

module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    echo Singularity not found, running using locally using bash.
    bash -c "${script_body}"
fi
