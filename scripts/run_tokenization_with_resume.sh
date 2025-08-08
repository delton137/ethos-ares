#!/bin/bash -l
#SBATCH --job-name=ethos_tokenize_resume::proj=IRB2023P002279,
#SBATCH --time=6:00:00
#SBATCH --partition=defq
#SBATCH --output=ethos_tokenize_resume.log

# Example script showing how to use the resume functionality
# Usage: ./scripts/run_tokenization_with_resume.sh [suffix] [resume]

# this script is intended to be run from the project root
suffix=$1
resume=${2:-true}  # Default to true for this example script

if [[ -n "$suffix" && ! "$suffix" =~ ^[-_] ]]; then
    suffix="-$suffix"
fi

input_dir="data/mimic-2.2-meds${suffix//_/-}/data"
output_dir="data/tokenized_datasets/mimic${suffix//-/_}"

echo "=== ETHOS Tokenization with Resume ==="
echo "Tokenization parameters:"
echo "  Input directory: $input_dir"
echo "  Output directory: $output_dir"
echo "  Resume: $resume"
echo

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
echo 'Starting train data tokenization...'
ethos_tokenize -m worker='range(0,7)' \
    input_dir=$input_dir/train \
    output_dir=$output_dir \
    out_fn=train \
    resume=$resume

echo 'Starting test data tokenization...'
ethos_tokenize -m worker='range(0,2)' \
    input_dir=$input_dir/test \
    vocab=$output_dir/train \
    output_dir=$output_dir \
    out_fn=test \
    resume=$resume

echo 'Tokenization completed successfully!'
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
