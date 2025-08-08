input_dir="/home/jupyter/workspaces/ehrtransformerbaseline/meds_data/data"
output_dir="/home/jupyter/workspaces/ehrtransformerbaseline/ethos_data"

ethos_tokenize -m worker='range(0,8)' \
    input_dir=$input_dir \
    output_dir=$output_dir \
    out_fn=train \
    resume=false \
    max_workers=8 \
    memory_limit_gb=10 \
    chunk_size=4 \
    use_memory_optimized=false
