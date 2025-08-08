import functools
import random
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Callable

import polars as pl
from MEDS_transforms.mapreduce.utils import rwlock_wrap

from ..inference.utils import wait_for_workers
from ..vocabulary import Vocabulary


def run_stage_optimized(
    in_fps,
    out_fps,
    *transform_fns,
    params={},
    vocab=None,
    agg_to=None,
    agg_params=None,
    worker=1,
    max_workers=None,
    chunk_size=4,
    memory_limit_gb=12,  # Per worker memory limit
    **kwargs,
):
    """Optimized version for high-performance parallel processing.
    
    Args:
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        chunk_size: Number of files to process per worker batch
        memory_limit_gb: Memory limit per worker in GB
    """
    
    # Set optimal defaults for 16 cores with 200GB RAM
    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 8)
    
    # Calculate optimal chunk size based on available memory
    available_memory_gb = 200  # Total system memory
    memory_per_worker = min(memory_limit_gb, available_memory_gb // max_workers)
    
    if vocab is not None:
        params = {"vocab": Vocabulary.from_path(vocab), **params}

    transforms_to_run = [
        functools.partial(transform_fn, **params) for transform_fn in transform_fns
    ]

    fps = list(zip(in_fps, out_fps))
    random.shuffle(fps)  # Randomize for better load balancing
    
    # Process files in chunks for better memory management
    def process_chunk(chunk_fps: List[Tuple[Path, Path]]) -> List[Path]:
        """Process a chunk of files with optimized memory usage."""
        processed_files = []
        
        for in_fp, out_fp in chunk_fps:
            try:
                # Use optimized parquet reading with memory-efficient settings
                rwlock_wrap(
                    in_fp,
                    out_fp,
                    functools.partial(
                        pl.read_parquet, 
                        use_pyarrow=True,
                        memory_map=True,  # Memory mapping for large files
                        parallel=True,     # Enable parallel reading
                    ),
                    lambda df, out_: df.write_parquet(
                        out_, 
                        use_pyarrow=True,
                        compression="snappy",  # Fast compression
                        row_group_size=100000,  # Optimize for large datasets
                    ),
                    compute_fn=lambda df: functools.reduce(
                        lambda df, fn: fn(df), transforms_to_run, df
                    ),
                )
                processed_files.append(out_fp)
                
            except Exception as e:
                print(f"Error processing {in_fp}: {e}")
                continue
                
        return processed_files

    # Split files into chunks for parallel processing
    chunks = [fps[i:i + chunk_size] for i in range(0, len(fps), chunk_size)]
    
    print(f"Processing {len(fps)} files in {len(chunks)} chunks with {max_workers} workers")
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        processed_files = []
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results = future.result()
                processed_files.extend(chunk_results)
                print(f"Completed chunk {chunk_idx + 1}/{len(chunks)}")
            except Exception as e:
                print(f"Chunk {chunk_idx} failed: {e}")

    # Handle aggregation if needed
    if agg_to is not None:
        agg_to = Path(agg_to)
        if worker == 1:
            wait_for_workers(out_fps[0].parent)
            transform_fns[-1].agg(in_fps=out_fps, out_fp=agg_to, **(agg_params or {}))
        else:
            while not agg_to.exists():
                time.sleep(1)  # Reduced sleep time for faster response
    
    return processed_files


def run_stage_memory_optimized(
    in_fps,
    out_fps,
    *transform_fns,
    params={},
    vocab=None,
    agg_to=None,
    agg_params=None,
    worker=1,
    **kwargs,
):
    """Memory-optimized version that processes files one at a time to minimize memory usage."""
    
    if vocab is not None:
        params = {"vocab": Vocabulary.from_path(vocab), **params}

    transforms_to_run = [
        functools.partial(transform_fn, **params) for transform_fn in transform_fns
    ]

    fps = list(zip(in_fps, out_fps))
    random.shuffle(fps)

    for in_fp, out_fp in fps:
        try:
            # Use memory-efficient settings
            rwlock_wrap(
                in_fp,
                out_fp,
                functools.partial(
                    pl.read_parquet, 
                    use_pyarrow=True,
                    memory_map=True,
                    parallel=False,  # Disable parallel reading to save memory
                ),
                lambda df, out_: df.write_parquet(
                    out_, 
                    use_pyarrow=True,
                    compression="snappy",
                    row_group_size=50000,  # Smaller row groups for memory efficiency
                ),
                compute_fn=lambda df: functools.reduce(
                    lambda df, fn: fn(df), transforms_to_run, df
                ),
            )
        except Exception as e:
            print(f"Error processing {in_fp}: {e}")
            continue

    if agg_to is not None:
        agg_to = Path(agg_to)
        if worker == 1:
            wait_for_workers(out_fps[0].parent)
            transform_fns[-1].agg(in_fps=out_fps, out_fp=agg_to, **(agg_params or {}))
        else:
            while not agg_to.exists():
                time.sleep(1)


# Backward compatibility - use optimized version by default
run_stage = run_stage_optimized
