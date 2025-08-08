import pickle
import shutil
import time
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ..constants import STATIC_DATA_FN
from ..constants import SpecialToken as ST
from ..datasets import TimelineDataset
from ..inference.utils import wait_for_workers
from ..vocabulary import Vocabulary
from .checkpoint import TokenizationCheckpoint, get_stage_output_files, verify_stage_outputs
from .run_stage import run_stage
from .utils import load_function

OmegaConf.register_new_resolver("default_if_null", lambda a, b: b if a is None else a)
OmegaConf.register_new_resolver("is_not_null", lambda v: v is not None)


@hydra.main(version_base=None, config_path="../configs", config_name="tokenization")
def main(cfg: DictConfig):
    if not (input_dir := Path(cfg.input_dir)).is_dir():
        raise FileNotFoundError(f"Data directory '{input_dir}' not found.")

    dataset = cfg.dataset.name
    out_fn = f"{dataset}_{input_dir.name}" if cfg.out_fn is None else cfg.out_fn
    output_dir = Path(cfg.output_dir) / out_fn
    # Update the real output directory path
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint system
    checkpoint = TokenizationCheckpoint(output_dir)
    resume_enabled = cfg.get('resume', False)
    
    if resume_enabled:
        checkpoint_loaded = checkpoint.load()
        if checkpoint_loaded:
            logger.info(f"Resuming tokenization from checkpoint")
        else:
            logger.info(f"No checkpoint found, starting fresh tokenization")
    else:
        # Clear any existing checkpoint if not resuming
        checkpoint.clear()

    logger.info(f"Tokenizing '{input_dir}' using the {dataset.upper()} preprocessing pipeline.")

    in_fps = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(in_fps)} patient splits in '{input_dir}'.")
    assert in_fps, "No parquet files found!"

    i = 1
    for stage_cfg in cfg.dataset.stages:
        stage_name = stage_cfg.name
        stage_dir = output_dir / f"{i:02}_{stage_name}"
        
        # Check if this stage should be skipped
        if stage_cfg.get("skip", False):
            logger.info(f"Skipping stage {stage_name}")
            i += 1
            continue
            
        # Check if stage is already completed
        if resume_enabled and checkpoint.is_stage_completed(stage_name):
            logger.info(f"Stage {stage_name} already completed, skipping")
            # Use the output files from the completed stage
            completed_files = checkpoint.get_completed_stage_files(stage_name)
            if completed_files:
                in_fps = [stage_dir / f for f in completed_files]
            i += 1
            continue
            
        # Set current stage for checkpointing
        if resume_enabled:
            checkpoint.set_current_stage(stage_name)
            
        stage_dir.mkdir(parents=True, exist_ok=True)
        out_fps = [stage_dir / fp.name for fp in in_fps]

        transform_fns = []
        if "transforms" in stage_cfg:
            for transform_name in stage_cfg.transforms:
                func = load_function(transform_name, f"ethos.tokenize.{dataset}")
                transform_fns.append(func)
        else:
            func = load_function(stage_cfg.name, "ethos.tokenize.common")
            transform_fns.append(func)

        run_stage(in_fps, out_fps, *transform_fns, worker=cfg.worker, **stage_cfg)

        # Make workers wait for lock file deletion to avoid moving on prematurely
        wait_for_workers(stage_dir)

        # Mark stage as completed and save checkpoint
        if resume_enabled:
            stage_files = get_stage_output_files(stage_dir)
            checkpoint.mark_stage_completed(stage_name, stage_files)

        if "agg_to" not in stage_cfg:
            in_fps = out_fps

        i += 1

    if cfg.worker == 1:
        if cfg.vocab is None:
            vocab = Vocabulary()
            with (output_dir / STATIC_DATA_FN).open("rb") as f:
                static_codes = sorted(
                    {
                        code
                        for pt_static_data in pickle.load(f).values()
                        for static_data_obj in pt_static_data.values()
                        for code in static_data_obj["code"]
                    }.difference([str(ST.DOB)])
                )
            vocab.add_words(static_codes)

            codes = pl.read_csv(output_dir / cfg.code_counts_fn, columns="code")["code"].to_list()
            vocab.add_words(codes)
        else:
            vocab = Vocabulary.from_path(cfg.vocab)
            if (quantile_fp := Path(cfg.vocab) / cfg.quantiles_fn).exists():
                logger.info(f"Copying quantile breaks from {quantile_fp} to {output_dir}")
                shutil.copy(quantile_fp, output_dir / cfg.quantiles_fn)

            if (intervals_fp := Path(cfg.vocab) / cfg.intervals_fn).exists():
                logger.info(f"Copying intervals from {intervals_fp} to {output_dir}")
                shutil.copy(intervals_fp, output_dir / cfg.intervals_fn)
            else:
                logger.warning(f"Intervals file not found at {intervals_fp}")

        vocab.dump(output_dir)
        for in_fp in in_fps:
            TimelineDataset.tensorize(in_fp, output_dir / in_fp.name, vocab)

    if __name__ == "__main__":
        main()
