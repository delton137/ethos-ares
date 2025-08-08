#!/usr/bin/env python3
"""
Convert OMOP data to MEDS format for ETHOS training.
This script reads OMOP CSV files and converts them to MEDS parquet format.
"""

import polars as pl
from pathlib import Path
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_condition_occurrence(df: pl.DataFrame, output_path: Path) -> None:
    """Convert condition_occurrence.csv to MEDS format"""
    logger.info("Converting condition_occurrence...")
    
    # Convert to MEDS format
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("condition_start_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("condition_start_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("CONDITION//"), pl.col("condition_source_value")]).alias("code"),
        pl.lit(1.0).alias("numeric_value")
    ]).filter(
        pl.col("subject_id").is_not_null() & 
        pl.col("time").is_not_null() & 
        pl.col("code").is_not_null()
    )
    
    # Convert time to microseconds since epoch (MEDS format) and ensure consistent types
    meds_df = meds_df.with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    # Save as parquet
    output_file = output_path / "condition_occurrence.parquet"
    meds_df.write_parquet(output_file)
    logger.info(f"Saved {len(meds_df)} condition records to {output_file}")


def convert_drug_exposure(df: pl.DataFrame, output_path: Path) -> None:
    """Convert drug_exposure.csv to MEDS format"""
    logger.info("Converting drug_exposure...")
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("drug_exposure_start_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("drug_exposure_start_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("DRUG//"), pl.col("drug_source_value")]).alias("code"),
        pl.col("quantity").fill_null(1.0).alias("numeric_value")
    ]).filter(
        pl.col("subject_id").is_not_null() & 
        pl.col("time").is_not_null() & 
        pl.col("code").is_not_null()
    )
    
    meds_df = meds_df.with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    output_file = output_path / "drug_exposure.parquet"
    meds_df.write_parquet(output_file)
    logger.info(f"Saved {len(meds_df)} drug records to {output_file}")


def convert_procedure_occurrence(df: pl.DataFrame, output_path: Path) -> None:
    """Convert procedure_occurrence.csv to MEDS format"""
    logger.info("Converting procedure_occurrence...")
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("procedure_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("procedure_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("PROCEDURE//"), pl.col("procedure_source_value")]).alias("code"),
        pl.col("quantity").fill_null(1.0).alias("numeric_value")
    ]).filter(
        pl.col("subject_id").is_not_null() & 
        pl.col("time").is_not_null() & 
        pl.col("code").is_not_null()
    )
    
    meds_df = meds_df.with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    output_file = output_path / "procedure_occurrence.parquet"
    meds_df.write_parquet(output_file)
    logger.info(f"Saved {len(meds_df)} procedure records to {output_file}")


def convert_measurement(df: pl.DataFrame, output_path: Path) -> None:
    """Convert measurement.csv to MEDS format"""
    logger.info("Converting measurement...")
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("measurement_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("measurement_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("LAB//"), pl.col("measurement_source_value")]).alias("code"),
        pl.col("value_as_number").fill_null(1.0).alias("numeric_value")
    ]).filter(
        pl.col("subject_id").is_not_null() & 
        pl.col("time").is_not_null() & 
        pl.col("code").is_not_null()
    )
    
    meds_df = meds_df.with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    output_file = output_path / "measurement.parquet"
    meds_df.write_parquet(output_file)
    logger.info(f"Saved {len(meds_df)} measurement records to {output_file}")


def add_birth_death_records(person_df: pl.DataFrame, death_df: pl.DataFrame, output_path: Path) -> None:
    """Add birth and death records"""
    logger.info("Adding birth and death records...")
    
    # Birth records
    birth_df = person_df.select([
        pl.col("person_id").alias("subject_id"),
        pl.lit(0).cast(pl.Int64).alias("time"),  # Birth at time 0
        pl.lit("MEDS_BIRTH").alias("code"),
        pl.lit(1.0).alias("numeric_value")
    ]).filter(pl.col("subject_id").is_not_null())
    
    # Ensure consistent data types
    birth_df = birth_df.with_columns([
        pl.col("subject_id").cast(pl.Float64),
        pl.col("time").cast(pl.Int64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    # Death records  
    death_df_converted = death_df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("death_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime).dt.timestamp("us").alias("time"),
        pl.lit("MEDS_DEATH").alias("code"),
        pl.lit(1.0).alias("numeric_value")
    ]).filter(
        pl.col("subject_id").is_not_null() & 
        pl.col("time").is_not_null()
    )
    
    # Ensure consistent data types
    death_df_converted = death_df_converted.with_columns([
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])
    
    # Combine birth and death
    vital_df = pl.concat([birth_df, death_df_converted])
    
    output_file = output_path / "vital_events.parquet"
    vital_df.write_parquet(output_file)
    logger.info(f"Saved {len(vital_df)} vital records to {output_file}")


def combine_all_meds_data(meds_dir: Path, output_dir: Path, split_ratio: float = 0.7) -> None:
    """Combine all MEDS parquet files and create train/test split"""
    logger.info("Combining all MEDS data...")
    
    # Read all parquet files
    all_dfs = []
    for parquet_file in meds_dir.glob("*.parquet"):
        logger.info(f"Reading {parquet_file}")
        df = pl.read_parquet(parquet_file)
        all_dfs.append(df)
    
    # Combine all data
    combined_df = pl.concat(all_dfs)
    logger.info(f"Combined dataset has {len(combined_df)} total records")
    
    # Sort by subject_id and time
    combined_df = combined_df.sort(["subject_id", "time"])
    
    # Get unique subjects for splitting
    unique_subjects = combined_df["subject_id"].unique().sort()
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * split_ratio)
    
    train_subjects = unique_subjects[:n_train]
    test_subjects = unique_subjects[n_train:]
    
    logger.info(f"Splitting {n_subjects} subjects: {n_train} train, {len(test_subjects)} test")
    
    # Create train/test splits
    train_df = combined_df.filter(pl.col("subject_id").is_in(train_subjects))
    test_df = combined_df.filter(pl.col("subject_id").is_in(test_subjects))
    
    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Shard the data into multiple files (similar to original structure)
    def shard_and_save(df, output_dir, prefix="data", shard_size=1000):
        for i, start_idx in enumerate(range(0, len(df), shard_size)):
            end_idx = min(start_idx + shard_size, len(df))
            shard_df = df.slice(start_idx, end_idx - start_idx)
            output_file = output_dir / f"{prefix}_{i}.parquet"
            shard_df.write_parquet(output_file)
        logger.info(f"Saved {i+1} shards to {output_dir}")
    
    shard_and_save(train_df, train_dir)
    shard_and_save(test_df, test_dir)
    
    logger.info(f"Train set: {len(train_df)} records")
    logger.info(f"Test set: {len(test_df)} records")


def main():
    parser = argparse.ArgumentParser(description="Convert OMOP data to MEDS format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to OMOP CSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output MEDS data")
    parser.add_argument("--split_ratio", type=float, default=0.7, help="Train/test split ratio")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create intermediate directory for individual parquet files
    meds_temp_dir = output_dir / "temp_meds"
    meds_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each OMOP table
    tables_to_convert = [
        ("condition_occurrence.csv", convert_condition_occurrence),
        ("drug_exposure.csv", convert_drug_exposure), 
        ("procedure_occurrence.csv", convert_procedure_occurrence),
        ("measurement.csv", convert_measurement),
    ]
    
    for csv_file, convert_func in tables_to_convert:
        csv_path = input_dir / csv_file
        if csv_path.exists():
            logger.info(f"Processing {csv_file}")
            df = pl.read_csv(csv_path, ignore_errors=True)
            convert_func(df, meds_temp_dir)
        else:
            logger.warning(f"File {csv_file} not found, skipping")
    
    # Add birth/death records
    person_path = input_dir / "person.csv"
    death_path = input_dir / "death.csv"
    
    if person_path.exists():
        person_df = pl.read_csv(person_path, ignore_errors=True)
        death_df = pl.read_csv(death_path, ignore_errors=True) if death_path.exists() else pl.DataFrame()
        add_birth_death_records(person_df, death_df, meds_temp_dir)
    
    # Combine and split data
    final_output_dir = output_dir / "data"
    combine_all_meds_data(meds_temp_dir, final_output_dir, args.split_ratio)
    
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
