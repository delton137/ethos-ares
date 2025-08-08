#!/usr/bin/env python3
"""
Convert All of Us OMOP data to MEDS format for ETHOS training.
This script reads All of Us Parquet files and converts them to MEDS parquet format.
"""

import polars as pl
from pathlib import Path
import argparse
from datetime import datetime
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_parquet_directory(data_dir: Path) -> pl.DataFrame:
    """Read all parquet files from a directory and combine them."""
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return pl.DataFrame()
    
    logger.info(f"Reading {len(parquet_files)} parquet files from {data_dir}")
    dfs = []
    for file in parquet_files:
        try:
            df = pl.read_parquet(file)
            dfs.append(df)
            logger.info(f"Read {len(df)} records from {file.name}")
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    if not dfs:
        return pl.DataFrame()
    
    combined_df = pl.concat(dfs)
    logger.info(f"Combined {len(combined_df)} total records from {data_dir}")
    return combined_df

def convert_condition_occurrence(data_dir: Path, output_path: Path) -> None:
    """Convert condition_occurrence data to MEDS format"""
    logger.info("Converting condition_occurrence...")
    
    df = read_parquet_directory(data_dir / "condition_occurrence")
    if df.is_empty():
        logger.warning("No condition_occurrence data found")
        return
    
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

def convert_drug_exposure(data_dir: Path, output_path: Path) -> None:
    """Convert drug_exposure data to MEDS format"""
    logger.info("Converting drug_exposure...")
    
    df = read_parquet_directory(data_dir / "drug_exposure")
    if df.is_empty():
        logger.warning("No drug_exposure data found")
        return
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("drug_exposure_start_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("drug_exposure_start_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("DRUG//"), pl.col("drug_source_value")]).alias("code"),
        pl.lit(1.0).alias("numeric_value")
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

def convert_procedure_occurrence(data_dir: Path, output_path: Path) -> None:
    """Convert procedure_occurrence data to MEDS format"""
    logger.info("Converting procedure_occurrence...")
    
    df = read_parquet_directory(data_dir / "procedure_occurrence")
    if df.is_empty():
        logger.warning("No procedure_occurrence data found")
        return
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("procedure_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("procedure_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("PROCEDURE//"), pl.col("procedure_source_value")]).alias("code"),
        pl.lit(1.0).alias("numeric_value")
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

def convert_measurement(data_dir: Path, output_path: Path) -> None:
    """Convert measurement data to MEDS format"""
    logger.info("Converting measurement...")
    
    df = read_parquet_directory(data_dir / "measurement")
    if df.is_empty():
        logger.warning("No measurement data found")
        return
    
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

def convert_observation(data_dir: Path, output_path: Path) -> None:
    """Convert observation data to MEDS format"""
    logger.info("Converting observation...")
    
    df = read_parquet_directory(data_dir / "observation")
    if df.is_empty():
        logger.warning("No observation data found")
        return
    
    meds_df = df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("observation_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .fill_null(pl.col("observation_date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime))
        .alias("time"),
        pl.concat_str([pl.lit("OBSERVATION//"), pl.col("observation_source_value")]).alias("code"),
        pl.coalesce(pl.col("value_as_number"), pl.lit(1.0)).alias("numeric_value")
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
    
    output_file = output_path / "observation.parquet"
    meds_df.write_parquet(output_file)
    logger.info(f"Saved {len(meds_df)} observation records to {output_file}")

def add_birth_death_records(data_dir: Path, output_path: Path) -> None:
    """Add birth and death records from person and death data"""
    logger.info("Adding birth and death records...")
    
    # Read person data
    person_df = read_parquet_directory(data_dir / "person")
    if person_df.is_empty():
        logger.warning("No person data found")
        return
    
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
    death_df = read_parquet_directory(data_dir / "death")
    death_df_converted = pl.DataFrame()
    
    if not death_df.is_empty():
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
            pl.col("time").cast(pl.Int64),
            pl.col("numeric_value").cast(pl.Float64)
        ])
    
    # Combine birth and death records
    combined_df = pl.concat([birth_df, death_df_converted])
    
    output_file = output_path / "birth_death.parquet"
    combined_df.write_parquet(output_file)
    logger.info(f"Saved {len(combined_df)} birth/death records to {output_file}")

def combine_all_meds_data(meds_dir: Path, output_dir: Path, split_ratio: float = 0.7) -> None:
    """Combine all MEDS data and split into train/test sets"""
    logger.info("Combining all MEDS data...")
    
    # Read all parquet files from the meds directory
    parquet_files = list(meds_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in meds directory")
        return
    
    all_dfs = []
    for file in parquet_files:
        try:
            df = pl.read_parquet(file)
            all_dfs.append(df)
            logger.info(f"Read {len(df)} records from {file.name}")
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    if not all_dfs:
        logger.error("No valid data found")
        return
    
    # Combine all data
    combined_df = pl.concat(all_dfs)
    logger.info(f"Combined {len(combined_df)} total records")
    
    # Get unique subjects
    unique_subjects = combined_df.select("subject_id").unique()
    logger.info(f"Found {len(unique_subjects)} unique subjects")
    
    # Split subjects into train/test
    n_train = int(len(unique_subjects) * split_ratio)
    train_subjects = unique_subjects.sample(n=n_train, seed=42)
    test_subjects = unique_subjects.filter(~pl.col("subject_id").is_in(train_subjects["subject_id"]))
    
    # Split data
    train_df = combined_df.filter(pl.col("subject_id").is_in(train_subjects["subject_id"]))
    test_df = combined_df.filter(pl.col("subject_id").is_in(test_subjects["subject_id"]))
    
    logger.info(f"Train set: {len(train_df)} records from {len(train_subjects)} subjects")
    logger.info(f"Test set: {len(test_df)} records from {len(test_subjects)} subjects")
    
    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data in chunks
    chunk_size = 10000  # Adjust based on memory
    for i, chunk in enumerate(train_df.iter_slices(chunk_size)):
        chunk_df = pl.DataFrame(chunk)
        output_file = train_dir / f"data_{i}.parquet"
        chunk_df.write_parquet(output_file)
        logger.info(f"Saved train chunk {i+1} with {len(chunk_df)} records")
    
    # Save test data in chunks
    for i, chunk in enumerate(test_df.iter_slices(chunk_size)):
        chunk_df = pl.DataFrame(chunk)
        output_file = test_dir / f"data_{i}.parquet"
        chunk_df.write_parquet(output_file)
        logger.info(f"Saved test chunk {i+1} with {len(chunk_df)} records")

def main():
    parser = argparse.ArgumentParser(description="Convert All of Us OMOP data to MEDS format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to All of Us OMOP data directory")
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
        ("condition_occurrence", convert_condition_occurrence),
        ("drug_exposure", convert_drug_exposure), 
        ("procedure_occurrence", convert_procedure_occurrence),
        ("measurement", convert_measurement),
        ("observation", convert_observation),
    ]
    
    for table_name, convert_func in tables_to_convert:
        table_dir = input_dir / table_name
        if table_dir.exists():
            logger.info(f"Processing {table_name}")
            convert_func(input_dir, meds_temp_dir)
        else:
            logger.warning(f"Directory {table_name} not found, skipping")
    
    # Add birth/death records
    add_birth_death_records(input_dir, meds_temp_dir)
    
    # Combine and split data
    final_output_dir = output_dir / "data"
    combine_all_meds_data(meds_temp_dir, final_output_dir, args.split_ratio)
    
    logger.info("Conversion complete!")

if __name__ == "__main__":
    main()
