#!/usr/bin/env python3
"""
Memory-efficient converter for All of Us OMOP data to MEDS format.
Processes data in chunks to avoid memory issues with large datasets.
"""

import polars as pl
from pathlib import Path
import argparse
from datetime import datetime
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_parquet_directory_in_chunks(data_dir: Path, output_path: Path, table_name: str, chunk_size: int = 1000000):
    """Process parquet files in chunks to avoid memory issues"""
    table_dir = data_dir / table_name
    parquet_files = list(table_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {table_dir}")
        return
    
    logger.info(f"Processing {len(parquet_files)} parquet files from {table_dir} in chunks")
    
    # Define conversion function based on table name
    if table_name == "condition_occurrence":
        convert_func = convert_condition_chunk
        output_file = output_path / "condition_occurrence.parquet"
    elif table_name == "drug_exposure":
        convert_func = convert_drug_chunk
        output_file = output_path / "drug_exposure.parquet"
    elif table_name == "procedure_occurrence":
        convert_func = convert_procedure_chunk
        output_file = output_path / "procedure_occurrence.parquet"
    elif table_name == "measurement":
        convert_func = convert_measurement_chunk
        output_file = output_path / "measurement.parquet"
    elif table_name == "observation":
        convert_func = convert_observation_chunk
        output_file = output_path / "observation.parquet"
    else:
        logger.error(f"Unknown table name: {table_name}")
        return
    
    # Process each file in chunks
    total_records = 0
    first_write = True
    
    for file in parquet_files:
        logger.info(f"Processing {file.name}")
        
        # Read file in chunks
        for chunk in pl.read_parquet(file).iter_slices(chunk_size):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            
            # Convert chunk
            converted_chunk = convert_func(chunk_df)
            if converted_chunk.is_empty():
                continue
            
            # Write chunk
            if first_write:
                converted_chunk.write_parquet(output_file)
                first_write = False
            else:
                converted_chunk.write_parquet(output_file, append=True)
            
            total_records += len(converted_chunk)
            logger.info(f"Processed {len(converted_chunk)} records from chunk")
            
            # Force garbage collection
            del chunk_df, converted_chunk
            gc.collect()
    
    logger.info(f"Total {total_records} records processed for {table_name}")

def convert_condition_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of condition_occurrence data to MEDS format"""
    return df.select([
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
    ).with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])

def convert_drug_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of drug_exposure data to MEDS format"""
    return df.select([
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
    ).with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])

def convert_procedure_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of procedure_occurrence data to MEDS format"""
    return df.select([
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
    ).with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])

def convert_measurement_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of measurement data to MEDS format"""
    return df.select([
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
    ).with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])

def convert_observation_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of observation data to MEDS format"""
    return df.select([
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
    ).with_columns([
        (pl.col("time").dt.timestamp("us")).alias("time"),
        pl.col("subject_id").cast(pl.Float64),
        pl.col("numeric_value").cast(pl.Float64)
    ])

def add_birth_death_records_memory_efficient(data_dir: Path, output_path: Path):
    """Add birth and death records efficiently"""
    logger.info("Adding birth and death records...")
    
    # Process person data in chunks
    person_dir = data_dir / "person"
    if not person_dir.exists():
        logger.warning("No person data found")
        return
    
    birth_records = []
    for file in person_dir.glob("*.parquet"):
        logger.info(f"Processing person data from {file.name}")
        for chunk in pl.read_parquet(file).iter_slices(100000):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            
            birth_chunk = chunk_df.select([
                pl.col("person_id").alias("subject_id"),
                pl.lit(0).cast(pl.Int64).alias("time"),
                pl.lit("MEDS_BIRTH").alias("code"),
                pl.lit(1.0).alias("numeric_value")
            ]).filter(pl.col("subject_id").is_not_null()).with_columns([
                pl.col("subject_id").cast(pl.Float64),
                pl.col("time").cast(pl.Int64),
                pl.col("numeric_value").cast(pl.Float64)
            ])
            
            birth_records.append(birth_chunk)
            del chunk_df, birth_chunk
            gc.collect()
    
    # Process death data
    death_dir = data_dir / "death"
    death_records = []
    if death_dir.exists():
        for file in death_dir.glob("*.parquet"):
            logger.info(f"Processing death data from {file.name}")
            for chunk in pl.read_parquet(file).iter_slices(100000):
                chunk_df = pl.DataFrame(chunk)
                if chunk_df.is_empty():
                    continue
                
                # Handle death_date column properly - convert to string first if it's a date
                death_chunk = chunk_df.select([
                    pl.col("person_id").alias("subject_id"),
                    pl.col("death_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime).dt.timestamp("us").alias("time"),
                    pl.lit("MEDS_DEATH").alias("code"),
                    pl.lit(1.0).alias("numeric_value")
                ]).filter(
                    pl.col("subject_id").is_not_null() & 
                    pl.col("time").is_not_null()
                ).with_columns([
                    pl.col("subject_id").cast(pl.Float64),
                    pl.col("time").cast(pl.Int64),
                    pl.col("numeric_value").cast(pl.Float64)
                ])
                
                death_records.append(death_chunk)
                del chunk_df, death_chunk
                gc.collect()
    
    # Combine and save
    if birth_records:
        birth_df = pl.concat(birth_records)
        birth_file = output_path / "birth_death.parquet"
        birth_df.write_parquet(birth_file)
        logger.info(f"Saved {len(birth_df)} birth records")
        del birth_df, birth_records
        gc.collect()
    
    if death_records:
        death_df = pl.concat(death_records)
        death_file = output_path / "death.parquet"
        death_df.write_parquet(death_file)
        logger.info(f"Saved {len(death_df)} death records")
        del death_df, death_records
        gc.collect()

def combine_meds_data_memory_efficient(meds_dir: Path, output_dir: Path, split_ratio: float = 0.7):
    """Combine MEDS data efficiently with train/test split"""
    logger.info("Combining MEDS data efficiently...")
    
    # First pass: count records and get unique subjects
    parquet_files = list(meds_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found")
        return
    
    # Get unique subjects from all files
    all_subjects = set()
    total_records = 0
    
    for file in parquet_files:
        logger.info(f"Scanning {file.name} for subjects...")
        for chunk in pl.read_parquet(file).iter_slices(100000):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            
            subjects = chunk_df.select("subject_id").unique()
            all_subjects.update(subjects["subject_id"].to_list())
            total_records += len(chunk_df)
            del chunk_df, subjects
            gc.collect()
    
    logger.info(f"Found {len(all_subjects)} unique subjects and {total_records} total records")
    
    # Split subjects
    all_subjects_list = list(all_subjects)
    n_train = int(len(all_subjects_list) * split_ratio)
    import random
    random.seed(42)
    train_subjects = set(random.sample(all_subjects_list, n_train))
    test_subjects = all_subjects - train_subjects
    
    logger.info(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    
    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Second pass: split data by subjects
    train_chunk_size = 0
    test_chunk_size = 0
    
    for file in parquet_files:
        logger.info(f"Processing {file.name} for train/test split...")
        train_chunks = []
        test_chunks = []
        
        for chunk in pl.read_parquet(file).iter_slices(50000):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            
            # Split chunk by subjects
            train_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(train_subjects)))
            test_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(test_subjects)))
            
            if not train_chunk.is_empty():
                train_chunks.append(train_chunk)
            if not test_chunk.is_empty():
                test_chunks.append(test_chunk)
            
            del chunk_df, train_chunk, test_chunk
            gc.collect()
        
        # Save train chunks
        if train_chunks:
            for i, chunk in enumerate(train_chunks):
                output_file = train_dir / f"data_{train_chunk_size + i}.parquet"
                chunk.write_parquet(output_file)
            train_chunk_size += len(train_chunks)
            del train_chunks
            gc.collect()
        
        # Save test chunks
        if test_chunks:
            for i, chunk in enumerate(test_chunks):
                output_file = test_dir / f"data_{test_chunk_size + i}.parquet"
                chunk.write_parquet(output_file)
            test_chunk_size += len(test_chunks)
            del test_chunks
            gc.collect()
    
    logger.info(f"Saved {train_chunk_size} train chunks and {test_chunk_size} test chunks")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient All of Us OMOP to MEDS converter")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to All of Us OMOP data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output MEDS data")
    parser.add_argument("--split_ratio", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create intermediate directory
    meds_temp_dir = output_dir / "temp_meds"
    meds_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each OMOP table
    tables_to_convert = [
        "condition_occurrence",
        "drug_exposure", 
        "procedure_occurrence",
        "measurement",
        "observation",
    ]
    
    for table_name in tables_to_convert:
        table_dir = input_dir / table_name
        if table_dir.exists():
            logger.info(f"Processing {table_name}")
            process_parquet_directory_in_chunks(input_dir, meds_temp_dir, table_name, args.chunk_size)
        else:
            logger.warning(f"Directory {table_name} not found, skipping")
    
    # Add birth/death records
    add_birth_death_records_memory_efficient(input_dir, meds_temp_dir)
    
    # Combine and split data
    final_output_dir = output_dir / "data"
    combine_meds_data_memory_efficient(meds_temp_dir, final_output_dir, args.split_ratio)
    
    logger.info("Conversion complete!")

if __name__ == "__main__":
    main()
