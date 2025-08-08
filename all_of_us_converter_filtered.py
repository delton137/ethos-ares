#!/usr/bin/env python3
"""
All of Us OMOP to MEDS converter with vocabulary filtering to remove rare codes.
"""

import polars as pl
from pathlib import Path
import argparse
import logging
import gc
from collections import Counter
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_condition_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of condition_occurrence data to MEDS format"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("condition_start_datetime").alias("time"),
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
        pl.col("drug_exposure_start_datetime").alias("time"),
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
        pl.col("procedure_datetime").alias("time"),
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
        pl.col("measurement_datetime").alias("time"),
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
        pl.col("observation_datetime").alias("time"),
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

def count_codes_pass(input_dir: Path, tables: list) -> Counter:
    """First pass: count all codes to determine which ones to keep"""
    logger.info("First pass: Counting all codes...")
    
    code_counter = Counter()
    
    for table_name in tables:
        table_dir = input_dir / table_name
        if not table_dir.exists():
            continue
            
        logger.info(f"Counting codes in {table_name}")
        parquet_files = list(table_dir.glob("*.parquet"))
        
        # Get conversion function
        convert_funcs = {
            "condition_occurrence": convert_condition_chunk,
            "drug_exposure": convert_drug_chunk,
            "procedure_occurrence": convert_procedure_chunk,
            "measurement": convert_measurement_chunk,
            "observation": convert_observation_chunk
        }
        
        if table_name not in convert_funcs:
            continue
            
        convert_func = convert_funcs[table_name]
        
        for file in parquet_files:
            logger.info(f"Counting codes in {file.name}")
            
            for chunk in pl.read_parquet(file).iter_slices(100000):  # Smaller chunks for counting
                chunk_df = pl.DataFrame(chunk)
                if chunk_df.is_empty():
                    continue
                
                # Convert chunk to get codes
                try:
                    converted_chunk = convert_func(chunk_df)
                    if not converted_chunk.is_empty():
                        codes = converted_chunk["code"].to_list()
                        code_counter.update(codes)
                        
                except Exception as e:
                    logger.warning(f"Error processing chunk in {file.name}: {e}")
                    continue
                
                del chunk_df, converted_chunk
                gc.collect()
    
    logger.info(f"Found {len(code_counter)} unique codes total")
    return code_counter

def filter_codes(code_counter: Counter, min_count: int = 10, max_vocab_size: int = 20000) -> set:
    """Filter codes based on frequency and vocabulary size limits"""
    logger.info(f"Filtering codes with min_count={min_count}, max_vocab_size={max_vocab_size}")
    
    # Filter by minimum count
    frequent_codes = {code for code, count in code_counter.items() if count >= min_count}
    logger.info(f"After min_count filter: {len(frequent_codes)} codes")
    
    # If still too many, keep only the most frequent ones
    if len(frequent_codes) > max_vocab_size:
        most_common_codes = code_counter.most_common(max_vocab_size)
        frequent_codes = {code for code, count in most_common_codes}
        logger.info(f"After max_vocab_size filter: {len(frequent_codes)} codes")
    
    # Always include essential codes
    essential_codes = {"MEDS_BIRTH", "MEDS_DEATH"}
    frequent_codes.update(essential_codes)
    
    logger.info(f"Final vocabulary size: {len(frequent_codes)} codes")
    return frequent_codes

def process_table_with_filter(input_dir: Path, output_dir: Path, table_name: str, 
                            allowed_codes: set, chunk_size: int = 500000):
    """Process a table, keeping only allowed codes"""
    table_dir = input_dir / table_name
    if not table_dir.exists():
        logger.warning(f"Directory {table_name} not found, skipping")
        return
    
    parquet_files = list(table_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {table_dir}")
        return
    
    logger.info(f"Processing {len(parquet_files)} files for {table_name} with filtering")
    
    # Get conversion function
    convert_funcs = {
        "condition_occurrence": convert_condition_chunk,
        "drug_exposure": convert_drug_chunk,
        "procedure_occurrence": convert_procedure_chunk,
        "measurement": convert_measurement_chunk,
        "observation": convert_observation_chunk
    }
    
    if table_name not in convert_funcs:
        logger.warning(f"No converter for {table_name}")
        return
    
    convert_func = convert_funcs[table_name]
    output_file = output_dir / f"{table_name}.parquet"
    
    total_records = 0
    filtered_records = 0
    first_write = True
    
    # Process each file
    for file_idx, file in enumerate(parquet_files):
        logger.info(f"Processing {file.name} ({file_idx + 1}/{len(parquet_files)})")
        
        try:
            for chunk_idx, chunk in enumerate(pl.read_parquet(file).iter_slices(chunk_size)):
                chunk_df = pl.DataFrame(chunk)
                if chunk_df.is_empty():
                    continue
                
                # Convert chunk
                converted_chunk = convert_func(chunk_df)
                if converted_chunk.is_empty():
                    continue
                
                # Filter by allowed codes
                original_count = len(converted_chunk)
                filtered_chunk = converted_chunk.filter(pl.col("code").is_in(list(allowed_codes)))
                filtered_count = len(filtered_chunk)
                
                total_records += original_count
                filtered_records += filtered_count
                
                if filtered_chunk.is_empty():
                    del chunk_df, converted_chunk, filtered_chunk
                    gc.collect()
                    continue
                
                # Write chunk
                if first_write:
                    filtered_chunk.write_parquet(output_file)
                    first_write = False
                else:
                    # Read existing, concat, and write back
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, filtered_chunk])
                    combined_df.write_parquet(output_file)
                    del existing_df, combined_df
                
                if chunk_idx % 20 == 0:  # Log every 20 chunks
                    logger.info(f"Processed {total_records} records, kept {filtered_records} for {table_name}")
                
                # Clean up
                del chunk_df, converted_chunk, filtered_chunk
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    filter_rate = (filtered_records / total_records * 100) if total_records > 0 else 0
    logger.info(f"Completed {table_name}: kept {filtered_records}/{total_records} records ({filter_rate:.1f}%)")

def create_birth_death_records(input_dir: Path, output_dir: Path):
    """Create birth and death records"""
    logger.info("Creating birth/death records...")
    
    person_dir = input_dir / "person"
    if person_dir.exists():
        birth_records = []
        for file in person_dir.glob("*.parquet"):
            logger.info(f"Processing person file {file.name}")
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
        
        if birth_records:
            birth_df = pl.concat(birth_records)
            birth_file = output_dir / "birth.parquet"
            birth_df.write_parquet(birth_file)
            logger.info(f"Created {len(birth_df)} birth records")
            del birth_df, birth_records
            gc.collect()

def split_data(output_dir: Path, final_dir: Path, split_ratio: float = 0.7):
    """Split data into train/test"""
    logger.info("Splitting data into train/test...")
    
    meds_files = list(output_dir.glob("*.parquet"))
    if not meds_files:
        logger.error("No MEDS files found to split")
        return
    
    # Get all unique subjects
    all_subjects = set()
    for file in meds_files:
        logger.info(f"Scanning {file.name} for subjects...")
        for chunk in pl.read_parquet(file).iter_slices(100000):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            subjects = chunk_df.select("subject_id").unique()["subject_id"].to_list()
            all_subjects.update(subjects)
            del chunk_df, subjects
            gc.collect()
    
    logger.info(f"Found {len(all_subjects)} unique subjects")
    
    # Split subjects
    import random
    random.seed(42)
    all_subjects_list = list(all_subjects)
    n_train = int(len(all_subjects_list) * split_ratio)
    train_subjects = set(random.sample(all_subjects_list, n_train))
    test_subjects = all_subjects - train_subjects
    
    logger.info(f"Train: {len(train_subjects)}, Test: {len(test_subjects)}")
    
    # Create output directories
    train_dir = final_dir / "train"
    test_dir = final_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Split each file
    for file in meds_files:
        logger.info(f"Splitting {file.name}...")
        
        train_chunks = []
        test_chunks = []
        
        for chunk in pl.read_parquet(file).iter_slices(50000):
            chunk_df = pl.DataFrame(chunk)
            if chunk_df.is_empty():
                continue
            
            train_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(train_subjects)))
            test_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(test_subjects)))
            
            if not train_chunk.is_empty():
                train_chunks.append(train_chunk)
            if not test_chunk.is_empty():
                test_chunks.append(test_chunk)
            
            del chunk_df, train_chunk, test_chunk
            gc.collect()
        
        if train_chunks:
            train_df = pl.concat(train_chunks)
            train_file = train_dir / f"data_{file.stem}.parquet"
            train_df.write_parquet(train_file)
            logger.info(f"Saved {len(train_df)} train records for {file.name}")
            del train_df, train_chunks
            gc.collect()
        
        if test_chunks:
            test_df = pl.concat(test_chunks)
            test_file = test_dir / f"data_{file.stem}.parquet"
            test_df.write_parquet(test_file)
            logger.info(f"Saved {len(test_df)} test records for {file.name}")
            del test_df, test_chunks
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="All of Us OMOP to MEDS converter with vocabulary filtering")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--min_count", type=int, default=10, help="Minimum code frequency to keep")
    parser.add_argument("--max_vocab_size", type=int, default=20000, help="Maximum vocabulary size")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create temp directory for MEDS files
    temp_dir = output_dir / "temp_meds"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Tables to process
    tables = ["condition_occurrence", "drug_exposure", "procedure_occurrence", "measurement", "observation"]
    
    # First pass: count all codes
    code_counter = count_codes_pass(input_dir, tables)
    
    # Save code counts for analysis
    with open(output_dir / "code_counts.json", "w") as f:
        json.dump(dict(code_counter.most_common()), f, indent=2)
    
    # Filter codes
    allowed_codes = filter_codes(code_counter, args.min_count, args.max_vocab_size)
    
    # Save filtered vocabulary
    with open(output_dir / "filtered_vocabulary.txt", "w") as f:
        for code in sorted(allowed_codes):
            f.write(f"{code}\n")
    
    logger.info(f"Final vocabulary saved to {output_dir / 'filtered_vocabulary.txt'}")
    
    # Second pass: process tables with filtering
    for table in tables:
        logger.info(f"\n=== Processing {table} with filtering ===")
        process_table_with_filter(input_dir, temp_dir, table, allowed_codes, args.chunk_size)
    
    # Create birth/death records
    create_birth_death_records(input_dir, temp_dir)
    
    # Split into train/test
    final_dir = output_dir / "data"
    split_data(temp_dir, final_dir, args.split_ratio)
    
    logger.info("Conversion complete!")
    logger.info(f"Final vocabulary size: {len(allowed_codes)} codes")

if __name__ == "__main__":
    main()
