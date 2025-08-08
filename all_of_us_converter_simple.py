#!/usr/bin/env python3
"""
Simple All of Us OMOP to MEDS converter that handles datetime columns properly.
Uses SNOMED CT concept IDs for standardized vocabulary.
"""

import polars as pl
from pathlib import Path
import argparse
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_condition_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of condition_occurrence data to MEDS format using SNOMED CT concepts"""
    logger.info(f"Condition schema: {df.schema}")
    
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("condition_start_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("CONDITION//"), pl.col("condition_concept_id").cast(pl.Utf8)]).alias("code"),
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
    """Convert a chunk of drug_exposure data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("drug_exposure_start_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("DRUG//"), pl.col("drug_concept_id").cast(pl.Utf8)]).alias("code"),
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
    """Convert a chunk of procedure_occurrence data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("procedure_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("PROCEDURE//"), pl.col("procedure_concept_id").cast(pl.Utf8)]).alias("code"),
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
    """Convert a chunk of measurement data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("measurement_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("LAB//"), pl.col("measurement_concept_id").cast(pl.Utf8)]).alias("code"),
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
    """Convert a chunk of observation data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("observation_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("OBSERVATION//"), pl.col("observation_concept_id").cast(pl.Utf8)]).alias("code"),
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

def convert_visit_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of visit_occurrence data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("visit_start_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("VISIT//"), pl.col("visit_concept_id").cast(pl.Utf8)]).alias("code"),
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

def convert_device_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a chunk of device_exposure data to MEDS format using SNOMED CT concepts"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("device_exposure_start_datetime").alias("time"),  # Already datetime
        pl.concat_str([pl.lit("DEVICE//"), pl.col("device_concept_id").cast(pl.Utf8)]).alias("code"),
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

def process_table_in_chunks(input_dir: Path, output_dir: Path, table_name: str, chunk_size: int = 500000):
    """Process a single table in chunks"""
    table_dir = input_dir / table_name
    if not table_dir.exists():
        logger.warning(f"Directory {table_name} not found, skipping")
        return
    
    parquet_files = list(table_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {table_dir}")
        return
    
    logger.info(f"Processing {len(parquet_files)} files for {table_name}")
    
    # Get conversion function
    convert_funcs = {
        "condition_occurrence": convert_condition_chunk,
        "drug_exposure": convert_drug_chunk,
        "procedure_occurrence": convert_procedure_chunk,
        "measurement": convert_measurement_chunk,
        "observation": convert_observation_chunk,
        "visit_occurrence": convert_visit_chunk,
        "device_exposure": convert_device_chunk
    }
    
    if table_name not in convert_funcs:
        logger.warning(f"No converter for {table_name}")
        return
    
    convert_func = convert_funcs[table_name]
    output_file = output_dir / f"{table_name}.parquet"
    
    total_records = 0
    first_write = True
    
    # Process each file
    for file_idx, file in enumerate(parquet_files):
        logger.info(f"Processing {file.name} ({file_idx + 1}/{len(parquet_files)})")
        
        try:
            # Read and process in chunks
            for chunk_idx, chunk in enumerate(pl.read_parquet(file).iter_slices(chunk_size)):
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
                    # Read existing, concat, and write back
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, converted_chunk])
                    combined_df.write_parquet(output_file)
                    del existing_df, combined_df
                
                total_records += len(converted_chunk)
                if chunk_idx % 10 == 0:  # Log every 10 chunks
                    logger.info(f"Processed {total_records} records so far for {table_name}")
                
                # Clean up
                del chunk_df, converted_chunk
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    logger.info(f"Completed {table_name}: {total_records} total records")

def create_birth_death_records(input_dir: Path, output_dir: Path):
    """Create birth and death records"""
    logger.info("Creating birth/death records...")
    
    # Birth records from person table
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
    
    # Find all MEDS files
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
            
            # Split by subjects
            train_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(train_subjects)))
            test_chunk = chunk_df.filter(pl.col("subject_id").is_in(list(test_subjects)))
            
            if not train_chunk.is_empty():
                train_chunks.append(train_chunk)
            if not test_chunk.is_empty():
                test_chunks.append(test_chunk)
            
            del chunk_df, train_chunk, test_chunk
            gc.collect()
        
        # Save train data
        if train_chunks:
            train_df = pl.concat(train_chunks)
            train_file = train_dir / f"data_{file.stem}.parquet"
            train_df.write_parquet(train_file)
            logger.info(f"Saved {len(train_df)} train records for {file.name}")
            del train_df, train_chunks
            gc.collect()
        
        # Save test data
        if test_chunks:
            test_df = pl.concat(test_chunks)
            test_file = test_dir / f"data_{file.stem}.parquet"
            test_df.write_parquet(test_file)
            logger.info(f"Saved {len(test_df)} test records for {file.name}")
            del test_df, test_chunks
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Simple All of Us OMOP to MEDS converter")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--chunk_size", type=int, default=500000)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create temp directory for MEDS files
    temp_dir = output_dir / "temp_meds"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each table
    tables = ["condition_occurrence", "drug_exposure", "procedure_occurrence", "measurement", "observation", "visit_occurrence", "device_exposure"]
    
    for table in tables:
        logger.info(f"\n=== Processing {table} ===")
        process_table_in_chunks(input_dir, temp_dir, table, args.chunk_size)
    
    # Create birth/death records
    create_birth_death_records(input_dir, temp_dir)
    
    # Split into train/test
    final_dir = output_dir / "data"
    split_data(temp_dir, final_dir, args.split_ratio)
    
    logger.info("Conversion complete!")

if __name__ == "__main__":
    main()
