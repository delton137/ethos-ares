# All of Us OMOP to MEDS Conversion Strategy

This document describes the conversion strategy for transforming All of Us OMOP CDM data into the MEDS format used by ETHOS for medical AI model training.

## Overview

All of Us uses the **OMOP Common Data Model (CDM) v5.4** with **SNOMED CT** as the primary vocabulary. This provides a standardized, semantically rich representation of clinical data that is ideal for machine learning applications.

## Data Structure

### All of Us OMOP Tables

The conversion processes **7 core OMOP tables**:

1. **`condition_occurrence`** - Diagnoses and conditions
2. **`drug_exposure`** - Medications and prescriptions  
3. **`procedure_occurrence`** - Medical procedures and interventions
4. **`measurement`** - Lab tests and clinical measurements
5. **`observation`** - Clinical observations and assessments
6. **`visit_occurrence`** - Hospital visits and encounters
7. **`device_exposure`** - Medical devices and equipment
8. **`person`** - Demographics (birth/death records)

### Key OMOP Columns

Each table contains these essential columns:
- **`person_id`** - Unique patient identifier
- **`{table}_concept_id`** - SNOMED CT concept ID (primary)
- **`{table}_source_value`** - Original source code (ICD, LOINC, etc.)
- **`{table}_start_datetime`** - Event timestamp
- **`value_as_number`** - Numeric values (for measurements)

## Conversion Strategy

### 1. SNOMED CT Concept ID Priority

**Primary Strategy**: Use SNOMED CT concept IDs (`{table}_concept_id`) as the primary code source.

**Rationale**:
- All of Us has already mapped everything to SNOMED CT
- SNOMED CT provides standardized, hierarchical vocabulary
- Concept IDs are numeric and efficient for ML
- Rich semantic relationships between concepts

### 2. Code Format

Convert to MEDS format with SNOMED CT concept IDs:

```
{table}_concept_id → {TABLE}//{concept_id}
```

**Examples**:
- `condition_concept_id: 72711` → `CONDITION//72711`
- `measurement_concept_id: 3036277` → `LAB//3036277`
- `drug_concept_id: 430193006` → `DRUG//430193006`

### 3. Data Quality Advantages

**All of Us Data Quality**:
- ✅ **100% SNOMED CT coverage** - All records have valid concept IDs
- ✅ **No null concept IDs** - Zero missing mappings
- ✅ **Standardized timestamps** - All datetime columns in UTC
- ✅ **Clean numeric values** - Proper handling of lab results

**Comparison with MIMIC**:
- MIMIC: Mixed ICD-9/ICD-10 codes, text descriptions
- All of Us: Pure SNOMED CT concept IDs

## Implementation Details

### Conversion Functions

Each table has a dedicated conversion function in `all_of_us_converter_simple.py`:

```python
def convert_condition_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Convert using SNOMED CT concept IDs"""
    return df.select([
        pl.col("person_id").alias("subject_id"),
        pl.col("condition_start_datetime").alias("time"),
        pl.concat_str([pl.lit("CONDITION//"), 
                      pl.col("condition_concept_id").cast(pl.Utf8)]).alias("code"),
        pl.lit(1.0).alias("numeric_value")
    ])
```

### MEDS Output Format

The converter produces MEDS format files with columns:
- **`subject_id`** - Patient identifier (Float64)
- **`time`** - Unix timestamp in microseconds (Float64)
- **`code`** - SNOMED CT concept code (String)
- **`numeric_value`** - Numeric value for measurements (Float64)

### Train/Test Split

- **70/30 split** at the subject level
- **Deterministic** using seed 42
- **No data leakage** between train/test sets

## Vocabulary Benefits

### SNOMED CT Advantages

1. **Hierarchical Structure**: Concepts organized in meaningful hierarchies
2. **Cross-Mapping**: Can map to ICD, LOINC, RxNorm, etc.
3. **Semantic Richness**: More detailed than ICD codes
4. **International Standard**: Used globally for clinical terminology

### Example Concept Hierarchy

```
SNOMED CT Concept 72711 (Diabetes mellitus)
├── Type 1 diabetes mellitus
├── Type 2 diabetes mellitus  
├── Gestational diabetes
└── Other specified diabetes
```

## ETHOS Integration

### Dataset Configuration

The `all_of_us.yaml` configuration includes:
- **7 table preprocessors** for each OMOP table
- **SNOMED CT-aware filtering** and processing
- **Standardized code prefixes** (CONDITION//, LAB//, etc.)

### Preprocessor Classes

Custom preprocessors in `src/ethos/tokenize/all_of_us/preprocessors.py`:
- `DemographicData` - Race and demographic processing
- `LabData` - Laboratory test processing
- `VisitData` - Visit type processing
- `DeviceData` - Medical device processing
- And more...

## Usage

### 1. Convert OMOP to MEDS

```bash
python all_of_us_converter_simple.py \
  --input_dir /path/to/omop_data \
  --output_dir /path/to/meds_data \
  --split_ratio 0.7
```

### 2. Tokenize with ETHOS

```bash
ethos_tokenize -m worker=2 \
  input_dir=meds_data/data \
  output_dir=ethos_data \
  out_fn=train \
  dataset=all_of_us
```

### 3. Train Model

```bash
ethos_train -m worker=1 \
  input_dir=ethos_data \
  output_dir=model_output \
  config=all_of_us_training
```

## Expected Vocabulary Size

With 7 OMOP tables and SNOMED CT concepts:
- **~10,000-50,000 unique codes** (vs. 100,000+ with mixed vocabularies)
- **Standardized concept IDs** (vs. mixed ICD/text codes)
- **Rich semantic relationships** for better model learning

## Advantages Over Other Approaches

### vs. MIMIC-style Conversion
- **No ICD-9 to ICD-10 mapping** needed
- **No text normalization** required
- **Consistent vocabulary** across all tables
- **Better semantic meaning** for ML models

### vs. Raw Source Values
- **Standardized across institutions**
- **Hierarchical relationships** preserved
- **International compatibility**
- **Future-proof** vocabulary

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use chunked processing for large datasets
2. **Schema Errors**: Ensure datetime columns are in UTC format
3. **Vocabulary Size**: Filter rare codes during tokenization if needed

### Data Quality Checks

```bash
# Check SNOMED CT coverage
python -c "import polars as pl; df = pl.read_parquet('condition_occurrence.parquet'); print('Null concept IDs:', df.select('condition_concept_id').null_count())"

# Check unique codes
python -c "import polars as pl; df = pl.read_parquet('condition_occurrence.parquet'); print('Unique codes:', df.select('code').n_unique())"
```

## Future Enhancements

1. **Concept Hierarchy Integration**: Use SNOMED CT hierarchies for better code organization
2. **Cross-Mapping**: Leverage SNOMED CT's ability to map to other vocabularies
3. **Semantic Embeddings**: Use SNOMED CT relationships for better model embeddings
4. **Vocabulary Filtering**: Implement frequency-based filtering for rare concepts

## References

- [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/)
- [SNOMED CT](https://www.snomed.org/)
- [All of Us Research Program](https://allofus.nih.gov/)
- [ETHOS Documentation](https://github.com/suinleelab/ethos)
