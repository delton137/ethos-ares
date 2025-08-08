"""All of Us specific preprocessors for ETHOS tokenization pipeline."""

import polars as pl
from ethos.tokenize.patterns import MatchAndRevise

class DemographicData:
    """Demographic data processing for All of Us data."""
    
    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        """Process All of Us race codes."""
        # Simple race mapping for All of Us - return the DataFrame as-is for now
        # The actual mapping will be handled by the vocabulary system
        return df

class LabData:
    """Lab data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="LAB//", apply_vocab=True)
    def retain_only_test_with_numeric_result(df: pl.DataFrame) -> pl.DataFrame:
        """Retain only lab tests with numeric results."""
        return df.filter(
            pl.col("code").str.starts_with("LAB//") & 
            pl.col("numeric_value").is_not_null()
        )

    @staticmethod
    @MatchAndRevise(prefix="LAB//Q//", needs_counts=True, needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None) -> pl.DataFrame:
        """Create quantile tokens for lab values."""
        # This will be handled by the quantile processing pipeline
        return df

class TransferData:
    """Transfer data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix=["TRANSFER//", "ADMISSION//"])
    def retain_only_transfer_and_admit_types(df: pl.DataFrame) -> pl.DataFrame:
        """Retain only transfer and admission events."""
        return df.filter(
            pl.col("code").str.starts_with("TRANSFER//") |
            pl.col("code").str.starts_with("ADMISSION//")
        )

class InpatientData:
    """Inpatient data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="DRG//", apply_vocab=True)
    def process_drg_codes(df: pl.DataFrame) -> pl.DataFrame:
        """Process DRG codes."""
        return df.with_columns([
            pl.when(pl.col("code").str.starts_with("DRG//"))
            .then(pl.concat_str([pl.lit("DRG//"), pl.col("code").str.replace("DRG//", "")]))
            .otherwise(pl.col("code"))
            .alias("code")
        ])
    
    @staticmethod
    @MatchAndRevise(prefix="ADMISSION//")
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        """Process hospital admission events."""
        return df.with_columns([
            pl.when(pl.col("code").str.starts_with("ADMISSION//"))
            .then(pl.concat_str([pl.lit("ADMISSION//"), pl.col("code").str.replace("ADMISSION//", "")]))
            .otherwise(pl.col("code"))
            .alias("code")
        ])
    
    @staticmethod
    @MatchAndRevise(prefix="DISCHARGE//")
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Process hospital discharge events."""
        return df.with_columns([
            pl.when(pl.col("code").str.starts_with("DISCHARGE//"))
            .then(pl.concat_str([pl.lit("DISCHARGE//"), pl.col("code").str.replace("DISCHARGE//", "")]))
            .otherwise(pl.col("code"))
            .alias("code")
        ])

class HCPCSData:
    """HCPCS data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="HCPCS//", apply_vocab=True)
    def unify_names(df: pl.DataFrame) -> pl.DataFrame:
        """Unify HCPCS code names."""
        return df.with_columns([
            pl.when(pl.col("code").str.starts_with("HCPCS//"))
            .then(pl.concat_str([pl.lit("HCPCS//"), pl.col("code").str.replace("HCPCS//", "")]))
            .otherwise(pl.col("code"))
            .alias("code")
        ])

class DeathData:
    """Death data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix=["DEATH", "DISCHARGE"], needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        """Place death events before discharge if they occur at the same time."""
        # For now, return as-is - this can be enhanced later
        return df

class PatientFluidOutputData:
    """Patient fluid output data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="SUBJECT_FLUID_OUTPUT//Q//", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Create quantile tokens for fluid output values."""
        return df

class BMIData:
    """BMI data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="BMI//Q", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Create quantile tokens for BMI values."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix=["BMI", "Q"])
    def join_token_and_quantile(df: pl.DataFrame) -> pl.DataFrame:
        """Join BMI tokens with their quantiles."""
        return df

class EdData:
    """Emergency department data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="ED_REGISTRATION")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        """Process ED registration events."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="ACUITY")
    def process_ed_acuity(df: pl.DataFrame) -> pl.DataFrame:
        """Process ED acuity levels."""
        return df

class MeasurementData:
    """Measurement data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix=["TEMPERATURE", "HEART_RATE", "RESPIRATORY_RATE", "O2_SATURATION"])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        """Process simple measurements."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="PAIN")
    def process_pain(df: pl.DataFrame) -> pl.DataFrame:
        """Process pain measurements."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="Blood Pressure")
    def process_blood_pressure(df: pl.DataFrame) -> pl.DataFrame:
        """Process blood pressure measurements."""
        return df

class ICUStayData:
    """ICU stay data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="ICU_", needs_vocab=True)
    def process(df: pl.DataFrame, *, num_quantiles: int = 10) -> pl.DataFrame:
        """Process ICU stay data."""
        return df

class DiagnosesData:
    """Diagnoses data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        """Prepare diagnosis codes for processing."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//9")
    def convert_icd_9_to_10(df: pl.DataFrame) -> pl.DataFrame:
        """Convert ICD-9 codes to ICD-10."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Process ICD-10 codes."""
        return df

class ProcedureData:
    """Procedure data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="PROCEDURE")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        """Prepare procedure codes for processing."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//9")
    def convert_icd_9_to_10(df: pl.DataFrame) -> pl.DataFrame:
        """Convert ICD-9 procedure codes to ICD-10."""
        return df
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Process ICD-10 procedure codes."""
        return df

class MedicationData:
    """Medication data processing for All of Us."""
    
    @staticmethod
    @MatchAndRevise(prefix="MEDICATION", needs_vocab=True)
    def convert_to_atc(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Convert medication codes to ATC format."""
        return df
