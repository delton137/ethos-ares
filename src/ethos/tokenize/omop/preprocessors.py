"""OMOP-specific preprocessors for ETHOS tokenization pipeline.

This module provides preprocessors that work with OMOP data that's already
been converted to MEDS format, avoiding MIMIC-specific column requirements.
"""

import polars as pl
from ethos.tokenize.patterns import MatchAndRevise


class DemographicData:
    """Demographic data processing for OMOP data."""
    
    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        """Process race codes for OMOP data."""
        race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
        race_minor = [
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", 
            "AMERICAN INDIAN/ALASKA NATIVE",
            "MULTIPLE RACE/ETHNICITY",
        ]
        
        race_priority_mapping = {"RACE//OTHER": 1, "RACE//UNKNOWN": 2}  # every other will get 0
        return (
            df.with_columns(
                code=pl.when(pl.col("text_value").is_in(race_unknown))
                .then(pl.lit("UNKNOWN"))
                .when(pl.col("text_value").is_in(race_minor))
                .then(pl.lit("OTHER"))
                .when(pl.col("text_value") == "SOUTH AMERICAN")
                .then(pl.lit("HISPANIC"))
                .when(pl.col("text_value") == "PORTUGUESE")
                .then(pl.lit("WHITE"))
                .when(pl.col("text_value").str.contains_any(["/", " "]))
                .then(pl.lit(None))
                .otherwise("text_value")
            )
            .with_columns(
                code=(
                    pl.lit("RACE//")
                    + pl.when(pl.col("code").is_null())
                    .then(pl.col("text_value").str.slice(0, pl.col("text_value").str.find("/| ")))
                    .otherwise("code")
                )
            )
            .group_by(MatchAndRevise.sort_cols[0], maintain_order=True)
            .agg(
                pl.col("code")
                .sort_by(
                    pl.col.code.replace_strict(
                        race_priority_mapping, default=0, return_dtype=pl.UInt8
                    )
                )
                .first(),
                pl.exclude("code").first(),
            )
            .select(df.columns)
        )

    @staticmethod
    @MatchAndRevise(prefix="MARITAL_STATUS", apply_vocab=True)
    def process_marital_status(df: pl.DataFrame) -> pl.DataFrame:
        """Process marital status codes for OMOP data."""
        return df.drop_nulls("text_value").with_columns(
            code=pl.lit("MARITAL//") + pl.col("text_value")
        )
