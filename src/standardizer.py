"""map columns and values"""

from typing import Dict, Any
import polars as pl


def map_column_names(df: pl.DataFrame, mapping_dict: Dict[str, str]) -> pl.DataFrame:
    """renames raw df columns to target schema names"""
    return df.rename(mapping_dict)


def normalize_categorical_values(
    df: pl.DataFrame, target_col: str, source_col: str, value_map: Dict[Any, Any]
) -> pl.DataFrame:
    """map raw categorical column values to target values as example"""
    return df.with_columns(
        pl.col(source_col)
        .cast(pl.String) # cast to string since goign from int -> str here
        .replace(value_map)
        .alias(target_col)
    ).drop(source_col)

def filter_most_recent(df: pl.DataFrame, patient_id_col: str, date_col: str) -> pl.DataFrame:
    """
    filter dataframe on most recent procedure date to deduplicate as an example 
    """
    return (
        df.with_columns(pl.col(date_col).str.to_date())
        .sort(by=[patient_id_col, date_col], descending=[False, True])
        .group_by(patient_id_col)
        .first()
    )
