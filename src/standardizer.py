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
        .cast(pl.String)  # cast to string since goign from int -> str here
        .replace(value_map)
        .alias(target_col)
    ).drop(source_col)


def filter_most_recent(
    df: pl.DataFrame, patient_id_col: str, date_col: str
) -> pl.DataFrame:
    """
    filter dataframe on most recent procedure date to deduplicate as an example
    """
    return (
        df.with_columns(pl.col(date_col).str.to_date())
        .sort(by=[patient_id_col, date_col], descending=[False, True])
        .group_by(patient_id_col)
        .first()
    )


def impute_quarantine_from_clean(
    clean_df: pl.DataFrame,
    quarantine_df: pl.DataFrame,
    target_cols: list[str],
    strata_col: str,
    sentinel_value: float = 999.0,
) -> pl.DataFrame:
    """
    replaces missing or out of range in quarantine with median from strata in clean df
    """
    # helper to create a "filtered" version of a column for median calculation
    # ensuring sentinel values do not pollute the baseline medians.
    def _clean_expr(col_name):
        return pl.col(col_name).filter(pl.col(col_name) != sentinel_value)

    # compute stratum-level medians 
    stratum_medians_lookup = clean_df.group_by(strata_col).agg(
        [_clean_expr(col).median().alias(f"{col}_ref_median") for col in target_cols]
    )

    # compute global medians as a final fallback
    global_median_map = {col: _clean_expr(col).median() for col in target_cols}

    # convert all sentinels to explicit Nulls to trigger .fill_null()
    quarantine_with_nulls = quarantine_df.with_columns(
        [
            pl.when(pl.col(col) != sentinel_value)
            .then(pl.col(col))
            .otherwise(None)  
            .alias(col)
            for col in target_cols
        ]
    )

    # Join the baseline lookup tables onto the quarantine set
    imputed_df = quarantine_with_nulls.join(
        stratum_medians_lookup, on=strata_col, how="left"
    )

    # create expressions to fill Nulls with stratum median, then global median
    imputation_exprs = []
    for col in target_cols:
        ref_med_col = f"{col}_ref_median"
        g_med = global_median_map[col]

        expr = (
            pl.col(col)
            .fill_null(pl.col(ref_med_col))  # Tier 1: Clean Stratum median lookup
            .fill_null(g_med)  # Tier 2: Clean Global median fallback
            .alias(col)
        )
        imputation_exprs.append(expr)

    # impute and drop intermediate reference columns
    cols_to_drop = [f"{col}_ref_median" for col in target_cols]
    return imputed_df.with_columns(imputation_exprs).drop(cols_to_drop)
