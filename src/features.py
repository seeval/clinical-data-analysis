"""create variables 'features' for analysis and modeling using polars"""

import polars as pl


def calculate_clinical_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """calculates age at procedure, BMI, and 90-day readmission status."""
    return df.with_columns(
        # calculate age at procedure in years
        # NOTE - procedure date converted to date at filter step
        (
            (
                pl.col("procedure_date") - pl.col("birth_date").str.to_date()
            ).dt.total_days()
            / 365.25
        )
        .round(1)
        .alias("age_at_procedure"),
        # calculate bmi (weight * 703) / (height ^ 2)
        # source: https://github.com/AlexTheAnalyst/PythonYouTubeSeries/blob/main/Python%20Project%20for%20Beginners%20-%20BMI%20Calculator.ipynb
        (((pl.col("weight_lbs") * 703) / (pl.col("height_inches") ** 2)))
        .round(1)
        .alias("bmi"),
        # calculate 90 day readmission flag
        # encode as boolean -> 1 | 0
        pl.when(
            (pl.col("readmission_date").str.to_date() - pl.col("procedure_date"))
            .dt.total_days()
            .is_between(0, 90)
        )
        .then(1)
        .otherwise(0)
        .alias("readmit_90day"),
    )


def flag_iqr_outliers(df: pl.DataFrame, column: str, k: float = 1.5) -> pl.DataFrame:
    """flag values outside of IQR"""
    # referenced this article for calculation -- would consult with team :)
    # source: ttps://medium.com/@morepravin1989/outlier-detection-with-the-iqr-method-a-complete-guide-c0199bbc10bd
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)

    return df.with_columns(
        ((pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)).alias(
            f"{column}_outlier_iqr"
        )
    )
