"example test for standardizer and imputing median for out of range value"

import polars as pl
import pytest
from src.standardizer import impute_quarantine_from_clean


def test_impute_quarantine_from_clean():
    """
    test invalid 999.0 value is nullified and imputed with mean from clean_df
    """

    # create test clean df
    clean_df = pl.DataFrame(
        {
            "joint_type": ["HIP", "HIP", "HIP", "KNEE", "KNEE", "KNEE"],
            "bmi": [28.0, 30.0, 32.0, 38.0, 40.0, 42.0],
        }
    )

    # create quarantine_df with one 999.0 and one null
    quarantine_df = pl.DataFrame({"joint_type": ["HIP", "KNEE"], "bmi": [999.0, None]})

    # test function
    result_df = impute_quarantine_from_clean(
        clean_df=clean_df,
        quarantine_df=quarantine_df,
        target_cols=["bmi"],
        strata_col="joint_type",
        sentinel_value=999.0,
    )

    # assert: verify values match expected medians
    expected_hip_bmi = 30.0
    expected_knee_bmi = 40.0

    hip_result = result_df.filter(pl.col("joint_type") == "HIP")["bmi"][0]
    knee_result = result_df.filter(pl.col("joint_type") == "KNEE")["bmi"][0]

    assert (
        hip_result == expected_hip_bmi
    ), f"Expected HIP BMI {expected_hip_bmi}, got {hip_result}"
    assert (
        knee_result == expected_knee_bmi
    ), f"Expected KNEE BMI {expected_knee_bmi}, got {knee_result}"

    # Ensure no sentinel values or nulls remain
    assert (result_df["bmi"] == 999.0).sum() == 0
    assert result_df["bmi"].is_null().sum() == 0
