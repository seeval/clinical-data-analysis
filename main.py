"""
main data analysis pipeline

1. loads configuration with schema
2. generates synthetic data
3. maps column names and value maps
4. merges with ledger
5. calculates new variables
6. validates schema
7. generates figures

"""

import numpy as np
import pandas as pd
import polars as pl

from src.config_loader import load_yaml_config
from src.standardizer import (
    map_column_names,
    normalize_categorical_values,
    filter_most_recent,
    impute_quarantine_from_clean,
)
from src.merger import MergeLedger
from src.features import calculate_clinical_metrics, flag_iqr_outliers
from src.data_validator import DataValidator
from src.reporter import generate_figures, generate_markdown_report

CONFIG_PATH = "config/schema_config.yml"


def main():
    # 1. load config
    config = load_yaml_config(filepath=CONFIG_PATH)

    # 2. generate synthetic data
    # for future unit testing, would move to another module
    np.random.seed(42)
    n = 1000

    # raw synthetic procedure data
    df_src1_raw = pl.DataFrame(
        {
            "raw_proc_id": [f"PRC_{i:06d}" for i in range(n)],
            # simulate repeat procedures per patient to test filtering
            "raw_pt_id": [f"PAT_{np.random.randint(1000, 1500):04d}" for _ in range(n)],
            "hosp_code": np.random.choice(
                [f"HOSP_{i:02d}" for i in range(1, 11)], size=n
            ),
            "surg_dt": pd.date_range("2025-01-01", periods=n, freq="h").strftime(
                "%Y-%m-%d"
            ),
            "joint_cd": np.random.choice([10, 20], size=n, p=[0.4, 0.6]),
            "surg_time_min": np.random.normal(120, 25, size=n).round(1),
        }
    )

    # raw synthetic readmission data
    df_src2_raw = pl.DataFrame(
        {
            "case_ref_id": [f"PRC_{i:06d}" for i in range(int(n * 0.90))],
            "re_dt": pd.date_range(
                "2025-01-15", periods=int(n * 0.90), freq="h"
            ).strftime("%Y-%m-%d"),
            "diag_cd": np.random.choice(["K40.9", "T84.0", "I10"], size=int(n * 0.90)),
        }
    )

    # raw synthetic patient data
    pts = df_src1_raw["raw_pt_id"].unique().to_list()
    df_src3_raw = pl.DataFrame(
        {
            "pt_ref_id": pts,
            "dob": (
                pd.Timestamp.now()
                - pd.to_timedelta(
                    np.random.normal(65 * 365.25, 10 * 365.25, len(pts)).clip(
                        18 * 365.25, 100 * 365.25
                    ),
                    unit="D",
                )
            ).strftime("%Y-%m-%d"),
            "ht_in": np.random.normal(67, 4, size=len(pts)).round(1),
            "wt_lbs": np.random.normal(195, 35, size=len(pts)).round(1),
        }
    )

    # 3. apply standardization
    col_maps = config["column_mappings"]
    value_maps = config["value_mappings"]

    df_proc = map_column_names(df_src1_raw, col_maps["source_1_procedures"])
    df_proc = normalize_categorical_values(
        df_proc,
        target_col="joint_type",
        source_col="joint_type_raw",
        value_map=value_maps["joint_type_raw"],
    )

    # as an example, filter procedure data and keep latest
    df_proc_filtered = filter_most_recent(
        df=df_proc, patient_id_col="patient_id", date_col="procedure_date"
    )

    # would move this to logging, embed in filter function
    print(f"Filtered {df_proc.height - df_proc_filtered.height} rows from procedure df")

    df_readmission = map_column_names(df_src2_raw, col_maps["source_2_readmissions"])
    df_pat = map_column_names(df_src3_raw, col_maps["source_3_baseline"])

    print(df_proc_filtered.head())
    print(df_readmission.head())
    print(df_pat.head())

    # 4. merge data and get merge_history
    # default merge how is left
    # more robust would have sources with full merge params
    # merge_priority, merge_how, merge_on and loop through merging
    ledger = MergeLedger()

    # primary merge
    df_merged = ledger.merge_data(
        left_df=df_proc_filtered,
        right_df=df_pat,
        on="patient_id",
        step_name="proc_JOIN_pat",
    )

    # secondary merge
    df_harmonized = ledger.merge_data(
        left_df=df_merged,
        right_df=df_readmission,
        on="procedure_id",
        step_name="procpat_JOIN_readmission",
    )

    print(df_harmonized.head())

    # calculate clinical variables and flag outliers via iqr bound detection
    df_features = calculate_clinical_metrics(df_harmonized)
    df_features = flag_iqr_outliers(df_features, column="surgical_duration_min")

    # print out df_features to view
    print(df_features.head())

    # filter on outliers to view, cols of interest
    # outlier col = surgical_duration_min_outlier_iqr
    # print out iqr for reference?
    print(
        df_features.filter(pl.col("surgical_duration_min_outlier_iqr")).select(
            ["surgical_duration_min_outlier_iqr", "surgical_duration_min"]
        )
    )

    # manaully add an out of range BMI value for random procedure id to check schema validation
    df_features = df_features.with_columns(
        pl.when(pl.col("procedure_id") == "PRC_000005")
        .then(pl.lit(999.0))
        .otherwise(pl.col("bmi"))
        .alias("bmi")
    )

    # apply schema validation
    validator = DataValidator(config)
    clean_df, quarantine_df = validator.validate_and_quarantine(df_features)

    # print out a quick audit summary
    # first print out merge
    print(f" --- Merge Ledger --- ")
    for entry in ledger.merge_history:
        print(
            f"Step: {entry['step_name']} | Left Input Shape: {entry['left_input_shape']} | Output Shape: {entry['output_shape']} | Unmatched Pct: {entry['unmatched_pct']:.2f}%"
        )

    # get metrics on valid vs invalid data
    # can see in invalid rows output, BMI of 999.0
    print(f" --- Data Output Metrics --- ")
    print(f"Len Valid Data: {clean_df.height}")
    print(f"Len Invalid Data: {quarantine_df.height}")

    # expect just one row at BMI but random seed could generate different values
    print(f"Invalid Rows:\n{quarantine_df}")

    # if you wanted to impute values from quarantine -
    # example nullify out of range float value (BMI) and impute strata median
    clean_bmi_df = impute_quarantine_from_clean(
        clean_df=clean_df,
        quarantine_df=quarantine_df,
        target_cols=["bmi"],
        strata_col="joint_type",
    )

    print(f"Cleaned BMI:\n{clean_bmi_df}")

    # generate figures first
    generate_figures(df=clean_df, output_image_path="reports/figures.png")

    # generate markdown and embed figures
    generate_markdown_report(
        df=clean_df,
        image_filename="figures.png",
        output_md_path="reports/clinical_data_summary.md",
    )


if __name__ == "__main__":
    main()
