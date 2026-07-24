"""write a markdown summary table and embed some simple plots"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


def generate_figures(df: pl.DataFrame, output_image_path: str) -> None:
    """create seaborn plots (box and scatter)"""

    # convert polars df to pandas for seaborn compatibility
    pdf = df.to_pandas()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.boxplot(
        data=pdf,
        x="joint_type",
        y="bmi",
        hue="readmit_90day",
        palette={0: "#1f77b4", 1: "#d62728"},
        ax=axes[0]
    )
    axes[0].set_title("BMI by Joint Type & Readmission")

    sns.scatterplot(
        data=pdf,
        x="age_at_procedure",
        y="surgical_duration_min",
        hue="joint_type",
        alpha=0.7,
        ax=axes[1]
    )
    axes[1].set_title("Age vs. Surgical Duration")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()


def generate_markdown_report(df: pl.DataFrame, image_filename: str, output_md_path: str) -> None:
    """
    create some descriptive statistics and write to markdown report
    """

    # calculate stats using polars aggregation
    stats = df.group_by("joint_type").agg(
        n=pl.len(),
        age_mean=pl.col("age_at_procedure").mean().round(1),
        bmi_median=pl.col("bmi").median().round(1),
        surg_min_mean=pl.col("surgical_duration_min").mean().round(1),
        readmit_n=pl.col("readmit_90day").sum()
    ).sort("joint_type")

    # Simple Markdown string construction
    md_content = f"""# Summary Report

## Cohort Overview

| Joint Type | N | Mean Age | Median BMI | Mean Surg Min | 90-Day Readmits |
|---|---|---|---|---|---|
"""

    for row in stats.iter_rows(named=True):
        md_content += f"| {row['joint_type']} | {row['n']} | {row['age_mean']} | {row['bmi_median']} | {row['surg_min_mean']} | {row['readmit_n']} |\n"

    # embed images into markdown 
    md_content += f"""
## Analytical Distributions

![Clinical Distributions]({image_filename})
"""

    # write markdown file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
