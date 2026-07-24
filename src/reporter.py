"""write a markdown summary table and embed some simple plots"""

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def generate_figures(df: pl.DataFrame, output_image_path: str) -> None:
    """
    generates bmi box plot and scatter on age x surgical duration
    """

    # convert polars df to pandas for seaborn compatibility
    pdf = df.to_pandas()
    
    # map readmission from binary to categorical
    pdf["readmit_label"] = pdf["readmit_90day"].map({0: "No Readmission", 1: "90-Day Readmit"})
    
    # set theme and subplot options
    sns.set_theme(style="ticks", font="sans-serif")
    fig, axes = plt.subplots(
            nrows=2, 
            ncols=1,
            figsize=(9, 10), 
            sharex=False,
            )

    # color palette
    colors_readmit = {"No Readmission": "#2b5c8f", "90-Day Readmit": "#d95f02"}
    colors_joint = {"HIP": "#1b9e77", "KNEE": "#7570b3"}

    # bmi by joint type and readmission
    sns.boxplot(
        data=pdf,
        x="joint_type",
        y="bmi",
        hue="readmit_label",
        palette=colors_readmit,
        width=0.5,
        linewidth=1.2,
        fliersize=3,
        ax=axes[0]
    )
    
    axes[0].set_title("BMI Distribution by Joint Type & Readmission Status", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Surgical Joint Type", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Body Mass Index (BMI, $\\mathrm{kg/m^2}$)", fontsize=10, labelpad=6)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # move legend to top right
    axes[0].legend(
        title="Clinical Outcome",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc"
    )
    
    # scatter of age x surgical_duration
    sns.scatterplot(
        data=pdf,
        x="age_at_procedure",
        y="surgical_duration_min",
        hue="joint_type",
        palette=colors_joint,
        alpha=0.6,
        s=35,
        edgecolor="none",
        ax=axes[1]
    )

    axes[1].set_title("Patient Age vs. Surgical Duration by Joint Type", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Age at Procedure (Years)", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Surgical Duration (Minutes)", fontsize=10, labelpad=6)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # move legend outside top right
    axes[1].legend(
        title="Joint Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc"
    )
    
    # set layout
    sns.despine(top=True, right=True) # remove top and right spines
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
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

    # create a markdown string
    md_content = f"""# Summary Report

## Overview

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
