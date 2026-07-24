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
    # get sample size
    total_n = df.height

    summary_df = df.group_by("joint_type").agg(
        n_count=pl.len(),
        age_mean=pl.col("age_at_procedure").mean().round(1),
        age_std=pl.col("age_at_procedure").std().round(1),
        bmi_median=pl.col("bmi").median().round(1),
        bmi_q25=pl.col("bmi").quantile(0.25).round(1),
        bmi_q75=pl.col("bmi").quantile(0.75).round(1),
        surg_mean=pl.col("surgical_duration_min").mean().round(1),
        surg_std=pl.col("surgical_duration_min").std().round(1),
        readmit_count=pl.col("readmit_90day").sum(),
        readmit_pct=(pl.col("readmit_90day").mean() * 100).round(1)
    ).sort("joint_type")

    hips = summary_df.filter(pl.col("joint_type") == "HIP")
    knees = summary_df.filter(pl.col("joint_type") == "KNEE")

    # get metrics by strata for formatting
    h_n, h_pct = hips["n_count"][0], (hips["n_count"][0] / total_n * 100)
    k_n, k_pct = knees["n_count"][0], (knees["n_count"][0] / total_n * 100)

    h_age = f"{hips['age_mean'][0]} (±{hips['age_std'][0]})"
    k_age = f"{knees['age_mean'][0]} (±{knees['age_std'][0]})"

    h_bmi = f"{hips['bmi_median'][0]} [{hips['bmi_q25'][0]}–{hips['bmi_q75'][0]}]"
    k_bmi = f"{knees['bmi_median'][0]} [{knees['bmi_q25'][0]}–{knees['bmi_q75'][0]}]"

    h_surg = f"{hips['surg_mean'][0]} (±{hips['surg_std'][0]})"
    k_surg = f"{knees['surg_mean'][0]} (±{knees['surg_std'][0]})"

    h_readm = f"{hips['readmit_count'][0]} ({hips['readmit_pct'][0]}%)"
    k_readm = f"{knees['readmit_count'][0]} ({knees['readmit_pct'][0]}%)"

    md_content = f"""# Clinical Summary Report

## Demographic & Procedural Characteristics

| Variable / Stratum | Hip (N = {h_n}) | Knee (N = {k_n}) | Total Cohort (N = {total_n}) |
| :--- | :---: | :---: | :---: |
| **Cohort Distribution, N (%)** | {h_n} ({h_pct:.1f}%) | {k_n} ({k_pct:.1f}%) | {total_n} (100.0%) |
| **Age at Procedure (Years), Mean ± SD** | {h_age} | {k_age} | -- |
| **Body Mass Index (BMI), Median [IQR]** | {h_bmi} | {k_bmi} | -- |
| **Surgical Duration (Minutes), Mean ± SD** | {h_surg} | {k_surg} | -- |
| **90-Day Readmission Rate, N (%)** | {h_readm} | {k_readm} | -- |

---

## Graph Distributions

![Figures]({image_filename})
"""

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
