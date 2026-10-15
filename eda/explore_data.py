"""
Exploratory data analysis for the SkyFinder dataset.
Focuses on temperature as the regression target:
  - Column overview and missing values
  - Temperature units and distribution
  - Per-camera temperature ranges (shows location-driven imbalance)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="SkyFinder temperature EDA")
    p.add_argument("--metadata", default="data/complete_table_with_mcr.csv",
                   help="Path to metadata CSV")
    p.add_argument("--fig_dir", default="eda/figures",
                   help="Directory to save figures")
    return p.parse_args()


def main():
    args = parse_args()
    import os
    os.makedirs(args.fig_dir, exist_ok=True)

    # Load
    df = pd.read_csv(args.metadata)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}\n")

    # Temperature columns
    for col in ["TempM", "TempI"]:
        raw = df[col]
        missing_sentinel = raw.isin([-9999, -9999.0, -999, -999.0]).sum()
        nan_count = raw.isna().sum()
        print(f"{col}:")
        print(f"  dtype={raw.dtype}, NaN={nan_count}, "
              f"sentinel(-9999/-999)={missing_sentinel}")
        valid = raw[~raw.isin([-9999, -9999.0, -999, -999.0]) & raw.notna()]
        print(f"  valid count={len(valid)}")
        print(f"  min={valid.min():.1f}, max={valid.max():.1f}, "
              f"mean={valid.mean():.1f}, median={valid.median():.1f}")
        print()

    df["temp_c"] = df["TempM"].replace([-9999, -9999.0, -999, -999.0], np.nan)
    valid_mask = df["temp_c"].notna()
    print(f"Usable rows (valid temp_c): {valid_mask.sum()} / {len(df)} "
          f"({valid_mask.mean()*100:.1f}%)\n")

    # Camera summary
    n_cams = df["CamId"].nunique()
    print(f"Number of unique cameras: {n_cams}")
    cam_counts = df["CamId"].value_counts()
    print(f"Images per camera: min={cam_counts.min()}, max={cam_counts.max()}, "
          f"median={cam_counts.median():.0f}\n")

    # Weather condition counts
    if "Conds" in df.columns:
        print("Weather conditions (top 15):")
        print(df["Conds"].value_counts().head(15).to_string())
        print()

    for flag in ["Fog", "Rain", "Snow", "night"]:
        if flag in df.columns:
            print(f"{flag}: {df[flag].sum()} rows flagged "
                  f"({df[flag].mean()*100:.1f}%)")
    print()

    # Plot 1: Overall temperature histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    temps = df.loc[valid_mask, "temp_c"]
    ax.hist(temps, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Number of images")
    ax.set_title("SkyFinder - Temperature distribution (all cameras)")
    ax.axvline(temps.median(), color="red", ls="--",
               label=f"median={temps.median():.1f} C")
    ax.legend()
    fig.tight_layout()
    path1 = os.path.join(args.fig_dir, "fig_temp_histogram.png")
    fig.savefig(path1, dpi=150)
    print(f"Saved: {path1}")

    # Plot 2: Temperature range per camera (box plot)
    df_valid = df.loc[valid_mask].copy()
    cam_order = (df_valid.groupby("CamId")["temp_c"]
                 .median().sort_values().index.astype(str))
    df_valid["CamId_str"] = df_valid["CamId"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_valid, x="CamId_str", y="temp_c", order=cam_order,
                fliersize=1, linewidth=0.6, ax=ax)
    ax.set_xlabel("Camera ID (sorted by median temp)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("SkyFinder - Temperature range per camera")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    path2 = os.path.join(args.fig_dir, "fig_temp_per_camera.png")
    fig.savefig(path2, dpi=150)
    print(f"Saved: {path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
