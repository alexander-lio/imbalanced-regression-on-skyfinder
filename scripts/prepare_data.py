"""
Build data/skyfinder.csv for DIR training from SkyFinder metadata.

Cleaning:
  - Drop TempM sentinels (-9999, -999).
  - Drop camera 19834 (text overlay = label leakage).
  - Drop rows where TempM is NaN.
  - Flag per-camera outliers (|temp - median| > 3*IQR, IQR floor 5 C).
  - Keep only rows whose image file exists on disk.

Labelling:
  - temp_c = round(TempM) to nearest integer C
  - label  = temp_c + 30  (non-negative index for DIR binning)

Splitting (time-based, season-balanced):
  - ISO week from filename -> week % 10:  0 = test pool, 5 = val pool, else train.
  - Balanced test/val: at most N images per 1 C bin (50 test, 25 val).
  - Leftover pool images are discarded, not added to train.

Output columns: path, label, temp_c, split, CamId, is_night
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Build data/skyfinder.csv from SkyFinder metadata.")
    p.add_argument("--metadata", default="data/complete_table_with_mcr.csv",
                   help="Path to complete_table_with_mcr.csv")
    p.add_argument("--images_dir", default="data/images",
                   help="Root directory containing <CamId>/<Filename> images")
    p.add_argument("--out", default="data/skyfinder.csv",
                   help="Output CSV path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_per_bin", type=int, default=50,
                   help="Max images per 1 C bin in test set")
    p.add_argument("--val_per_bin", type=int, default=25,
                   help="Max images per 1 C bin in val set")
    return p.parse_args()


SENTINEL_VALUES = {-9999, -999}
DROP_CAMERAS = {19834}
OUTLIER_IQR_MULT = 3
IQR_FLOOR = 5.0
LABEL_OFFSET = 30


def filename_to_week(filename):
    """Extract ISO week number from YYYYMMDD_HHMMSS.jpg filename."""
    date_str = filename.split("_")[0]
    dt = datetime.strptime(date_str, "%Y%m%d")
    return dt.isocalendar()[1]


def assign_pool(week):
    mod = week % 10
    if mod == 0:
        return "test_pool"
    elif mod == 5:
        return "val_pool"
    return "train"


def balanced_sample(pool_df, max_per_bin, name, seed):
    """Sample at most max_per_bin images per temp_c bin."""
    sampled = []
    for _, grp in pool_df.groupby("temp_c"):
        if len(grp) <= max_per_bin:
            sampled.append(grp)
        else:
            sampled.append(grp.sample(n=max_per_bin, random_state=seed))
    result = pd.concat(sampled) if sampled else pd.DataFrame()
    print(f"  {name}: {len(pool_df)} pool -> {len(result)} sampled "
          f"({len(pool_df) - len(result)} discarded)")
    return result


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load
    print("Loading CSV...")
    df = pd.read_csv(args.metadata)
    print(f"  Raw rows: {len(df)}")

    # Drop cameras with label leakage
    n_before = len(df)
    df = df[~df["CamId"].isin(DROP_CAMERAS)]
    print(f"  Dropped cameras {DROP_CAMERAS}: {n_before - len(df)} rows removed")

    # Drop sentinel TempM
    n_before = len(df)
    df = df[~df["TempM"].isin(SENTINEL_VALUES)]
    print(f"  Dropped TempM sentinels {SENTINEL_VALUES}: "
          f"{n_before - len(df)} rows removed")

    # Drop NaN TempM
    n_before = len(df)
    df = df[df["TempM"].notna()]
    print(f"  Dropped NaN TempM: {n_before - len(df)} rows removed")

    # Keep only rows with images on disk
    df["path"] = df.apply(
        lambda r: os.path.join(str(r["CamId"]), r["Filename"]), axis=1)
    df["img_exists"] = df["path"].apply(
        lambda p: os.path.isfile(os.path.join(args.images_dir, p)))

    n_total = len(df)
    n_found = df["img_exists"].sum()
    coverage = n_found / n_total if n_total else 0

    # Warn loudly if image coverage is low
    if coverage < 0.90:
        all_cams = set(df["CamId"].unique())
        cams_on_disk = set()
        if os.path.isdir(args.images_dir):
            cams_on_disk = {
                int(d) for d in os.listdir(args.images_dir)
                if os.path.isdir(os.path.join(args.images_dir, d)) and d.isdigit()
            }
        missing_cams = sorted(all_cams - cams_on_disk)
        print(f"\n  WARNING: only {coverage:.1%} of metadata rows have an "
              f"image on disk ({n_found}/{n_total})")
        print(f"   Missing cameras ({len(missing_cams)}): "
              f"{missing_cams}")
        print(f"   Run: python scripts/download_images.py "
              f"--images_dir {args.images_dir} \n")

    n_before = len(df)
    df = df[df["img_exists"]].drop(columns=["img_exists"])
    print(f"  Kept rows with images on disk: {len(df)} "
          f"(dropped {n_before - len(df)})")

    if len(df) == 0:
        print("ERROR: no images found. Check --images_dir.", file=sys.stderr)
        sys.exit(1)

    # Per-camera outlier flagging
    print(f"\nPer-camera outlier detection (|temp - median| > {OUTLIER_IQR_MULT} * IQR):")
    outlier_mask = pd.Series(False, index=df.index)
    for cam, grp in df.groupby("CamId"):
        med = grp["TempM"].median()
        q1 = grp["TempM"].quantile(0.25)
        q3 = grp["TempM"].quantile(0.75)
        iqr = max(q3 - q1, IQR_FLOOR)
        is_outlier = (grp["TempM"] - med).abs() > OUTLIER_IQR_MULT * iqr
        n_out = is_outlier.sum()
        if n_out > 0:
            print(f"  Camera {cam}: dropping {n_out} outlier(s), "
                  f"median={med:.1f}, IQR={iqr:.1f}")
            for idx in grp.index[is_outlier]:
                row = df.loc[idx]
                print(f"    CamId={cam}  Filename={row['Filename']}  "
                      f"TempM={row['TempM']}")
        outlier_mask |= is_outlier

    n_outliers = outlier_mask.sum()
    df = df[~outlier_mask]
    print(f"  Total outliers dropped: {n_outliers}")
    print(f"  Remaining rows: {len(df)}")

    # is_night flag
    df["is_night"] = (df["night"] > 0.5).astype(int)
    print(f"\nNight images: {df['is_night'].sum()} / {len(df)} "
          f"({100 * df['is_night'].mean():.1f}%)")

    # Label construction
    df["temp_c"] = df["TempM"].round(0).astype(int)
    df["label"] = df["temp_c"] + LABEL_OFFSET
    print(f"\nLabel range: {df['label'].min()} to {df['label'].max()} "
          f"(temp_c: {df['temp_c'].min()} to {df['temp_c'].max()})")

    # Time-based split
    df["week"] = df["Filename"].apply(filename_to_week)
    df["split"] = df["week"].apply(assign_pool)
    print(f"\nPool assignment (before balancing):")
    print(f"  train:     {(df['split'] == 'train').sum()}")
    print(f"  test_pool: {(df['split'] == 'test_pool').sum()}")
    print(f"  val_pool:  {(df['split'] == 'val_pool').sum()}")

    # Balanced test/val sampling
    print("\nBalanced sampling:")
    train_df = df[df["split"] == "train"].copy()
    train_df["split"] = "train"

    test_df = balanced_sample(
        df[df["split"] == "test_pool"], args.test_per_bin, "test", args.seed)
    test_df["split"] = "test"

    val_df = balanced_sample(
        df[df["split"] == "val_pool"], args.val_per_bin, "val", args.seed)
    val_df["split"] = "val"

    # Combine and save
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out = out[["path", "label", "temp_c", "split", "CamId", "is_night"]]

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SPLIT SIZES")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        s = out[out["split"] == split]
        print(f"  {split:5s}: {len(s):5d} images, "
              f"temp range [{s['temp_c'].min()}, {s['temp_c'].max()}] C, "
              f"{s['temp_c'].nunique()} unique bins")

    train_labels = out[out["split"] == "train"]["label"].astype(int)
    label_counts = Counter(train_labels)
    many = sum(1 for v in label_counts.values() if v > 100)
    medium = sum(1 for v in label_counts.values() if 20 <= v <= 100)
    few = sum(1 for v in label_counts.values() if v < 20)
    print(f"\nShot groups (train, 1 C bins):")
    print(f"  Many (>100):     {many} bins")
    print(f"  Medium (20-100): {medium} bins")
    print(f"  Few (<20):       {few} bins")

    # Per-split histogram (text)
    print("\nPer-split label distribution (temp_c):")
    for split in ["train", "val", "test"]:
        s = out[out["split"] == split]
        hist = s["temp_c"].value_counts().sort_index()
        print(f"\n  [{split}]")
        for temp, cnt in hist.items():
            bar = "#" * min(cnt, 80)
            print(f"    {temp:4d} C: {cnt:4d} {bar}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
    print(f"Total rows: {len(out)}")

    # Histogram figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, split in zip(axes, ["train", "val", "test"]):
            s = out[out["split"] == split]
            bins = range(s["temp_c"].min(), s["temp_c"].max() + 2)
            ax.hist(s["temp_c"], bins=bins, edgecolor="black", alpha=0.7)
            ax.set_title(f"{split} (n={len(s)})")
            ax.set_xlabel("Temperature (C)")
            ax.set_ylabel("Count")
        plt.suptitle("SkyFinder Label Distribution by Split",
                     fontweight="bold")
        plt.tight_layout()
        fig_dir = os.path.dirname(args.out) or "."
        fig_path = os.path.join(fig_dir, "label_distribution.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Histogram saved to {fig_path}")
    except ImportError:
        print("matplotlib not available, skipping histogram figure")


if __name__ == "__main__":
    main()
