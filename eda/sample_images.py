"""
Sample images from downloaded cameras across different conditions.
Creates a contact sheet showing visual diversity: night, fog, snow, clear, warm, cold.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from PIL import Image
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a sample image contact sheet")
    p.add_argument("--metadata", default="data/complete_table_with_mcr.csv",
                   help="Path to metadata CSV")
    p.add_argument("--images_dir", default="data/images",
                   help="Root directory containing <CamId>/<Filename> images")
    p.add_argument("--fig_dir", default="eda/figures",
                   help="Directory to save the output figure")
    return p.parse_args()


def main():
    args = parse_args()
    import os
    os.makedirs(args.fig_dir, exist_ok=True)

    IMG_DIR = Path(args.images_dir)
    df = pd.read_csv(args.metadata)
    df["temp_c"] = df["TempM"].replace([-9999, -999], np.nan)

    # Only keep rows for cameras we downloaded
    downloaded_cams = [int(d.name) for d in IMG_DIR.iterdir() if d.is_dir()]
    df = df[df["CamId"].isin(downloaded_cams)].copy()
    print(f"Rows for downloaded cameras: {len(df)}")

    conditions = {
        "Snow (cold)":   df[(df["Snow"] == 1) & (df["temp_c"] < 0)],
        "Fog":           df[df["Fog"] == 1],
        "Rain":          df[df["Rain"] == 1],
        "Night (dark)":  df[df["Conds"].str.contains("Clear", na=False)
                            & (df["daylight"] < 0.3)],
        "Clear (warm)":  df[(df["Conds"] == "Clear") & (df["temp_c"] > 20)],
        "Clear (cold)":  df[(df["Conds"] == "Clear") & (df["temp_c"] < 0)],
        "Overcast":      df[df["Conds"] == "Overcast"],
        "Partly Cloudy": df[df["Conds"] == "Partly Cloudy"],
    }

    n_per = 4
    rows = []
    for label, sub in conditions.items():
        if len(sub) == 0:
            print(f"  {label}: no matches, skipping")
            continue
        sample = sub.sample(n=min(n_per, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            img_path = IMG_DIR / str(row["CamId"]) / row["Filename"]
            if img_path.exists():
                rows.append((label, row["Filename"], row["CamId"],
                             row["temp_c"], row.get("Conds", ""), img_path))

    print(f"\nTotal sample images found: {len(rows)}")

    n_cols = n_per
    n_rows_grid = len(conditions)
    fig, axes = plt.subplots(n_rows_grid, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows_grid))

    grouped = defaultdict(list)
    for label, fname, cam, temp, cond, path in rows:
        grouped[label].append((fname, cam, temp, cond, path))

    for row_idx, (label, items) in enumerate(grouped.items()):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(items):
                fname, cam, temp, cond, path = items[col_idx]
                img = Image.open(path)
                ax.imshow(img)
                temp_str = f"{temp:.0f}C" if not np.isnan(temp) else "?"
                ax.set_title(f"cam{cam} {temp_str}\n{cond}", fontsize=7)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=80, va="center")

    fig.suptitle("SkyFinder - Sample images across conditions",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(args.fig_dir, "fig_sample_images.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
