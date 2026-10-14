"""
Post-hoc analysis of a training run's test predictions.

Reads <run_dir>/test_preds.csv (columns: path, CamId, is_night, true_temp_c,
pred_temp_c) and the full split CSV (--csv) to get training-set label counts.
Saves four artifacts into <run_dir>/analysis/:

  1. pred_vs_true.png   – scatter + y=x line + binned-mean curve
  2. stats.txt          – MAE, std, slope, baselines, day/night, per-camera
  3. worst_20.png       – grid of 20 highest-error test images
  4. error_by_bin.png   – MAE per 1 °C bin with train-count second axis
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


LABEL_OFFSET = 30


def parse_args():
    p = argparse.ArgumentParser(description="Analyze a training run's test predictions")
    p.add_argument("--run_dir", required=True, help="folder containing test_preds.csv")
    p.add_argument("--csv", default="data/skyfinder.csv", help="path to the split CSV")
    p.add_argument("--images_dir", default="data/images", help="root dir with <CamId>/<file>.jpg")
    return p.parse_args()


def load_data(args):
    preds_path = os.path.join(args.run_dir, "test_preds.csv")
    df_pred = pd.read_csv(preds_path)
    df_full = pd.read_csv(args.csv)
    return df_pred, df_full


# ── 1. pred_vs_true.png ─────────────────────────────────────────────────────

def plot_pred_vs_true(df, out_dir):
    true = df["true_temp_c"].values
    pred = df["pred_temp_c"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=1, alpha=0.15, rasterized=True)

    lo = min(true.min(), pred.min()) - 2
    hi = max(true.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")

    # binned-mean curve
    bin_edges = np.arange(np.floor(true.min()), np.ceil(true.max()) + 1, 1)
    bin_idx = np.digitize(true, bin_edges) - 1
    bin_centers, bin_means = [], []
    for i in range(len(bin_edges) - 1):
        mask = bin_idx == i
        if mask.sum() >= 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(pred[mask].mean())
    ax.plot(bin_centers, bin_means, "r-", lw=1.5, label="binned mean")

    ax.set_xlabel("True temperature (°C)")
    ax.set_ylabel("Predicted temperature (°C)")
    ax.set_title("Predicted vs True Temperature")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_true.png"), dpi=150)
    plt.close(fig)


# ── 2. stats.txt ─────────────────────────────────────────────────────────────

def write_stats(df, df_full, out_dir):
    true = df["true_temp_c"].values
    pred = df["pred_temp_c"].values
    errors = np.abs(pred - true)
    mae = errors.mean()

    # constant-mean baseline: predict train-set mean temp
    train_mean = (df_full.loc[df_full["split"] == "train", "label"] - LABEL_OFFSET).mean()
    baseline_mae = np.mean(np.abs(true - train_mean))

    # linear fit slope
    slope = np.polyfit(true, pred, 1)[0]

    lines = [
        f"MAE:             {mae:.3f} °C",
        f"std(pred):       {pred.std():.3f}",
        f"std(true):       {true.std():.3f}",
        f"linear fit slope (pred on true): {slope:.4f}",
        f"constant-mean baseline MAE:      {baseline_mae:.3f} °C  (train mean = {train_mean:.2f} °C)",
        "",
    ]

    # day vs night
    lines.append("MAE by day/night:")
    for night_val, label in [(0, "day"), (1, "night")]:
        mask = df["is_night"] == night_val
        n = mask.sum()
        if n > 0:
            lines.append(f"  {label:6s}: MAE {errors[mask].mean():.3f} °C  (n={n})")
        else:
            lines.append(f"  {label:6s}: n/a  (n=0)")
    lines.append("")

    # per-camera MAE sorted descending
    lines.append("MAE per camera (sorted by MAE desc):")
    cam_groups = df.groupby("CamId").apply(
        lambda g: pd.Series({"mae": np.abs(g["pred_temp_c"] - g["true_temp_c"]).mean(), "n": len(g)}),
        include_groups=False,
    ).sort_values("mae", ascending=False)
    for cam, row in cam_groups.iterrows():
        lines.append(f"  {str(cam):>8s}: MAE {row['mae']:.3f}  (n={int(row['n'])})")

    with open(os.path.join(out_dir, "stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── 3. worst_20.png ──────────────────────────────────────────────────────────

def plot_worst_20(df, images_dir, out_dir):
    df = df.copy()
    df["abs_error"] = np.abs(df["pred_temp_c"] - df["true_temp_c"])
    worst = df.nlargest(20, "abs_error")

    ncols = 5
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12))
    axes = axes.flatten()

    for i, (_, row) in enumerate(worst.iterrows()):
        ax = axes[i]
        img_path = os.path.join(images_dir, row["path"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(
            f"{row['CamId']}\ntrue={row['true_temp_c']:.1f}  pred={row['pred_temp_c']:.1f}\nerr={row['abs_error']:.1f}",
            fontsize=7,
        )
        ax.axis("off")

    fig.suptitle("20 Worst Predictions (largest |error|)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "worst_20.png"), dpi=150)
    plt.close(fig)


# ── 4. error_by_bin.png ──────────────────────────────────────────────────────

def plot_error_by_bin(df, df_full, out_dir):
    true = df["true_temp_c"].values
    pred = df["pred_temp_c"].values

    # train label counts per 1 °C bin
    train_labels = df_full.loc[df_full["split"] == "train", "label"].values
    train_temp_c = train_labels - LABEL_OFFSET

    lo = int(np.floor(min(true.min(), train_temp_c.min())))
    hi = int(np.ceil(max(true.max(), train_temp_c.max())))
    bin_edges = np.arange(lo, hi + 2, 1) - 0.5
    bin_centers = bin_edges[:-1] + 0.5

    # train counts per bin
    train_counts, _ = np.histogram(train_temp_c, bins=bin_edges)

    # test MAE per bin
    errors = np.abs(pred - true)
    bin_idx = np.digitize(true, bin_edges) - 1
    mae_per_bin = np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        mask = bin_idx == i
        if mask.sum() > 0:
            mae_per_bin[i] = errors[mask].mean()

    fig, ax1 = plt.subplots(figsize=(10, 4))

    color_bar = "#c0c0c0"
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, train_counts, width=0.9, color=color_bar, alpha=0.6, label="train count")
    ax2.set_ylabel("Train count per bin", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    valid = ~np.isnan(mae_per_bin)
    ax1.plot(bin_centers[valid], mae_per_bin[valid], "b.-", lw=1.2, markersize=3, label="test MAE", zorder=3)
    ax1.set_xlabel("True temperature (°C)")
    ax1.set_ylabel("MAE (°C)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    ax1.set_title("Test MAE per 1 °C Bin vs Train Density")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_by_bin.png"), dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    df_pred, df_full = load_data(args)
    print(f"Loaded {len(df_pred)} test predictions from {args.run_dir}")

    plot_pred_vs_true(df_pred, out_dir)
    print("  saved pred_vs_true.png")

    write_stats(df_pred, df_full, out_dir)
    print("  saved stats.txt")

    plot_worst_20(df_pred, args.images_dir, out_dir)
    print("  saved worst_20.png")

    plot_error_by_bin(df_pred, df_full, out_dir)
    print("  saved error_by_bin.png")

    print(f"All outputs in {out_dir}")


if __name__ == "__main__":
    main()
