"""
Compare all runs on the balanced test set.

Reads results/<run>-analysis/test_preds.csv for each run and
results/train_counts_per_bin.csv (train images per 1 °C bin, from prepare_data.py).
Writes into results/:
  results_table.md               - MAE per group for every method (+ metadata baseline)
  comparison_error_by_bin.png    - test MAE per 1 °C bin, all methods on one axis,
                                   with train counts in a separate panel below
  comparison_calibration.png     - mean prediction per true-temperature bin vs y = x
Shot groups follow the DIR paper: train count per bin >100 many, 20-100 medium, <20 few.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"
RUNS = [  # (folder prefix, label, color, marker)
    ("reg",     "Baseline (ERM)",   "#2a78d6", "o"),
    ("lds",     "LDS",       "#eb6834", "s"),
    ("fds",     "FDS",       "#1baf7a", "^"),
    ("lds-fds", "LDS + FDS", "#eda100", "D"),
]
# Metadata-only baseline (scripts/metadata_baseline.py output, same test set)
METADATA = {"all": 4.712, "many": 4.511, "med": 11.710, "few": 15.984,
            "cold": 13.584, "hot": 8.137, "slope": float("nan")}

counts = pd.read_csv(os.path.join(RES, "train_counts_per_bin.csv")).set_index("temp_c")["train_count"]
def shot(t):
    c = counts.get(t, 0)
    return "many" if c > 100 else ("few" if c < 20 else "med")

def metrics(df):
    t = df["true_temp_c"].round().astype(int)
    e = (df["pred_temp_c"] - df["true_temp_c"]).abs()
    g = t.map(shot)
    return {
        "all": e.mean(), "many": e[g == "many"].mean(), "med": e[g == "med"].mean(),
        "few": e[g == "few"].mean(), "cold": e[t < -10].mean(), "hot": e[t > 35].mean(),
        "slope": np.polyfit(df["true_temp_c"], df["pred_temp_c"], 1)[0],
        "n": {k: int((g == k).sum()) for k in ["many", "med", "few"]} | {"cold": int((t < -10).sum()), "hot": int((t > 35).sum())},
    }

data, rows = {}, []
for key, label, color, marker in RUNS:
    p = os.path.join(RES, f"{key}-analysis", "test_preds.csv")
    if not os.path.exists(p):
        print(f"skip {key}: no test_preds.csv"); continue
    df = pd.read_csv(p)
    data[key] = df
    rows.append((label, metrics(df)))
n = rows[0][1]["n"]

# table
hdr = (f"| Method | All | Many (n={n['many']}) | Medium (n={n['med']}) | Few (n={n['few']}) "
       f"| < −10 °C (n={n['cold']}) | > 35 °C (n={n['hot']}) | Slope |")
lines = ["Test MAE (°C) on the balanced test set (2,213 images). Lower is better; slope of prediction on truth, ideal 1.0.", "",
         hdr, "|" + "---|" * 8]
def fmt(v): return "n/a" if v != v else f"{v:.2f}"
lines.append("| Camera + month mean (no image) | " + " | ".join(fmt(METADATA[k]) for k in ["all","many","med","few","cold","hot","slope"]) + " |")
for label, m in rows:
    lines.append(f"| {label} | " + " | ".join(fmt(m[k]) for k in ["all","many","med","few","cold","hot"]) + f" | {m['slope']:.3f} |")
open(os.path.join(RES, "results_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# error by bin (two stacked panels, shared x; no dual axis)
fig, (ax, axc) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
for a in (ax, axc):
    for t, c in counts.items():
        if c < 20:   a.axvspan(t - .5, t + .5, color="#e34948", alpha=.06, lw=0)
        elif c <= 100: a.axvspan(t - .5, t + .5, color="#eda100", alpha=.06, lw=0)
for key, label, color, marker in RUNS:
    if key not in data: continue
    df = data[key]
    t = df["true_temp_c"].round().astype(int)
    e = (df["pred_temp_c"] - df["true_temp_c"]).abs()
    s = e.groupby(t).mean()
    ax.plot(s.index, s.values, color=color, lw=2, marker=marker, ms=4, label=label)
ax.set_ylabel("Test MAE (°C)")
ax.set_ylim(bottom=0)
ax.grid(axis="y", color="#e6e6e3", lw=.8); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc="upper center")
ax.set_title("Test MAE per 1 °C bin (shaded bins: medium-shot in yellow, few-shot in red)", fontsize=11, loc="left")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False); axc.spines[sp].set_visible(False)
axc.bar(counts.index, counts.values, width=.9, color="#9a9a94")
axc.set_ylabel("Train images")
axc.set_xlabel("True temperature (°C)")
fig.savefig(os.path.join(RES, "comparison_error_by_bin.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# calibration: mean prediction per true bin
fig, ax = plt.subplots(figsize=(6.5, 6))
lo, hi = -25, 50
ax.plot([lo, hi], [lo, hi], color="#555", lw=1, ls="--", label="perfect (y = x)")
for key, label, color, marker in RUNS:
    if key not in data: continue
    df = data[key]
    t = df["true_temp_c"].round().astype(int)
    s = df.groupby(t)["pred_temp_c"].mean()
    s = s[df.groupby(t).size() >= 2]
    slope = np.polyfit(df["true_temp_c"], df["pred_temp_c"], 1)[0]
    ax.plot(s.index, s.values, color=color, lw=2, marker=marker, ms=3.5, label=f"{label} (slope {slope:.2f})")
ax.set_xlabel("True temperature (°C)"); ax.set_ylabel("Mean predicted temperature (°C)")
ax.set_title("Mean prediction per true-temperature bin", fontsize=11, loc="left")
ax.grid(color="#e6e6e3", lw=.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
fig.savefig(os.path.join(RES, "comparison_calibration.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# summary: MAE by group, grouped bars
grp = [("many", "Many-shot"), ("med", "Medium-shot"), ("few", "Few-shot"), ("cold", "< −10 °C"), ("hot", "> 35 °C")]
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.18; xs = np.arange(len(grp))
for i, ((key, label, color, marker), (_, m)) in enumerate(zip([r for r in RUNS if r[0] in data], rows)):
    vals = [m[g] for g, _ in grp]
    bars = ax.bar(xs + (i - 1.5) * w, vals, width=w - 0.02, color=color, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color="#444")
ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={n[g]})" for g, l in grp])
ax.set_ylabel("Test MAE (°C)"); ax.set_ylim(0, max(m["few"] for _, m in rows) + 1.5)
ax.set_title("Test MAE by shot group and temperature tail (lower is better)", fontsize=11, loc="left")
ax.grid(axis="y", color="#e6e6e3", lw=.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left")
fig.savefig(os.path.join(RES, "comparison_by_group.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
print("saved results/results_table.md, comparison_error_by_bin.png, comparison_calibration.png, comparison_by_group.png")
