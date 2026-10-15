"""Train (natural, imbalanced) vs test (balanced) label distribution, with DIR shot regions.
Inputs: results/train_counts_per_bin.csv, results/reg-analysis/test_preds.csv. Output: results/fig_split_distribution.png"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = pd.read_csv("results/train_counts_per_bin.csv").set_index("temp_c")["train_count"]
te = pd.read_csv("results/reg-analysis/test_preds.csv")["true_temp_c"].round().astype(int).value_counts().sort_index()
col = {"many": "#2a78d6", "med": "#eda100", "few": "#e34948"}
shot = lambda t: "many" if c.get(t, 0) > 100 else ("few" if c.get(t, 0) < 20 else "med")
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"hspace": .3})
a.bar(c.index, c.values, width=.9, color=[col[shot(t)] for t in c.index])
a.set_title(f"(a) Training set: natural, imbalanced ({c.sum():,} images). Bars colored by shot group: many (>100), medium (20 to 100), few (<20)", fontsize=10, loc="left")
a.set_ylabel("# images")
b.bar(te.index, te.values, width=.9, color=[col[shot(t)] for t in te.index])
b.set_title(f"(b) Test set: balanced, at most 50 images per 1 °C bin ({te.sum():,} images). Tails stay sparse because few images exist", fontsize=10, loc="left")
b.set_ylabel("# images"); b.set_xlabel("Temperature (°C)"); b.set_xlim(-30, 50)
for ax in (a, b):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=.8); ax.set_axisbelow(True)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=col[k], label=l) for k, l in [("many", "Many-shot"), ("med", "Medium-shot"), ("few", "Few-shot")]], frameon=False, fontsize=9)
fig.savefig("results/fig_split_distribution.png", dpi=160, bbox_inches="tight")
print("saved results/fig_split_distribution.png")
