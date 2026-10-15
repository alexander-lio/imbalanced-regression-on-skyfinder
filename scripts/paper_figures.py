"""
Paper-style tables and figures (Yang et al., ICML 2021) from saved test predictions.
No GPU needed. Inputs: results/<run>-analysis/test_preds.csv, results/train_counts_per_bin.csv.
Outputs (results/):
  table_paper_style.md     - Table 1 format: MAE and GM for All / Many / Medium / Few, + best-DIR-vs-baseline row
  fig_density_vs_error.png - Fig. 2/3 analogue: empirical vs LDS-smoothed train density vs baseline test error (Pearson r)
  fig_mae_gain.png         - Fig. 7 analogue: per-bin absolute MAE gain of LDS+FDS (and LDS) over the ERM baseline
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"
RUNS = [("reg", "Baseline"), ("lds", "+ LDS"), ("fds", "+ FDS"), ("lds-fds", "+ LDS + FDS")]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
REGION = {"few": "#e34948", "med": "#eda100"}

# data
c = pd.read_csv(os.path.join(RES, "train_counts_per_bin.csv")).set_index("temp_c")["train_count"]
idx = np.arange(-35, 56)
emp = pd.Series(0.0, index=idx); emp.loc[c.index] = c.values
# Exact repo LDS kernel: utils.get_lds_kernel_window("gaussian", ks=5, sigma=2), i.e. scipy gaussian_filter1d
# of a length-5 delta (reflect boundary) normalized to max 1. Values hardcoded so this script needs no scipy.
k = np.array([0.858, 0.946, 1.0, 0.946, 0.858])
eff = pd.Series(np.convolve(emp.values, k, mode="same"), index=idx)
def shot(t):
    n = c.get(t, 0)
    return "many" if n > 100 else ("few" if n < 20 else "med")

preds = {key: pd.read_csv(os.path.join(RES, f"{key}-analysis", "test_preds.csv")) for key, _ in RUNS}
def errs(df):
    t = df["true_temp_c"].round().astype(int)
    return t, (df["pred_temp_c"] - df["true_temp_c"]).abs()
def gmean(e): return float(np.exp(np.log(e).mean()))

# Table 1 style
groups = ["all", "many", "med", "few"]
tab = {}
for key, label in RUNS:
    t, e = errs(preds[key]); g = t.map(shot)
    sel = {"all": e, "many": e[g == "many"], "med": e[g == "med"], "few": e[g == "few"]}
    tab[label] = {**{f"mae_{s}": sel[s].mean() for s in groups}, **{f"gm_{s}": gmean(sel[s]) for s in groups}}
    n = {s: len(sel[s]) for s in groups}
cols = [f"mae_{s}" for s in groups] + [f"gm_{s}" for s in groups]
best = {col: min(tab[l][col] for _, l in RUNS[1:]) for col in cols}
L = ["Benchmarking results on SkyFinder-DIR (balanced test set, seed 0). Baseline = ERM, unweighted L1 loss. Lower is better; best DIR value in bold.", "",
     f"| Metrics | MAE All | MAE Many | MAE Med. | MAE Few | GM All | GM Many | GM Med. | GM Few |",
     "|---|---|---|---|---|---|---|---|---|",
     f"| Shot (n) | {n['all']} | {n['many']} | {n['med']} | {n['few']} | {n['all']} | {n['many']} | {n['med']} | {n['few']} |"]
for _, label in RUNS:
    cells = []
    for col in cols:
        v = tab[label][col]
        cells.append(f"**{v:.2f}**" if label != "Baseline" and abs(v - best[col]) < 1e-9 else f"{v:.2f}")
    L.append(f"| {label if label == 'Baseline' else 'Baseline ' + label} | " + " | ".join(cells) + " |")
gain = [tab["Baseline"][col] - best[col] for col in cols]
L.append("| **Best DIR vs. Baseline** (positive = improvement) | " + " | ".join(f"{g:+.2f}" for g in gain) + " |")
L += ["", "Pearson correlation between per-bin test MAE and training label density (paper Fig. 2 and 3 analysis):", "",
      "| Model | r(error, empirical density) | r(error, LDS-smoothed density) |", "|---|---|---|"]
for key, label in RUNS:
    t, e = errs(preds[key]); m = e.groupby(t).mean()
    r1 = np.corrcoef(m.values, emp[m.index].values)[0, 1]; r2 = np.corrcoef(m.values, eff[m.index].values)[0, 1]
    L.append(f"| {label if label == 'Baseline' else 'Baseline ' + label} | {r1:+.2f} | {r2:+.2f} |")
open(os.path.join(RES, "table_paper_style.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

def style(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
def shade(ax):
    for tt in idx:
        s = shot(tt) if tt in c.index else "few"
        if s != "many" and -30 <= tt <= 50:
            ax.axvspan(tt - .5, tt + .5, color=REGION[s], alpha=.07, lw=0)

# Fig. 2/3 analogue
t, e = errs(preds["reg"]); m = e.groupby(t).mean()
r_emp = np.corrcoef(m.values, emp[m.index].values)[0, 1]; r_eff = np.corrcoef(m.values, eff[m.index].values)[0, 1]
fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": .35})
axs[0].bar(idx, emp.values, width=.9, color=BLUE)
axs[0].set_title(f"(a) Empirical training label density. Pearson r with baseline test error: {r_emp:+.2f}", fontsize=10, loc="left", color=INK)
axs[1].bar(idx, eff.values, width=.9, color=BLUE)
axs[1].set_title(f"(b) LDS-smoothed effective density (Gaussian kernel, size 5, σ = 2). Pearson r: {r_eff:+.2f}", fontsize=10, loc="left", color=INK)
axs[2].bar(m.index, m.values, width=.9, color=ORANGE)
axs[2].set_title("(c) Baseline (ERM): test MAE per 1 °C bin on the balanced test set", fontsize=10, loc="left", color=INK)
axs[0].set_ylabel("# train images", color=MUTED); axs[1].set_ylabel("effective count", color=MUTED); axs[2].set_ylabel("MAE (°C)", color=MUTED)
axs[2].set_xlabel("Temperature (°C)", color=MUTED); axs[2].set_xlim(-30, 50)
for a in axs: style(a)
fig.savefig(os.path.join(RES, "fig_density_vs_error.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# Fig. 7 analogue
def per_bin(key):
    t, e = errs(preds[key]); return e.groupby(t).mean()
base = per_bin("reg")
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2], "hspace": .12})
shade(ax0); shade(ax1)
ax0.bar(idx, emp.values, width=.9, color="#9a9a94"); ax0.set_ylabel("# train images", color=MUTED)
ax0.set_title("Absolute MAE gain over the ERM baseline per 1 °C bin (positive = better; shaded bins: medium-shot yellow, few-shot red)",
              fontsize=10, loc="left", color=INK)
g = (base - per_bin("lds-fds")).dropna()
ax1.bar(g.index, g.values, width=.8, color=np.where(g.values >= 0, BLUE, ORANGE), label="+ LDS + FDS (bars)")
gl = (base - per_bin("lds")).dropna()
ax1.plot(gl.index, gl.values, color=INK, lw=1.2, marker="o", ms=3, label="+ LDS (line)")
ax1.axhline(0, color=MUTED, lw=1)
ax1.set_ylabel("MAE gain (°C)", color=MUTED); ax1.set_xlabel("True temperature (°C)", color=MUTED)
ax1.set_xlim(-30, 50); ax1.legend(frameon=False, fontsize=9, loc="upper center")
for a in (ax0, ax1): style(a)
fig.savefig(os.path.join(RES, "fig_mae_gain.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
print("saved table_paper_style.md, fig_density_vs_error.png, fig_mae_gain.png")
