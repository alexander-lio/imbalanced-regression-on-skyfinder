Benchmarking results on SkyFinder-DIR (balanced test set, seed 0). Baseline = ERM, unweighted L1 loss. Lower is better; best DIR value in bold.

| Metrics | MAE All | MAE Many | MAE Med. | MAE Few | GM All | GM Many | GM Med. | GM Few |
|---|---|---|---|---|---|---|---|---|
| Shot (n) | 2213 | 2163 | 30 | 20 | 2213 | 2163 | 30 | 20 |
| Baseline | 3.53 | 3.43 | 7.38 | 8.55 | 2.20 | 2.15 | 4.96 | 7.19 |
| Baseline + LDS | 3.63 | 3.59 | 5.23 | **4.92** | 2.23 | 2.21 | **3.22** | 3.33 |
| Baseline + FDS | **3.53** | **3.44** | 7.11 | 7.85 | **2.21** | **2.17** | 4.99 | 5.69 |
| Baseline + LDS + FDS | 3.62 | 3.59 | **5.08** | 5.06 | 2.23 | 2.21 | 3.53 | **2.86** |
| **Best DIR vs. Baseline** (positive = improvement) | -0.00 | -0.01 | +2.30 | +3.63 | -0.01 | -0.02 | +1.75 | +4.32 |

Pearson correlation between per-bin test MAE and training label density (paper Fig. 2 and 3 analysis):

| Model | r(error, empirical density) | r(error, LDS-smoothed density) |
|---|---|---|
| Baseline | -0.47 | -0.48 |
| Baseline + LDS | -0.26 | -0.27 |
| Baseline + FDS | -0.48 | -0.48 |
| Baseline + LDS + FDS | -0.30 | -0.31 |
