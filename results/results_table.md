Test MAE (°C) on the balanced test set (2,213 images). Lower is better; slope of prediction on truth, ideal 1.0.

| Method | All | Many (n=2163) | Medium (n=30) | Few (n=20) | < −10 °C (n=50) | > 35 °C (n=72) | Slope |
|---|---|---|---|---|---|---|---|
| Camera + month mean (no image) | 4.71 | 4.51 | 11.71 | 15.98 | 13.58 | 8.14 | n/a |
| Baseline (ERM) | 3.53 | 3.43 | 7.38 | 8.55 | 7.59 | 4.21 | 0.871 |
| LDS | 3.63 | 3.59 | 5.23 | 4.92 | 4.96 | 3.79 | 0.926 |
| FDS | 3.53 | 3.44 | 7.11 | 7.85 | 6.53 | 4.57 | 0.873 |
| LDS + FDS | 3.62 | 3.59 | 5.08 | 5.06 | 4.78 | 3.98 | 0.934 |
