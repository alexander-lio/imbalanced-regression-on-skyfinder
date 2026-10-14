"""
No-image baseline: predict temperature as the mean training temperature
for each (CamId, Month) group. Reports MAE overall, by shot group
(Many/Medium/Few), and by temperature range (<-10, -10..35, >35 C).
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import gmean


LABEL_OFFSET = 30


def parse_args():
    p = argparse.ArgumentParser(description="Metadata-only temperature baseline")
    p.add_argument("--csv", default="data/skyfinder.csv", help="path to split CSV")
    return p.parse_args()


def shot_metrics_simple(preds_c, labels_c, train_labels_int, label_offset=30):
    """MAE by Many/Median/Low shot groups (thresholds 100/20 on train counts)."""
    preds_lbl = preds_c + label_offset
    labels_lbl = labels_c + label_offset
    train_counts = Counter(train_labels_int)

    groups = {'many': [], 'median': [], 'low': []}
    for lbl in np.unique(labels_lbl):
        mask = labels_lbl == lbl
        errs = np.abs(preds_lbl[mask] - labels_lbl[mask])
        tc = train_counts.get(int(lbl), 0)
        if tc > 100:
            groups['many'].extend(errs)
        elif tc < 20:
            groups['low'].extend(errs)
        else:
            groups['median'].extend(errs)

    result = {}
    for name, errs in groups.items():
        if errs:
            result[name] = {'l1': np.mean(errs), 'gmean': gmean(errs).astype(float), 'n': len(errs)}
        else:
            result[name] = {'l1': float('nan'), 'gmean': float('nan'), 'n': 0}
    return result


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df['temp_c'] = df['label'] - LABEL_OFFSET
    # Extract month from path: <CamId>/YYYYMMDD_HHMMSS.jpg
    df['month'] = df['path'].apply(lambda p: int(p.split('/')[-1][4:6]))

    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    # Build lookup: mean train temp per (CamId, Month)
    lookup = train.groupby(['CamId', 'month'])['temp_c'].mean().to_dict()
    # Fallback: per-camera mean, then global mean
    cam_mean = train.groupby('CamId')['temp_c'].mean().to_dict()
    global_mean = train['temp_c'].mean()

    preds = []
    for _, row in test.iterrows():
        key = (row['CamId'], row['month'])
        if key in lookup:
            preds.append(lookup[key])
        elif row['CamId'] in cam_mean:
            preds.append(cam_mean[row['CamId']])
        else:
            preds.append(global_mean)

    preds_c = np.array(preds)
    labels_c = test['temp_c'].values

    mae = np.mean(np.abs(preds_c - labels_c))
    gm = gmean(np.abs(preds_c - labels_c) + 1e-8).astype(float)

    print(f"Metadata baseline: predict mean train temp per (CamId, Month)")
    print(f"Test set: {len(test)} images")
    print(f"\nOverall MAE: {mae:.3f} C   G-Mean: {gm:.3f}")

    # Shot groups
    train_labels_int = train['label'].astype(int).values
    shot = shot_metrics_simple(preds_c, labels_c, train_labels_int)
    for name in ['many', 'median', 'low']:
        s = shot[name]
        print(f"  {name.capitalize():7s}: MAE {s['l1']:.3f}  G-Mean {s['gmean']:.3f}  (n={s['n']})")

    # Temperature-range groups
    print(f"\nTemp-range breakdown:")
    for rname, mask in [('< -10 C', labels_c < -10),
                        ('-10..35 C', (labels_c >= -10) & (labels_c <= 35)),
                        ('> 35 C', labels_c > 35)]:
        if mask.sum() > 0:
            rmae = np.mean(np.abs(preds_c[mask] - labels_c[mask]))
            print(f"  {rname:10s}: MAE {rmae:.3f} (n={mask.sum()})")
        else:
            print(f"  {rname:10s}: n/a (n=0)")


if __name__ == "__main__":
    main()
