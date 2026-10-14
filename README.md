# SkyFinder Temperature Prediction (DIR)

Predict air temperature (C) from a single webcam image using the
[SkyFinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) dataset, comparing a
vanilla CNN baseline against
[Deep Imbalanced Regression](https://github.com/YyzHarry/imbalanced-regression)
(LDS / FDS, Yang et al. ICML 2021).

## Repository layout

```
data/                        # metadata CSV + images (not committed, see below)
eda/                         # exploratory data analysis scripts and figures
scripts/
  download_images.py         # fetch metadata + camera zips from Valdosta server
  prepare_data.py            # clean metadata -> data/skyfinder.csv
dir_skyfinder/               # training code (adapted from agedb-dir/)
notebooks/                   # Colab notebooks (training happens here)
results/                     # saved predictions, tables, plots
NOTES.md                     # decision log
```

## Setup

### 1. Get the data

The metadata CSV (88 MB) and camera images (~10 GB) are not committed.

```bash
# Download metadata only (works on Mac or Colab)
python scripts/download_images.py --metadata_only

# Download all camera images (run on Colab for speed/disk)
python scripts/download_images.py --images_dir data/images

# Optional: resize to save disk
python scripts/download_images.py --images_dir data/images --resize 256
```

### 2. Build the training CSV

```bash
python scripts/prepare_data.py \
    --metadata data/complete_table_with_mcr.csv \
    --images_dir data/images \
    --out data/skyfinder.csv
```

### 3. Train (on Colab)

```bash
cd dir_skyfinder
python train.py --data_dir ../data --dataset skyfinder --loss l1 --epoch 30
```

## Environment

- Mac: EDA, data prep, code editing (no GPU needed)
- Colab: training, evaluation (GPU required)
- Python 3.14.2 locally; see `requirements.txt`
