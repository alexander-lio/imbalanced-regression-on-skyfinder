# Deep Imbalanced Regression on SkyFinder

Predicting air temperature from a single outdoor webcam image, and testing whether
the label and feature distribution smoothing methods (LDS and FDS) from
[Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554)
(Yang et al., ICML 2021) help on the rare cold and hot temperatures.

The training code is the authors' [`agedb-dir`](https://github.com/YyzHarry/imbalanced-regression/tree/main/agedb-dir)
code, adapted to SkyFinder. The first commit in this repository is that code unmodified,
so `git diff 1119ef2 -- dir_skyfinder/` shows every change. Changed lines are marked `# CHANGED`.

## Task

- **Input:** one image from one of the SkyFinder webcams. The model sees pixels only.
- **Target:** air temperature in °C (`TempM` in the SkyFinder metadata, a nearby
  weather-station reading matched to the image by time).
- **Why it is a DIR problem:** most training images are between 0 and 30 °C. Images below
  −10 °C or above 35 °C are rare, and a standard model predicts them too close to the middle.

## Results

Test MAE in °C on the balanced test set (2,213 images), seed 0. Shot groups follow the
paper: training images per 1 °C bin above 100 (many), 20 to 100 (medium), below 20 (few).

| Method | All | Many (n=2163) | Medium (n=30) | Few (n=20) | < −10 °C (n=50) | > 35 °C (n=72) |
|---|---|---|---|---|---|---|
| Camera + month mean (no image) | 4.71 | 4.51 | 11.71 | 15.98 | 13.58 | 8.14 |
| Baseline (ERM, unweighted L1) | 3.53 | 3.43 | 7.38 | 8.55 | 7.59 | 4.21 |
| + LDS | 3.63 | 3.59 | 5.23 | 4.92 | 4.96 | 3.79 |
| + FDS | 3.53 | 3.44 | 7.11 | 7.85 | 6.53 | 4.57 |
| + LDS + FDS | 3.62 | 3.59 | 5.08 | 5.06 | 4.78 | 3.98 |

In short: LDS (with the square-root-inverse reweighting it is paired with) roughly halves
the error on rare temperatures at a small cost on common ones. FDS alone has a small effect,
and adding it to LDS changes little. SkyFinder's label histogram is already smooth, so the
smoothing step itself changes the density estimate very little; most of the LDS gain likely
comes from the reweighting. The medium and few-shot groups are small and only one seed was
run, so differences under about 1 °C in those groups are not reliable.

More tables and figures are in [`results/`](results/) (see "Figures" below).

## Data

[SkyFinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) (Mihail et al., WACV 2016) has about
90,000 images from 53 static webcams, built for sky segmentation. Its metadata includes weather
data for each frame. Processing (`scripts/prepare_data.py`):

1. Drop camera 19834, which prints the temperature on the image (label leakage).
2. Drop missing temperatures (sentinel −9999 and blank values).
3. Keep images that exist on disk. 6 of the 53 cameras could not be downloaded (HTTP 404).
4. Drop per-camera outliers: |t − camera median| > 3 × IQR, IQR floored at 5 °C (3 rows).
   Two were 50 °C labels that the raw METAR reports show are −2 °C and 9 °C (a visibility
   field misparsed as temperature); the third was a genuine 33 °C reading.
5. Labels: temperature rounded to 1 °C and shifted by +30, because the DIR code uses the
   integer label as a bin index (81 bins, −27 to 49 °C observed).
6. Split by ISO week of the filename timestamp (week % 10: 0 test, 5 validation, rest training),
   so near-identical frames from the same day never cross splits.
7. Balanced validation and test sets: at most 25 and 50 images per 1 °C bin. Leftover images
   from the held-out weeks are discarded.

Result: 68,319 images from 46 cameras (64,816 train, 1,290 validation, 2,213 test).

## Training setup

- ResNet-50 from the repository's `resnet.py`, initialized from ImageNet weights, ImageNet
  normalization, 224 × 224 input.
- Adam, learning rate 1e-4 (divided by 10 at epochs 10 and 13), 15 epochs, batch size 64,
  L1 loss, fp32, seed 0. The checkpoint with the lowest validation MAE is evaluated once on
  the test set.
- LDS and FDS use the settings from the authors' AgeDB-DIR commands: Gaussian kernel of size 5
  with σ = 2, square-root-inverse reweighting for LDS, momentum 0.9 for FDS.
- This differs from the paper, which trains from scratch for 90 epochs; the schedule was
  shortened for the pretrained backbone and a single GPU.
- Experiments were run on a single NVIDIA T4 GPU (Google Colab).

## Repository layout

```
dir_skyfinder/        training code adapted from the authors' agedb-dir (train.py, datasets.py, fds.py, resnet.py, loss.py, utils.py)
scripts/
  download_images.py  download the metadata CSV and per-camera image archives
  prepare_data.py     clean the metadata and build data/skyfinder.csv (labels and splits)
  metadata_baseline.py  camera + month mean reference model (no images)
  analyze_run.py      per-run analysis of test predictions (plots and stats)
  compare_runs.py     results table and comparison figures across runs
  paper_figures.py    paper-style table (MAE and GM by shot) and figures
  split_figure.py     train vs balanced test label distribution figure
eda/                  exploratory analysis of the raw metadata
results/              tables, figures, and per-run predictions and metrics
```

## Reproducing

Install dependencies (Python 3.10 or newer):

```bash
pip install -r requirements.txt
```

1. Download data (the images take several GB; resizing to 256 px on the short side saves space):

```bash
python scripts/download_images.py --metadata_only
python scripts/download_images.py --images_dir data/images --resize 256
```

A few SkyFinder JPEGs are truncated. They load with PIL's `LOAD_TRUNCATED_IMAGES`, which the
training code enables; files that still cannot be read should be removed before step 2.

2. Build the training CSV:

```bash
python scripts/prepare_data.py --metadata data/complete_table_with_mcr.csv \
    --images_dir data/images --out data/skyfinder.csv
```

3. Reference model without images:

```bash
python scripts/metadata_baseline.py --csv data/skyfinder.csv
```

4. Train (from `dir_skyfinder/`). Each run writes checkpoints, `training.log`,
   `test_preds.csv` and `test_metrics.json` to `<store_root>/<run name>/`.

```bash
cd dir_skyfinder
COMMON="--data_dir ../data/images --csv ../data/skyfinder.csv --store_root ../runs \
  --epoch 15 --lr 1e-4 --schedule 10 13 --batch_size 64 --workers 2 --seed 0"
LDS="--reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2"
FDS="--fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --start_update 0 --start_smooth 1"

python train.py $COMMON --store_name baseline
python train.py $COMMON --store_name lds $LDS
python train.py $COMMON --store_name fds $FDS
python train.py $COMMON --store_name lds_fds $LDS $FDS
cd ..
```

5. Analyse each run, then collect the results. The comparison scripts expect each run's
   `test_preds.csv`, `test_metrics.json` and analysis outputs in `results/<name>-analysis/`,
   with names `reg` (baseline), `lds`, `fds` and `lds-fds`:

```bash
python scripts/analyze_run.py --run_dir runs/<run folder> --csv data/skyfinder.csv --images_dir data/images
# copy <run folder>/analysis/*, test_preds.csv and test_metrics.json to results/<name>-analysis/
python scripts/compare_runs.py
python scripts/paper_figures.py
python scripts/split_figure.py
```

`results/train_counts_per_bin.csv` holds the training images per 1 °C bin produced by step 2;
the figure scripts read it so they can run without the image data.

## Figures

| File | Shows |
|---|---|
| `results/fig_split_distribution.png` | Training (imbalanced) vs balanced test label distribution, coloured by shot group |
| `results/comparison_by_group.png` | Test MAE by shot group and temperature tail for each method |
| `results/comparison_calibration.png` | Mean prediction per true temperature vs y = x: the baseline predicts cold days too warm |
| `results/fig_density_vs_error.png` | Empirical vs LDS-smoothed training density against baseline error (the paper's Fig. 2/3 check) |
| `results/fig_mae_gain.png` | Per-bin MAE gain of + LDS and + LDS + FDS over the baseline |
| `results/comparison_error_by_bin.png` | Per-bin test MAE for all methods |
| `results/reg-analysis/worst_20.png` | The baseline's 20 largest errors (blank frames, night scenes, possible label errors) |
| `eda/figures/fig_temp_per_camera.png` | Temperature range per camera in the raw metadata |

## Limitations

- One seed per method; the medium and few-shot test groups are small.
- The time-based split puts every camera in both training and test, so results describe
  known locations. The model appears to rely partly on recognizing the camera.
- Labels are weather-station readings, not measurements at the camera, and some are wrong.
- A square-root-inverse reweighting run without LDS was not performed, so the claim that
  the reweighting drives the LDS gain is an inference from the weights, not a tested result.

## References

- Y. Yang, K. Zha, Y.-C. Chen, H. Wang, D. Katabi. Delving into Deep Imbalanced Regression. ICML 2021.
  Code: https://github.com/YyzHarry/imbalanced-regression
- R. P. Mihail, S. Workman, Z. Bessinger, N. Jacobs. Sky Segmentation in the Wild: An Empirical Study. WACV 2016.

If you use this code, please cite the original DIR paper and the SkyFinder dataset:

```bibtex
@inproceedings{yang2021delving,
  title={Delving into Deep Imbalanced Regression},
  author={Yang, Yuzhe and Zha, Kaiwen and Chen, Ying-Cong and Wang, Hao and Katabi, Dina},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}

@inproceedings{mihail2016sky,
  title={Sky Segmentation in the Wild: An Empirical Study},
  author={Mihail, Radu P. and Workman, Scott and Bessinger, Zach and Jacobs, Nathan},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2016}
}
```
