# Deep Imbalanced Regression on SkyFinder

Predicting temperature from an outdoor webcam image, and testing whether
the label and feature distribution smoothing methods (LDS and FDS) from
[Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554)
(Yang et al., ICML 2021) help on the rare cold and hot temperatures.

The training code is [`agedb-dir`](https://github.com/YyzHarry/imbalanced-regression/tree/main/agedb-dir)
code, adapted to SkyFinder.

## Task

- **Input:** one image from one of the SkyFinder webcams. The model sees pixels only.
- **Target:** air temperature in °C (`TempM` in the SkyFinder metadata, a nearby
  weather-station reading matched to the image by time).
- **Why it is a DIR problem:** most training images are between 0 and 30 °C. Images below
  −10 °C or above 35 °C are rare, and a standard model predicts them too close to the middle.

## Data

[SkyFinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) (Mihail et al., WACV 2016) has about
90,000 images from 53 static webcams, built for sky segmentation. Its metadata includes weather
data for each frame. Processing is done in `scripts/prepare_data.py`.

Result: 68,319 images from 46 cameras (64,816 train, 1,290 validation, 2,213 test).

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
