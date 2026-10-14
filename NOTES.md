# Decision Log

## Data cleaning (prepare_data.py)

- **Drop camera 19834**: temperature is burned into the image as a text overlay,
  creating label leakage. 1,328 rows removed.
- **TempM sentinels**: values -9999 and -999 mean "missing" in the SkyFinder
  metadata. 325 rows affected (0.35%). Dropped.
- **Blank TempM**: 10 rows with NaN TempM dropped (after sentinel removal).
- **Label encoding**: `label = round(TempM) + 30`. The +30 offset ensures all
  labels are non-negative, which the DIR repo requires (it indexes bins by
  `int(label)`). The shift cancels in MAE, but any printed predictions must be
  shifted back to degrees C.
- **Time-based split**: images are split by ISO week number to avoid
  near-duplicate leakage (consecutive frames from the same day). Week % 10 == 0
  goes to test pool, == 5 to val pool, rest to train. Whole weeks are assigned
  to one split, and the modular scheme ensures all seasons are represented.
- **Balanced val/test**: from each pool, at most 50 (test) or 25 (val) images
  per 1 degree C bin are sampled. Leftover pool images are discarded (not added
  to train) to keep the evaluation sets balanced.
- **Per-camera outlier rule**: flag rows where |temp - camera_median| > 3 * IQR.
  IQR floor of 5 degrees C prevents cameras with very narrow ranges from having
  zero IQR (which would flag everything). Outlier rule 3xIQR drops 2x 50 C
  readings on cam 10870 and 1x 33 C on cam 9708.
- **is_night column**: derived from the metadata `night` score (> 0.5 = night).
  41.5% of images are flagged as night. Kept for now; can be used to filter or
  analyze separately.

## Data split with only 3 cameras

With only cameras 858, 3888, and 4795 downloaded:
- Train: 2,050 images, temp range [-11, 35] C, 47 unique bins
- Val: 279 images, temp range [0, 29] C, 30 bins
- Test: 220 images, temp range [-6, 33] C, 39 bins
- Val set is missing sub-zero temperatures because camera 4795 (sole source of
  negatives) has no images in week % 10 == 5 weeks.
- Shot groups: 1 Many bin, 37 Medium bins, 9 Few bins.
- Full download (all ~47 cameras) will improve coverage significantly.
