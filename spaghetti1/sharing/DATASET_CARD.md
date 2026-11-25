# SPAGHETTI-1 cleaned dataset (`spaghetti1_clean_v1`)

This package is the official AN2DL 2026/27 competition data after the project's cleaning step. It was produced with `python -m spaghetti1 preprocess` and `python -m spaghetti1 export` from the team repository. The full explanation is in `docs/03_PREPROCESSING.md` in the repository.

## Contents

| Path | Contents |
| --- | --- |
| `train/` | 3,995 training PNGs (256 × 256 RGB). 332 alien-overlay images and 1 blank frame are left out. |
| `test/` | All 14,736 test PNGs. Every one needs a prediction, including the 2 blank frames. |
| `manifests/train_manifest.csv` | All 4,328 training rows: label, `keep`, `exclude_reason`, `fold`, duplicate group, `is_copy`, `original_label`, and quality metrics. |
| `manifests/test_manifest.csv` | All test rows with quality metrics and the 9 test images that duplicate a training image. |
| `manifests/preprocessing_summary.json` | Counts behind every cleaning decision. |
| `train.csv`, `sample_submission.csv` | Official files, unchanged. |
| `dataset_info.json` | Version, file counts, and manifest checksums used by `prepare_dataset` to verify the copy. |

## What to know before training

1. Use only rows with `keep == True`. Their images are in `train/`.
2. Use the `fold` column for validation (5 folds, `StratifiedGroupKFold`, seed 42). Copies of the same sky cutout are always in the same fold.
3. 721 cutouts appear more than once: one clean original and 1–5 brightened, noisier copies. In 719 of these groups, every copy has the neighbouring label (Gnocchi↔Ravioli or barred↔tight spiral). The default policy (`label_policy="original"`) trains on copies with their original's label.
4. Validate on originals and unique images only (`split_fold` does this by default).

## Loading

```python
from spaghetti1.sharing.fetch import prepare_dataset
from spaghetti1.preprocessing.loaders import build_dataloaders

data_dir = prepare_dataset("<Google Drive link or path to spaghetti1_clean_v1.zip>")
loaders = build_dataloaders(data_dir, fold=0, batch_size=64, image_size=224)
```
