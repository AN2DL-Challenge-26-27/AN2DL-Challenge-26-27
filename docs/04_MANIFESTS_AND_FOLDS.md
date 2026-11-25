# 04 · Manifests and folds

Previous: [03 · Preprocessing](03_PREPROCESSING.md) · Next: [05 · Augmentation and transforms](05_AUGMENTATION_AND_TRANSFORMS.md)

The manifests are the **single source of truth** for which images we train on, which fold each one belongs to, and which label to use. They are small CSV files tracked in git, so every teammate uses identical folds. They are also copied into the Google Drive package.

Code: [spaghetti1/preprocessing/manifest.py](../spaghetti1/preprocessing/manifest.py) writes them; [spaghetti1/preprocessing/dataset.py](../spaghetti1/preprocessing/dataset.py) reads them.

## `manifests/train_manifest.csv` (4,328 rows, one per official training image)

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | str | `train_00000` … `train_04327` |
| `label` | int | Official label 0–9 |
| `class_name` | str | Name of the official label |
| `path` | str | Relative image path, e.g. `train/train_00000.png` (same in raw data and package) |
| `keep` | bool | `False` for removed images (their PNG is not in the Drive package) |
| `exclude_reason` | str | `alien_overlay`, `blank_frame`, or empty |
| `fold` | int | CV fold 0–4 for kept images, −1 for removed ones |
| `dup_group` | str | Duplicate group id; unique images have their own group |
| `dup_group_size` | int | Kept images in the group (1 = unique) |
| `is_copy` | bool | `True` for the brightened, noisy copies; `False` for originals and unique images |
| `original_id` | str | Id of the group's original (the image itself if it is an original) |
| `original_label` | int | Label of the group's original; used by the default label policy |
| `label_conflict` | bool | The group contains more than one label |
| `test_duplicates` | str | Test ids that duplicate this image, separated by `;` |
| `alien_skin_px`, `alien_eye_px`, `is_alien` | int, int, bool | Alien detector measurements and decision |
| `contrast`, `is_blank` | float, bool | 99th percentile minus background, blank-frame decision |
| `noise_sigma`, `is_noisy` | float, bool | Robust noise estimate (0–255 scale), σ ≥ 12 flag |
| `background_r`, `background_g`, `background_b` | float | Per-channel median (background level and colour cast) |

## `manifests/test_manifest.csv` (14,736 rows)

`id`, `path`, the same quality columns, and `train_duplicate`, `train_duplicate_label`, `train_duplicate_correlation` for the 9 test images that match a training image. The row order equals `sample_submission.csv`, so predictions can be written in this order.

## Cross-validation folds

* 5 folds, `StratifiedGroupKFold(shuffle=True, random_state=42)` from scikit-learn.
* **Grouped** by `dup_group`: an original and all its copies are always in the same fold.
* **Stratified** by `original_label`, so every fold has the same class proportions (Penne: 11–12 validation images per fold).
* Each fold has 799 kept images. About 600 of them are originals or unique images, and only those are used for validation by default.

Per-class validation counts per fold are in `outputs/preprocessing/class_counts.csv`.

## Using the manifests in code

```python
from spaghetti1.preprocessing.dataset import load_train_manifest, split_fold

train = load_train_manifest(manifest_dir)                 # kept rows only
training, validation = split_fold(train, fold=0)          # label_policy="original" by default
training, validation = split_fold(train, fold=0, label_policy="drop_copies")
```

`split_fold` returns two DataFrames:

* **training**: the other four folds, with the label policy applied. Its `label` column is the label to train on; `given_label` keeps the official one.
* **validation**: the fold's originals and unique images. Pass `validate_on_copies=True` to include the copies.

Label policies (`"original"`, `"drop_copies"`, `"as_given"`) are explained in [03 · Preprocessing, section 4](03_PREPROCESSING.md#4--copies-carry-the-neighbouring-label).

## Regenerating

`python -m spaghetti1 preprocess` rebuilds the manifests deterministically from the raw data (same seed, same folds). Commit the new CSVs if anything changes, so the team keeps using one version. Change the number of folds with `--folds` and the seed with `--seed`.
