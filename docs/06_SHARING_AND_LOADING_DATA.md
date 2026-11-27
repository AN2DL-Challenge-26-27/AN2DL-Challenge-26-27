# 06 · Sharing and loading data (Google Drive and Colab)

Previous: [05 · Augmentation and transforms](05_AUGMENTATION_AND_TRANSFORMS.md) · Next: [07 · Next steps](07_NEXT_STEPS.md)

The cleaned dataset is shared as **one zip file on Google Drive**. Every teammate uses the same images and manifests without downloading from Kaggle or re-running the preprocessing.

Code: [spaghetti1/sharing/export.py](../spaghetti1/sharing/export.py), [spaghetti1/sharing/fetch.py](../spaghetti1/sharing/fetch.py), [spaghetti1/preprocessing/loaders.py](../spaghetti1/preprocessing/loaders.py). Notebook: [notebooks/01_load_preprocessed_data.ipynb](../notebooks/01_load_preprocessed_data.ipynb).

The trained models are shared the same way in a second zip, `spaghetti1_weights_v1.zip` (756 MB); see [09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md).

## The team folder on Google Drive

All shared files live in one Drive folder, which the notebooks use by default:

```text
My Drive/SPAGHETTI-1 | Image Classification/
├── spaghetti1_clean_v1.zip      # cleaned dataset (this page)
├── spaghetti1_weights_v1.zip    # trained models (docs/09)
├── e0_convnext_tiny_original_fold0/, ensemble_tiny224_tiny384/   # submissions
└── colab_runs/<plan>/           # written by the Colab training notebook (docs/11)
```

If the folder was shared with you, open it in Drive and choose *Organize → Add shortcut → My Drive*. It then appears under the same path in Colab (`/content/drive/MyDrive/SPAGHETTI-1 | Image Classification`).

## The package: `spaghetti1_clean_v1.zip` (2.6 GB)

```text
spaghetti1_clean_v1/
├── dataset_info.json          # version, file counts, manifest checksums
├── DATASET_CARD.md            # short description of the cleaning and how to load
├── manifests/                 # train_manifest.csv, test_manifest.csv, preprocessing_summary.json
├── train/                     # 3,995 kept training PNGs (aliens and blank frame left out)
├── test/                      # all 14,736 test PNGs
├── train.csv                  # official, unchanged
└── sample_submission.csv      # official, unchanged
```

## A · For the person who builds and uploads the package

1. Run steps 1–4 from [00 · Start here](00_START_HERE.md) (download, verify, preprocess).
2. `python -m spaghetti1 export` writes `outputs/export/spaghetti1_clean_v1.zip` (about 20 seconds).
3. Upload the zip to Google Drive, for example into a shared folder `AN2DL/`.
4. Right-click the file → **Share** → General access: **Anyone with the link**, role **Viewer** → **Copy link**.
5. Paste the link into the `DATA_SOURCE` cell of notebook 01 (and into the team chat).

When the manifests change, export a new version with `python -m spaghetti1 export --version v2` and upload it next to the old one, so running experiments are not affected.

## B · For every teammate (Colab)

Open [notebooks/01_load_preprocessed_data.ipynb](../notebooks/01_load_preprocessed_data.ipynb) in Colab (File → Open notebook → GitHub), select a GPU runtime, paste the Drive link into `DATA_SOURCE`, and run all cells. The notebook:

1. clones the repository (private repositories: add a `GITHUB_PAT` Colab secret);
2. downloads the zip with `gdown`, extracts it to `/content/spaghetti1_data`, and verifies the file counts;
3. prints the cleaning summary and class counts;
4. builds the DataLoaders for fold 0 and shows an augmented batch.

`prepare_dataset` accepts three kinds of source:

| Source | Example | Notes |
| --- | --- | --- |
| Drive share link or file id | `https://drive.google.com/file/d/<ID>/view?usp=sharing` | Downloads with `gdown` (pre-installed on Colab) |
| Zip on a mounted Drive | `/content/drive/MyDrive/AN2DL/spaghetti1_clean_v1.zip` | Mount first with `drive.mount("/content/drive")`; the zip is copied to local disk because reading many small files from the mount is slow |
| Extracted folder | `/content/spaghetti1_data/spaghetti1_clean_v1` | Used as is after verification |

Running it again finds the verified copy and returns immediately. If the download fails, check that the file is shared as "Anyone with the link".

From a terminal: `python -m spaghetti1 fetch --source "<link or path>"` (default target: `/content/spaghetti1_data` on Colab, `data/clean/` locally).

## C · DataLoaders in your own notebook

```python
from spaghetti1.sharing.fetch import prepare_dataset
from spaghetti1.preprocessing.loaders import build_dataloaders

data_dir = prepare_dataset("<Drive link>")
loaders = build_dataloaders(data_dir, fold=0, batch_size=64, image_size=224)

loaders.train            # augmented, shuffled; label policy "original"
loaders.validation       # originals and unique images of the fold, no augmentation
loaders.test             # 14,736 test images in submission order, label -1
loaders.class_weights    # tensor for CrossEntropyLoss(weight=...)
loaders.train_frame      # the rows behind each loader (ids, labels, flags)
loaders.config           # everything needed to record the experiment
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `fold` | 0 | Validation fold 0–4; `-1` trains on all 3,995 kept images (final model) |
| `image_size` | 256 | Resize target, e.g. 224 for most ImageNet backbones |
| `label_policy` | `"original"` | `"drop_copies"` or `"as_given"` for comparison runs (see 03) |
| `balance_power` | 0.0 | > 0 over-samples rare classes (1.0 = equal frequency) |
| `preprocess` | `PreprocessConfig(image_size)` | Denoising and background normalisation switches (see 05) |
| `augment` | `AugmentConfig()` | `None` turns augmentation off |
| `num_workers` | 2 | Colab has 2 CPU cores |
| `cache_images` | False | `True` keeps decoded training and validation images in RAM (~0.8 GB) and speeds up epochs |
| `seed` | 42 | Shuffling and sampling seed |

The loaders also work on the raw Kaggle folder (`data/raw`). In that case the repository's `manifests/` folder is used, and the excluded images are simply skipped.

## Writing a submission

```python
submission = pd.DataFrame({"id": loaders.test_frame["id"], "label": predictions})
assert len(submission) == 14736 and submission["label"].between(0, 9).all()
submission.to_csv("submission.csv", index=False)
```
