# SPAGHETTI-1 | AN2DL 2026/27 galaxy classification

Classify 256 × 256 RGB galaxy images into 10 classes. Submissions are scored with **weighted F1**. The official data has 4,328 labelled training images and 14,736 test images. [Competition page](https://www.kaggle.com/competitions/an2dl2627)

**Start with [docs/00_START_HERE.md](docs/00_START_HERE.md).** The documentation is numbered 00–10 in reading order.

## Quick start for teammates (Colab)

1. Open [notebooks/01_load_preprocessed_data.ipynb](notebooks/01_load_preprocessed_data.ipynb) in Colab and select a GPU runtime.
2. Paste the Google Drive link of `spaghetti1_clean_v1.zip` into `DATA_SOURCE`.
3. Run all cells. You get verified data and ready DataLoaders:

```python
from spaghetti1.sharing.fetch import prepare_dataset
from spaghetti1.preprocessing.loaders import build_dataloaders

data_dir = prepare_dataset("<Google Drive link>")
loaders = build_dataloaders(data_dir, fold=0, batch_size=64, image_size=224)
```

## Local setup

Use Python 3.12 in a virtual environment and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
cp .env.example .env            # optional: paths, workers, seed (PowerShell: Copy-Item .env.example .env)
```

| Step | Command | Result |
| --- | --- | --- |
| 1 | `python -m spaghetti1 download` | Official data in `data/raw/` (needs a Kaggle token, see [docs/01](docs/01_DATASET_AND_TASK.md)) |
| 2 | `python -m spaghetti1 verify` | CSVs checked against the PNG files |
| 3 | `python -m spaghetti1 eda` | Statistics and plots in `outputs/eda/` |
| 4 | `python -m spaghetti1 preprocess` | `manifests/` (cleaned lists, folds) and review sheets in `outputs/preprocessing/` |
| 5 | `python -m spaghetti1 augment-preview` | Grid of augmented training images |
| 6 | `python -m spaghetti1 export` | `outputs/export/spaghetti1_clean_v1.zip` for Google Drive |
| 7 | `python -m spaghetti1 fetch --source <link or path>` | Downloaded, extracted, and verified package |
| 8 | `python -m spaghetti1 train --run-name <name>` | Trained model and metrics in `outputs/runs/<name>/` (see [docs/08](docs/08_TRAINING_AND_RESULTS.md)) |
| 9 | `python -m spaghetti1 evaluate --run-dir <run>` | Validation score of saved weights, without retraining |
| 10 | `python -m spaghetti1 predict --run-dirs <run> ... --output-dir <dir>` | `submission.csv` from one model or an ensemble |
| 11 | `python -m spaghetti1 export-weights` / `fetch-weights --source <link>` | Model-weights zip for Google Drive (see [docs/09](docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md)) |
| 12 | `python -m spaghetti1 export-results` | Lightweight results snapshot in `results/` |

## What the preprocessing found

| Finding | Decision |
| --- | --- |
| 332 training images carry an alien overlay; none in test | Removed |
| 1 blank training frame (2 in test) | Removed from training |
| 721 sky cutouts appear 2–6 times: a clean original plus brightened, noisy copies | Kept in the same CV fold |
| 994 of the 996 copies carry the neighbouring label (2↔3 or 5↔6) | By default, copies are trained with their original's label |
| Test images are corrupted in colour, brightness, noise, and overlays | Training augmentation simulates these corruptions |

Evidence and alternatives: [docs/03_PREPROCESSING.md](docs/03_PREPROCESSING.md).

## Results so far (fold 0 validation, weighted F1)

| Model | Score |
| --- | ---: |
| ConvNeXt-Tiny 224 (baseline) | 0.836 |
| ConvNeXt-Tiny 384, copies dropped | 0.847 |
| ConvNeXt-Small 288, copies dropped, TTA | 0.847 |
| Ensemble Tiny-224 + Tiny-384 + Small-288 | **0.856** |

Details: [docs/08](docs/08_TRAINING_AND_RESULTS.md). Weights and reproduction without retraining: [docs/09](docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md). Next experiments after the research review: [docs/10](docs/10_RESEARCH_REVIEW_AND_PLAN.md).

## Repository layout

```text
docs/           numbered documentation (00-10), research reports, and figures
manifests/      train/test manifests with flags and CV folds (tracked; identical for everyone)
notebooks/      01_load_preprocessed_data.ipynb (Colab entry point)
results/        configs, learning curves, metrics, confusion matrices, submissions of every run
experiments/    scripts for planned experiment rounds
spaghetti1/     Python package: ingestion/, eda/, preprocessing/, sharing/, training/, config.py, __main__.py
data/, outputs/ local only, ignored by git
```

`.gitignore` keeps credentials (`.env`, Kaggle tokens), `data/`, and `outputs/` out of git.
