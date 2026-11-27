# 00 · Start here

SPAGHETTI-1 is our team's solution to the AN2DL 2026/27 Kaggle challenge: classify 256 × 256 galaxy images into 10 classes, scored with **weighted F1**. This page explains how the repository is organised and in which order to read it.

## Reading order

Read the documents in number order. Each one builds on the previous one.

| # | Document | Read it to learn |
| --- | --- | --- |
| 00 | [Start here](00_START_HERE.md) | Repository map, quick start, reading order |
| 01 | [Dataset and task](01_DATASET_AND_TASK.md) | Classes, metric, files, and the official rules we must respect |
| 02 | [EDA findings](02_EDA_FINDINGS.md) | What the exploratory analysis measured and how it shaped the plan |
| 03 | [Preprocessing](03_PREPROCESSING.md) | Every cleaning decision (aliens, blank frames, noise, duplicates, label conflicts) and its evidence |
| 04 | [Manifests and folds](04_MANIFESTS_AND_FOLDS.md) | Columns of `train_manifest.csv` / `test_manifest.csv`, CV folds, label policies |
| 05 | [Augmentation and transforms](05_AUGMENTATION_AND_TRANSFORMS.md) | What happens to an image between disk and model, and the switches to experiment with |
| 06 | [Sharing and loading data](06_SHARING_AND_LOADING_DATA.md) | Google Drive package, Colab notebook, `build_dataloaders` |
| 07 | [Next steps](07_NEXT_STEPS.md) | Experiment roadmap for the baseline and model phases |
| 08 | [Training and results](08_TRAINING_AND_RESULTS.md) | How to train a model, where results are saved, what was tested, and the results |
| 09 | [Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md) | Weights package on Google Drive; checking scores and creating submissions without retraining |
| 10 | [Research review and plan](10_RESEARCH_REVIEW_AND_PLAN.md) | External deep-research findings checked against our data, and the experiments chosen next |

If you only want to train a model, read 00, 03 (summary table only), and 06. Then open [notebooks/01_load_preprocessed_data.ipynb](../notebooks/01_load_preprocessed_data.ipynb).

## Repository map

```text
├── docs/                       # numbered documentation (this folder) + figures/
├── manifests/                  # cleaned image lists and CV folds (tracked in git, shared by everyone)
│   ├── train_manifest.csv
│   ├── test_manifest.csv
│   └── preprocessing_summary.json
├── results/                    # lightweight results of every run + submission files (tracked in git)
├── configs/                    # experiment config files (.env format) and shared recipes
├── experiments/                # scripts for planned experiment rounds (e.g. research_round1.ps1)
├── notebooks/
│   └── 01_load_preprocessed_data.ipynb   # Colab entry point for every teammate
├── spaghetti1/                 # Python package; run it with `python -m spaghetti1 <command>`
│   ├── config.py               # paths, class names, official counts, number of folds
│   ├── __main__.py             # command-line interface (all commands below)
│   ├── ingestion/              # step 1-2: download and verify the official data
│   ├── eda/                    # step 3: exploratory statistics and plots
│   ├── preprocessing/          # step 4-5: cleaning, manifests, folds, transforms, DataLoaders
│   ├── sharing/                # step 6-7: pack the cleaned data for Google Drive and fetch it back
│   └── training/               # step 8-10: train, evaluate, and predict; track results
├── data/  outputs/             # local only (ignored by git): raw data, reports, review sheets, zip
├── requirements.txt
└── .env.example
```

## Pipeline commands

Run them from the repository root with Python 3.12. The numbers match the code comments and the documents.

| Step | Command | Output | Who runs it |
| --- | --- | --- | --- |
| 1 | `python -m spaghetti1 download` | `data/raw/` (official data, 2.7 GB) | Whoever builds the package |
| 2 | `python -m spaghetti1 verify` | Console check of CSVs against files | Whoever builds the package |
| 3 | `python -m spaghetti1 eda` | `outputs/eda/` | Optional |
| 4 | `python -m spaghetti1 preprocess` | `manifests/` + `outputs/preprocessing/` (~4.5 min) | Whoever builds the package |
| 5 | `python -m spaghetti1 augment-preview` | `outputs/preprocessing/augmentation_preview.png` | Anyone changing augmentation |
| 6 | `python -m spaghetti1 export` | `outputs/export/spaghetti1_clean_v1.zip` (2.6 GB) | Whoever uploads to Drive |
| 7 | `python -m spaghetti1 fetch --source <Drive link>` | Extracted package, verified | Every teammate (or use the notebook) |
| 8 | `python -m spaghetti1 train --config configs/experiments/<name>.env` | `outputs/runs/<name>/` + a row in `outputs/runs/results.csv` | Whoever runs an experiment |
| 9 | `python -m spaghetti1 evaluate --run-dir <run>` | Validation score of saved weights | Anyone checking a result |
| 10 | `python -m spaghetti1 predict --run-dirs <run> ... --output-dir <dir>` | `submission.csv` from one model or an ensemble | Whoever submits |
| 11 | `python -m spaghetti1 export-weights` / `fetch-weights --source <link>` | Weights zip for Google Drive / verified local copy | Uploader / everyone |
| 12 | `python -m spaghetti1 export-results` | `results/` snapshot for git | Whoever finishes experiments |

Steps 1–6 have already been run. Teammates only need step 7, or notebook 01, which runs it for them.

## Results of the preprocessing in one table

| Finding | Count | Decision |
| --- | ---: | --- |
| Training images | 4,328 | |
| Alien overlays (train only, none in test) | 332 | Removed |
| Blank frames (no object) | 1 train, 2 test | Removed from train; test still gets a prediction |
| Kept training images | 3,995 | |
| Sky cutouts that appear more than once | 721 groups (1,717 images) | Kept in the same CV fold |
| Copies whose label differs from their original | 994 of 996 | Default: train with the original's label |
| Noisy kept images (noise σ ≥ 12) | 671, of which 657 are copies | Kept |
| Test images that duplicate a training image | 9 | Reported only |

Details and evidence are in [03 · Preprocessing](03_PREPROCESSING.md).
