# Data ingestion and validation

- `download.py` downloads the competition dataset from Kaggle, validates its layout, and places it in `data/raw/`.
- `dataset.py` reads image identifiers and training labels from `train.csv`, reads test identifiers from `sample_submission.csv`, and checks that those identifiers match the files in both splits.

From the project root, run `python -m spaghetti1 download` and `python -m spaghetti1 verify` with the virtual environment described in the main guide. This stage provides the raw dataset and a validated image manifest. Documentation: [docs/01_DATASET_AND_TASK.md](../../docs/01_DATASET_AND_TASK.md) · [Main README](../../README.md)
