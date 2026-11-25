# Set up SPAGHETTI-1

The full, numbered documentation starts at [00_START_HERE.md](00_START_HERE.md). This page is the short version.

## Teammates: use the cleaned data from Google Drive

Open [notebooks/01_load_preprocessed_data.ipynb](../notebooks/01_load_preprocessed_data.ipynb) in Colab, paste the Drive link of `spaghetti1_clean_v1.zip`, and run all cells. Details: [06_SHARING_AND_LOADING_DATA.md](06_SHARING_AND_LOADING_DATA.md).

## Local setup

Use Python 3.12 in a virtual environment. Create it with `python -m venv .venv`, then activate it: `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install -r requirements.txt
python -m spaghetti1 download         # official data -> data/raw/ (Kaggle token required)
python -m spaghetti1 verify           # CSVs vs PNG files
python -m spaghetti1 eda              # outputs/eda/
python -m spaghetti1 preprocess       # manifests/ + outputs/preprocessing/
python -m spaghetti1 augment-preview  # outputs/preprocessing/augmentation_preview.png
python -m spaghetti1 export           # outputs/export/spaghetti1_clean_v1.zip for Google Drive
python -m spaghetti1 fetch --source "<Drive link or zip path>"
```

For the Kaggle token, accept the competition rules and set `KAGGLE_API_TOKEN` or create `~/.kaggle/access_token`. An optional `.env` (copy `.env.example`) sets paths, workers, and the seed. [Competition rules](https://www.kaggle.com/competitions/an2dl2627/rules) · [Kaggle API settings](https://www.kaggle.com/settings/api)

Keep credentials, downloaded data, and generated outputs in the ignored local paths (`.env`, `data/`, `outputs/`).
