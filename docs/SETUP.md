# Set up SPAGHETTI-1

Run the pipeline from the root of this repository. Use Python 3.12 in an existing virtual environment, or create one with `python -m venv .venv` and activate it. In PowerShell, run `.\.venv\Scripts\Activate.ps1`. On macOS or Linux, run `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
```

## Get and check the dataset

Use a Kaggle account eligible for the competition and accept its rules. Set `KAGGLE_API_TOKEN` in your environment or place the token in `~/.kaggle/access_token`. An optional local `.env` can be copied from `.env.example` to configure paths, workers, and the random seed. [Competition rules](https://www.kaggle.com/competitions/an2dl2627/rules) ? [Kaggle API settings](https://www.kaggle.com/settings/api)

```bash
python -m spaghetti1 download
python -m spaghetti1 verify
python -m spaghetti1 eda
```

The download command places the official files under `data/raw/` and reuses a valid local copy. If the dataset is already extracted elsewhere, skip the download and pass `--data-dir` to both `verify` and `eda`. Point it at the directory containing `train.csv` and `sample_submission.csv`.

The analysis scans the images and writes CSV tables, a JSON summary, and PNG plots to `outputs/eda/` by default. Start with `summary.json`, `class_distribution.png`, and `domain_shift.png`. Keep credentials, downloaded data, and generated outputs in the ignored local paths.

## Code layout

- `spaghetti1/ingestion/` downloads the dataset and checks the CSVs and file layout. [Ingestion guide](../spaghetti1/ingestion/README.md)
- `spaghetti1/preprocessing/` decodes 256 ? 256 RGB PNGs and returns their original pixel values. [Preprocessing guide](../spaghetti1/preprocessing/README.md)
- `spaghetti1/eda/` calculates per-image statistics, detects exact duplicates, and creates reports and plots. [EDA guide](../spaghetti1/eda/README.md)

For findings and the validation plan, see the [project progress report](README.md).
