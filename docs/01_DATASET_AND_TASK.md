# 01 · Dataset and task

Previous: [00 · Start here](00_START_HERE.md) · Next: [02 · EDA findings](02_EDA_FINDINGS.md)

## Task

Classify each 256 × 256 RGB image from the SPAGHETTI-1 telescope into one of 10 galaxy morphology classes. The competition metric is the **weighted F1-score**: each class's F1 is weighted by its share of the true labels. The public leaderboard uses one part of the test set, and the final ranking uses a hidden part. Validate locally and do not tune against the leaderboard. [Competition page](https://www.kaggle.com/competitions/an2dl2627)

| Label | Class | Visual description | Training images (official) |
| ---: | --- | --- | ---: |
| 0 | Spaghetti alla rinfusa | Irregular, tangled | 213 |
| 1 | Gemelli | Two interacting objects | 345 |
| 2 | Gnocchi | Round and smooth | 744 |
| 3 | Ravioli | Intermediate round and smooth | 685 |
| 4 | Penne | Smooth, cigar-shaped | 83 |
| 5 | Fusilli con la sbarra | Barred spiral | 584 |
| 6 | Fusilli stretti | Tightly wound spiral | 565 |
| 7 | Riccioli | Loosely wound spiral | 477 |
| 8 | Linguine | Thin, flat, edge-on | 276 |
| 9 | Lasagna | Edge-on with a central bulge | 356 |

The pairs 2/3 (round vs intermediate round) and 5/6 (barred vs tight spiral) are visually close. They matter again in [03 · Preprocessing](03_PREPROCESSING.md).

## Files

| File | Contents |
| --- | --- |
| `train/train_00000.png` … `train_04327.png` | 4,328 labelled images |
| `test/test_00000.png` … `test_14735.png` | 14,736 unlabelled images |
| `train.csv` | `id,label` for the training images |
| `sample_submission.csv` | All test ids. Its labels are placeholders. |

The submission is `submission.csv` with header `id,label` and exactly 14,736 rows, one integer label 0–9 per test id.

## Getting the official data (steps 1–2)

You need a Kaggle account that has accepted the competition rules. Store the API token in `~/.kaggle/access_token`, or set `KAGGLE_API_TOKEN` in the environment or in the local `.env` (copy `.env.example`). [Kaggle API settings](https://www.kaggle.com/settings/api)

```bash
python -m pip install -r requirements.txt
python -m spaghetti1 download   # -> data/raw/, validated; reuses a valid copy
python -m spaghetti1 verify     # CSV headers, unique ids, labels 0-9, every id has a PNG
```

Code: [spaghetti1/ingestion/download.py](../spaghetti1/ingestion/download.py) and [spaghetti1/ingestion/dataset.py](../spaghetti1/ingestion/dataset.py).

Most teammates do not need the raw data. They use the cleaned package described in [06 · Sharing and loading data](06_SHARING_AND_LOADING_DATA.md).
