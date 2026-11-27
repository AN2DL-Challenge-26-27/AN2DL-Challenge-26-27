# 11 · Training on Kaggle (2 × T4)

Previous: [10 · Research review and plan](10_RESEARCH_REVIEW_AND_PLAN.md)

Kaggle notebooks offer two free T4 GPUs (16 GB each), with about 30 GPU hours per week and at most 12 hours per session. For us that is faster than the laptop, for three reasons:

* **No cooling pauses:** on the laptop the temperature guard pauses training about half the time.
* **Two runs at once:** one per GPU.
* **More memory:** 16 GB instead of 8 GB.

Expect roughly 2–3× more throughput for multi-run plans such as 5-fold training (to be confirmed on the first run).

Notebook: [notebooks/02_kaggle_training.ipynb](../notebooks/02_kaggle_training.ipynb). Queue script: [experiments/kaggle_queue.py](../experiments/kaggle_queue.py).

## One-time setup

1. On kaggle.com: **Create → New Notebook → File → Import Notebook**, and upload `notebooks/02_kaggle_training.ipynb` (or open it from GitHub).
2. In the right panel, *Session options*: **Accelerator = GPU T4 x2** and **Internet = on**. Internet needs a phone-verified account.
3. **Add Input → Competitions → an2dl2627.** Nothing has to be uploaded: the queue reads the raw competition files and takes the team's manifests and folds from the repository.
4. If the repository is private: *Add-ons → Secrets* → `GITHUB_PAT` = a GitHub token with read access.

## Running

1. In step 3 of the notebook, choose `PLAN`:

   | Plan | Runs | Approximate time |
   | --- | --- | --- |
   | `dinov2_5fold` | DINOv2-Base 336, folds 1–4 | 2–3 h |
   | `dinov2_5fold_all` | DINOv2-Base 336, folds 0–4 | 3–4 h |
   | `dinov3_5fold` | DINOv3 ViT-B/16 384, folds 0–4 | 3–4 h |
   | `custom` | your own list in `CUSTOM_CONFIGS` | |

2. For a quick check, run the cells one by one. For the real run, use **Save Version → Save & Run All (Commit)**: it runs in the background (browser may be closed) and keeps the output files.
3. Progress is printed every minute (run, GPU, latest epoch). A failed run does not stop the others. Running the training cell again continues unfinished runs.

## What the queue does

* Spreads the config files over the GPUs (round-robin); each GPU trains its runs one after another.
* Trains every run with `--predict-test --no-test-tta`, so each run saves test probabilities without TTA. TTA lowered DINOv2's score, and the ensembles score the same either way.
* Uses **float16 with gradient scaling** automatically. T4 and P100 cannot use bfloat16, which the laptop uses. The setting `PRECISION=auto` picks the right one, and every log line and `metrics.json` records which was used.
* Groups runs whose names differ only in `_fold<k>`. For each group it reports per-fold and **out-of-fold** weighted F1: each training image is scored once, by the fold model that did not see it (about 3,000 images). It also writes a **fold-ensemble submission** (average of the folds' test probabilities).

## Results to download

After a committed run the *Output* tab shows:

| File | Contents | Size | What to do with it |
| --- | --- | --- | --- |
| `spaghetti1_results_<tag>.zip` | Everything except the weights: `runs/results.csv`, logs, `config.env`/`config.json`, metrics, learning curves, confusion matrices, validation and test probabilities, per-run `submission.csv`, `summary/summary.json`, fold-ensemble submissions | small | Send this one for analysis and ensembling |
| `spaghetti1_weights_<tag>.zip` | Trained models (`best.pt`) with SHA-256 checksums, in the format of [09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md) | ~350 MB per DINOv2/DINOv3 run | Upload to the team Drive; `python -m spaghetti1 fetch-weights --source <zip>` |

`summary/fold_ensemble_<group>/submission.csv` can be submitted to the competition directly.

## Using Kaggle results locally

* **Weights:** `python -m spaghetti1 fetch-weights --source spaghetti1_weights_<tag>.zip` extracts and verifies them. `evaluate` and `predict` then work on `data/weights/<package>/runs/<run>`.
* **Probabilities only (no GPU needed):** extract the results zip. Each run folder holds `val_probabilities.npy` (validation fold, no TTA, `split_fold` order), `val_probabilities_tta.npy`, and `test_probabilities.npy` (sample-submission order). These are the inputs for new ensembles, for example with [experiments/combine_submissions.py](../experiments/combine_submissions.py).

## Verified

On 2026-11-27, the full queue (training, test predictions, fold summary, both exports, and re-fetching the weights zip) was run end to end on the laptop GPU with `--precision fp16`. The fold summary was unit-tested with perfect predictions: out-of-fold F1 1.0 over 2,999 images, and a 14,736-row fold-ensemble file. Not yet verified on Kaggle itself: the notebook's clone and install cell, and the actual T4 speed.
