# 11 · Training in the cloud (Google Colab or Kaggle)

Previous: [10 · Research review and plan](10_RESEARCH_REVIEW_AND_PLAN.md) · Next: [12 · Concepts and techniques explained](12_CONCEPTS_EXPLAINED.md)

The laptop GPU needs cooling pauses about half the time, which roughly doubles every run (see 10). Long plans such as 5-fold training are better run in the cloud. Two options use the same code and the same config files:

| | Google Colab (notebook 03) | Kaggle (notebook 02) |
| --- | --- | --- |
| GPU | 1 × T4 (free); L4 or A100 with Colab Pro | 2 × T4 (free), two runs at once |
| Limits | Free: disconnects after ~90 min idle, sessions up to ~12 h | ~30 GPU hours per week, 12 h per session |
| Data | Team Drive package, or a download from Kaggle with your API token | Competition data attached directly |
| Results | Written straight to your Google Drive | Download from the *Output* tab |
| After a disconnect | Run all cells again: continues from the last epoch | Committed runs continue in the background |
| Precision | T4: float16 (automatic); L4/A100: bfloat16 | float16 (automatic) |

Queue script used by both: [experiments/kaggle_queue.py](../experiments/kaggle_queue.py).

## Google Colab

Notebook: [notebooks/03_colab_training.ipynb](../notebooks/03_colab_training.ipynb).

1. Open it in Colab: **File → Open notebook → GitHub** → the team repository → `notebooks/03_colab_training.ipynb`. If the repository is private, use the GitHub tab with your account, or upload the file.
2. **Runtime → Change runtime type → T4 GPU** (or L4/A100 with Colab Pro).
3. **Secrets** (key icon, left sidebar). Add `GITHUB_PAT` if the repository is private, and `KAGGLE_API_TOKEN` only for data option B. Allow notebook access when asked.
4. Run the cells in order:
   * **Step 1:** mounts Google Drive and opens the team folder `My Drive/SPAGHETTI-1 | Image Classification/` (if it was shared with you: *Add shortcut → My Drive* first). All outputs go to its `colab_runs/<plan>/` subfolder.
   * **Step 2:** clones the repository and upgrades `timm`.
   * **Step 3:** gets the data. Option A (default, nothing to change): `spaghetti1_clean_v1.zip` from the team folder. Option B: `DATA_SOURCE = "kaggle"`.
   * **Step 4:** choose `PLAN` (`dinov2_5fold`, `dinov2_5fold_all`, `dinov3_5fold`, or `custom`).
   * **Step 5:** trains. Progress is printed every minute.
   * **Steps 6–7:** show the results and offer the small results zip as a download.
5. If Colab disconnects: reconnect, select the GPU again if needed, and **run all cells again**. Finished runs are skipped. An interrupted run continues from its last saved epoch, because the run folders live on Drive (`colab_runs/<plan>/runs/`).

Free Colab has 2 CPU cores, so data loading uses 2 workers per run. Saving `last.pt` to Drive after every epoch adds some seconds per epoch; it is what makes the runs survive disconnects.

## Kaggle

Notebook: [notebooks/02_kaggle_training.ipynb](../notebooks/02_kaggle_training.ipynb).

1. kaggle.com: **Create → New Notebook → File → Import Notebook** → upload `notebooks/02_kaggle_training.ipynb`.
2. *Session options*: **Accelerator = GPU T4 x2**, **Internet = on** (needs a phone-verified account).
3. **Add Input → Competitions → an2dl2627.**
4. If the repository is private: *Add-ons → Secrets* → `GITHUB_PAT`.
5. Choose `PLAN` in step 3, then **Save Version → Save & Run All (Commit)**. It runs in the background; the zip files appear under *Output*.

## What the queue does (both platforms)

* Spreads the config files over the available GPUs (one on Colab, two on Kaggle); each GPU trains its runs one after another.
* Trains every run with `--predict-test --no-test-tta`, so each run also saves test probabilities without TTA. TTA lowered DINOv2's score.
* Uses **float16 with gradient scaling** on GPUs without bfloat16 (T4, P100) via `PRECISION=auto`, and records the precision in every log and `metrics.json`.
* Groups runs whose names differ only in `_fold<k>`. It reports per-fold and **out-of-fold** weighted F1 (each training image scored once, by the fold model that did not see it, about 3,000 images) and writes a **fold-ensemble submission** (average of the folds' test probabilities).

## Results to bring back

| File | Contents | What to do with it |
| --- | --- | --- |
| `spaghetti1_results_<tag>.zip` | Everything except weights: `runs/results.csv`, logs, `config.env`/`config.json`, metrics, curves, confusion matrices, validation and test probabilities, per-run `submission.csv`, `summary/summary.json`, fold-ensemble submissions | **Send this one** (small) for analysis and new ensembles |
| `spaghetti1_weights_<tag>.zip` | Trained models (`best.pt`) with SHA-256 checksums (format of [09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)) | Share via Drive; `python -m spaghetti1 fetch-weights --source <zip>` |

`summary/fold_ensemble_<group>/submission.csv` can be submitted to the competition directly.

## Using cloud results locally

* **Weights:** `python -m spaghetti1 fetch-weights --source spaghetti1_weights_<tag>.zip`. `evaluate` and `predict` then work on `data/weights/<package>/runs/<run>`.
* **Probabilities only (no GPU needed):** extract the results zip. Each run folder holds `val_probabilities.npy` (validation fold, no TTA, `split_fold` order), `val_probabilities_tta.npy`, and `test_probabilities.npy` (sample-submission order). These are the inputs for new ensembles, for example with [experiments/combine_submissions.py](../experiments/combine_submissions.py).

## Verified

On 2026-11-27, the full queue (training, test predictions, fold summary, both exports, and re-fetching the weights zip) was run end to end on the laptop GPU with `--precision fp16`, the mode a T4 uses. The fold summary was unit-tested with perfect predictions: out-of-fold F1 1.0 over 2,999 images, and a 14,736-row fold-ensemble file. Every notebook cell parses. Not yet verified on Colab or Kaggle themselves: the clone, install, and data cells, and the real GPU speed.
