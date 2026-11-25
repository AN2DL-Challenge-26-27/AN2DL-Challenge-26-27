# `spaghetti1/training`: model training and experiment tracking

Full explanation: [docs/08_TRAINING_AND_RESULTS.md](../../docs/08_TRAINING_AND_RESULTS.md).

| # | Module | Role |
| --- | --- | --- |
| 1 | `metrics.py` | Weighted F1 (competition metric), macro F1, accuracy, per-class scores, confusion-matrix plot |
| 2 | `engine.py` | `timm` model with a 10-class head, AdamW with a faster head, warm-up + cosine schedule, one training epoch, prediction with optional 8-way TTA |
| 3 | `run.py` | `TrainConfig` (every setting of a run) and `run_experiment`: train, pick the best epoch on validation, evaluate, save outputs, append to `outputs/runs/results.csv` |
| 4 | `predict.py` | `evaluate_run` and `predict_test`: re-score saved weights and create (ensemble) submissions without retraining; `load_run` for your own code |

Commands: `python -m spaghetti1 train --run-name <name> [--option value ...]`, `evaluate --run-dir <run>`, `predict --run-dirs <run> ... --output-dir <dir>`. Every `TrainConfig` field is a command-line option (`--label-policy drop_copies`, `--model tf_efficientnetv2_s.in21k_ft_in1k`, `--no-tta`, ...).
