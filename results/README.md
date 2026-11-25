# Experiment results (snapshot)

Lightweight outputs of every training run, kept in git so results can be inspected without the model weights. The weights themselves are in `spaghetti1_weights_v1.zip` on Google Drive (see [docs/09](../docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md)). Explanations and conclusions are in [docs/08](../docs/08_TRAINING_AND_RESULTS.md).

| Path | Contents |
| --- | --- |
| `results.csv` | One row per run: model, input size, label policy, best epoch, validation weighted F1 with and without TTA, macro F1, accuracy, per-class F1, minutes |
| `runs/<run_name>/config.json` | Every training setting |
| `runs/<run_name>/history.csv` | Per-epoch learning rate, training loss and accuracy, validation loss, weighted F1, macro F1, accuracy |
| `runs/<run_name>/metrics.json` | Validation metrics of the best epoch, per class, with the confusion matrix, without and with TTA |
| `runs/<run_name>/confusion_matrix*.png` | Confusion matrices (rows = true class) |
| `runs/<run_name>/val_predictions.csv` | Per validation image: true label, prediction, class probabilities |
| `submissions/<name>/submission.csv` | Files ready to upload to Kaggle (`id,label`, 14,736 rows) |
| `submissions/<name>/ensemble_info.json` | Models in the submission and its validation weighted F1 |

All runs use CV fold 0 (599 validation images: originals and unique images). Regenerate this folder with `python -m spaghetti1 export-results`.
