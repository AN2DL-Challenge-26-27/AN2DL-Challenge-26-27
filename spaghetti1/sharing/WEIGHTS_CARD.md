# SPAGHETTI-1 model weights (`spaghetti1_weights_v1`)

Trained models from the SPAGHETTI-1 experiments, for evaluating and predicting **without retraining**. Full instructions are in `docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md` in the team repository, and every result is in `docs/08_TRAINING_AND_RESULTS.md`.

## Models (all trained on CV fold 0; validation = 599 originals and unique images)

| Run folder | Backbone (`timm`, ImageNet-pretrained) | Input | Label policy | Val weighted F1 | + 8-way TTA |
| --- | --- | ---: | --- | ---: | ---: |
| `e0_convnext_tiny_original_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | original | 0.836 | 0.835 |
| `e1_convnext_tiny_drop_copies_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | drop_copies | 0.843 | 0.844 |
| `e1_convnext_tiny_as_given_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | as_given | 0.836 | 0.829 |
| `e2_convnext_tiny_288_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 288 | drop_copies | 0.846 | 0.833 |
| `e2_convnext_tiny_384_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 384 | drop_copies | 0.847 | 0.846 |
| `e3_convnext_small_288_fold0` | `convnext_small.fb_in22k_ft_in1k` | 288 | drop_copies | 0.841 | 0.847 |

**Best validated ensemble:** average of the 8-way-TTA probabilities of `e1_convnext_tiny_drop_copies_fold0`, `e2_convnext_tiny_384_fold0`, and `e3_convnext_small_288_fold0`: validation weighted F1 **0.856**.

## Files per run

* `best.pt`: PyTorch `state_dict` of the epoch with the best validation weighted F1 (SHA-256 in `weights_info.json`)
* `config.json`: every training setting; the loader rebuilds the network and the exact preprocessing from it
* `metrics.json`, `history.csv`, `val_predictions.csv`: scores, learning curves, validation probabilities
* `test_probabilities.npy` (some runs): test-set probabilities in `sample_submission.csv` order

## Use

```bash
python -m spaghetti1 fetch-weights --source "<Drive link or path to spaghetti1_weights_v1.zip>"
python -m spaghetti1 evaluate --run-dir data/weights/spaghetti1_weights_v1/runs/e2_convnext_tiny_384_fold0
python -m spaghetti1 predict --output-dir outputs/submissions/best_ensemble --run-dirs \
    data/weights/spaghetti1_weights_v1/runs/e1_convnext_tiny_drop_copies_fold0 \
    data/weights/spaghetti1_weights_v1/runs/e2_convnext_tiny_384_fold0 \
    data/weights/spaghetti1_weights_v1/runs/e3_convnext_small_288_fold0
```

In Python, `spaghetti1.training.predict.load_run(run_dir, device)` returns the ready model and its config.
