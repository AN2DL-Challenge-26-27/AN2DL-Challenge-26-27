# 08 · Training and results

Previous: [07 · Next steps](07_NEXT_STEPS.md) · Next: [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)

Code: [spaghetti1/training/](../spaghetti1/training/), read in this order: `metrics.py` → `engine.py` → `run.py`.

## Training a model

```bash
python -m spaghetti1 train --run-name e2_convnext_tiny_288_fold0 --image-size 288 --label-policy drop_copies --epochs 30
```

`--data-dir` points to the raw Kaggle folder (default `data/raw`) or to the extracted Google Drive package. Every field of `TrainConfig` in `run.py` is a command-line option:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `convnext_tiny.fb_in22k_ft_in1k` | Any `timm` model name (general-purpose pretrained weights only, see 07) |
| `--fold` | 0 | Validation fold; `-1` trains on all kept images (final model) |
| `--epochs`, `--batch-size`, `--image-size` | 25, 32, 224 | Images are upsampled from 256 when `image-size` is larger |
| `--lr`, `--head-lr-multiplier` | 1e-4, 10 | AdamW; the new classification head learns 10× faster |
| `--weight-decay`, `--warmup-epochs`, `--drop-path-rate` | 0.05, 1, 0.1 | Linear warm-up, then cosine decay to 1% |
| `--label-smoothing` | 0.1 | |
| `--class-weight-power` | 0.5 | Loss weights `(1/n_c)^power`; 0 = unweighted |
| `--label-policy` | `original` | `original`, `drop_copies`, or `as_given` (see 03) |
| `--balance-power` | 0 | > 0 over-samples rare classes instead |
| `--augment / --no-augment`, `--background-normalize`, `--denoise` | on, off, none | See 05 |
| `--tta / --no-tta` | on | Also evaluate with 8-way rotation/flip test-time augmentation |
| `--predict-test` | off | Predict the 14,736 test images and write `submission.csv` |
| `--resume` | off | Continue from `last.pt` after a crash or stop |

Mixed precision (bfloat16) is used automatically on GPUs that support it. On a laptop RTX 4060 (8 GB), ConvNeXt-Tiny at 224 px takes about 20 s per epoch and about 10 minutes per run.

## Exact commands used for every run

These six commands produced every result on this page; all other settings are the `TrainConfig` defaults (also saved in `results/runs/<run>/config.json`). Each run needs the dataset (`data/raw` or `--data-dir <package>`) and took 8–20 minutes on a laptop RTX 4060.

```bash
python -m spaghetti1 train --run-name e0_convnext_tiny_original_fold0 --label-policy original --predict-test
python -m spaghetti1 train --run-name e1_convnext_tiny_drop_copies_fold0 --label-policy drop_copies
python -m spaghetti1 train --run-name e1_convnext_tiny_as_given_fold0 --label-policy as_given
python -m spaghetti1 train --run-name e2_convnext_tiny_288_fold0 --image-size 288 --label-policy drop_copies --epochs 30 --resume
python -m spaghetti1 train --run-name e2_convnext_tiny_384_fold0 --image-size 384 --batch-size 16 --label-policy drop_copies --epochs 30 --resume
python -m spaghetti1 train --run-name e3_convnext_small_288_fold0 --model convnext_small.fb_in22k_ft_in1k --image-size 288 --batch-size 24 --label-policy drop_copies --epochs 30 --resume
```

Shared defaults: fold 0, AdamW (lr 1e-4, head ×10, weight decay 0.05), 1 warm-up epoch then cosine decay, label smoothing 0.1, loss weights `(1/n_c)^0.5`, drop path 0.1, bfloat16 mixed precision, seed 42, 25 epochs unless set. Retraining gives very close but not bit-identical numbers (GPU non-determinism); to use the exact trained models, load the shared weights instead ([09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)).

## What a run saves (`outputs/runs/<run_name>/`)

| File | Contents |
| --- | --- |
| `config.json` | All settings |
| `history.csv` | Per epoch: learning rate, training loss/accuracy, validation loss, weighted F1, macro F1, accuracy, seconds |
| `metrics.json` | Validation metrics of the best epoch (per class and confusion matrix), without and with TTA |
| `confusion_matrix.png`, `confusion_matrix_tta.png` | Row-normalised confusion matrices |
| `val_predictions.csv` | Per validation image: true label, prediction, class probabilities (for ensembling and error analysis) |
| `best.pt` | Weights of the epoch with the highest validation weighted F1 |
| `last.pt` | Full training state after the last finished epoch (for `--resume`) |
| `test_probabilities.npy`, `submission.csv` | With `--predict-test` |

Each finished run appends one row to `outputs/runs/results.csv`: weighted F1 with and without TTA, macro F1, accuracy, per-class F1, and minutes. That file is the experiment log to compare runs.

The best epoch is chosen on the validation fold, which makes its score slightly optimistic. Compare runs on equal terms and confirm important decisions on all 5 folds.

## Results so far (fold 0, 599 validation images)

With 599 images, differences below about 1.5 points are within noise.

| Run | Model | Size | Label policy | Weighted F1 | + TTA | F1 irregular (0) | F1 tight (6) | F1 loose (7) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| E0 | ConvNeXt-Tiny | 224 | original | 0.836 | 0.835 | 0.54 | 0.73 | 0.69 |
| E1a | ConvNeXt-Tiny | 224 | drop_copies | **0.843** | **0.844** | 0.51 | 0.78 | 0.70 |
| E1b | ConvNeXt-Tiny | 224 | as_given | 0.836 | 0.829 | 0.44 | 0.78 | 0.67 |
| E2a | ConvNeXt-Tiny | 288 | drop_copies | 0.846 | 0.833 | 0.35 | 0.80 | 0.69 |
| E2b | ConvNeXt-Tiny | 384 | drop_copies | **0.847** | 0.846 | 0.54 | 0.75 | 0.72 |
| E3 | ConvNeXt-Small | 288 | drop_copies | 0.841 | **0.847** | 0.60 | 0.79 | 0.71 |
| Ensemble | E1a + E2b + E3 (average of TTA probabilities) | | | | **0.856** | 0.57 | 0.78 | 0.70 |

### Findings

1. **Label policy:** the three policies are within noise. `drop_copies` is marginally best and fastest, and it avoids contradictory labels, so it is the working default for the next experiments. `as_given` is clearly worse on the irregular class.
2. **Robustness:** applying the test-like corruptions to the validation images costs at most about 1 point (colour, darkening, blur, noise) and about 2 points with overlays. The test-domain shift is not the main problem.
3. **Where the errors are:** irregular ↔ loose spiral ↔ tight spiral. Of 92 errors, 36 are made with confidence above 0.8, and visual review shows several look mislabelled or ambiguous. Galaxy Zoo labels come from volunteer votes, so part of this error is irreducible.
4. **Fitting:** training accuracy 0.92 vs validation 0.84; validation stops improving after about epoch 15.
5. **TTA and ensembling of these runs:** +0.2 and +0.3 points only. The models are too similar to correct each other.

6. **Resolution and backbone (E2, E3):** every single model lands at 0.84–0.85. Larger images (288, 384) and ConvNeXt-Small do not clearly beat ConvNeXt-Tiny at 224 on this fold; the differences are within noise. ConvNeXt-Small fits the training set almost perfectly (training accuracy 0.97) and overfits more.
7. **Ensembling different models works:** averaging Tiny-224, Tiny-384, and Small-288 gives 0.856 (+1 point over the best single model).
8. **Shared errors:** the models make mostly the same mistakes (75 of 107 errors are shared by Tiny-384 and Small-288). Even counting an image as correct when *any* of the six models gets it right gives only 0.90, so about 10% of validation images are wrong for every model. Many of these look ambiguous or mislabelled.

### Conclusions so far

* A single fine-tuned ImageNet model plateaus at about 0.845 on fold 0.
* The gains left must come from diversity and data: ensembles of different architectures over all 5 folds, stronger regularisation (EMA, MixUp/CutMix) for the larger models, and pseudo-labelling of the 14,736 test images.
* With only 599 validation images, the next comparisons should use out-of-fold predictions over all 5 folds (about 3,000 images) to be reliable.

## What has been implemented and tested (summary)

| Area | Implemented | Tested | Outcome |
| --- | --- | --- | --- |
| Data cleaning | Alien and blank-frame removal, star-field duplicate grouping, original/copy marking | All 19,064 images, review sheets | 3,995 kept training images; see 03 |
| Validation | 5-fold `StratifiedGroupKFold`, validation on originals and unique images | Fold 0 | Used by every run |
| Label policies | `original`, `drop_copies`, `as_given` | E0, E1a, E1b | Within noise; `drop_copies` kept as default |
| Augmentation | Rotation, flips, scale, test-like colour, brightness, noise, blur, overlays | Robustness check | Test-like corruptions cost at most 1–2 points |
| Backbones | ConvNeXt-Tiny, ConvNeXt-Small (ImageNet-22k pretrained, `timm`) | E0–E3 | Single models plateau at about 0.845 |
| Input size | 224, 288, 384 px | E1a, E2a, E2b | No clear gain on fold 0 |
| Test-time augmentation | 8-way rotation/flip averaging | Every run | −1.3 to +0.6 points; not reliable alone |
| Ensembling | Average of class probabilities | Fold-0 validation | **+1 point** (0.856 with three models) |
| Reproduction | `evaluate` / `predict` from saved weights, weights package with checksums | CPU re-scoring | TTA score reproduced exactly (0.8441); see 09 |
| Not yet tested | EMA, MixUp/CutMix, 5-fold training, other architecture families, pseudo-labelling, class-weight and threshold tuning | | Planned in 07 |

## Submissions

| File (`results/submissions/`) | Models | Val wF1 |
| --- | --- | ---: |
| `e0_convnext_tiny_original_fold0/submission.csv` | ConvNeXt-Tiny 224 | 0.836 |
| `ensemble_tiny224_tiny384/submission.csv` | ConvNeXt-Tiny 224 + 384, TTA | 0.846 |
| *(to create, needs a GPU)* | + ConvNeXt-Small 288 | 0.856 |

How to recreate every number and file from the shared weights is described in [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md). The improvement plan (EMA, multi-architecture 5-fold ensemble, pseudo-labelling, per-class probability tuning) is in [07 · Next steps](07_NEXT_STEPS.md).
