# 07 · Next steps (experiment roadmap)

Previous: [06 · Sharing and loading data](06_SHARING_AND_LOADING_DATA.md) · Next: [08 · Training and results](08_TRAINING_AND_RESULTS.md)

The data side is ready. The next phases follow the task plan: baseline, then optimisation, error analysis, and final inference. Each experiment changes **one thing** and is compared on the same folds with **validation weighted F1** (also record macro F1, accuracy, per-class F1, and the confusion matrix).

## Competition rule on pretrained models

General-purpose pretrained models (ImageNet and similar) are allowed. Models designed specifically for space imagery (for example Zoobot or other Galaxy Zoo-pretrained networks) are **not** allowed.

## Rules for every experiment

* Use `build_dataloaders` and the committed manifests. Never create a new random split.
* Run at least fold 0 for quick decisions and all 5 folds before a change is adopted.
* Save `loaders.config`, the model name, hyperparameters, metrics, and the checkpoint path with each run.
* Leaderboard scores are a sanity check only. Decisions come from local validation.

## Planned experiments, in order

| # | Experiment | Options | Why first |
| --- | --- | --- | --- |
| E0 | Baseline | Small pretrained CNN (e.g. ResNet-18 or EfficientNet-B0), default loaders, cross-entropy with `class_weights` | Reference score for everything else |
| E1 | **Label policy** | `"original"` vs `"drop_copies"` vs `"as_given"` | Decides how 996 images and the 2/3 and 5/6 boundaries are learned |
| E2 | Class balancing | Loss weights vs `balance_power` 0.5 / 1.0 vs none | Penne has only 56 images |
| E3 | Augmentation strength | Default vs no photometric vs stronger artifacts | Checks that the test-like corruptions help |
| E4 | Preprocessing switches | `background_normalize`, `denoise="bilateral"` | Off by default; keep only with measured gain |
| E5 | Backbones and input size | ConvNeXt-T, EfficientNet-B2/B3, ViT/DINOv2 at 224–256 | Larger is not automatically better |
| E6 | Regularisation | Label smoothing, MixUp/CutMix, dropout, weight decay | Small dataset, risk of over-fitting |
| E7 | Test-time augmentation and ensembling | D4 flips/rotations, fold ensemble | Usually a reliable final gain |

For E1, also report F1 on the 2/3 and 5/6 classes separately. That is where the policies differ.

## Points to watch in error analysis

* Confusions between 2↔3 and 5↔6, the pairs affected by the copy labels.
* Penne (class 4) recall.
* Performance on test-like corruptions: evaluate validation images with augmentation turned on as a robustness check.
* The two blank test frames (test_13385, test_13646): they get whatever the model predicts. That is 2 of 14,736 rows and not worth special handling.

## Status (2026-11-26)

Done: E0 (baseline), E1 (label policies), resolution 288/384, ConvNeXt-Small, TTA, and a first 3-model ensemble (see [08](08_TRAINING_AND_RESULTS.md)). Everything in the revised plan below is still open. The plan was refined after an external research review; the current order is in [10](10_RESEARCH_REVIEW_AND_PLAN.md).

## Revised plan after the first results (2026-11-26)

The first runs (see [08](08_TRAINING_AND_RESULTS.md)) showed that the label policy is within noise, the test-like corruptions cost little, and the remaining errors are fine-morphology confusions (irregular / loose / tight spiral). The plan is therefore re-ordered:

| Order | Step | Expected gain |
| ---: | --- | --- |
| 1 | Higher resolution: 288 and 384 px | +1 to 3 points |
| 2 | Stronger general-purpose backbone (ConvNeXt-Small, ConvNeXt-V2, EfficientNetV2-S) | +1 to 2 |
| 3 | Weight EMA, 40 epochs, layer-wise learning-rate decay | +0.5 to 1 |
| 4 | Ensemble of different architectures over all 5 folds, with TTA | +1 to 2 |
| 5 | Pseudo-labelling of confident test predictions | +0.5 to 1.5 |
| 6 | Unweighted loss and per-class probability multipliers fitted on out-of-fold predictions | +0.5 |

These gains overlap, so they do not simply add up. Together they are the realistic route towards a weighted F1 of about 0.88–0.90.
