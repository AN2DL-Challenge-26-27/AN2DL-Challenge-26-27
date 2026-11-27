# 08 · Training and results

Previous: [07 · Next steps](07_NEXT_STEPS.md) · Next: [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)

Code: [spaghetti1/training/](../spaghetti1/training/), read in this order: `models.py` (model registry) → `config.py` (settings and config files) → `metrics.py` → `engine.py` (training mechanics) → `run.py` (experiment flow) → `predict.py`.

## Training a model

Every run is described by a config file in [configs/experiments/](../configs/experiments/):

```bash
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env --dry-run   # check
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env             # train
```

`--data-dir` points to the raw Kaggle folder (default `data/raw`, or `SPAGHETTI_DATA_DIR` in `.env`) or to the extracted Google Drive package. Mixed precision (bfloat16) is used automatically on GPUs that support it. On a laptop RTX 4060 (8 GB), ConvNeXt-Tiny at 224 px takes about 20 s per epoch and about 10 minutes per run.

## Configuration files and model presets

### Choosing a model

`MODEL=` (or `--model`) takes a **preset name** or any `timm` model name. Run `python -m spaghetti1 list-models` to see the presets:

| Preset | `timm` checkpoint | Preset defaults | Tested |
| --- | --- | --- | --- |
| `convnext_tiny` | `convnext_tiny.fb_in22k_ft_in1k` | 224 px, batch 32 | yes (E0–E2) |
| `convnext_small` | `convnext_small.fb_in22k_ft_in1k` | 288 px, batch 24 | yes (E3) |
| `convnextv2_tiny` | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 288 px, batch 32 | no |
| `efficientnetv2_s` | `tf_efficientnetv2_s.in21k_ft_in1k` | 288 px, batch 32 | no |
| `swin_base_384` | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | 384 px, batch 8 × 4, gradient checkpointing | no |
| `dinov2_base` | `vit_base_patch14_reg4_dinov2.lvd142m` | 336 px, batch 8 × 4, gradient checkpointing, LR 2e-4, layer decay 0.75 | no (R2) |
| `eva02_base` | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 336 px, batch 8 × 4, gradient checkpointing, LR 2e-4, layer decay 0.75 | no |

All presets are general-purpose pretrained models, as the rules require. To add a model, add one entry to `MODEL_PRESETS` in [models.py](../spaghetti1/training/models.py), or simply write its `timm` name in `MODEL=`.

### Config files

A config file lists only what differs from the defaults, one `KEY=VALUE` per line. Keys are the `TrainConfig` fields in upper case. `BASE=` inherits a shared recipe:

```ini
# configs/experiments/r1_convnext_tiny_224_recipe_fold0.env
BASE=../recipes/research_v2.env
DESCRIPTION=ConvNeXt-Tiny 224 with the research recipe
MODEL=convnext_tiny
EPOCHS=30
LR=3e-4
LAYER_DECAY=0.85
```

Settings are resolved in this order (later wins): `TrainConfig` defaults → `SPAGHETTI_NUM_WORKERS` from `.env` → model preset → `BASE` file(s) → the experiment file → command-line flags. Unknown keys stop the run with the list of valid keys. Rules and examples: [configs/README.md](../configs/README.md).

### All settings

| Key | Default | Meaning |
| --- | --- | --- |
| `RUN_NAME` | config file name | Output folder `outputs/runs/<run name>/` |
| `DESCRIPTION` | | Free text, saved with the run |
| `MODEL` | `convnext_tiny` | Preset or `timm` name |
| `FOLD`, `SEED` | 0, 42 | Validation fold (`-1` = train on all kept images); random seed |
| `EPOCHS`, `BATCH_SIZE`, `ACCUMULATION_STEPS` | 25, 32, 1 | Effective batch = batch size × accumulation steps |
| `LR`, `HEAD_LR_MULTIPLIER`, `LAYER_DECAY` | 1e-4, 10, 0 | Without layer decay the head learns 10× faster; with layer decay `LR` is the head's rate and every earlier block gets another factor `LAYER_DECAY` |
| `WEIGHT_DECAY`, `WARMUP_EPOCHS`, `DROP_PATH_RATE` | 0.05, 1, 0.1 | AdamW; linear warm-up then cosine decay to 1% |
| `GRAD_CHECKPOINTING`, `EMA_DECAY` | false, 0 | Memory saving for large models; > 0 scores and saves a moving average of the weights |
| `LABEL_SMOOTHING`, `CLASS_WEIGHT_POWER` | 0.1, 0.5 | Loss weights `(1/n_c)^power`; 0 = unweighted |
| `LABEL_POLICY`, `BALANCE_POWER` | `original`, 0 | `original`, `drop_copies`, `as_given` (see 03); > 0 over-samples rare classes |
| `MIXUP_ALPHA`, `MIXUP_PROB`, `MIXUP_OFF_EPOCHS` | 0, 0.5, 5 | MixUp (never CutMix); off in the last epochs |
| `IMAGE_SIZE`, `AUGMENT`, `BACKGROUND_NORMALIZE`, `DENOISE` | 224, true, false, none | See 05 |
| `NUM_WORKERS` | 4 | DataLoader workers (machine setting, best set in `.env`) |
| `PRECISION` | `auto` | `auto` = bfloat16 if the GPU supports it, else float16 with gradient scaling (Kaggle T4/P100); or `bf16`, `fp16`, `fp32` |
| `TTA`, `PREDICT_TEST`, `TEST_TTA`, `RESUME` | true, false, true, false | 8-way TTA evaluation; write `submission.csv` and `test_probabilities.npy`; use TTA for the test predictions; continue from `last.pt` |
| `LIMIT_BATCHES` | 0 | > 0: only this many batches per epoch, for quick smoke tests |

### Traceability

Each run saves `config.env`, with every resolved setting and its source as a comment:

```ini
MODEL=dinov2_base  # configs/experiments/r2_dinov2_base_336_recipe_fold0.env
LR=0.0002  # preset dinov2_base
MIXUP_ALPHA=0.3  # configs/recipes/research_v2.env
EPOCHS=3  # command line
```

It also saves `config.json` with the same values plus a `sources` map. `python -m spaghetti1 train --config outputs/runs/<run>/config.env` repeats a run exactly. The shared `.env` holds only machine settings (paths, workers, runs folder); experiment parameters always live in the config files.

## Config files of every reported run

Each result on this page has its config file in `configs/experiments/` (they resolve to exactly the settings saved with the runs):

| Run | Config file |
| --- | --- |
| E0 | `e0_convnext_tiny_original_fold0.env` |
| E1a | `e1_convnext_tiny_drop_copies_fold0.env` |
| E1b | `e1_convnext_tiny_as_given_fold0.env` |
| E2a | `e2_convnext_tiny_288_fold0.env` |
| E2b | `e2_convnext_tiny_384_fold0.env` |
| E3 | `e3_convnext_small_288_fold0.env` |
| R1 | `r1_convnext_tiny_224_recipe_fold0.env` |
| R2 | `r2_dinov2_base_336_recipe_fold0.env` |

Retraining gives very close but not bit-identical numbers (GPU non-determinism). To use the exact trained models, load the shared weights instead ([09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)).

## What a run saves (`outputs/runs/<run_name>/`)

| File | Contents |
| --- | --- |
| `config.env` | All resolved settings with their source; re-runnable with `--config` |
| `config.json` | The same settings plus a `sources` map |
| `history.csv` | Per epoch: learning rate, training loss/accuracy, validation loss, weighted F1, macro F1, accuracy, seconds |
| `metrics.json` | Validation metrics of the best epoch (per class and confusion matrix), without and with TTA |
| `confusion_matrix.png`, `confusion_matrix_tta.png` | Row-normalised confusion matrices |
| `val_predictions.csv` | Per validation image: true label, prediction, class probabilities (for ensembling and error analysis) |
| `val_probabilities.npy`, `val_probabilities_tta.npy` | Raw validation probabilities without and with TTA (`split_fold` order) |
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
| R1 | ConvNeXt-Tiny | 224 | drop_copies, research recipe | 0.831 | 0.828 | 0.39 | 0.75 | 0.70 |
| R2 | DINOv2-Base | 336 | drop_copies, research recipe | **0.852** | 0.839 | 0.38 | 0.81 | 0.72 |
| Ensemble | E1a + E2b + E3 (average of TTA probabilities) | | | | 0.856 | 0.57 | 0.78 | 0.70 |
| Ensemble | E1a + E2b + E3 + R2 | | | | **0.859** | 0.59 | 0.78 | 0.72 |

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
| Not yet tested | EMA, MixUp, layer-wise LR decay, DINOv2/EVA-02, 5-fold training, pseudo-labelling, class-weight and threshold tuning | | Planned in 10 (research review) |

## Submissions

| File (`results/submissions/`) | Models | Val wF1 | Kaggle public |
| --- | --- | ---: | ---: |
| `e0_convnext_tiny_original_fold0/submission.csv` | ConvNeXt-Tiny 224 | 0.836 | |
| `ensemble_tiny224_tiny384/submission.csv` | ConvNeXt-Tiny 224 + 384, TTA | 0.846 | |
| `ensemble_tiny224_tiny384_small288/submission.csv` | + ConvNeXt-Small 288 | 0.856 | **0.8371** |
| `lb1_a_dinov2_alone/submission.csv` | DINOv2-Base 336 alone (no TTA) | 0.851 | 0.8321 |
| `lb1_b_4models_convnext3_dinov2/submission.csv` | 3 ConvNeXt + DINOv2 | 0.859 | 0.8410 |
| `lb1_c_tiny384_small288_dinov2/submission.csv` | Tiny-384 + Small-288 + DINOv2 | 0.860 | 0.8424 |
| `n2_dinov3_base_384_recipe_fold0/submission.csv` | DINOv3-Base 384 alone (no TTA) | 0.832 | 0.8253 |
| `lb1_d_5models_convnext3_dinov2_dinov3/submission.csv` | 3 ConvNeXt + DINOv2 + DINOv3 | 0.855 (test-like 0.861) | **0.8441** |

**Validation vs leaderboard (2026-11-27).** The 3-model ensemble scored 0.8371 on the public leaderboard, 1.9 points below its fold-0 validation score. The test set shows the same predicted class mix as validation (no class or label-convention shift), but the models are slightly less confident on test images (mean top probability 0.813 vs 0.829), most of all on images least similar to the training set. The likely causes are: selection of epochs and ensemble on the same 599 images (optimistic), validation noise (±1.5 points), mild test-domain shift, and fold-0 models trained on 80% of the data. Expect leaderboard scores about 2 points below fold-0 validation; 5-fold out-of-fold scores, full-data models, and pseudo-labelling are the planned remedies (see 10).

How to recreate every number and file from the shared weights is described in [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md). The improvement plan (EMA, multi-architecture 5-fold ensemble, pseudo-labelling, per-class probability tuning) is in [07 · Next steps](07_NEXT_STEPS.md).
