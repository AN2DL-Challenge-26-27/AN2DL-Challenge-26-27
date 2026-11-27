# 09 · Model weights and reproducing the results

Previous: [08 · Training and results](08_TRAINING_AND_RESULTS.md) · Next: [10 · Research review and plan](10_RESEARCH_REVIEW_AND_PLAN.md)

Every trained model is shared as **one zip file on Google Drive**, next to the image dataset. With it, anyone can check the reported validation scores and create the submission files **without retraining**.

Code: [spaghetti1/training/predict.py](../spaghetti1/training/predict.py) (evaluate and predict) and [spaghetti1/sharing/weights.py](../spaghetti1/sharing/weights.py) (pack, fetch, verify).

## What is shared where

| Item | Where | Size |
| --- | --- | ---: |
| Cleaned images + manifests (`spaghetti1_clean_v1.zip`) | Google Drive | 2.6 GB |
| Model weights (`spaghetti1_weights_v2.zip`, all 10 fold-0 models; replaces v1 with 6) | Google Drive | 1.67 GB |
| Code, docs, manifests | GitHub | small |
| Lightweight results: configs, learning curves, metrics, confusion matrices, validation predictions, submissions | GitHub, [results/](../results/) | ~4 MB |

## The weights package: `spaghetti1_weights_v2.zip`

```text
spaghetti1_weights_v2/
├── weights_info.json     # per model: backbone, input size, fold, label policy, validation scores, SHA-256 of best.pt
├── WEIGHTS_CARD.md       # short description of every model and how to use it
└── runs/<run_name>/
    ├── best.pt           # weights (PyTorch state_dict) of the best validation epoch
    ├── config.json, config.env   # every training setting: rebuilds the network and its exact preprocessing
    ├── metrics.json, history.csv, val_predictions.csv
    └── val_probabilities*.npy, test_probabilities.npy   # where saved (newer runs)
```

| Run folder | Backbone | Input | Label policy | Val wF1 | + TTA |
| --- | --- | ---: | --- | ---: | ---: |
| `e0_convnext_tiny_original_fold0` | ConvNeXt-Tiny | 224 | original | 0.836 | 0.835 |
| `e1_convnext_tiny_drop_copies_fold0` | ConvNeXt-Tiny | 224 | drop_copies | 0.843 | 0.844 |
| `e1_convnext_tiny_as_given_fold0` | ConvNeXt-Tiny | 224 | as_given | 0.836 | 0.829 |
| `e2_convnext_tiny_288_fold0` | ConvNeXt-Tiny | 288 | drop_copies | 0.846 | 0.833 |
| `e2_convnext_tiny_384_fold0` | ConvNeXt-Tiny | 384 | drop_copies | 0.847 | 0.846 |
| `e3_convnext_small_288_fold0` | ConvNeXt-Small | 288 | drop_copies | 0.841 | 0.847 |
| `r1_convnext_tiny_224_recipe_fold0` | ConvNeXt-Tiny | 224 | drop_copies, research recipe | 0.831 | 0.828 |
| `r2_dinov2_base_336_recipe_fold0` | DINOv2-Base/14 | 336 | drop_copies, research recipe | **0.852** | 0.839 |
| `n1_dinov3_convnext_tiny_288_fold0` | DINOv3 ConvNeXt-Tiny | 288 | drop_copies | 0.840 | 0.833 |
| `n2_dinov3_base_384_recipe_fold0` | DINOv3 ViT-B/16 | 384 | drop_copies, research recipe | 0.832 | 0.843 |

All models are trained on CV fold 0 and scored on its 599 originals and unique images. The backbones are general-purpose pretrained `timm` models (ImageNet-22k ConvNeXt, DINOv2 LVD-142M, DINOv3 LVD-1689M web images), as the competition rules require. `last.pt` (the optimizer state for resuming training) is not included. It is three times larger and is not needed to evaluate or predict.

**Best combination on Kaggle (0.8441):** `e1_convnext_tiny_drop_copies_fold0` + `e2_convnext_tiny_384_fold0` + `e3_convnext_small_288_fold0` (TTA probabilities) + `r2_dinov2_base_336_recipe_fold0` + `n2_dinov3_base_384_recipe_fold0` (probabilities without TTA). Validation 0.855, test-like validation 0.861. `weights_info.json` records this and the two other Kaggle-scored ensembles.

## A · Uploading (done once)

```bash
python -m spaghetti1 export-weights      # -> outputs/export/spaghetti1_weights_v2.zip (about 10 seconds)
```

Upload the zip to the same Google Drive folder as `spaghetti1_clean_v1.zip` and share it the same way (see [06](06_SHARING_AND_LOADING_DATA.md)). After new runs, pass them with `--runs <folder> ...` and `--version v2`.

## B · Getting the weights

```bash
python -m spaghetti1 fetch-weights --source "<Drive link, or path to spaghetti1_weights_v2.zip>"
```

The weights land in `data/weights/spaghetti1_weights_v2/` locally (ignored by git) or `/content/spaghetti1_weights/spaghetti1_weights_v2/` on Colab. Every `best.pt` is checked against its SHA-256, and running the command again reuses a verified copy. The same three source types as the dataset work: Drive link, zip on a mounted Drive, or an extracted folder. You also need the dataset itself (`fetch`, or the raw Kaggle data in `data/raw`).

## C · Checking a reported score (no training)

```bash
python -m spaghetti1 evaluate --run-dir data/weights/spaghetti1_weights_v2/runs/e1_convnext_tiny_drop_copies_fold0
```

The command rebuilds the network from `config.json`, loads `best.pt`, scores the run's validation fold with the same preprocessing, and prints the new score next to the one recorded during training. Add `--no-tta` for the plain score. No internet access is needed; the pretrained ImageNet weights are not downloaded again.

Verified on 2026-11-26 (CPU): `e1_convnext_tiny_drop_copies_fold0` gives 0.8441 with TTA, exactly as recorded. Without TTA it gives 0.8409 vs 0.8425 recorded, because the CPU computes in float32 and the training GPU in bfloat16, which flips one or two borderline images. On a GPU the numbers match exactly.

## D · Creating a submission from the weights (no training)

```bash
W=data/weights/spaghetti1_weights_v2/runs
python -m spaghetti1 predict --output-dir outputs/submissions/best_ensemble --run-dirs \
    $W/e1_convnext_tiny_drop_copies_fold0 $W/e2_convnext_tiny_384_fold0 $W/e3_convnext_small_288_fold0
```

(PowerShell: `$W = "data/weights/spaghetti1_weights_v2/runs"`, and write the arguments on one line.)

`predict` writes into the output folder:

| File | Contents |
| --- | --- |
| `submission.csv` | `id,label` for all 14,736 test images, checked before saving |
| `ensemble_info.json` | Models used, TTA on/off, predicted class counts, validation weighted F1 of each model and of **this exact ensemble** |
| `test_probabilities_<run>.npy`, `val_probabilities_<run>.npy` | Cached probabilities per model; an interrupted call continues from them |
| `test_probabilities_ensemble.npy` | Averaged test probabilities |

One `--run-dirs` entry gives a single-model submission. Any combination can be tried; `ensemble_info.json` shows its validation score.

### Run time

| Hardware | ConvNeXt-Tiny 224 | ConvNeXt-Tiny 384 | ConvNeXt-Small 288 |
| --- | ---: | ---: | ---: |
| Laptop RTX 4060 GPU, 14,736 test images, 8-way TTA | 2.5 min | 7 min | ~6 min |
| CPU (8 threads), 599 validation images, 8-way TTA | 4 min | ~10 min | ~10 min |

The full test set on a CPU with TTA takes hours (ConvNeXt-Small about 4 hours). Use a GPU (Colab is enough) or `--no-tta` for a quick check.

## E · Using a model in your own code

```python
import torch
from spaghetti1.training.predict import load_run
from spaghetti1.preprocessing.transforms import GalaxyTransform, PreprocessConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config = load_run("data/weights/spaghetti1_weights_v2/runs/e2_convnext_tiny_384_fold0", device)
transform = GalaxyTransform(PreprocessConfig(image_size=config["image_size"]), None)   # evaluation transform
# probabilities = model(transform(pixels)[None].to(device)).softmax(1)   # pixels: uint8 (256, 256, 3)
```

## Submissions made so far

All files are in `results/submissions/` (with `ensemble_info.json` and, on Drive, the members' probabilities). Rebuild ensembles from saved probabilities with `experiments/combine_submissions.py`.

| Folder | Models | Val wF1 | Kaggle public |
| --- | --- | ---: | ---: |
| `lb1_d_5models_convnext3_dinov2_dinov3` | 3 ConvNeXt + DINOv2 + DINOv3 | 0.855 (test-like 0.861) | **0.8441** |
| `lb1_c_tiny384_small288_dinov2` | ConvNeXt Tiny-384 + Small-288 + DINOv2 | 0.860 | 0.8424 |
| `lb1_b_4models_convnext3_dinov2` | 3 ConvNeXt + DINOv2 | 0.859 | 0.8410 |
| `ensemble_tiny224_tiny384_small288` | 3 ConvNeXt (TTA) | 0.856 | 0.8371 |
| `lb1_a_dinov2_alone` | DINOv2-Base 336 | 0.851 | 0.8321 |
| `n2_dinov3_base_384_recipe_fold0` | DINOv3 ViT-B/16 384 | 0.832 | 0.8253 |
| `ensemble_tiny224_tiny384` | ConvNeXt Tiny-224 + Tiny-384 (TTA) | 0.846 | |
| `e0_convnext_tiny_original_fold0` | ConvNeXt-Tiny 224 baseline | 0.836 | |
