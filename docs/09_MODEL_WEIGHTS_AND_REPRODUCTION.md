# 09 · Model weights and reproducing the results

Previous: [08 · Training and results](08_TRAINING_AND_RESULTS.md) · Next: [10 · Research review and plan](10_RESEARCH_REVIEW_AND_PLAN.md)

Every trained model is shared as **one zip file on Google Drive**, next to the image dataset. With it, anyone can check the reported validation scores and create the submission files **without retraining**.

Code: [spaghetti1/training/predict.py](../spaghetti1/training/predict.py) (evaluate and predict) and [spaghetti1/sharing/weights.py](../spaghetti1/sharing/weights.py) (pack, fetch, verify).

> **Update 2026-11-27: `spaghetti1_weights_v2.zip` (1.67 GB) supersedes v1.** It contains all 10 fold-0 runs: the six below plus R1, R2 (DINOv2), N1, and N2 (DINOv3), with their saved validation and test probabilities. It also records the three ensembles with Kaggle scores (best: 0.8441). Paths in the commands below then read `spaghetti1_weights_v2`. The model list is in the package's `WEIGHTS_CARD.md`.

## What is shared where

| Item | Where | Size |
| --- | --- | ---: |
| Cleaned images + manifests (`spaghetti1_clean_v1.zip`) | Google Drive | 2.6 GB |
| Model weights (`spaghetti1_weights_v1.zip`) | Google Drive | 756 MB |
| Code, docs, manifests | GitHub | small |
| Lightweight results: configs, learning curves, metrics, confusion matrices, validation predictions, submissions | GitHub, [results/](../results/) | 1.6 MB |

## The weights package: `spaghetti1_weights_v1.zip`

```text
spaghetti1_weights_v1/
├── weights_info.json     # per model: backbone, input size, fold, label policy, validation scores, SHA-256 of best.pt
├── WEIGHTS_CARD.md       # short description of every model and how to use it
└── runs/<run_name>/
    ├── best.pt           # weights (PyTorch state_dict) of the best validation epoch
    ├── config.json       # every training setting: rebuilds the network and its exact preprocessing
    ├── metrics.json, history.csv, val_predictions.csv
    └── test_probabilities.npy   # only for runs that predicted the test set during training
```

| Run folder | Backbone | Input | Label policy | Val wF1 | + TTA |
| --- | --- | ---: | --- | ---: | ---: |
| `e0_convnext_tiny_original_fold0` | ConvNeXt-Tiny | 224 | original | 0.836 | 0.835 |
| `e1_convnext_tiny_drop_copies_fold0` | ConvNeXt-Tiny | 224 | drop_copies | 0.843 | 0.844 |
| `e1_convnext_tiny_as_given_fold0` | ConvNeXt-Tiny | 224 | as_given | 0.836 | 0.829 |
| `e2_convnext_tiny_288_fold0` | ConvNeXt-Tiny | 288 | drop_copies | 0.846 | 0.833 |
| `e2_convnext_tiny_384_fold0` | ConvNeXt-Tiny | 384 | drop_copies | 0.847 | 0.846 |
| `e3_convnext_small_288_fold0` | ConvNeXt-Small | 288 | drop_copies | 0.841 | 0.847 |

All models are trained on CV fold 0 and scored on its 599 originals and unique images. The backbones are general-purpose ImageNet-pretrained `timm` models (`convnext_tiny.fb_in22k_ft_in1k`, `convnext_small.fb_in22k_ft_in1k`), as the competition rules require. `last.pt` (the optimizer state for resuming training) is not included. It is three times larger and is not needed to evaluate or predict.

**Best validated combination:** the average of the TTA probabilities of `e1_convnext_tiny_drop_copies_fold0` + `e2_convnext_tiny_384_fold0` + `e3_convnext_small_288_fold0`, validation weighted F1 **0.856**.

## A · Uploading (done once)

```bash
python -m spaghetti1 export-weights      # -> outputs/export/spaghetti1_weights_v1.zip (about 10 seconds)
```

Upload the zip to the same Google Drive folder as `spaghetti1_clean_v1.zip` and share it the same way (see [06](06_SHARING_AND_LOADING_DATA.md)). After new runs, pass them with `--runs <folder> ...` and `--version v2`.

## B · Getting the weights

```bash
python -m spaghetti1 fetch-weights --source "<Drive link, or path to spaghetti1_weights_v1.zip>"
```

The weights land in `data/weights/spaghetti1_weights_v1/` locally (ignored by git) or `/content/spaghetti1_weights/spaghetti1_weights_v1/` on Colab. Every `best.pt` is checked against its SHA-256, and running the command again reuses a verified copy. The same three source types as the dataset work: Drive link, zip on a mounted Drive, or an extracted folder. You also need the dataset itself (`fetch`, or the raw Kaggle data in `data/raw`).

## C · Checking a reported score (no training)

```bash
python -m spaghetti1 evaluate --run-dir data/weights/spaghetti1_weights_v1/runs/e1_convnext_tiny_drop_copies_fold0
```

The command rebuilds the network from `config.json`, loads `best.pt`, scores the run's validation fold with the same preprocessing, and prints the new score next to the one recorded during training. Add `--no-tta` for the plain score. No internet access is needed; the pretrained ImageNet weights are not downloaded again.

Verified on 2026-11-26 (CPU): `e1_convnext_tiny_drop_copies_fold0` gives 0.8441 with TTA, exactly as recorded. Without TTA it gives 0.8409 vs 0.8425 recorded, because the CPU computes in float32 and the training GPU in bfloat16, which flips one or two borderline images. On a GPU the numbers match exactly.

## D · Creating a submission from the weights (no training)

```bash
W=data/weights/spaghetti1_weights_v1/runs
python -m spaghetti1 predict --output-dir outputs/submissions/best_ensemble --run-dirs \
    $W/e1_convnext_tiny_drop_copies_fold0 $W/e2_convnext_tiny_384_fold0 $W/e3_convnext_small_288_fold0
```

(PowerShell: `$W = "data/weights/spaghetti1_weights_v1/runs"`, and write the arguments on one line.)

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
model, config = load_run("data/weights/spaghetti1_weights_v1/runs/e2_convnext_tiny_384_fold0", device)
transform = GalaxyTransform(PreprocessConfig(image_size=config["image_size"]), None)   # evaluation transform
# probabilities = model(transform(pixels)[None].to(device)).softmax(1)   # pixels: uint8 (256, 256, 3)
```

## Submissions made so far

| Folder in `results/submissions/` | Models | Val wF1 | Notes |
| --- | --- | ---: | --- |
| `e0_convnext_tiny_original_fold0` | ConvNeXt-Tiny 224, label policy `original` | 0.836 | First baseline |
| `ensemble_tiny224_tiny384` | ConvNeXt-Tiny 224 + 384 (copies dropped), TTA | 0.846 | Best file available so far |
| `ensemble_tiny224_tiny384_small288` | + ConvNeXt-Small 288, TTA | 0.856 | Created 2026-11-27 after the GPU restart |

The third model's test predictions could not be made on the training laptop: Windows blocked GPU access after the long training session ("GPU access blocked by the operating system"). After a restart, or on Colab, command D creates this file in about 6 minutes, reusing the cached predictions of the other two models if they are in the same output folder.
