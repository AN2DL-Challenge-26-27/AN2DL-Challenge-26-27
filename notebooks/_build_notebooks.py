"""Generate ``01_load_preprocessed_data.ipynb`` and ``02_kaggle_training.ipynb`` from the cell lists below.

Keeping the notebook source in this script makes changes easy to review in git.
Run ``python notebooks/_build_notebooks.py`` after editing the cells.
"""

import json
from pathlib import Path

CELLS = [
    ("markdown", """# Notebook 01: Load the preprocessed SPAGHETTI-1 data

This notebook is the starting point for every teammate. It downloads the **cleaned dataset package** from Google Drive and checks it. It then builds the PyTorch DataLoaders that every model notebook uses.

Read the documentation in this order: `docs/00_START_HERE.md`, then `docs/03_PREPROCESSING.md` (what was cleaned and why), then `docs/06_SHARING_AND_LOADING_DATA.md` (this notebook).

| Step | What happens |
| --- | --- |
| 1 | Get the repository code and the two packages Colab may lack |
| 2 | Download and verify the cleaned dataset from Google Drive |
| 3 | Read the manifests and the cleaning summary |
| 4 | Build the DataLoaders for one cross-validation fold |
| 5 | Look at one augmented training batch |
| 6 | Copy the snippet into your training notebook |

In Colab, choose **Runtime → Change runtime type → GPU** before running."""),
    ("markdown", """## Step 1: Get the code

On Colab, this cell clones the team repository. If the repository is private, add a GitHub token as the Colab secret `GITHUB_PAT` (key icon in the left sidebar). When you run the notebook from a local clone, the cell uses that clone instead."""),
    ("code", """import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "github.com/AN2DL-Challenge-26-27/AN2DL-Challenge-26-27.git"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    REPO_DIR = Path("/content/AN2DL-Challenge-26-27")
    if not REPO_DIR.exists():
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_PAT")
        except Exception:
            token = None
        url = f"https://{token}@{REPO_URL}" if token else f"https://{REPO_URL}"
        subprocess.run(["git", "clone", "--depth", "1", url, str(REPO_DIR)], check=True)
    # Colab already has torch, torchvision, OpenCV, scikit-learn, pandas, and matplotlib.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-dotenv", "gdown"], check=True)
else:
    REPO_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("Repository:", REPO_DIR)"""),
    ("markdown", """## Step 2: Download the cleaned dataset

Paste the Google Drive link of `spaghetti1_clean_v1.zip` below. The file must be shared as **Anyone with the link → Viewer**. Other accepted sources:

* the path of the zip on a mounted Drive, for example `/content/drive/MyDrive/AN2DL/spaghetti1_clean_v1.zip` (run `from google.colab import drive; drive.mount("/content/drive")` first);
* a folder where the package is already extracted.

The zip (~2.6 GB) is extracted to `/content/spaghetti1_data` on Colab's fast local disk. Re-running the cell reuses the verified copy."""),
    ("code", """from spaghetti1.sharing.fetch import prepare_dataset

DATA_SOURCE = "PASTE_THE_GOOGLE_DRIVE_LINK_HERE"

data_dir = prepare_dataset(DATA_SOURCE)
print("Dataset folder:", data_dir)"""),
    ("markdown", """## Step 3: Read the manifests

`train_manifest.csv` has one row per official training image, including the excluded ones (`keep == False`, with `exclude_reason`). The most important columns:

| Column | Meaning |
| --- | --- |
| `keep` | `False` for the 332 alien overlays and 1 blank frame; their PNGs are not in the package |
| `fold` | Cross-validation fold 0–4 (grouped by duplicate group, stratified by class) |
| `dup_group`, `is_copy` | Copies of the same sky cutout share a group; the cleanest image is the original |
| `label`, `original_label` | Official label, and the label of the group's original |

`docs/04_MANIFESTS_AND_FOLDS.md` describes every column."""),
    ("code", """import json

import pandas as pd

from spaghetti1.config import CLASS_NAMES
from spaghetti1.preprocessing.dataset import load_test_manifest, load_train_manifest

manifest_dir = data_dir / "manifests"
summary = json.loads((manifest_dir / "preprocessing_summary.json").read_text())
print(json.dumps({key: summary[key] for key in ("train_kept", "excluded", "duplicate_groups", "copies",
                                                 "copies_with_label_different_from_original", "unique_images")}, indent=2))

train_all = load_train_manifest(manifest_dir, kept_only=False)
train = load_train_manifest(manifest_dir)
test = load_test_manifest(manifest_dir)
counts = pd.DataFrame({
    "class": [CLASS_NAMES[label] for label in range(10)],
    "official": train_all["label"].value_counts().sort_index().to_numpy(),
    "kept": train["label"].value_counts().reindex(range(10), fill_value=0).to_numpy(),
    "kept, original labels": train["original_label"].value_counts().reindex(range(10), fill_value=0).to_numpy(),
})
print(f"{len(train):,} kept training images, {len(test):,} test images")
counts"""),
    ("markdown", """## Step 4: Build the DataLoaders for one fold

`build_dataloaders` applies the agreed pipeline:

* **Training set:** the kept images outside the fold. Copies get their original's label (`label_policy="original"`) and the training augmentations are on.
* **Validation set:** the fold's originals and unique images, with no augmentation.
* **Test set:** all 14,736 images in manifest order, so predictions line up with `test_frame["id"]`.

Change `image_size` to match your backbone (for example 224 for most ImageNet models). Use `fold=-1` to train the final model on all kept images."""),
    ("code", """from spaghetti1.preprocessing.loaders import build_dataloaders

FOLD = 0
loaders = build_dataloaders(
    data_dir,
    fold=FOLD,
    batch_size=64,
    image_size=224,
    label_policy="original",   # or "drop_copies" / "as_given" for comparison runs
    balance_power=0.0,         # > 0 over-samples rare classes; 0 = plain shuffling
    num_workers=2,
    cache_images=True,         # keep the decoded training images in RAM (~0.8 GB)
)
print(f"train {len(loaders.train_frame):,} | validation {len(loaders.validation_frame):,} | test {len(loaders.test_frame):,}")
print("class weights for the loss:", loaders.class_weights.numpy().round(2))"""),
    ("markdown", """## Step 5: Look at one augmented training batch

The images should look like test images: random orientation, darker or brighter, colour casts, some noise or blur, and now and then a coloured stripe or streak. The galaxy shape must stay recognisable."""),
    ("code", """import matplotlib.pyplot as plt
import torch

from spaghetti1.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD

images, labels = next(iter(loaders.train))
mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
figure, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for axis, image, label in zip(axes.flat, images, labels):
    axis.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
    axis.set_title(f"{label.item()}: {CLASS_NAMES[label.item()]}", fontsize=7)
    axis.axis("off")
plt.tight_layout()
plt.show()"""),
    ("markdown", """## Step 6: Use the loaders in a training notebook

```python
criterion = torch.nn.CrossEntropyLoss(weight=loaders.class_weights.to(device), label_smoothing=0.1)
for epoch in range(num_epochs):
    for images, labels in loaders.train:
        ...
    # evaluate on loaders.validation with weighted F1 (sklearn.metrics.f1_score(average="weighted"))

# test predictions, in the same order as loaders.test_frame["id"]
submission = pd.DataFrame({"id": loaders.test_frame["id"], "label": predictions})
submission.to_csv("submission.csv", index=False)
```

The first experiments are listed in `docs/07_NEXT_STEPS.md`. Start with the label-policy comparison, because it decides how 996 training images are used."""),
]


KAGGLE_CELLS = [
    ("markdown", """# Notebook 02: Train SPAGHETTI-1 models on Kaggle (2 x T4)

This notebook trains a list of experiment configs from the team repository on Kaggle's free GPUs and exports the results as two zip files you can download. Full guide: `docs/11_KAGGLE_TRAINING.md`.

**Before you run it** (right-hand panel, *Session options*):

1. **Accelerator: GPU T4 x2.**
2. **Internet: on.** This needs a phone-verified Kaggle account. It is used to clone the repository and download the pretrained weights.
3. **Add Input → Competitions → `an2dl2627`** (the official data; nothing needs to be uploaded).
4. Only if the repository is private: *Add-ons → Secrets* → add your GitHub token as `GITHUB_PAT`.

For long runs use **Save Version → Save & Run All (Commit)**. The notebook then runs in the background, also with the browser closed, for up to 12 hours. The zip files appear under *Output* when it finishes."""),
    ("markdown", """## Step 1: Get the code and two packages

Kaggle already has PyTorch, OpenCV, scikit-learn, and pandas. `timm` is upgraded because DINOv3 needs timm 1.0.20 or newer."""),
    ("code", """import os
import subprocess
import sys
from pathlib import Path

REPO = "github.com/AN2DL-Challenge-26-27/AN2DL-Challenge-26-27.git"
REPO_DIR = Path("/kaggle/working/AN2DL-Challenge-26-27")

if not REPO_DIR.exists():
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_PAT")
    except Exception:
        token = None
    url = f"https://{token}@{REPO}" if token else f"https://{REPO}"
    subprocess.run(["git", "clone", "--depth", "1", url, str(REPO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm>=1.0.22", "python-dotenv"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
import timm, torch
print("repository:", REPO_DIR, "| commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
print("torch", torch.__version__, "| timm", timm.__version__)
for gpu in range(torch.cuda.device_count()):
    print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)} | bf16 supported: {torch.cuda.is_bf16_supported()}")"""),
    ("markdown", """## Step 2: Find the competition data

The code reads the raw competition files directly and uses the team's manifests and folds from the repository (`manifests/`), so the results are comparable with every earlier run."""),
    ("code", """sys.path.insert(0, str(REPO_DIR / "experiments"))
from kaggle_queue import find_data_dir

DATA_DIR = find_data_dir()
print("data:", DATA_DIR, "|", len(list((DATA_DIR / "train").glob("*.png"))), "train /", len(list((DATA_DIR / "test").glob("*.png"))), "test images")"""),
    ("markdown", """## Step 3: Choose what to train

Pick one plan, or write your own list of config files from `configs/experiments/`. The configs are spread over both GPUs. Rough durations on 2 x T4 (to be confirmed on the first run):

| Plan | Runs | Approximate time |
| --- | --- | --- |
| `dinov2_5fold` | DINOv2-Base 336, folds 1–4 (fold 0 already trained on the laptop) | 2–3 h |
| `dinov2_5fold_all` | DINOv2-Base 336, folds 0–4 | 3–4 h |
| `dinov3_5fold` | DINOv3 ViT-B/16 384, folds 0–4 | 3–4 h |
| `custom` | your own `CUSTOM_CONFIGS` list | |

Every run also predicts the 14,736 test images (without TTA, which did not help DINOv2)."""),
    ("code", """PLAN = "dinov2_5fold"          # dinov2_5fold | dinov2_5fold_all | dinov3_5fold | custom
CUSTOM_CONFIGS = []             # e.g. ["configs/experiments/n1_dinov3_convnext_tiny_288_fold0.env"]

PLANS = {
    "dinov2_5fold": [f"configs/experiments/r2_dinov2_base_336_recipe_fold{k}.env" for k in (1, 2, 3, 4)],
    "dinov2_5fold_all": [f"configs/experiments/r2_dinov2_base_336_recipe_fold{k}.env" for k in range(5)],
    "dinov3_5fold": [f"configs/experiments/n2_dinov3_base_384_recipe_fold{k}.env" for k in range(5)],
    "custom": CUSTOM_CONFIGS,
}
CONFIGS = PLANS[PLAN]
assert CONFIGS, "empty plan"
for config in CONFIGS:
    assert (REPO_DIR / config).is_file(), f"missing {config}"
print(f"{len(CONFIGS)} runs:")
for config in CONFIGS:
    print("  ", config)
# Inspect the full settings of the first run (no training):
subprocess.run([sys.executable, "-m", "spaghetti1", "train", "--config", CONFIGS[0], "--dry-run"])"""),
    ("markdown", """## Step 4: Train

Progress is printed every minute: the current run on each GPU and its latest epoch. The full logs are in `/kaggle/working/runs/logs/`. A run that fails does not stop the others. If the session is interrupted, running this cell again continues unfinished runs (`RESUME=true`)."""),
    ("code", """TAG = f"kaggle_{PLAN}"
command = [sys.executable, "-u", "experiments/kaggle_queue.py", "--configs", *CONFIGS,
           "--data-dir", str(DATA_DIR), "--work-dir", "/kaggle/working", "--tag", TAG]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="")
print("exit code:", process.wait())"""),
    ("markdown", """## Step 5: Results

`summary/summary.json` lists every run, its status, and for each group of folds the per-fold and **out-of-fold** weighted F1 (about 3,000 images, more reliable than one fold). It also contains a fold-ensemble submission (average of the folds' test probabilities)."""),
    ("code", """import json
import pandas as pd

summary = json.loads(Path("/kaggle/working/summary/summary.json").read_text())
print(json.dumps(summary["fold_groups"], indent=2))
results = pd.read_csv("/kaggle/working/runs/results.csv")
results[["run_name", "fold", "image_size", "best_epoch", "weighted_f1", "weighted_f1_tta", "minutes"]]"""),
    ("markdown", """## Step 6: Download and share the results

Two files are written to `/kaggle/working/` (shown under *Output* after a committed run):

| File | Contents | Use |
| --- | --- | --- |
| `spaghetti1_results_<tag>.zip` | Everything except model weights: results, logs, configs, curves, validation and test probabilities, submissions, summary | Small: send this one for analysis and ensembling |
| `spaghetti1_weights_<tag>.zip` | The trained models (`best.pt`) with checksums | Large: upload to the team Drive; `python -m spaghetti1 fetch-weights --source <zip>` |

A fold-ensemble `submission.csv` can be submitted directly from `summary/fold_ensemble_<group>/`."""),
    ("code", """from IPython.display import FileLink, display

for path in sorted(Path("/kaggle/working").glob("spaghetti1_*.zip")):
    print(f"{path.name}: {path.stat().st_size / 2**20:.0f} MB")
    display(FileLink(str(path.relative_to("/kaggle/working"))))
for path in sorted(Path("/kaggle/working/summary").glob("fold_ensemble_*/submission.csv")):
    print("fold-ensemble submission:", path)"""),
]


def build(path: Path, cells: list = CELLS) -> None:
    def cell(kind: str, source: str) -> dict:
        lines = source.splitlines(keepends=True)
        base = {"cell_type": kind, "metadata": {}, "source": lines}
        return {**base, "execution_count": None, "outputs": []} if kind == "code" else base

    notebook = {
        "cells": [cell(kind, source) for kind, source in cells],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).with_name("01_load_preprocessed_data.ipynb"))
    build(Path(__file__).with_name("02_kaggle_training.ipynb"), KAGGLE_CELLS)
