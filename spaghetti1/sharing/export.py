"""Package the cleaned dataset as one zip file for sharing on Google Drive.

Command: ``python -m spaghetti1 export``. Documentation: ``docs/06_SHARING_AND_LOADING_DATA.md``.

The zip has this layout (``spaghetti1_clean_v1/`` is the root inside the archive)::

    spaghetti1_clean_v1/
    ├── dataset_info.json         # version, file counts, checksums of the manifests
    ├── DATASET_CARD.md           # what was cleaned and how to load it
    ├── manifests/
    │   ├── train_manifest.csv    # all 4,328 training rows with flags, groups, and folds
    │   ├── test_manifest.csv     # all 14,736 test rows with quality flags
    │   └── preprocessing_summary.json
    ├── train/                    # only the 3,995 kept training PNGs (no aliens, no blank)
    ├── test/                     # all 14,736 test PNGs (every one needs a prediction)
    ├── train.csv                 # official files, unchanged
    └── sample_submission.csv

PNGs are already compressed, so the archive stores them without recompression (fast).
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

PACKAGE_NAME = "spaghetti1_clean"
PACKAGE_VERSION = "v1"
INFO_FILE = "dataset_info.json"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def export_package(data_dir: Path, manifest_dir: Path, output_dir: Path, version: str = PACKAGE_VERSION) -> Path:
    """Write ``<output_dir>/spaghetti1_clean_<version>.zip`` and return its path."""
    train = pd.read_csv(manifest_dir / "train_manifest.csv")
    test = pd.read_csv(manifest_dir / "test_manifest.csv")
    kept = train.loc[train["keep"]]
    root = f"{PACKAGE_NAME}_{version}"
    manifest_files = ["train_manifest.csv", "test_manifest.csv", "preprocessing_summary.json"]

    info = {
        "name": PACKAGE_NAME,
        "version": version,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {"train_png": len(kept), "test_png": len(test), "train_manifest_rows": len(train)},
        "manifest_sha256": {name: _sha256(manifest_dir / name) for name in manifest_files},
        "note": "train/ holds only rows with keep=True; excluded rows stay in the manifest for reference.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root}.zip"
    partial = archive.with_suffix(".zip.partial")
    card = Path(__file__).with_name("DATASET_CARD.md")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        bundle.writestr(f"{root}/{INFO_FILE}", json.dumps(info, indent=2) + "\n")
        bundle.write(card, f"{root}/DATASET_CARD.md")
        for name in manifest_files:
            bundle.write(manifest_dir / name, f"{root}/manifests/{name}")
        for name in ("train.csv", "sample_submission.csv"):
            bundle.write(data_dir / name, f"{root}/{name}")
        for count, path in enumerate(pd.concat([kept["path"], test["path"]]), start=1):
            bundle.write(data_dir / path, f"{root}/{path}")
            if count % 5000 == 0:
                print(f"  {count:,} images packed", flush=True)
    shutil.move(partial, archive)
    return archive
