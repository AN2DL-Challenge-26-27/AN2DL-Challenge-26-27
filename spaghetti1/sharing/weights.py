"""Package trained model weights for Google Drive, fetch them back, and snapshot results for git.

Commands (documentation: ``docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md``):

* ``python -m spaghetti1 export-weights``  -> ``outputs/export/spaghetti1_weights_v1.zip``
* ``python -m spaghetti1 fetch-weights --source <Drive link | zip path | folder>``
* ``python -m spaghetti1 export-results``  -> lightweight copy of every run in ``results/`` (tracked in git)

Layout of the weights package::

    spaghetti1_weights_v1/
    ├── weights_info.json        # version, per-run model/size/fold/scores, SHA-256 of every best.pt
    ├── WEIGHTS_CARD.md          # what each model is and how to use it
    └── runs/<run_name>/
        ├── best.pt              # model weights (state_dict) of the best validation epoch
        ├── config.json          # every training setting (model name, image size, fold, ...)
        ├── metrics.json, history.csv, val_predictions.csv
        └── test_probabilities.npy   # only for runs that predicted the test set during training

``last.pt`` (optimizer state for resuming training) is left out on purpose: it is three times
larger than ``best.pt`` and is not needed to evaluate or predict.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

from .fetch import obtain_package

WEIGHTS_NAME = "spaghetti1_weights"
WEIGHTS_VERSION = "v1"
WEIGHTS_INFO_FILE = "weights_info.json"
RUN_FILES = ("best.pt", "config.env", "config.json", "metrics.json", "history.csv", "val_predictions.csv",
             "val_probabilities.npy", "val_probabilities_tta.npy", "test_probabilities.npy", "submission.csv")
RESULT_FILES = ("config.env", "config.json", "metrics.json", "history.csv", "val_predictions.csv",
                "confusion_matrix.png", "confusion_matrix_tta.png")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_summary(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    validation = metrics.get("validation") or {}
    validation_tta = metrics.get("validation_tta") or {}
    return {
        "model": config["model"], "image_size": config["image_size"], "fold": config["fold"],
        "label_policy": config["label_policy"], "epochs": config["epochs"], "best_epoch": metrics.get("best_epoch"),
        "validation_weighted_f1": round(validation.get("weighted_f1", float("nan")), 4),
        "validation_weighted_f1_tta": round(validation_tta.get("weighted_f1", float("nan")), 4),
        "best_pt_sha256": _file_sha256(run_dir / "best.pt"),
        "best_pt_megabytes": round((run_dir / "best.pt").stat().st_size / 2**20, 1),
    }


def export_weights(run_dirs: list[Path], output_dir: Path, *, version: str = WEIGHTS_VERSION,
                   ensembles: dict | None = None) -> Path:
    """Write ``<output_dir>/spaghetti1_weights_<version>.zip`` with the given runs; return its path."""
    root = f"{WEIGHTS_NAME}_{version}"
    runs = {Path(run).name: _run_summary(Path(run)) for run in run_dirs}
    info = {
        "name": WEIGHTS_NAME, "version": version,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs, "ensembles": ensembles or {},
        "note": "best.pt files are PyTorch state_dicts for timm models; see WEIGHTS_CARD.md.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root}.zip"
    partial = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        bundle.writestr(f"{root}/{WEIGHTS_INFO_FILE}", json.dumps(info, indent=2) + "\n")
        bundle.write(Path(__file__).with_name("WEIGHTS_CARD.md"), f"{root}/WEIGHTS_CARD.md")
        for run in map(Path, run_dirs):
            for name in RUN_FILES:
                if (run / name).is_file():
                    bundle.write(run / name, f"{root}/runs/{run.name}/{name}")
            print(f"  packed {run.name}", flush=True)
    shutil.move(partial, archive)
    return archive


def verify_weights(root: Path) -> dict:
    """Check that every listed ``best.pt`` exists and matches its SHA-256; return the info."""
    info = json.loads((root / WEIGHTS_INFO_FILE).read_text(encoding="utf-8"))
    for run, summary in info["runs"].items():
        weights = root / "runs" / run / "best.pt"
        if not weights.is_file():
            raise RuntimeError(f"{root}: missing runs/{run}/best.pt")
        if _file_sha256(weights) != summary["best_pt_sha256"]:
            raise RuntimeError(f"{root}: checksum mismatch for runs/{run}/best.pt (corrupted download?)")
    return info


def default_weights_dir() -> Path:
    """``/content/spaghetti1_weights`` on Colab, else ``data/weights`` in the project (ignored by git)."""
    colab = Path("/content")
    if colab.is_dir():
        return colab / "spaghetti1_weights"
    from ..config import PROJECT_DIR

    return PROJECT_DIR / "data" / "weights"


def prepare_weights(source: str | Path, target_dir: Path | None = None, *, keep_zip: bool = False) -> Path:
    """Make the weights package available locally (Drive link, zip, or folder); return its root.

    The run folders are ``<root>/runs/<run_name>``; pass them to ``evaluate`` and ``predict``.
    """
    root = obtain_package(source, Path(target_dir) if target_dir else default_weights_dir(),
                          info_file=WEIGHTS_INFO_FILE, verify=verify_weights,
                          zip_name="spaghetti1_weights.zip", keep_zip=keep_zip)
    info = json.loads((root / WEIGHTS_INFO_FILE).read_text(encoding="utf-8"))
    print(f"Ready: {info['name']} {info['version']} at {root} ({len(info['runs'])} models, checksums verified)")
    for run, summary in info["runs"].items():
        print(f"  {run}: {summary['model']} {summary['image_size']} px, "
              f"validation wF1 {summary['validation_weighted_f1']} (TTA {summary['validation_weighted_f1_tta']})")
    return root


def export_results(runs_dir: Path, results_dir: Path, submissions_dir: Path | None = None) -> Path:
    """Copy the lightweight outputs of every run (no weights) into ``results_dir`` for git."""
    results_dir.mkdir(parents=True, exist_ok=True)
    if (runs_dir / "results.csv").is_file():
        shutil.copy2(runs_dir / "results.csv", results_dir / "results.csv")
    for run in sorted(path for path in runs_dir.iterdir() if (path / "config.json").is_file()):
        target = results_dir / "runs" / run.name
        target.mkdir(parents=True, exist_ok=True)
        for name in RESULT_FILES:
            if (run / name).is_file():
                shutil.copy2(run / name, target / name)
        if (run / "submission.csv").is_file():
            submission_dir = results_dir / "submissions" / run.name
            submission_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / "submission.csv", submission_dir / "submission.csv")
    if submissions_dir and submissions_dir.is_dir():
        for folder in sorted(path for path in submissions_dir.iterdir() if (path / "submission.csv").is_file()):
            target = results_dir / "submissions" / folder.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("submission.csv", "ensemble_info.json"):
                if (folder / name).is_file():
                    shutil.copy2(folder / name, target / name)
    return results_dir
