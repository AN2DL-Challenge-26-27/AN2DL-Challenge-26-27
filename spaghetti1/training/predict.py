"""Reproduce results from saved weights, without retraining.

Commands (documentation: ``docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md``):

* ``python -m spaghetti1 evaluate --run-dir <run>``: re-score one model on its validation
  fold and compare with the score recorded when it was trained.
* ``python -m spaghetti1 predict --run-dirs <run> [<run> ...] --output-dir <dir>``: predict
  the 14,736 test images with one model or an ensemble (average of class probabilities) and
  write ``submission.csv``. The same models are also scored on the validation fold, so the
  exact submission comes with its own validation score.

A *run folder* is ``outputs/runs/<run_name>/`` after training, or ``runs/<run_name>/`` inside
the extracted weights package. It must contain ``best.pt`` and ``config.json``.

Per-model probabilities are cached in the output folder, so an interrupted ``predict`` call
continues where it stopped. The GPU is used when available; the CPU works but is slow
(about 8-26 images per second, times 8 with test-time augmentation).
"""

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import NUM_CLASSES
from ..preprocessing.dataset import GalaxyDataset, load_test_manifest, load_train_manifest, split_fold
from ..preprocessing.loaders import manifest_dir_for
from ..preprocessing.transforms import GalaxyTransform, PreprocessConfig
from .engine import create_model, predict
from .metrics import compute_metrics


def _device() -> tuple[torch.device, torch.dtype | None]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    print("No usable GPU found: running on the CPU, which is much slower.", flush=True)
    return torch.device("cpu"), None


def load_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the network described by ``config.json`` and load ``best.pt`` (no internet needed)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = create_model(config["model"], config.get("drop_path_rate", 0.1), pretrained=False,
                         image_size=config["image_size"])
    state = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval(), config


def _transform(config: dict) -> GalaxyTransform:
    """The exact evaluation preprocessing the model was trained with."""
    return GalaxyTransform(PreprocessConfig(
        image_size=config["image_size"], denoise=config.get("denoise", "none"),
        background_normalize=config.get("background_normalize", False),
    ), None)


def _loader(frame: pd.DataFrame, data_dir: Path, config: dict, num_workers: int) -> DataLoader:
    return DataLoader(GalaxyDataset(frame, data_dir, _transform(config)), batch_size=64, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def validation_frame(data_dir: Path, fold: int) -> pd.DataFrame:
    """Originals and unique images of ``fold``: the set every run was scored on."""
    _, validation = split_fold(load_train_manifest(manifest_dir_for(data_dir)), fold)
    return validation


def _cached_probabilities(path: Path, compute) -> np.ndarray:
    if path.exists():
        print(f"  reusing {path.name}", flush=True)
        return np.load(path)
    probabilities = compute()
    np.save(path, probabilities.astype(np.float32))
    return probabilities


def evaluate_run(run_dir: Path, data_dir: Path, *, tta: bool = True, num_workers: int = 4) -> dict:
    """Score a saved model on its validation fold; print it next to the recorded score."""
    device, amp = _device()
    model, config = load_run(run_dir, device)
    frame = validation_frame(data_dir, config["fold"])
    started = time.time()
    probabilities, labels, _ = predict(model, _loader(frame, data_dir, config, num_workers), device,
                                       tta=tta, amp_dtype=amp)
    metrics = compute_metrics(labels, probabilities.argmax(1))
    recorded_file = Path(run_dir) / "metrics.json"
    recorded = None
    if recorded_file.exists():
        saved = json.loads(recorded_file.read_text(encoding="utf-8"))
        block = saved.get("validation_tta" if tta else "validation") or {}
        recorded = block.get("weighted_f1")
    print(f"{config['run_name']} | fold {config['fold']} | {len(frame)} validation images | TTA {tta}")
    print(f"  weighted F1 now {metrics['weighted_f1']:.4f}"
          + (f" | recorded at training time {recorded:.4f}" if recorded is not None else "")
          + f" | {time.time() - started:.0f} s on {device.type}")
    return {"run_name": config["run_name"], "tta": tta, "weighted_f1": metrics["weighted_f1"],
            "recorded_weighted_f1": recorded, "metrics": metrics}


def predict_test(run_dirs: list[Path], data_dir: Path, output_dir: Path, *, tta: bool = True,
                 num_workers: int = 4) -> Path:
    """Average the test probabilities of ``run_dirs``, write ``submission.csv``, and return its path.

    Validation probabilities of the same models are averaged too, so ``ensemble_info.json``
    records the validation weighted F1 of exactly this ensemble (when the runs share a fold).
    """
    device, amp = _device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = load_test_manifest(manifest_dir_for(data_dir))
    suffix = "" if tta else "_notta"
    test_probs, val_probs, configs = [], [], []
    for run_dir in map(Path, run_dirs):
        model, config = load_run(run_dir, device)
        configs.append(config)
        name = config["run_name"]
        print(f"{name} ({config['model']}, {config['image_size']} px)", flush=True)
        started = time.time()
        val = validation_frame(data_dir, config["fold"])
        val_probs.append(_cached_probabilities(
            output_dir / f"val_probabilities_{name}{suffix}.npy",
            lambda: predict(model, _loader(val, data_dir, config, num_workers), device, tta=tta, amp_dtype=amp)[0]))
        test_probs.append(_cached_probabilities(
            output_dir / f"test_probabilities_{name}{suffix}.npy",
            lambda: predict(model, _loader(test, data_dir, config, num_workers), device, tta=tta, amp_dtype=amp)[0]))
        print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = np.mean(test_probs, axis=0)
    np.save(output_dir / "test_probabilities_ensemble.npy", ensemble.astype(np.float32))
    submission = pd.DataFrame({"id": test["id"], "label": ensemble.argmax(1).astype(int)})
    if len(submission) != len(test) or not submission["label"].between(0, NUM_CLASSES - 1).all():
        raise RuntimeError("Submission check failed: wrong row count or labels outside 0-9")
    submission.to_csv(output_dir / "submission.csv", index=False)

    info = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "tta": tta,
            "runs": [c["run_name"] for c in configs], "test_rows": len(submission),
            "predicted_class_counts": submission["label"].value_counts().sort_index().to_dict()}
    folds = {c["fold"] for c in configs}
    if len(folds) == 1 and -1 not in folds:
        labels = validation_frame(data_dir, folds.pop())["label"].to_numpy()
        info["validation_weighted_f1_per_model"] = {
            c["run_name"]: round(compute_metrics(labels, p.argmax(1))["weighted_f1"], 4) for c, p in zip(configs, val_probs)}
        info["validation_weighted_f1_ensemble"] = round(compute_metrics(labels, np.mean(val_probs, 0).argmax(1))["weighted_f1"], 4)
        print(f"Validation weighted F1 of this ensemble: {info['validation_weighted_f1_ensemble']:.4f}")
    (output_dir / "ensemble_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_dir / 'submission.csv'} ({len(submission):,} rows)")
    return output_dir / "submission.csv"
