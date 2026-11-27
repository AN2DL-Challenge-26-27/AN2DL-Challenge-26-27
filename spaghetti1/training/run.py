"""Run one training experiment end to end and save everything needed to compare it.

Command: ``python -m spaghetti1 train --config configs/experiments/<name>.env [--flag value ...]``.
Documentation: ``docs/08_TRAINING_AND_RESULTS.md`` and ``configs/README.md``. The settings
themselves (``TrainConfig``) live in ``config.py``; the model registry in ``models.py``.

Every run writes ``<runs dir>/<run_name>/`` (``outputs/runs`` unless ``SPAGHETTI_RUNS_DIR`` is set):

* ``config.env``           the fully resolved settings, re-runnable with ``--config``
* ``config.json``          the same settings plus the source of every value
* ``history.csv``          per epoch: learning rate, train loss/accuracy, validation loss and F1
* ``metrics.json``         validation metrics of the best epoch, without and with 8-way TTA
* ``confusion_matrix.png`` and ``confusion_matrix_tta.png``
* ``val_predictions.csv``  per validation image: true label and class probabilities (for ensembling)
* ``best.pt``              weights of the epoch with the highest validation weighted F1
* ``last.pt``              full training state after the latest epoch (model, optimizer, step,
                           history, RNG); ``--resume`` continues from it after a crash or stop
* ``test_probabilities.npy`` and ``submission.csv`` when ``predict_test`` is on

and appends one summary row to ``outputs/runs/results.csv``.
"""

from dataclasses import asdict, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from timm.data import Mixup
from timm.utils import ModelEmaV3
from torch import nn

from ..config import NUM_CLASSES, PROJECT_DIR
from ..preprocessing.dataset import class_weights
from ..preprocessing.loaders import build_dataloaders
from ..preprocessing.transforms import AugmentConfig, PreprocessConfig
from .config import TrainConfig, format_config
from .engine import build_optimizer, predict, train_one_epoch
from .models import create_model, resolve_model
from .metrics import compute_metrics, plot_confusion

__all__ = ["TrainConfig", "run_experiment", "runs_dir"]


def runs_dir() -> Path:
    """Where runs are written: ``SPAGHETTI_RUNS_DIR`` from the project ``.env``, else outputs/runs."""
    return PROJECT_DIR / Path(os.getenv("SPAGHETTI_RUNS_DIR", "outputs/runs"))


RUNS_DIR = runs_dir()  # kept for code written against the earlier module


class LimitedLoader:
    """A DataLoader view with at most ``limit`` batches (``LIMIT_BATCHES`` smoke tests)."""

    def __init__(self, loader, limit: int):
        self.loader, self.limit = loader, limit

    def __len__(self) -> int:
        return min(len(self.loader), self.limit)

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_training_state(path: Path, model, optimizer, epoch: int, step: int, history: list,
                        best_f1: float, best_epoch: int, ema=None) -> None:
    """Write everything needed to continue training; the temporary file makes the save atomic."""
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
        "ema": ema.module.state_dict() if ema is not None else None,
        "history": history, "best_f1": best_f1, "best_epoch": best_epoch,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_training_state(path: Path, model, optimizer, ema=None) -> tuple[int, int, list, float, int]:
    """Restore a ``last.pt``; return (next epoch, step, history, best F1, best epoch)."""
    state = torch.load(path, map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema") is not None:
        ema.module.load_state_dict(state["ema"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["epoch"] + 1, state["step"], state["history"], state["best_f1"], state["best_epoch"]


def _row(run: TrainConfig, metrics: dict, metrics_tta: dict | None, best_epoch: int, minutes: float) -> dict:
    row = {
        "run_name": run.run_name, "finished": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": run.model,
        "fold": run.fold, "label_policy": run.label_policy, "image_size": run.image_size, "epochs": run.epochs,
        "best_epoch": best_epoch, "weighted_f1": round(metrics["weighted_f1"], 4),
        "weighted_f1_tta": round(metrics_tta["weighted_f1"], 4) if metrics_tta else None,
        "macro_f1": round(metrics["macro_f1"], 4), "accuracy": round(metrics["accuracy"], 4),
    }
    best = metrics_tta or metrics
    row.update({f"f1_{c}": round(best["per_class"][c]["f1"], 3) for c in range(NUM_CLASSES)})
    row["minutes"] = round(minutes, 1)
    return row


def run_experiment(run: TrainConfig, data_dir: Path, sources: dict | None = None) -> dict:
    """Train on the fold's training part, select the best epoch on validation, evaluate, and save.

    ``sources`` (from ``config.build_config``) records where every setting came from.
    """
    started = time.time()
    timm_name, preset, _ = resolve_model(run.model)  # accept a preset name in ``model``
    run = replace(run, model=timm_name, preset=preset or run.preset)
    sources = sources or {name: "TrainConfig" for name in asdict(run)}
    seed_everything(run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    torch.backends.cudnn.benchmark = True

    output = runs_dir() / run.run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.env").write_text(format_config(run, sources), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({**asdict(run), "sources": sources}, indent=2) + "\n",
                                        encoding="utf-8")

    preprocess = PreprocessConfig(image_size=run.image_size, denoise=run.denoise,
                                  background_normalize=run.background_normalize)
    loaders = build_dataloaders(
        data_dir, fold=run.fold, batch_size=run.batch_size, image_size=run.image_size,
        label_policy=run.label_policy, balance_power=run.balance_power, preprocess=preprocess,
        augment=AugmentConfig() if run.augment else None, num_workers=run.num_workers, seed=run.seed,
    )
    if run.limit_batches > 0:  # smoke test: a few batches per epoch only
        loaders.train = LimitedLoader(loaders.train, run.limit_batches)
        loaders.validation = LimitedLoader(loaders.validation, run.limit_batches)
        loaders.validation_frame = loaders.validation_frame.head(run.limit_batches * run.batch_size)
        loaders.test = LimitedLoader(loaders.test, run.limit_batches)
        loaders.test_frame = loaders.test_frame.head(run.limit_batches * run.batch_size)
    print(f"[{run.run_name}] model {run.model}" + (f" (preset {run.preset})" if run.preset else "")
          + f" | {run.image_size} px | fold {run.fold}", flush=True)
    print(f"[{run.run_name}] train {len(loaders.train_frame):,} | validation {len(loaders.validation_frame):,} "
          f"| device {device} | amp {amp_dtype}", flush=True)

    model = create_model(run.model, run.drop_path_rate, image_size=run.image_size).to(device)
    if run.grad_checkpointing:
        model.set_grad_checkpointing(True)
    weights = class_weights(loaders.train_frame["label"], run.class_weight_power) if run.class_weight_power else None
    weights = None if weights is None else weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=run.label_smoothing)
    mixup = (Mixup(mixup_alpha=run.mixup_alpha, cutmix_alpha=0.0, prob=run.mixup_prob,
                   label_smoothing=run.label_smoothing, num_classes=NUM_CLASSES)
             if run.mixup_alpha > 0 else None)
    optimizer = build_optimizer(model, run.lr, run.head_lr_multiplier, run.weight_decay, run.layer_decay)
    ema = ModelEmaV3(model, decay=run.ema_decay, use_warmup=True) if run.ema_decay > 0 else None
    steps_per_epoch = math.ceil(len(loaders.train) / run.accumulation_steps)
    total_steps = run.epochs * steps_per_epoch
    warmup_steps = int(run.warmup_epochs * steps_per_epoch)

    history, best_f1, best_epoch, step, start_epoch = [], -1.0, 0, 0, 1
    last_state = output / "last.pt"
    if run.resume and last_state.exists():
        start_epoch, step, history, best_f1, best_epoch = load_training_state(last_state, model, optimizer, ema)
        print(f"[{run.run_name}] resumed after epoch {start_epoch - 1} (best wF1 {best_f1:.4f} at epoch {best_epoch})",
              flush=True)
    elif run.resume:
        print(f"[{run.run_name}] no last.pt found; starting from epoch 1", flush=True)
    has_validation = len(loaders.validation_frame) > 0
    for epoch in range(start_epoch, run.epochs + 1):
        epoch_start = time.time()
        mix_this_epoch = mixup if epoch <= run.epochs - run.mixup_off_epochs else None
        train_loss, train_acc, step = train_one_epoch(
            model, loaders.train, criterion, optimizer, device,
            step=step, total_steps=total_steps, warmup_steps=warmup_steps, amp_dtype=amp_dtype,
            mixup=mix_this_epoch, class_weight=weights, ema=ema, accumulation_steps=run.accumulation_steps,
        )
        evaluated = ema.module if ema is not None else model  # the weights that are scored and saved
        record = {"epoch": epoch, "lr": max(group["lr"] for group in optimizer.param_groups),
                  "train_loss": train_loss, "train_acc": train_acc, "mixup": mix_this_epoch is not None}
        if has_validation:
            probs, labels, val_loss = predict(evaluated, loaders.validation, device, amp_dtype=amp_dtype)
            scores = compute_metrics(labels, probs.argmax(1))
            record.update(val_loss=val_loss, val_weighted_f1=scores["weighted_f1"], val_macro_f1=scores["macro_f1"],
                          val_acc=scores["accuracy"])
            if scores["weighted_f1"] > best_f1:
                best_f1, best_epoch = scores["weighted_f1"], epoch
                torch.save(evaluated.state_dict(), output / "best.pt")
        record["seconds"] = round(time.time() - epoch_start, 1)
        history.append(record)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        save_training_state(last_state, model, optimizer, epoch, step, history, best_f1, best_epoch, ema)
        print(f"[{run.run_name}] epoch {epoch:2d}/{run.epochs} loss {train_loss:.3f} acc {train_acc:.3f}"
              + (f" | val loss {val_loss:.3f} wF1 {scores['weighted_f1']:.4f} acc {scores['accuracy']:.3f}"
                 if has_validation else "") + f" | {record['seconds']:.0f}s", flush=True)

    if not has_validation:  # fold = -1: final model trained on everything
        torch.save((ema.module if ema is not None else model).state_dict(), output / "best.pt")
        best_epoch = run.epochs
    model.load_state_dict(torch.load(output / "best.pt", map_location=device))

    summary = {"run_name": run.run_name, "best_epoch": best_epoch}
    metrics = metrics_tta = None
    if has_validation:
        probs, labels, _ = predict(model, loaders.validation, device, amp_dtype=amp_dtype)
        metrics = compute_metrics(labels, probs.argmax(1))
        plot_confusion(metrics["confusion_matrix"], f"{run.run_name} (epoch {best_epoch}) wF1 {metrics['weighted_f1']:.4f}",
                       output / "confusion_matrix.png")
        predictions = loaders.validation_frame[["id", "label", "original_label", "dup_group_size"]].copy()
        if run.tta:
            probs_tta, _, _ = predict(model, loaders.validation, device, tta=True, amp_dtype=amp_dtype)
            metrics_tta = compute_metrics(labels, probs_tta.argmax(1))
            plot_confusion(metrics_tta["confusion_matrix"], f"{run.run_name} + TTA wF1 {metrics_tta['weighted_f1']:.4f}",
                           output / "confusion_matrix_tta.png")
            probs = probs_tta
        predictions["predicted"] = probs.argmax(1)
        for c in range(NUM_CLASSES):
            predictions[f"p{c}"] = probs[:, c].round(5)
        predictions.to_csv(output / "val_predictions.csv", index=False)
        summary.update(validation=metrics, validation_tta=metrics_tta)

    if run.predict_test:
        test_probs, _, _ = predict(model, loaders.test, device, tta=run.tta, amp_dtype=amp_dtype)
        np.save(output / "test_probabilities.npy", test_probs.astype(np.float32))
        pd.DataFrame({"id": loaders.test_frame["id"], "label": test_probs.argmax(1)}).to_csv(output / "submission.csv", index=False)

    minutes = (time.time() - started) / 60
    summary["minutes"] = round(minutes, 1)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if metrics:
        results = runs_dir() / "results.csv"
        row = pd.DataFrame([_row(run, metrics, metrics_tta, best_epoch, minutes)])
        row.to_csv(results, mode="a", header=not results.exists(), index=False)
        report = metrics_tta or metrics
        print(f"[{run.run_name}] best epoch {best_epoch}: weighted F1 {metrics['weighted_f1']:.4f}"
              + (f" | with TTA {metrics_tta['weighted_f1']:.4f}" if metrics_tta else "")
              + f" | {minutes:.1f} min", flush=True)
        print("  per-class F1: " + ", ".join(f"{c}:{report['per_class'][c]['f1']:.2f}" for c in range(NUM_CLASSES)))
    return summary
