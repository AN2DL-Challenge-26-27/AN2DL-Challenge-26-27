"""Train a list of experiment configs on all available GPUs, then summarise and export the results.

Made for Kaggle notebooks (2 x T4) but works on any machine. Documentation: docs/11_KAGGLE_TRAINING.md.

    python experiments/kaggle_queue.py --configs configs/experiments/r2_dinov2_base_336_recipe_fold1.env ... \\
        --data-dir /kaggle/input/an2dl2627 --work-dir /kaggle/working

What it does:

1. Spreads the configs over the GPUs (round-robin). Each GPU runs its configs one after the
   other as separate ``python -m spaghetti1 train --config ...`` processes, with
   ``--predict-test`` so every run also saves test probabilities (without TTA unless
   ``--test-tta`` is given). Logs go to ``<work-dir>/runs/logs/<run>.log``.
2. Summarises fold groups: runs whose names differ only in ``_fold<k>`` form one group. For
   each group it reports the per-fold and out-of-fold (OOF) weighted F1 and writes a
   fold-ensemble submission (average of the folds' test probabilities).
3. Exports two zip files into ``<work-dir>``:
   * ``spaghetti1_weights_<tag>.zip``: weights package (``fetch-weights`` reads it locally);
   * ``spaghetti1_results_<tag>.zip``: everything except the weights (small): results.csv,
     logs, configs, metrics, curves, validation/test probabilities, submissions, summary.

A run that crashes does not stop the others; it is listed as failed in the summary. Configs
use ``RESUME=true`` (research recipe), so re-running the same command continues unfinished runs.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find_data_dir(start: Path = Path("/kaggle/input")) -> Path:
    """The folder that holds ``train.csv`` and ``train/`` (the competition data on Kaggle)."""
    for csv in sorted(start.rglob("train.csv")):
        if (csv.parent / "train").is_dir() and (csv.parent / "test").is_dir():
            return csv.parent
    raise FileNotFoundError(f"No folder with train.csv, train/ and test/ under {start}; add the competition data")


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def run_name_of(config: Path, extra: list[str]) -> str:
    if "--run-name" in extra:
        return extra[extra.index("--run-name") + 1]
    from spaghetti1.training.config import build_config

    return build_config(config)[0].run_name


def worker(gpu: int, configs: list[Path], args, status: dict, lock: threading.Lock) -> None:
    for config in configs:
        name = run_name_of(config, args.extra)
        log_path = args.runs_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "spaghetti1", "train", "--config", str(config),
                   "--data-dir", str(args.data_dir), "--predict-test",
                   "--test-tta" if args.test_tta else "--no-test-tta", *args.extra]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "SPAGHETTI_RUNS_DIR": str(args.runs_dir),
               "SPAGHETTI_NUM_WORKERS": str(args.num_workers), "PYTHONWARNINGS": "ignore"}
        with lock:
            status[name] = f"running on GPU {gpu}"
        started = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - started) / 60
        with lock:
            status[name] = f"{'finished' if code == 0 else f'FAILED (exit {code})'} on GPU {gpu} in {minutes:.0f} min"
            print(f"[queue] {name}: {status[name]}", flush=True)


def last_epoch_line(runs_dir: Path, name: str) -> str:
    log = runs_dir / "logs" / f"{name}.log"
    if not log.is_file():
        return ""
    lines = [line.strip() for line in log.read_text(encoding="utf-8", errors="ignore").splitlines() if "epoch" in line]
    return lines[-1].split("] ", 1)[-1] if lines else ""


def summarise_folds(runs_dir: Path, data_dir: Path, out_dir: Path) -> list[dict]:
    """Per-fold and OOF weighted F1 per fold group; fold-ensemble submissions."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score

    from spaghetti1.preprocessing.dataset import load_test_manifest, load_train_manifest, split_fold
    from spaghetti1.preprocessing.loaders import manifest_dir_for

    manifests = manifest_dir_for(data_dir)
    train = load_train_manifest(manifests)
    test_ids = load_test_manifest(manifests)["id"]
    groups: dict[str, dict[int, Path]] = {}
    for run in sorted(p for p in runs_dir.iterdir() if (p / "config.json").is_file()):
        fold = json.loads((run / "config.json").read_text())["fold"]
        groups.setdefault(re.sub(r"_fold\d+$", "", run.name), {})[fold] = run

    summary = []
    for group, runs in groups.items():
        labels, predictions, per_fold, tests = [], [], {}, []
        for fold, run in sorted(runs.items()):
            if fold < 0 or not (run / "val_probabilities.npy").is_file():
                continue
            y = split_fold(train, fold)[1]["label"].to_numpy()
            p = np.load(run / "val_probabilities.npy")
            if len(p) != len(y):  # e.g. a LIMIT_BATCHES smoke run
                print(f"[queue] {run.name}: {len(p)} validation predictions for {len(y)} images, skipped", flush=True)
                continue
            per_fold[fold] = round(f1_score(y, p.argmax(1), average="weighted"), 4)
            labels.append(y)
            predictions.append(p)
            test_file = run / "test_probabilities.npy"
            if test_file.is_file() and len(test_probs := np.load(test_file)) == len(test_ids):
                tests.append(test_probs)
        if not per_fold:
            continue
        oof = f1_score(np.concatenate(labels), np.concatenate(predictions).argmax(1), average="weighted")
        entry = {"group": group, "folds": sorted(per_fold), "per_fold_weighted_f1": per_fold,
                 "oof_weighted_f1": round(oof, 4), "oof_images": int(sum(len(y) for y in labels))}
        if len(tests) == len(per_fold) and len(tests) > 1:
            ensemble = np.mean(tests, axis=0)
            folder = out_dir / f"fold_ensemble_{group}"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "test_probabilities.npy", ensemble.astype(np.float32))
            pd.DataFrame({"id": test_ids, "label": ensemble.argmax(1)}).to_csv(folder / "submission.csv", index=False)
            entry["fold_ensemble_submission"] = str(folder / "submission.csv")
        summary.append(entry)
    return summary


def export(runs_dir: Path, work_dir: Path, finished: list[Path], tag: str, summary_dir: Path) -> list[Path]:
    from spaghetti1.sharing.weights import export_weights

    outputs = []
    if finished:
        outputs.append(export_weights(finished, work_dir, version=tag))
    results_zip = work_dir / f"spaghetti1_results_{tag}.zip"
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(runs_dir.rglob("*")):
            if path.is_file() and path.suffix not in (".pt", ".tmp"):
                bundle.write(path, f"runs/{path.relative_to(runs_dir).as_posix()}")
        for path in sorted(summary_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, f"summary/{path.relative_to(summary_dir).as_posix()}")
    outputs.append(results_zip)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: found under /kaggle/input")
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--gpus", type=str, default=None, help="comma-separated GPU ids (default: all)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per run (Kaggle: 4 CPUs)")
    parser.add_argument("--test-tta", action="store_true", help="use 8-way TTA for test predictions (slower)")
    parser.add_argument("--tag", default=None, help="name used in the zip files (default: kaggle_<date_time>)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags for every train call")
    args = parser.parse_args()

    args.data_dir = (args.data_dir or find_data_dir()).resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir = (args.work_dir / "runs").resolve()
    configs = [(ROOT / c if not c.is_absolute() else c).resolve() for c in args.configs]
    missing = [str(c) for c in configs if not c.is_file()]
    if missing:
        parser.error(f"config files not found: {missing}")
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(max(gpu_count(), 1)))
    tag = args.tag or f"kaggle_{datetime.now():%Y%m%d_%H%M}"
    plan = {gpu: configs[index::len(gpus)] for index, gpu in enumerate(gpus)}
    print(f"[queue] data {args.data_dir} | outputs {args.work_dir} | GPUs {gpus}", flush=True)
    for gpu, items in plan.items():
        print(f"[queue] GPU {gpu}: {', '.join(c.stem for c in items)}", flush=True)

    status: dict[str, str] = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, items, args, status, lock), daemon=True)
               for gpu, items in plan.items() if items]
    started = time.time()
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(60)
        with lock:
            running = {name: state for name, state in status.items() if state.startswith("running")}
        elapsed = (time.time() - started) / 60
        for name, state in running.items():
            print(f"[queue] {elapsed:4.0f} min | {name} ({state}) | {last_epoch_line(args.runs_dir, name)}", flush=True)

    summary_dir = args.work_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    finished = [args.runs_dir / name for name, state in status.items()
                if state.startswith("finished") and (args.runs_dir / name / "best.pt").is_file()]
    folds = summarise_folds(args.runs_dir, args.data_dir, summary_dir)
    report = {"tag": tag, "minutes": round((time.time() - started) / 60, 1), "runs": status, "fold_groups": folds}
    (summary_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if (args.runs_dir / "results.csv").is_file():
        (summary_dir / "results.csv").write_bytes((args.runs_dir / "results.csv").read_bytes())
    print(json.dumps(report, indent=2), flush=True)
    for path in export(args.runs_dir, args.work_dir, finished, tag, summary_dir):
        print(f"[queue] export: {path} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
