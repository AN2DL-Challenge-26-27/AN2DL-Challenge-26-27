"""Command-line interface: ``python -m spaghetti1 <command>``.

Commands in pipeline order (documentation in ``docs/00_START_HERE.md``):

1. ``download``        get the official data from Kaggle into ``data/raw/``
2. ``verify``          check the CSVs against the PNG files
3. ``eda``             exploratory statistics and plots in ``outputs/eda/``
4. ``preprocess``      flag artifacts, group duplicates, write ``manifests/`` and folds
5. ``augment-preview`` save a grid of augmented training images
6. ``export``          pack the cleaned dataset into one zip for Google Drive
7. ``fetch``           download or copy that zip and extract it on any machine
8. ``train``           train and evaluate one model from a config file; results in ``outputs/runs/``
   ``list-models``     show the model presets that ``MODEL=`` accepts
9. ``evaluate``        re-score a saved model on its validation fold (no retraining)
10. ``predict``        test predictions + submission.csv from one model or an ensemble
11. ``export-weights`` pack trained models into one zip for Google Drive
12. ``fetch-weights``  download or copy that zip, extract it, and verify checksums
13. ``export-results`` copy configs, metrics, curves, and submissions of every run to ``results/``
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DATA_DIR, NUM_FOLDS, OUTPUT_DIR, PREPROCESS_DIR, PROJECT_DIR
from .ingestion.dataset import load_manifest, validate_layout


# Validated ensembles of the fold-0 runs (validation weighted F1 with 8-way TTA), recorded in the
# weights package so everyone knows which combination to use.
KNOWN_ENSEMBLES = {
    # Validated combinations of the fold-0 runs, recorded in the weights package. Members use the same
    # TTA choice as the submissions: ConvNeXt with 8-way TTA, DINOv2/DINOv3 without TTA.
    "best_kaggle": {
        "runs": ["e1_convnext_tiny_drop_copies_fold0", "e2_convnext_tiny_384_fold0", "e3_convnext_small_288_fold0",
                 "r2_dinov2_base_336_recipe_fold0", "n2_dinov3_base_384_recipe_fold0"],
        "validation_weighted_f1": 0.8547, "test_like_validation_weighted_f1": 0.8609, "kaggle_public": 0.8441,
        "submission": "lb1_d_5models_convnext3_dinov2_dinov3",
    },
    "tiny384_small288_dinov2": {
        "runs": ["e2_convnext_tiny_384_fold0", "e3_convnext_small_288_fold0", "r2_dinov2_base_336_recipe_fold0"],
        "validation_weighted_f1": 0.8596, "test_like_validation_weighted_f1": 0.8597, "kaggle_public": 0.8424,
        "submission": "lb1_c_tiny384_small288_dinov2",
    },
    "convnext3": {
        "runs": ["e1_convnext_tiny_drop_copies_fold0", "e2_convnext_tiny_384_fold0", "e3_convnext_small_288_fold0"],
        "validation_weighted_f1": 0.8561, "kaggle_public": 0.8371,
        "submission": "ensemble_tiny224_tiny384_small288",
    },
}


def main() -> int:
    load_dotenv(PROJECT_DIR / ".env", encoding="utf-8-sig", interpolate=False)
    data_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_DATA_DIR", str(DATA_DIR)))
    output_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_OUTPUT_DIR", str(OUTPUT_DIR)))
    preprocess_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_PREPROCESS_DIR", str(PREPROCESS_DIR)))
    manifest_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_MANIFEST_DIR", "manifests"))

    parser = argparse.ArgumentParser(description="SPAGHETTI-1 dataset preparation and EDA")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download the competition dataset from Kaggle")
    download.add_argument("--data-dir", type=Path, default=data_dir)

    verify = commands.add_parser("verify", help="validate CSVs and match every ID to a PNG")
    eda = commands.add_parser("eda", help="scan all images and save tables and plots")
    for command in (verify, eda):
        command.add_argument("--data-dir", type=Path, default=data_dir)
        command.add_argument("--allow-subset", action="store_true", help="skip official counts for local fixtures")
    eda.add_argument("--output-dir", type=Path, default=output_dir)
    eda.add_argument("--workers", type=int, default=os.getenv("SPAGHETTI_EDA_WORKERS", "8"))
    eda.add_argument("--seed", type=int, default=os.getenv("SPAGHETTI_EDA_SEED", "42"))

    preprocess = commands.add_parser("preprocess", help="flag artifacts, group duplicates, write manifests and folds")
    preprocess.add_argument("--data-dir", type=Path, default=data_dir)
    preprocess.add_argument("--manifest-dir", type=Path, default=manifest_dir)
    preprocess.add_argument("--review-dir", type=Path, default=preprocess_dir)
    preprocess.add_argument("--workers", type=int, default=os.getenv("SPAGHETTI_EDA_WORKERS", "8"))
    preprocess.add_argument("--seed", type=int, default=os.getenv("SPAGHETTI_EDA_SEED", "42"))
    preprocess.add_argument("--folds", type=int, default=NUM_FOLDS)
    preprocess.add_argument("--allow-subset", action="store_true", help="skip official counts for local fixtures")

    preview = commands.add_parser("augment-preview", help="save a grid of augmented training images")
    preview.add_argument("--data-dir", type=Path, default=data_dir)
    preview.add_argument("--manifest-dir", type=Path, default=manifest_dir)
    preview.add_argument("--output", type=Path, default=preprocess_dir / "augmentation_preview.png")
    preview.add_argument("--images", type=int, default=8)
    preview.add_argument("--variants", type=int, default=7)
    preview.add_argument("--seed", type=int, default=os.getenv("SPAGHETTI_EDA_SEED", "42"))
    preview.add_argument("--denoise", choices=("none", "bilateral", "nlm"), default="none")
    preview.add_argument("--background-normalize", action="store_true")

    export = commands.add_parser("export", help="pack kept images and manifests into one zip for Google Drive")
    export.add_argument("--data-dir", type=Path, default=data_dir)
    export.add_argument("--manifest-dir", type=Path, default=manifest_dir)
    export.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "export")
    export.add_argument("--version", default="v1", help="package version, e.g. v2 after the manifests change")

    fetch = commands.add_parser("fetch", help="download or copy the cleaned package and extract it locally")
    fetch.add_argument("--source", required=True, help="Google Drive link/file id, zip path, or extracted folder")
    fetch.add_argument("--target-dir", type=Path, default=None)

    from dataclasses import fields

    from .training.config import TrainConfig

    train = commands.add_parser(
        "train", help="train one model; settings from --config <file.env> and/or flags (see configs/README.md)")
    train.add_argument("--data-dir", type=Path, default=data_dir, help="raw data folder or extracted Drive package")
    train.add_argument("--config", type=Path, default=None,
                       help="experiment config file, e.g. configs/experiments/r2_dinov2_base_recipe_fold0.env")
    train.add_argument("--dry-run", action="store_true", help="print the resolved settings and their sources, then stop")
    # Flags override the config file only when given (default=SUPPRESS keeps them out of args).
    for option in fields(TrainConfig):
        if option.name == "preset":
            continue
        flag = "--" + option.name.replace("_", "-")
        if option.type is bool:
            train.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            train.add_argument(flag, type=option.type, default=argparse.SUPPRESS)
    commands.add_parser("list-models", help="show the model presets accepted by MODEL= / --model")

    runs_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_RUNS_DIR", "outputs/runs"))
    best_runs = [runs_dir / name for name in (  # every reported fold-0 run (docs/08, docs/10)
        "e0_convnext_tiny_original_fold0", "e1_convnext_tiny_drop_copies_fold0", "e1_convnext_tiny_as_given_fold0",
        "e2_convnext_tiny_288_fold0", "e2_convnext_tiny_384_fold0", "e3_convnext_small_288_fold0",
        "r1_convnext_tiny_224_recipe_fold0", "r2_dinov2_base_336_recipe_fold0",
        "n1_dinov3_convnext_tiny_288_fold0", "n2_dinov3_base_384_recipe_fold0")]

    evaluate = commands.add_parser("evaluate", help="re-score a saved model on its validation fold")
    evaluate.add_argument("--run-dir", type=Path, required=True, help="folder with best.pt and config.json")
    predict = commands.add_parser("predict", help="predict the test set with one model or an ensemble")
    predict.add_argument("--run-dirs", type=Path, nargs="+", required=True, help="one or more run folders")
    predict.add_argument("--output-dir", type=Path, required=True)
    for command in (evaluate, predict):
        command.add_argument("--data-dir", type=Path, default=data_dir, help="raw data folder or extracted Drive package")
        command.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--num-workers", type=int, default=4)

    export_weights = commands.add_parser("export-weights", help="pack trained models into a zip for Google Drive")
    export_weights.add_argument("--runs", type=Path, nargs="+", default=best_runs)
    export_weights.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "export")
    export_weights.add_argument("--version", default="v2")
    fetch_weights = commands.add_parser("fetch-weights", help="download or copy the weights zip and verify it")
    fetch_weights.add_argument("--source", required=True, help="Google Drive link/file id, zip path, or extracted folder")
    fetch_weights.add_argument("--target-dir", type=Path, default=None)
    export_results = commands.add_parser("export-results", help="copy lightweight run outputs to results/ for git")
    export_results.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()

    try:
        if args.command == "download":
            from .ingestion.download import download_dataset

            path = download_dataset(args.data_dir)
            print(f"Dataset ready: {path}")
            return 0

        if args.command == "preprocess":
            if args.workers < 1 or args.seed < 0 or args.folds < 2:
                parser.error("--workers must be positive, --seed nonnegative, and --folds at least 2")
            from .preprocessing.manifest import build_manifests

            summary = build_manifests(
                args.data_dir, args.manifest_dir, args.review_dir, workers=args.workers, seed=args.seed,
                n_folds=args.folds, check_counts=not args.allow_subset,
            )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "augment-preview":
            from .preprocessing.preview import save_augmentation_preview

            path = save_augmentation_preview(
                args.data_dir, args.manifest_dir, args.output, images=args.images, variants=args.variants,
                seed=args.seed, denoise=args.denoise, background_normalize=args.background_normalize,
            )
            print(f"Saved {path}")
            return 0

        if args.command == "export":
            from .sharing.export import export_package

            print(f"Saved {export_package(args.data_dir, args.manifest_dir, args.output_dir, args.version)}")
            return 0

        if args.command == "fetch":
            from .sharing.fetch import prepare_dataset

            print(f"Dataset folder: {prepare_dataset(args.source, args.target_dir)}")
            return 0

        if args.command == "evaluate":
            from .training.predict import evaluate_run

            evaluate_run(args.run_dir, args.data_dir, tta=args.tta, num_workers=args.num_workers)
            return 0

        if args.command == "predict":
            from .training.predict import predict_test

            predict_test(args.run_dirs, args.data_dir, args.output_dir, tta=args.tta, num_workers=args.num_workers)
            return 0

        if args.command == "export-weights":
            from .sharing.weights import export_weights as pack_weights

            print(f"Saved {pack_weights(args.runs, args.output_dir, version=args.version, ensembles=KNOWN_ENSEMBLES)}")
            return 0

        if args.command == "fetch-weights":
            from .sharing.weights import prepare_weights

            print(f"Weights folder: {prepare_weights(args.source, args.target_dir)}")
            return 0

        if args.command == "export-results":
            from .sharing.weights import export_results as snapshot

            print(f"Saved {snapshot(runs_dir, args.results_dir, PROJECT_DIR / 'outputs' / 'submissions')}")
            return 0

        if args.command == "list-models":
            from .training.models import describe_presets

            print(describe_presets())
            print("\nAny other timm model name also works (MODEL=<timm name>), without preset defaults.")
            return 0

        if args.command == "train":
            from .training.config import build_config, format_config

            overrides = {option.name: getattr(args, option.name) for option in fields(TrainConfig)
                         if hasattr(args, option.name)}
            config, sources = build_config(args.config, overrides)
            if args.dry_run:
                print(format_config(config, sources), end="")
                return 0
            from .training.run import run_experiment

            run_experiment(config, args.data_dir, sources)
            return 0

        if args.command == "verify":
            manifest = load_manifest(args.data_dir)
            summary = validate_layout(args.data_dir, manifest, check_counts=not args.allow_subset)
        else:
            if args.workers < 1 or args.seed < 0:
                parser.error("--workers must be positive and --seed must be nonnegative")
            from .eda.report import run_eda

            summary = run_eda(
                args.data_dir, args.output_dir, args.workers, args.seed, check_counts=not args.allow_subset
            )
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
