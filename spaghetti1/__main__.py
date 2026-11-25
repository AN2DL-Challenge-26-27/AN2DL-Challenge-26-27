"""Command-line interface: ``python -m spaghetti1 <command>``.

Commands in pipeline order (documentation in ``docs/00_START_HERE.md``):

1. ``download``        get the official data from Kaggle into ``data/raw/``
2. ``verify``          check the CSVs against the PNG files
3. ``eda``             exploratory statistics and plots in ``outputs/eda/``
4. ``preprocess``      flag artifacts, group duplicates, write ``manifests/`` and folds
5. ``augment-preview`` save a grid of augmented training images
6. ``export``          pack the cleaned dataset into one zip for Google Drive
7. ``fetch``           download or copy that zip and extract it on any machine
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DATA_DIR, NUM_FOLDS, OUTPUT_DIR, PREPROCESS_DIR, PROJECT_DIR
from .ingestion.dataset import load_manifest, validate_layout


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
