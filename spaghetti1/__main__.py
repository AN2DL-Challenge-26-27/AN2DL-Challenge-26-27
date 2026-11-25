import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DATA_DIR, OUTPUT_DIR, PROJECT_DIR
from .ingestion.dataset import load_manifest, validate_layout


def main() -> int:
    load_dotenv(PROJECT_DIR / ".env", encoding="utf-8-sig", interpolate=False)
    data_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_DATA_DIR", str(DATA_DIR)))
    output_dir = PROJECT_DIR / Path(os.getenv("SPAGHETTI_OUTPUT_DIR", str(OUTPUT_DIR)))

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
    args = parser.parse_args()

    try:
        if args.command == "download":
            from .ingestion.download import download_dataset

            path = download_dataset(args.data_dir)
            print(f"Dataset ready: {path}")
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
