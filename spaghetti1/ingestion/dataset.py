"""Read the official CSV files and check them against the PNG files on disk.

Used by ``python -m spaghetti1 verify`` and by every later step. Documentation:
``docs/01_DATASET_AND_TASK.md``.
"""

from pathlib import Path

import pandas as pd

from ..config import CLASS_NAMES, EXPECTED_COUNTS, SPLIT_CSVS


def load_manifest(data_dir: Path) -> pd.DataFrame:
    """Return one row per image (``id``, ``label``, ``split``) from train.csv and sample_submission.csv.

    Raises ``ValueError`` for a wrong header, malformed or duplicate ids, or labels outside 0-9.
    Test labels are set to missing, because the sample submission only holds placeholders.
    """
    splits = []
    for split, csv_name in SPLIT_CSVS.items():
        frame = pd.read_csv(data_dir / csv_name, dtype={"id": "string"})
        if list(frame.columns) != ["id", "label"]:
            raise ValueError(f"{csv_name}: expected the header id,label")
        if not frame["id"].str.fullmatch(rf"{split}_[0-9]{{5}}").fillna(False).all():
            raise ValueError(f"{csv_name}: missing or malformed image IDs")
        if frame.empty or frame["id"].duplicated().any():
            raise ValueError(f"{csv_name}: empty manifest or duplicate image IDs")
        if split == "train" and not frame["label"].isin(CLASS_NAMES).all():
            raise ValueError(f"{csv_name}: expected integer class labels 0–9")
        if split == "test":
            frame["label"] = pd.NA
        frame["label"] = frame["label"].astype("Int64")
        splits.append(frame.assign(split=split))
    return pd.concat(splits, ignore_index=True)


def validate_layout(
    data_dir: Path, manifest: pd.DataFrame, *, check_counts: bool = True
) -> dict[str, dict[str, int]]:
    """Check that every CSV id has a PNG and every PNG has a CSV id; return counts per split."""
    summary = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        folder = data_dir / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image directory: {folder}")
        ids = set(manifest.loc[manifest["split"] == split, "id"])
        file_ids = {path.stem for path in folder.glob("*.png") if path.is_file()}
        if check_counts and len(ids) != expected_count:
            raise ValueError(f"{split}: expected {expected_count:,} CSV rows, found {len(ids):,}")
        missing, unexpected = sorted(ids - file_ids), sorted(file_ids - ids)
        if missing or unexpected:
            raise ValueError(
                f"{split}: {len(missing):,} missing PNGs {missing[:5]}, "
                f"{len(unexpected):,} unexpected PNGs {unexpected[:5]}"
            )
        summary[split] = {"rows": len(ids), "png_files": len(file_ids)}
    return summary
