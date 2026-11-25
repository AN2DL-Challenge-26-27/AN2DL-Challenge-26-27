import json
from pathlib import Path

import pandas as pd

from ..config import CLASS_NAMES
from ..ingestion.dataset import load_manifest, validate_layout
from .image_stats import METRICS, scan_images
from .plots import plot_distribution, plot_domain_shift, plot_grid


def class_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = manifest["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    table = counts.rename_axis("label").rename("count").reset_index()
    table["class_name"] = table["label"].map(CLASS_NAMES)
    table["percentage"] = 100 * table["count"] / counts.sum()
    return table


def save_sample_grids(data_dir: Path, valid: pd.DataFrame, output_dir: Path, seed: int) -> None:
    train = valid.loc[valid["split"] == "train"]
    test = valid.loc[valid["split"] == "test"]
    class_examples = []
    for label in CLASS_NAMES:
        group = train.loc[train["label"] == label]
        class_examples.append(group.sample(n=min(5, len(group)), random_state=seed))
    plot_grid(data_dir, class_examples, 5, "Training examples by class", output_dir / "class_samples.png")

    sample = test.sample(n=min(12, len(test)), random_state=seed)
    plot_grid(data_dir, [sample.iloc[:6], sample.iloc[6:]], 6, "Test examples", output_dir / "test_samples.png")

    outliers = [frame.nlargest(6, "hf_energy") for frame in (train, test)]
    plot_grid(data_dir, outliers, 6, "Highest high-frequency energy", output_dir / "outliers.png")


def run_eda(data_dir: Path, output_dir: Path, workers: int, seed: int, check_counts: bool) -> dict:
    manifest = load_manifest(data_dir)
    layout = validate_layout(data_dir, manifest, check_counts=check_counts)
    print(f"Scanning {len(manifest):,} images with {workers} workers...", flush=True)
    stats = scan_images(data_dir, manifest, workers)
    invalid = stats.loc[stats["error"].notna()]
    valid = stats.loc[stats["error"].isna()].copy()
    valid[list(METRICS)] = valid[list(METRICS)].astype(float)

    duplicates = valid.loc[valid["sha256"].duplicated(keep=False)].copy()
    groups = duplicates.groupby("sha256")
    duplicates["cross_split"] = groups["split"].transform("nunique").gt(1)
    duplicates["conflicting_train_labels"] = groups["label"].transform("nunique").gt(1)
    low_contrast = valid.loc[valid["std"].lt(2.0)]
    domain = valid.groupby("split")[list(METRICS)].agg(["count", "mean", "std", "min", "max"])
    domain.columns = [f"{metric}_{statistic}" for metric, statistic in domain.columns]
    counts = class_distribution(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "class_distribution": counts,
        "image_stats": stats,
        "invalid_images": invalid,
        "exact_duplicates": duplicates.sort_values(["sha256", "split", "id"]),
        "low_contrast": low_contrast,
        "domain_shift": domain.reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    plot_distribution(counts, output_dir / "class_distribution.png")
    plot_domain_shift(valid, output_dir / "domain_shift.png")
    save_sample_grids(data_dir, valid, output_dir, seed)

    positive_counts = counts.loc[counts["count"] > 0, "count"]
    summary = {
        "data_dir": str(data_dir.resolve()),
        "layout": layout,
        "invalid_images": len(invalid),
        "low_contrast_images": len(low_contrast),
        "exact_duplicate_groups": int(duplicates["sha256"].nunique()),
        "cross_split_duplicate_groups": int(duplicates.loc[duplicates["cross_split"], "sha256"].nunique()),
        "conflicting_train_label_groups": int(
            duplicates.loc[duplicates["conflicting_train_labels"], "sha256"].nunique()
        ),
        "largest_to_smallest_present_class_ratio": float(positive_counts.max() / positive_counts.min()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
