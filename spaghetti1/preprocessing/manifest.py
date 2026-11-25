"""Build the cleaned training manifest, the test manifest, and group-aware CV folds.

Command: ``python -m spaghetti1 preprocess``. Documentation: ``docs/03_PREPROCESSING.md``.

What this module decides, and why (numbers measured on the full dataset):

1. **Alien overlays are removed from training** (332 images, 7.7%). They are spread evenly
   over the 10 classes and never occur in the test set, so they only hide galaxy features.
2. **Blank frames are removed from training** (1 image; 2 more in test still need a prediction).
3. **Near-duplicates are grouped.** The training set contains 721 sky cutouts that appear
   2-6 times: one clean *original* plus brightened, noisier *copies*. Copies never leave
   their group's fold, which prevents validation leakage.
4. **Copies carry the neighbouring label.** In 719 of 721 groups every copy has the other
   label of an ambiguous pair (Gnocchi/Ravioli 2<->3 or barred/tight spiral 5<->6). The
   manifest keeps the given label, marks ``is_copy``, and records ``original_label``;
   ``dataset.apply_label_policy`` decides how training uses them.
5. **Noisy images are kept.** 657 of the 671 noisy kept images (noise sigma >= 12) are
   copies from step 4; with the default label policy they get their original's label.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CLASS_NAMES, NUM_CLASSES
from ..ingestion.dataset import load_manifest, validate_layout
from .duplicates import (
    CROSS_SPLIT_MIN_CORRELATION,
    DUPLICATE_MIN_CORRELATION,
    connected_groups,
    find_duplicate_pairs,
    star_field_fingerprint,
)
from .images import read_rgb_png
from .quality import measure_quality
from .review import save_image_sheet

QUALITY_COLUMNS = [
    "alien_skin_px", "alien_eye_px", "is_alien", "contrast", "is_blank",
    "noise_sigma", "is_noisy", "background_r", "background_g", "background_b",
]
TRAIN_COLUMNS = [
    "id", "label", "class_name", "path", "keep", "exclude_reason", "fold",
    "dup_group", "dup_group_size", "is_copy", "original_id", "original_label", "label_conflict",
    "test_duplicates", *QUALITY_COLUMNS,
]
TEST_COLUMNS = [
    "id", "path", "train_duplicate", "train_duplicate_label", "train_duplicate_correlation", *QUALITY_COLUMNS,
]


# --------------------------------------------------------------------------- step helpers

def _inspect(path: Path) -> tuple[dict, np.ndarray]:
    pixels = read_rgb_png(path)
    return measure_quality(pixels), star_field_fingerprint(pixels)


def scan_quality(data_dir: Path, frame: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-image quality metrics and star-field fingerprints, in ``frame`` order."""
    paths = [data_dir / path for path in frame["path"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_inspect, paths))
    quality = pd.DataFrame.from_records([result[0] for result in results], index=frame.index)
    fingerprints = np.stack([result[1] for result in results])
    return quality, fingerprints


def _exclude_reason(row) -> str:
    if row.is_alien:
        return "alien_overlay"
    if row.is_blank:
        return "blank_frame"
    return ""


def mark_originals_and_copies(train: pd.DataFrame) -> None:
    """Within each duplicate group, the least noisy kept image is the original; others are copies.

    Adds ``is_copy``, ``original_id``, and ``original_label`` (the original's label).
    """
    kept = train.loc[train["keep"]]
    originals = kept.sort_values(["noise_sigma", "id"]).drop_duplicates("dup_group").set_index("dup_group")
    train["original_id"] = train["dup_group"].map(originals["id"]).where(train["keep"], train["id"])
    train["original_label"] = train["dup_group"].map(originals["label"]).where(train["keep"], train["label"])
    train["original_label"] = train["original_label"].astype(int)
    train["is_copy"] = train["keep"] & train["original_id"].ne(train["id"])


def assign_folds(train: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """K-fold assignment stratified by the original's label and grouped by duplicate group."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=train.index, dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(train, train["original_label"], train["dup_group"])):
        folds.iloc[validation] = fold
    return folds


# ------------------------------------------------------------------------------ main step

def build_manifests(
    data_dir: Path,
    manifest_dir: Path,
    review_dir: Path,
    *,
    workers: int,
    seed: int,
    n_folds: int,
    check_counts: bool = True,
) -> dict:
    """Run the whole cleaning step and write manifests, review sheets, and a summary."""
    # Step 1 - read the official CSVs and check that every listed PNG exists.
    manifest = load_manifest(data_dir)
    validate_layout(data_dir, manifest, check_counts=check_counts)
    manifest["path"] = manifest["split"] + "/" + manifest["id"] + ".png"

    # Step 2 - quality flags (alien, blank, noise) and fingerprints for every image.
    print(f"Scanning {len(manifest):,} images with {workers} workers...", flush=True)
    quality, fingerprints = scan_quality(data_dir, manifest, workers)
    frame = pd.concat([manifest, quality], axis=1)
    frame["exclude_reason"] = [_exclude_reason(row) for row in frame.itertuples()]

    is_train = (frame["split"] == "train").to_numpy()
    train = frame.loc[is_train].reset_index(drop=True)
    test = frame.loc[~is_train].reset_index(drop=True)
    train_prints, test_prints = fingerprints[is_train], fingerprints[~is_train]
    train["keep"] = train["exclude_reason"].eq("")

    # Step 3 - near-duplicates among kept training images and between training and test.
    print("Matching star-field fingerprints...", flush=True)
    kept = np.flatnonzero(train["keep"].to_numpy())
    within = find_duplicate_pairs(train_prints[kept], train_prints[kept], same_set=True)
    within["a"], within["b"] = kept[within.pop("query_index")], kept[within.pop("reference_index")]
    across = find_duplicate_pairs(train_prints[kept], test_prints, threshold=CROSS_SPLIT_MIN_CORRELATION)
    across["a"], across["b"] = kept[across.pop("query_index")], across.pop("reference_index")

    # Step 4 - duplicate groups, originals vs copies, and label conflicts.
    groups = connected_groups(len(train), list(zip(within["a"], within["b"])))
    train["dup_group"] = [f"g{group:05d}" for group in groups]
    kept_rows = train.loc[train["keep"]].groupby("dup_group")
    train["dup_group_size"] = kept_rows["id"].transform("size").reindex(train.index).fillna(1).astype(int)
    train["label_conflict"] = kept_rows["label"].transform("nunique").reindex(train.index).fillna(1).gt(1)
    mark_originals_and_copies(train)

    best_match = across.sort_values("correlation", ascending=False).drop_duplicates("b").set_index("b")
    test["train_duplicate"] = pd.Series(train["id"].to_numpy()[best_match["a"]], index=best_match.index)
    test["train_duplicate_label"] = pd.Series(
        train["label"].to_numpy()[best_match["a"]], index=best_match.index
    ).astype("Int64")
    test["train_duplicate_correlation"] = best_match["correlation"]
    test_ids = test["id"].to_numpy()
    train["test_duplicates"] = (
        across.groupby("a")["b"].agg(lambda b: ";".join(test_ids[b.to_numpy()])).reindex(train.index).fillna("")
    )

    # Step 5 - group-aware, stratified folds over the kept images.
    train["fold"] = -1
    train.loc[train["keep"], "fold"] = assign_folds(train.loc[train["keep"]], n_folds, seed)
    train["class_name"] = train["label"].map(CLASS_NAMES)

    # Step 6 - write the manifests (tracked in git) and the review material (local only).
    manifest_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    train[TRAIN_COLUMNS].round(4).to_csv(manifest_dir / "train_manifest.csv", index=False)
    test[TEST_COLUMNS].round(4).to_csv(manifest_dir / "test_manifest.csv", index=False)
    duplicate_pair_table(train, test, within, across).to_csv(review_dir / "duplicate_pairs.csv", index=False)
    class_count_table(train, n_folds).to_csv(review_dir / "class_counts.csv", index=False)
    save_review_sheets(data_dir, train, test, review_dir, seed)

    summary = build_summary(train, test, n_folds, seed)
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (manifest_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------- reports

def duplicate_pair_table(train, test, within: pd.DataFrame, across: pd.DataFrame) -> pd.DataFrame:
    """One row per matched pair with ids, labels, correlation, and matching orientation."""
    def describe(pairs: pd.DataFrame, kind: str, other: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "kind": kind,
            "id_a": train["id"].to_numpy()[pairs["a"]],
            "id_b": other["id"].to_numpy()[pairs["b"]],
            "label_a": train["label"].to_numpy()[pairs["a"]],
            "label_b": other["label"].to_numpy()[pairs["b"]],
            "correlation": pairs["correlation"].round(4).to_numpy(),
            "dihedral": pairs["dihedral"].to_numpy(),
        })

    table = pd.concat([describe(within, "train-train", train), describe(across, "train-test", test)])
    return table.sort_values(["kind", "correlation"], ascending=[True, False])


def build_summary(train: pd.DataFrame, test: pd.DataFrame, n_folds: int, seed: int) -> dict:
    kept = train.loc[train["keep"]]
    copies = kept.loc[kept["is_copy"]]
    conflicts = kept.loc[kept["label_conflict"]]
    pattern = conflicts.groupby("dup_group")["label"].agg(lambda s: "-".join(map(str, sorted(set(s)))))
    return {
        "train_images": len(train),
        "train_kept": len(kept),
        "excluded": train["exclude_reason"].replace("", np.nan).dropna().value_counts().to_dict(),
        "alien_overlays_in_test": int(test["is_alien"].sum()),
        "blank_frames_in_test": test.loc[test["is_blank"], "id"].tolist(),
        "duplicate_min_correlation": {"train-train": DUPLICATE_MIN_CORRELATION, "train-test": CROSS_SPLIT_MIN_CORRELATION},
        "duplicate_groups": int(kept.loc[kept["dup_group_size"].gt(1), "dup_group"].nunique()),
        "train_images_in_duplicate_groups": int(kept["dup_group_size"].gt(1).sum()),
        "copies": len(copies),
        "copies_with_label_different_from_original": int(copies["label"].ne(copies["original_label"]).sum()),
        "conflicting_label_sets": pattern.value_counts().to_dict(),
        "unique_images": int(kept["dup_group_size"].eq(1).sum()),
        "noisy_train_kept": int(kept["is_noisy"].sum()),
        "noisy_train_copies": int(copies["is_noisy"].sum()),
        "noisy_test": int(test["is_noisy"].sum()),
        "test_images_with_train_duplicate": int(test["train_duplicate"].notna().sum()),
        "folds": {"n_folds": n_folds, "seed": seed, "strategy": "StratifiedGroupKFold(original_label, dup_group)"},
        "fold_sizes": kept["fold"].value_counts().sort_index().to_dict(),
    }


def class_count_table(train: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Per-class counts: original, removed, kept, after the default label policy, and per fold."""
    def count(frame: pd.DataFrame, column: str = "label") -> np.ndarray:
        return frame[column].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_numpy()

    table = pd.DataFrame({"label": range(NUM_CLASSES)})
    table["class_name"] = table["label"].map(CLASS_NAMES)
    table["original"] = count(train)
    for reason in ("alien_overlay", "blank_frame"):
        table[f"removed_{reason}"] = count(train.loc[train["exclude_reason"].eq(reason)])
    kept = train.loc[train["keep"]]
    table["kept"] = count(kept)
    table["kept_with_original_labels"] = count(kept, "original_label")
    table["originals_and_unique"] = count(kept.loc[~kept["is_copy"]])
    for fold in range(n_folds):
        table[f"fold_{fold}_validation"] = count(kept.loc[kept["fold"].eq(fold) & ~kept["is_copy"]])
    return table


def save_review_sheets(data_dir: Path, train: pd.DataFrame, test: pd.DataFrame, review_dir: Path, seed: int) -> None:
    """Contact sheets of every decision, so each one can be checked by eye."""
    def item(row, caption: str):
        return data_dir / row.path, caption

    aliens = train.loc[train["is_alien"]].sample(frac=1, random_state=seed).head(40)
    save_image_sheet([item(r, f"{r.id}\nL{r.label} eyes {r.alien_eye_px}") for r in aliens.itertuples()], 10,
                     f"Removed: alien overlays ({int(train['is_alien'].sum())} in train, sample)",
                     review_dir / "removed_alien_overlays.png")

    near_misses = pd.concat([train, test]).query("~is_alien").nlargest(20, "alien_skin_px")
    save_image_sheet([item(r, f"{r.id}\nskin {r.alien_skin_px} eyes {r.alien_eye_px}") for r in near_misses.itertuples()],
                     10, "Kept: largest skin-coloured regions without alien eyes (should be galaxies)",
                     review_dir / "kept_alien_near_misses.png")

    blanks = pd.concat([train, test]).query("is_blank")
    save_image_sheet([item(r, f"{r.id}\ncontrast {r.contrast:.1f}") for r in blanks.itertuples()], 6,
                     "Blank frames (removed from train; test still needs a prediction)", review_dir / "blank_frames.png")

    noisy = train.loc[train["keep"]].nlargest(20, "noise_sigma")
    save_image_sheet([item(r, f"{r.id} L{r.label}\nnoise {r.noise_sigma:.1f}") for r in noisy.itertuples()], 10,
                     "Kept: noisiest training images", review_dir / "kept_noisiest.png")

    groups = train.loc[train["is_copy"], "dup_group"].drop_duplicates()
    groups = groups.sample(n=min(10, len(groups)), random_state=seed)
    members = train.loc[train["dup_group"].isin(groups) & train["keep"]]
    members = members.assign(order=members["is_copy"].astype(int)).sort_values(["dup_group", "order", "id"])
    save_image_sheet(
        [item(r, f"{r.id} L{r.label}\n{'copy' if r.is_copy else 'ORIGINAL'} n{r.noise_sigma:.0f}") for r in members.itertuples()],
        8, "Duplicate groups: original first, then its copies (sample)", review_dir / "duplicate_groups.png",
    )

    matched = test.loc[test["train_duplicate"].notna()].head(12)
    by_id = train.set_index("id")
    pairs = []
    for r in matched.itertuples():
        source = by_id.loc[r.train_duplicate]
        pairs += [(data_dir / source.path, f"{r.train_duplicate} L{source.label}"),
                  item(r, f"{r.id}\nr={r.train_duplicate_correlation:.2f}")]
    save_image_sheet(pairs, 8, "Test images that duplicate a training image (pairs)", review_dir / "train_test_duplicates.png")
