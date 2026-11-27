"""Build ensemble submissions from probabilities that were already saved (no GPU needed).

Run from the repository root after the ``predict`` calls listed in docs/10 ("Leaderboard
round 1"):

    python experiments/combine_submissions.py

Each member of an ensemble is a pair of saved arrays: validation probabilities (fold 0, same
order as ``split_fold``) and test probabilities (``sample_submission.csv`` order). The script
averages them, reports the validation weighted F1 of exactly that combination, checks the
submission format, and writes ``submission.csv`` + ``ensemble_info.json`` per ensemble into
``outputs/submissions/<name>/``.
"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = ROOT / "outputs" / "submissions"
CONVNEXT = SUBMISSIONS / "ensemble_tiny224_tiny384_small288"  # 8-way TTA probabilities
DINOV2 = SUBMISSIONS / "dinov2_base_336_notta"                 # no TTA (TTA lowered its score)

MEMBERS = {
    "convnext_tiny_224": (CONVNEXT / "val_probabilities_e1_convnext_tiny_drop_copies_fold0.npy",
                          CONVNEXT / "test_probabilities_e1_convnext_tiny_drop_copies_fold0.npy"),
    "convnext_tiny_384": (CONVNEXT / "val_probabilities_e2_convnext_tiny_384_fold0.npy",
                          CONVNEXT / "test_probabilities_e2_convnext_tiny_384_fold0.npy"),
    "convnext_small_288": (CONVNEXT / "val_probabilities_e3_convnext_small_288_fold0.npy",
                           CONVNEXT / "test_probabilities_e3_convnext_small_288_fold0.npy"),
    "dinov2_base_336": (DINOV2 / "val_probabilities_r2_dinov2_base_336_recipe_fold0_notta.npy",
                        DINOV2 / "test_probabilities_r2_dinov2_base_336_recipe_fold0_notta.npy"),
}

ENSEMBLES = {
    "lb1_a_dinov2_alone": ["dinov2_base_336"],
    "lb1_b_4models_convnext3_dinov2": ["convnext_tiny_224", "convnext_tiny_384", "convnext_small_288", "dinov2_base_336"],
    "lb1_c_tiny384_small288_dinov2": ["convnext_tiny_384", "convnext_small_288", "dinov2_base_336"],
}


def validation_labels() -> np.ndarray:
    import sys

    sys.path.insert(0, str(ROOT))
    from spaghetti1.preprocessing.dataset import load_train_manifest, split_fold

    _, validation = split_fold(load_train_manifest(), 0)
    return validation["label"].to_numpy()


def main() -> None:
    labels = validation_labels()
    ids = pd.read_csv(ROOT / "manifests" / "test_manifest.csv")["id"]
    reference = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")["id"]
    loaded = {name: (np.load(val), np.load(test)) for name, (val, test) in MEMBERS.items()}
    for name, (val, test) in loaded.items():
        print(f"member {name:20s} validation wF1 {f1_score(labels, val.argmax(1), average='weighted'):.4f}")

    for name, members in ENSEMBLES.items():
        val = np.mean([loaded[m][0] for m in members], axis=0)
        test = np.mean([loaded[m][1] for m in members], axis=0)
        score = f1_score(labels, val.argmax(1), average="weighted")
        submission = pd.DataFrame({"id": ids, "label": test.argmax(1).astype(int)})
        assert len(submission) == 14736 and submission["id"].is_unique and (submission["id"] == reference).all()
        assert submission["label"].between(0, 9).all() and not submission.isna().any().any()
        folder = SUBMISSIONS / name
        folder.mkdir(parents=True, exist_ok=True)
        submission.to_csv(folder / "submission.csv", index=False)
        info = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "members": members,
                "member_files": {m: [str(p.relative_to(ROOT)) for p in MEMBERS[m]] for m in members},
                "validation_weighted_f1_ensemble": round(score, 4), "test_rows": len(submission),
                "predicted_class_counts": submission["label"].value_counts().sort_index().to_dict()}
        (folder / "ensemble_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"{name:34s} validation wF1 {score:.4f} -> {folder / 'submission.csv'}")


if __name__ == "__main__":
    main()
