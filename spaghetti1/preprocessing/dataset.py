"""PyTorch datasets, fold splits, label policies, and class balancing.

Documentation: ``docs/04_MANIFESTS_AND_FOLDS.md`` and ``docs/06_SHARING_AND_LOADING_DATA.md``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..config import NUM_CLASSES, PROJECT_DIR
from .images import read_rgb_png

MANIFEST_DIR = PROJECT_DIR / "manifests"


def load_train_manifest(manifest_dir: Path = MANIFEST_DIR, *, kept_only: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(manifest_dir / "train_manifest.csv")
    return frame.loc[frame["keep"]].reset_index(drop=True) if kept_only else frame


def load_test_manifest(manifest_dir: Path = MANIFEST_DIR) -> pd.DataFrame:
    return pd.read_csv(manifest_dir / "test_manifest.csv")


LABEL_POLICIES = ("original", "drop_copies", "as_given")


def apply_label_policy(train: pd.DataFrame, policy: str = "original") -> pd.DataFrame:
    """Decide how the brightened, noisy *copies* of duplicated cutouts are used for training.

    * ``"original"`` (default): keep the copies but give them their original's label.
      They then act as extra noisy views of the same galaxy with one consistent label.
    * ``"drop_copies"``: train on originals and unique images only.
    * ``"as_given"``: keep the official labels. The same galaxy then appears with two
      different labels (2<->3 or 5<->6), so use this only as a comparison run.

    The ``label`` column holds the label used for training; ``given_label`` keeps the official one.
    """
    if policy not in LABEL_POLICIES:
        raise ValueError(f"policy must be one of {LABEL_POLICIES}, got {policy!r}")
    frame = train.assign(given_label=train["label"])
    if policy == "drop_copies":
        frame = frame.loc[~frame["is_copy"]]
    elif policy == "original":
        frame["label"] = frame["original_label"]
    return frame.reset_index(drop=True)


def split_fold(
    train: pd.DataFrame, fold: int, *, label_policy: str = "original", validate_on_copies: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training part, validation part) for one group-aware CV fold.

    Validation uses originals and unique images only (they look like the test set and
    carry one label per galaxy), unless ``validate_on_copies`` is True.
    """
    is_validation = train["fold"].eq(fold)
    training = apply_label_policy(train.loc[~is_validation], label_policy)
    validation = train.loc[is_validation]
    if not validate_on_copies:
        validation = validation.loc[~validation["is_copy"]]
    return training, validation.reset_index(drop=True)


class GalaxyDataset(Dataset):
    """Yields (image tensor, label). Test rows have label -1."""

    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform, *, cache: bool = False):
        self.paths = [Path(data_dir) / path for path in frame["path"]]
        labels = frame["label"] if "label" in frame else pd.Series(-1, index=frame.index)
        self.labels = torch.as_tensor(labels.fillna(-1).astype(int).to_numpy())
        self.ids = frame["id"].tolist()
        self.transform = transform
        # 4,000 training images are ~0.8 GB as uint8; caching avoids PNG decoding every epoch.
        self.cache = [read_rgb_png(path) for path in self.paths] if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.cache[index] if self.cache is not None else read_rgb_png(self.paths[index])
        return self.transform(pixels), self.labels[index]


def class_weights(labels, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency class weights ``(N / (C * n_c)) ** power``; power 0.5 is a softened version."""
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(float)
    weights = (counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))) ** power
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels, power: float = 0.5, seed: int = 42) -> WeightedRandomSampler:
    """Sampler that over-draws rare classes; power 1 gives equal expected class frequency."""
    per_class = class_weights(labels, power).numpy()
    sample_weights = torch.as_tensor(per_class[np.asarray(labels)], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that makes NumPy randomness reproducible per worker."""
    np.random.seed(torch.initial_seed() % 2**32)
