"""One call that turns the dataset folder into ready PyTorch DataLoaders.

Documentation: ``docs/06_SHARING_AND_LOADING_DATA.md``. Example::

    loaders = build_dataloaders(data_dir, fold=0, batch_size=64, image_size=224)
    for images, labels in loaders.train:
        ...

``data_dir`` is either the extracted Google Drive package (it has ``manifests/`` inside)
or the raw Kaggle folder ``data/raw`` (then the repository's ``manifests/`` folder is used).
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import (
    MANIFEST_DIR,
    GalaxyDataset,
    balanced_sampler,
    class_weights,
    load_test_manifest,
    load_train_manifest,
    seed_worker,
    split_fold,
)
from .transforms import AugmentConfig, GalaxyTransform, PreprocessConfig


@dataclass
class GalaxyLoaders:
    """Everything a training script needs for one fold."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    class_weights: torch.Tensor
    config: dict = field(default_factory=dict)


def manifest_dir_for(data_dir: Path) -> Path:
    """Use the package's own manifests when present, otherwise the repository's ``manifests/``."""
    packaged = Path(data_dir) / "manifests"
    return packaged if (packaged / "train_manifest.csv").is_file() else MANIFEST_DIR


def build_dataloaders(
    data_dir: Path,
    *,
    fold: int = 0,
    batch_size: int = 64,
    image_size: int = 256,
    label_policy: str = "original",
    balance_power: float = 0.0,
    preprocess: PreprocessConfig | None = None,
    augment: AugmentConfig | None = AugmentConfig(),
    num_workers: int = 2,
    cache_images: bool = False,
    seed: int = 42,
) -> GalaxyLoaders:
    """Build train/validation/test loaders for one CV fold.

    ``fold=-1`` trains on all kept images (final model); validation is then empty.
    ``balance_power`` > 0 over-samples rare classes (1.0 = equal class frequency); with 0 the
    training loader simply shuffles and ``class_weights`` can be used in the loss instead.
    ``augment=None`` turns training augmentation off.
    """
    data_dir = Path(data_dir)
    manifest_dir = manifest_dir_for(data_dir)
    preprocess = preprocess or PreprocessConfig(image_size=image_size)
    train_transform, eval_transform = GalaxyTransform(preprocess, augment), GalaxyTransform(preprocess, None)

    # fold = -1 matches no row, so every kept image is used for training.
    train_frame, validation_frame = split_fold(load_train_manifest(manifest_dir), fold, label_policy=label_policy)
    test_frame = load_test_manifest(manifest_dir)

    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "worker_init_fn": seed_worker,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    train_dataset = GalaxyDataset(train_frame, data_dir, train_transform, cache=cache_images)
    if balance_power > 0:
        sampler = balanced_sampler(train_frame["label"], balance_power, seed)
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=True, **common)
    validation_loader = DataLoader(GalaxyDataset(validation_frame, data_dir, eval_transform, cache=cache_images),
                                   shuffle=False, **common)
    test_loader = DataLoader(GalaxyDataset(test_frame, data_dir, eval_transform), shuffle=False, **common)

    return GalaxyLoaders(
        train=train_loader, validation=validation_loader, test=test_loader,
        train_frame=train_frame, validation_frame=validation_frame, test_frame=test_frame,
        class_weights=class_weights(train_frame["label"]),
        config={"fold": fold, "batch_size": batch_size, "image_size": preprocess.image_size,
                "label_policy": label_policy, "balance_power": balance_power, "preprocess": preprocess,
                "augment": augment, "seed": seed, "manifest_dir": str(manifest_dir)},
    )
