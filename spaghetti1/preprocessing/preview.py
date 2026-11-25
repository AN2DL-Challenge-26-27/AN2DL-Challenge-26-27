"""Render augmented samples so the augmentation strength can be checked by eye.

Command: ``python -m spaghetti1 augment-preview``. Documentation: ``docs/05_AUGMENTATION_AND_TRANSFORMS.md``.
"""

from pathlib import Path

import numpy as np
import torch

from .dataset import load_train_manifest
from .images import read_rgb_png
from .review import save_image_sheet
from .transforms import AugmentConfig, GalaxyTransform, PreprocessConfig


def save_augmentation_preview(
    data_dir: Path,
    manifest_dir: Path,
    output: Path,
    *,
    images: int,
    variants: int,
    seed: int,
    denoise: str = "none",
    background_normalize: bool = False,
) -> Path:
    torch.manual_seed(seed)
    train = load_train_manifest(manifest_dir)
    samples = train.groupby("label").sample(n=1, random_state=seed).sample(n=min(images, 10), random_state=seed)
    preprocess = PreprocessConfig(denoise=denoise, background_normalize=background_normalize)
    clean = GalaxyTransform(preprocess, None)
    augmented = GalaxyTransform(preprocess, AugmentConfig())

    def to_pixels(image: torch.Tensor) -> np.ndarray:
        return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    cells = []
    for row in samples.itertuples():
        pixels = read_rgb_png(Path(data_dir) / row.path)
        cells.append((pixels, f"{row.id} L{row.label}\noriginal"))
        cells.append((to_pixels(clean.to_unit(pixels)), "preprocessed"))
        cells += [(to_pixels(augmented.to_unit(pixels)), f"augmented {k + 1}") for k in range(variants)]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image_sheet(cells, variants + 2, "Training augmentation preview", output)
    return output
