from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from ..config import IMAGE_SIZE


def read_rgb_png(source: Path | BinaryIO) -> np.ndarray:
    """Decode a competition PNG and return its original RGB pixel values."""
    with Image.open(source) as image:
        if (image.format, image.mode, image.size) != ("PNG", "RGB", IMAGE_SIZE):
            width, height = IMAGE_SIZE
            raise ValueError(f"Expected {width}×{height} RGB PNG, found {image.format} {image.mode} {image.size}")
        return np.asarray(image)
