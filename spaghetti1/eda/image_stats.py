from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..preprocessing.images import read_rgb_png

METRICS = (
    "mean", "std", "zero_fraction", "saturated_fraction", "red_mean",
    "green_mean", "blue_mean", "hf_energy", "center_mean", "border_mean",
)


def inspect_image(path: Path) -> dict:
    result = dict.fromkeys(("sha256", "error", *METRICS))
    try:
        raw = path.read_bytes()
        result["sha256"] = sha256(raw).hexdigest()
        pixels = read_rgb_png(BytesIO(raw)).astype(np.float32)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        result["error"] = str(exc)
        return result

    gray = pixels.mean(axis=2)
    height_difference = np.abs(np.diff(gray, axis=0)).mean()
    width_difference = np.abs(np.diff(gray, axis=1)).mean()
    edges = (gray[:16], gray[-16:], gray[:, :16], gray[:, -16:])
    red, green, blue = pixels.mean(axis=(0, 1))
    result.update(
        mean=pixels.mean(),
        std=pixels.std(),
        zero_fraction=(pixels == 0).mean(),
        saturated_fraction=(pixels == 255).mean(),
        red_mean=red,
        green_mean=green,
        blue_mean=blue,
        hf_energy=(height_difference + width_difference) / 2,
        center_mean=gray[96:160, 96:160].mean(),
        border_mean=np.mean([edge.mean() for edge in edges]),
    )
    return result


def scan_images(data_dir: Path, manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    paths = (data_dir / row.split / f"{row.id}.png" for row in manifest.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(inspect_image, paths)
        stats = pd.DataFrame.from_records(results)
    return pd.concat([manifest.reset_index(drop=True), stats], axis=1)
