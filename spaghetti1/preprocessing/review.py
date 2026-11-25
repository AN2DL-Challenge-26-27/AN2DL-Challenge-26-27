"""Contact sheets for visually reviewing the images flagged by the cleaning step.

The sheets are written to ``outputs/preprocessing/``; ``docs/03_PREPROCESSING.md`` lists them.
"""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .images import read_rgb_png


def save_image_sheet(
    images: list[tuple[np.ndarray | Path, str]], columns: int, title: str, path: Path
) -> None:
    """Save a grid of images; each item is (pixels or PNG path, caption)."""
    if not images:
        return
    rows = (len(images) + columns - 1) // columns
    figure = Figure(figsize=(1.9 * columns, 2.1 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (source, caption) in zip(axes.flat, images):
        axis.imshow(read_rgb_png(source) if isinstance(source, Path) else source)
        axis.set_title(caption, fontsize=7)
    figure.suptitle(title)
    FigureCanvasAgg(figure).print_figure(path, dpi=110, bbox_inches="tight")
