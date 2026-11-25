"""Plots for the exploratory analysis. Documentation: ``docs/02_EDA_FINDINGS.md``."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from ..config import CLASS_NAMES
from ..preprocessing.images import read_rgb_png


def save_plot(figure: Figure, path: Path) -> None:
    """Render a matplotlib figure to a PNG without needing a display."""
    FigureCanvasAgg(figure).print_figure(path, dpi=140, bbox_inches="tight")


def plot_distribution(counts: pd.DataFrame, path: Path) -> None:
    """Bar chart of the training class counts."""
    figure = Figure(figsize=(11, 5), layout="constrained")
    axis = figure.subplots()
    bars = axis.bar(counts["class_name"], counts["count"], color="steelblue")
    axis.bar_label(bars, padding=3)
    axis.tick_params(axis="x", labelrotation=35)
    axis.set(ylabel="Images", title="Training class distribution")
    axis.margins(y=0.15)
    save_plot(figure, path)


def plot_domain_shift(valid: pd.DataFrame, path: Path) -> None:
    """Overlaid train/test histograms of brightness, contrast, sharpness, and channel means."""
    columns = ("mean", "std", "hf_energy", "red_mean", "green_mean", "blue_mean")
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 3)
    for axis, metric in zip(axes.flat, columns):
        values = valid[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            axis.set(title=metric)
            continue
        bins = np.histogram_bin_edges(values, bins=50)
        for split, frame in valid.groupby("split"):
            axis.hist(frame[metric].to_numpy(dtype=float), bins=bins, density=True, alpha=0.55, label=split)
        axis.set(title=metric, ylabel="Density")
        axis.legend()
    figure.suptitle("Train and test image statistics")
    save_plot(figure, path)


def plot_grid(data_dir: Path, rows: list[pd.DataFrame], width: int, title: str, path: Path) -> None:
    """Grid of images: one list of samples per row, captioned with class and sharpness."""
    figure = Figure(figsize=(2.3 * width, 2.3 * len(rows)), layout="constrained")
    axes = figure.subplots(len(rows), width, squeeze=False)
    for axis_row, samples in zip(axes, rows):
        for axis in axis_row:
            axis.set_axis_off()
        for axis, sample in zip(axis_row, samples.itertuples()):
            axis.imshow(read_rgb_png(data_dir / sample.split / f"{sample.id}.png"))
            caption = CLASS_NAMES[int(sample.label)] if sample.split == "train" else "Test"
            axis.set_title(f"{caption}\n{sample.id} | HF {sample.hf_energy:.1f}", fontsize=8)
    figure.suptitle(title)
    save_plot(figure, path)
