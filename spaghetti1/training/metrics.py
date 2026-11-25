"""Evaluation metrics for the competition (weighted F1 first) and the confusion-matrix plot.

Documentation: ``docs/08_TRAINING_AND_RESULTS.md``.
"""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..config import CLASS_NAMES, NUM_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Weighted F1 (competition metric), macro F1, accuracy, and per-class precision/recall/F1."""
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            int(c): {"precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c]), "support": int(support[c])}
            for c in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(matrix: list[list[int]], title: str, path: Path) -> None:
    """Row-normalised confusion matrix (rows = true class) with raw counts in each cell."""
    counts = np.asarray(matrix)
    normalised = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    figure = Figure(figsize=(8.5, 7.5), layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    names = [f"{c} {CLASS_NAMES[c][:14]}" for c in range(NUM_CLASSES)]
    axis.set_xticks(range(NUM_CLASSES), names, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(NUM_CLASSES), names, fontsize=8)
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            if counts[row, column]:
                colour = "white" if normalised[row, column] > 0.6 else "black"
                axis.text(column, row, counts[row, column], ha="center", va="center", fontsize=7, color=colour)
    figure.colorbar(image, ax=axis, shrink=0.8, label="Share of true class")
    FigureCanvasAgg(figure).print_figure(path, dpi=120)
