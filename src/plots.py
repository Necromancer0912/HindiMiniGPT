"""Shared matplotlib style so every notebook's figures read as one system.

Palette source: dataviz skill reference palette (validated categorical order,
sequential blue ramp, blue<->red diverging pair). Light-surface only -- these are
static notebook/report figures, not a theme-switching web page.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DIR

# Categorical, in fixed validated order -- always assign by series index, never cycle/reorder.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
            "figure.dpi": 110,
            "savefig.dpi": 150,
        }
    )


def savefig(fig, name: str, subdir: str = "") -> Path:
    out_dir = FIG_DIR / subdir if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_train_val_curves(steps, train_vals, val_vals, ylabel: str, title: str, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, train_vals, color=CATEGORICAL[0], linewidth=2, label="train")
    ax.plot(steps, val_vals, color=CATEGORICAL[5], linewidth=2, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(cm: np.ndarray, class_names, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = SURFACE if cm[i, j] > thresh else INK_PRIMARY
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_pareto(x, y, labels, xlabel: str, ylabel: str, title: str):
    """Quality-vs-cost scatter (e.g. perplexity vs latency) with each point labeled."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.scatter(xi, yi, s=90, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
