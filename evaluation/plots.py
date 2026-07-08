"""
evaluation/plots.py
--------------------
Generates all evaluation visualisations:
    1. Confusion matrix heatmaps (one per run)
    2. Training/validation loss curves with LR schedule overlay
    3. Hidden size comparison bar chart (F1 vs hidden_size per variant)
    4. Ablation comparison table heatmap
    5. ROC-AUC comparison bar chart
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_DIR = Path("evaluation/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette
VARIANT_COLORS = {
    "price_only": "#378ADD",
    "text_only":  "#1D9E75",
    "fusion":     "#7F77DD",
}


def plot_confusion_matrix(
    cm: List[List[int]],
    run_name: str,
    save_dir: str = "evaluation/plots",
) -> str:
    """Plot and save a seaborn confusion matrix heatmap."""
    save_path = Path(save_dir) / f"confusion_matrix_{run_name}.png"
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Down", "Pred: Up"],
        yticklabels=["True: Down", "True: Up"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix\n{run_name}", fontsize=11, pad=10)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_loss_curves(
    history: List,   # List[EpochMetrics]
    run_name: str,
    save_dir: str = "evaluation/plots",
) -> str:
    """Plot train/val loss curves with LR schedule overlay."""
    save_path = Path(save_dir) / f"loss_curve_{run_name}.png"

    epochs      = [m.epoch for m in history]
    train_loss  = [m.train_loss for m in history]
    val_loss    = [m.val_loss for m in history]
    lrs         = [m.lr for m in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(epochs, train_loss, label="Train Loss", color="#378ADD", linewidth=1.8)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#E05C4B", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves — {run_name}", fontsize=11)
    ax1.legend(loc="upper left")

    # LR schedule on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#AAAAAA", linewidth=1.0,
             linestyle="--", alpha=0.7, label="LR")
    ax2.set_ylabel("Learning Rate", color="#AAAAAA")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax2.tick_params(axis="y", labelcolor="#AAAAAA")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_hidden_size_comparison(
    results_df: pd.DataFrame,
    metric: str = "f1_macro",
    save_dir: str = "evaluation/plots",
) -> str:
    """Grouped bar chart: metric vs hidden_size for each variant."""
    save_path = Path(save_dir) / f"hidden_size_comparison_{metric}.png"

    # Only rows with actual training (text_only deduped to h64)
    df = results_df[results_df["note"].fillna("") == ""].copy()
    df["hidden_size"] = df["hidden_size"].astype(str)

    variants = ["price_only", "text_only", "fusion"]
    hidden_sizes = ["64", "128", "256"]
    x = np.arange(len(hidden_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, variant in enumerate(variants):
        vdf = df[df["variant"] == variant].set_index("hidden_size")
        values = [
            vdf.loc[h, metric] if h in vdf.index else 0.0
            for h in hidden_sizes
        ]
        bars = ax.bar(
            x + i * width, values, width,
            label=variant,
            color=VARIANT_COLORS[variant],
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"h={h}" for h in hidden_sizes])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Hidden Size Comparison — {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_ablation_heatmap(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    save_dir: str = "evaluation/plots",
) -> str:
    """Heatmap comparing all variants across key metrics."""
    save_path = Path(save_dir) / "ablation_heatmap.png"

    if metrics is None:
        metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]

    # Deduplicate text_only rows
    df = results_df[results_df["note"].fillna("") == ""].copy()
    df["label"] = df["variant"] + "\nh=" + df["hidden_size"].astype(str)

    pivot = df.set_index("label")[metrics]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.7)))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0.4,
        vmax=0.75,
    )
    ax.set_title("Ablation Study — Test Metrics Heatmap", fontsize=12, pad=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_roc_auc_comparison(
    results_df: pd.DataFrame,
    save_dir: str = "evaluation/plots",
) -> str:
    """Horizontal bar chart of ROC-AUC per model."""
    save_path = Path(save_dir) / "roc_auc_comparison.png"

    df = results_df[results_df["note"].fillna("") == ""].copy()
    df["label"] = df["variant"] + " h=" + df["hidden_size"].astype(str)
    df = df.sort_values("roc_auc", ascending=True)

    colors = [VARIANT_COLORS.get(v, "#888888") for v in df["variant"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.55)))
    bars = ax.barh(df["label"], df["roc_auc"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7,
               label="Random baseline")
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("ROC-AUC Comparison Across Models")
    ax.legend()
    ax.set_xlim(0.3, min(1.0, df["roc_auc"].max() + 0.08))
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)