"""Visual evaluation plots for pediatric bone age models.

Generates:
- Bland-Altman agreement plots with 95% limits of agreement and bias lines
- Predicted vs. Actual scatter plots with identity line and +/- 6/12 month bounds
- Stratified pediatric age cohort error distributions
- Paired model comparison distribution charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ml.eval.metrics import PEDIATRIC_COHORT_BINS, PEDIATRIC_COHORT_LABELS, compute_bland_altman

logger = logging.getLogger(__name__)


def plot_bland_altman(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    save_path: Union[Path, str],
    title: str = "Bland-Altman Agreement Plot",
    model_name: Optional[str] = None,
) -> None:
    """Generate and save publication-quality Bland-Altman agreement plot."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    means = (y_true + y_pred) / 2.0
    diffs = y_pred - y_true
    stats = compute_bland_altman(y_true, y_pred)

    mean_bias = stats["mean_bias"]
    lower_lim = stats["lower_limit"]
    upper_lim = stats["upper_limit"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Inliers vs Outliers
    outliers = (diffs < lower_lim) | (diffs > upper_lim)
    ax.scatter(means[~outliers], diffs[~outliers], alpha=0.45, color="steelblue", s=25, label="Inliers")
    ax.scatter(means[outliers], diffs[outliers], alpha=0.85, color="crimson", s=35, label=f"Outliers ({stats['outlier_pct']}%)")

    # Reference lines
    ax.axhline(mean_bias, color="navy", linestyle="-", linewidth=1.8, label=f"Mean Bias: {mean_bias:+.2f} mo")
    ax.axhline(upper_lim, color="red", linestyle="--", linewidth=1.5, label=f"+1.96 SD: {upper_lim:+.2f} mo")
    ax.axhline(lower_lim, color="red", linestyle="--", linewidth=1.5, label=f"-1.96 SD: {lower_lim:+.2f} mo")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)

    # Shaded agreement region
    ax.axhspan(lower_lim, upper_lim, alpha=0.08, color="blue")

    header = f"{title}\n({model_name})" if model_name else title
    ax.set_title(header, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Bone Age (True + Predicted) / 2 (months)", fontsize=11)
    ax.set_ylabel("Difference (Predicted - True) (months)", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved Bland-Altman plot to {save_path}")


def plot_predictions_vs_ground_truth(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    save_path: Union[Path, str],
    title: str = "Predicted vs. Actual Bone Age",
    model_name: Optional[str] = None,
) -> None:
    """Generate and save scatter plot of predictions vs. ground truth."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    min_val = 0.0
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), 240.0) + 10.0

    residuals = np.abs(y_pred - y_true)
    mae = float(np.mean(residuals))
    r2 = 1.0 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Scatter
    scatter = ax.scatter(y_true, y_pred, alpha=0.4, c=residuals, cmap="viridis", s=25, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error (months)", fontsize=10)

    # Identity Line
    ax.plot([min_val, max_val], [min_val, max_val], "k-", linewidth=1.5, label="Identity (y = x)")

    # +/- 6 and +/- 12 month bounds
    ax.plot([min_val, max_val], [min_val + 6, max_val + 6], "b--", linewidth=1.0, alpha=0.7, label="+/- 6 months")
    ax.plot([min_val, max_val], [min_val - 6, max_val - 6], "b--", linewidth=1.0, alpha=0.7)
    ax.plot([min_val, max_val], [min_val + 12, max_val + 12], "r:", linewidth=1.0, alpha=0.7, label="+/- 12 months")
    ax.plot([min_val, max_val], [min_val - 12, max_val - 12], "r:", linewidth=1.0, alpha=0.7)

    # Stat annotation
    stats_text = f"MAE: {mae:.2f} mo\n$R^2$: {r2:.4f}\nN = {len(y_true):,}"
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    header = f"{title}\n({model_name})" if model_name else title
    ax.set_title(header, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Chronological Ground Truth Bone Age (months)", fontsize=11)
    ax.set_ylabel("Model Predicted Bone Age (months)", fontsize=11)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved scatter plot to {save_path}")


def plot_cohort_error_distribution(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    save_path: Union[Path, str],
    title: str = "MAE across Pediatric Age Cohorts",
) -> None:
    """Generate bar chart of MAE stratified across 5 pediatric age cohorts."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_errors = np.abs(y_pred - y_true)

    cohort_maes = []
    cohort_counts = []
    labels = []

    for i in range(len(PEDIATRIC_COHORT_BINS) - 1):
        low, high = PEDIATRIC_COHORT_BINS[i], PEDIATRIC_COHORT_BINS[i + 1]
        mask = (y_true >= low) & (y_true < high)
        labels.append(PEDIATRIC_COHORT_LABELS[i].split(" ")[0])
        cnt = int(np.sum(mask))
        cohort_counts.append(cnt)
        cohort_maes.append(float(np.mean(abs_errors[mask])) if cnt > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(labels, cohort_maes, color="cornflowerblue", edgecolor="navy", alpha=0.85, width=0.55)

    # Annotate bar values and counts
    for bar, mae, cnt in zip(bars, cohort_maes, cohort_counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.4,
            f"{mae:.2f} mo\n(N={cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Pediatric Age Cohort", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (months)", fontsize=11)
    ax.set_ylim(0, max(cohort_maes) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved cohort error distribution plot to {save_path}")
