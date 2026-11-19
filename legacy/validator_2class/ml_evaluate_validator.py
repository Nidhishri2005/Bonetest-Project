"""Independent evaluation of HandXRayValidator on held-out test data.

Computes:
- Accuracy
- Precision (macro, weighted, per-class)
- Recall (macro, weighted, per-class)
- F1-Score (macro, weighted, per-class)
- Confusion Matrix
- Test Loss

Saves results to:
- JSON: backend/metrics/validator_evaluation_results.json
- Plot: backend/metrics/plots/validator_confusion_matrix.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES
from ml.validator_dataset import HandValidationDataset, build_synthetic_validator_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_validator(
    checkpoint_path: Path = BACKEND_ROOT / "checkpoints" / "hand_validator_best.pt",
    metrics_dir: Path = BACKEND_ROOT / "metrics",
    samples_per_class: int = 120,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    torch.set_num_threads(8)
    device_obj = torch.device(device)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train first with ml.train_validator.")

    logger.info("Loading test dataset (seed=%d)...", seed)
    _, _, test_data = build_synthetic_validator_dataset(
        samples_per_class=samples_per_class, seed=seed
    )
    test_loader = DataLoader(
        HandValidationDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info("Loading model from %s...", checkpoint_path)
    model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES))
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device_obj)
            labels = labels.to(device_obj)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_loss = total_loss / len(y_true)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(VALIDATOR_CLASSES))))

    per_class_results = {}
    for i, cls_name in enumerate(VALIDATOR_CLASSES):
        per_class_results[cls_name] = {
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support_per_class[i]),
        }

    results = {
        "model_architecture": "ResNet18-Validator",
        "checkpoint": str(checkpoint_path.name),
        "test_samples": len(y_true),
        "test_loss": round(float(test_loss), 4),
        "accuracy": round(acc, 4),
        "macro_metrics": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f1_macro), 4),
        },
        "weighted_metrics": {
            "precision": round(float(p_weighted), 4),
            "recall": round(float(r_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
        },
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "classes": VALIDATOR_CLASSES,
    }

    # Save metrics JSON
    out_json = metrics_dir / "validator_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation metrics to %s", out_json)

    # Generate Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6.5))
    display_labels = [c.replace("_", " ").title() for c in VALIDATOR_CLASSES]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        cbar=True,
    )
    ax.set_title("Hand X-Ray Validation Classifier — Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_plot_path = plots_dir / "validator_confusion_matrix.png"
    fig.savefig(cm_plot_path, dpi=200)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", cm_plot_path)

    # Print summary report
    print("\n" + "=" * 70)
    print("HAND X-RAY VALIDATOR EVALUATION SUMMARY (Held-Out Test Set)")
    print("=" * 70)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Macro Precision:   {p_macro * 100:.2f}%")
    print(f"Macro Recall:      {r_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print("-" * 70)
    print(f"{'Class':<28} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 70)
    for cls_name, m in per_class_results.items():
        print(f"{cls_name:<28} {m['precision']*100:>8.2f}%    {m['recall']*100:>8.2f}%    {m['f1_score']*100:>8.2f}%    {m['support']:>5}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    evaluate_validator()
