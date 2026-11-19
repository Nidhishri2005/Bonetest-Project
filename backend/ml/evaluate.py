"""Evaluate all trained models on validation or test set.

Usage:
    python -m ml.evaluate --data-dir /path/to/rsna --checkpoints-dir checkpoints --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.new_models import MultiscaleCNNFeatureExtractor, MultiscaleCNNRFWrapper
from ml.dataset import create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bone age models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=BACKEND_ROOT / "metrics"
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split to benchmark on",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive pediatric bone age regression metrics.

    Includes:
    1. MAE (months)
    2. MSE
    3. RMSE (months)
    4. R2 Score
    5. Mean Error (systematic bias, months)
    6. Median Absolute Error (months)
    7. % Within 6 Months
    8. % Within 12 Months
    9. Age-cohort stratified error breakdown
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    # R2 Score
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    mean_error = float(np.mean(errors))
    median_ae = float(np.median(abs_errors))
    within_6m = float(np.mean(abs_errors <= 6.0) * 100.0)
    within_12m = float(np.mean(abs_errors <= 12.0) * 100.0)

    # Age cohort breakdown (months)
    cohort_definitions = [
        ("infancy_0_24", 0.0, 24.0),
        ("early_childhood_24_72", 24.0, 72.0),
        ("mid_childhood_72_144", 72.0, 144.0),
        ("adolescence_144_216", 144.0, 216.0),
        ("post_pubertal_216_plus", 216.0, 300.0),
    ]

    cohorts = {}
    for name, low, high in cohort_definitions:
        mask = (y_true >= low) & (y_true < high)
        if np.any(mask):
            cohort_mae = float(np.mean(abs_errors[mask]))
            cohort_count = int(np.sum(mask))
        else:
            cohort_mae = None
            cohort_count = 0
        cohorts[name] = {"mae": cohort_mae, "count": cohort_count}

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mean_error": mean_error,
        "median_ae": median_ae,
        "within_6m_pct": within_6m,
        "within_12m_pct": within_12m,
        "cohorts": cohorts,
    }


@torch.no_grad()
def evaluate_model(model_type: str, loader, device, checkpoints_dir: Path):
    meta = MODEL_REGISTRY[model_type]
    ckpt_path = checkpoints_dir / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        print(f"Skipping {model_type}: checkpoint not found at {ckpt_path}")
        return None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_type in {"cnn_rf", "multiscale_cnn_rf"}:
        if model_type == "cnn_rf":
            cnn = CNNFeatureExtractor(pretrained=False)
            rf_path = BACKEND_ROOT / "rf_models" / "cnn_rf.joblib"
        else:
            cnn = MultiscaleCNNFeatureExtractor(pretrained=False)
            rf_path = BACKEND_ROOT / "rf_models" / "multiscale_cnn_rf.joblib"

        if not rf_path.exists():
            print(f"Skipping {model_type}: random forest weights not found at {rf_path}")
            return None, None, None

        cnn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        cnn.to(device).eval()
        rf = joblib.load(rf_path)
        preds, targets, ids = [], [], []
        for batch in loader:
            imgs = batch["image"].to(device)
            feats = cnn(imgs).cpu().numpy()
            batch_preds = rf.predict(feats)
            preds.extend(batch_preds.tolist())
            targets.extend(batch["bone_age"].numpy().tolist())
            ids.extend(batch["id"])
        return np.array(targets), np.array(preds), ids

    model = meta["class"](pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()

    preds, targets, ids = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        if meta.get("requires_gender", False):
            gender = batch["male"].unsqueeze(-1).to(device)
            out = model(imgs, gender)
        else:
            out = model(imgs)
        preds.extend(out.cpu().numpy().tolist())
        targets.extend(batch["bone_age"].numpy().tolist())
        ids.extend(batch["id"])

    return np.array(targets), np.array(preds), ids


def plot_comparison(results: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}
    sns.set_theme(style="whitegrid")

    models = list(results.keys())
    metrics = ["mae", "mse", "rmse"]
    labels = ["MAE (months)", "MSE", "RMSE (months)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = sns.color_palette("muted", len(models))

    for ax, metric, label in zip(axes, metrics, labels):
        values = [results[m]["metrics"][metric] for m in models]
        names = [results[m]["display_name"].replace(" + ", "\n+ ") for m in models]
        bars = ax.bar(names, values, color=palette)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    comparison_path = output_dir / "metrics_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["metrics_comparison"] = str(comparison_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    for model_type, data in results.items():
        df = pd.read_csv(data["predictions_csv"])
        ax.scatter(
            df["actual"],
            df["predicted"],
            alpha=0.35,
            s=14,
            label=f"{data['display_name']} (MAE: {data['metrics']['mae']:.2f})",
        )
    lims = [0, 230]
    ax.plot(lims, lims, "k--", alpha=0.6, label="Ideal (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Bone Age (months)", fontsize=11)
    ax.set_ylabel("Predicted Bone Age (months)", fontsize=11)
    ax.set_title("Actual vs Predicted Bone Age", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    scatter_path = output_dir / "scatter_comparison.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["scatter_comparison"] = str(scatter_path)

    return plot_paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(
        args.data_dir,
        batch_size=32,
        return_test=True,
    )
    train_loader, val_loader, test_loader, train_df, val_df, test_df = loaders
    target_loader = test_loader if args.split == "test" else val_loader
    split_name = "test" if args.split == "test" else "validation"

    print(f"Benchmarking models on {split_name} split ({len(target_loader.dataset)} samples)...")

    all_results = {}
    for model_type in SUPPORTED_MODELS:
        print(f"Evaluating {model_type}...")
        y_true, y_pred, ids = evaluate_model(
            model_type, target_loader, device, args.checkpoints_dir
        )
        if y_true is None:
            continue

        metrics = compute_metrics(y_true, y_pred)
        meta = MODEL_REGISTRY[model_type]
        pred_csv = args.metrics_dir / f"predictions_{model_type}.csv"
        pd.DataFrame(
            {"id": ids, "actual": y_true, "predicted": y_pred}
        ).to_csv(pred_csv, index=False)

        all_results[model_type] = {
            "model_type": model_type,
            "display_name": meta["display_name"],
            "metrics": metrics,
            "num_samples": len(y_true),
            "predictions_csv": str(pred_csv),
        }
        print(
            f"  MAE={metrics['mae']:.2f} mo, RMSE={metrics['rmse']:.2f} mo, "
            f"R2={metrics['r2']:.3f}, <=12m={metrics['within_12m_pct']:.1f}%"
        )

    if not all_results:
        print("No valid checkpoints found to evaluate.")
        return

    best = min(all_results.items(), key=lambda x: x[1]["metrics"]["mae"])
    plots_dir = args.metrics_dir / "plots"
    plot_paths = plot_comparison(all_results, plots_dir)

    output = {
        "comparison": {
            "models": [
                {
                    "model_type": k,
                    "display_name": v["display_name"],
                    "metrics": v["metrics"],
                    "num_samples": v["num_samples"],
                }
                for k, v in all_results.items()
            ],
            "best_model": best[0],
            "split": split_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "plots": {k: str(v) for k, v in plot_paths.items()},
        "details": {
            "dataset": "RSNA Pediatric Bone Age",
            "split_evaluated": split_name,
            "image_size": 512,
            "normalization": {"mean": 0.4523, "std": 0.2118},
        },
    }

    out_path = args.metrics_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved updated evaluation results to {out_path}")


if __name__ == "__main__":
    main()
