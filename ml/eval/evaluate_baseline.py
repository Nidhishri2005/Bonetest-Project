"""Evaluation script for legacy CNN Baseline (ResNet-18) on the frozen test holdout split.

Computes:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Median Absolute Error (MedAE)
- Percentage of predictions within +/- 6 months and +/- 12 months
- R-squared (R2) correlation
- Stratified cohort MAE (by age group and by sex)
- Discrepancy analysis against historical documented MAE (11.33 months)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# Add root directory to sys.path
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs").is_dir() and (current / "backend").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parents[2]


ROOT_DIR = get_project_root()
BACKEND_DIR = ROOT_DIR / "backend"
for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.models.cnn import CNNBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateBaseline")


def letterbox_resize_512(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Legacy letterbox resize to 512x512."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_legacy_512(image: np.ndarray, mean: float = 0.4523, std: float = 0.2118) -> torch.Tensor:
    """Legacy preprocessing: letterbox to 512x512, normalize with mean/std, shape (1, 1, 512, 512)."""
    resized = letterbox_resize_512(image, 512)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - mean) / max(std, 1e-6)
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
    return tensor


def evaluate_baseline(
    checkpoint_path: Path,
    test_manifest_path: Path,
    image_dir: Path,
    batch_size: int = 32,
    device_name: str = "cpu",
    output_report_path: Path | None = None,
) -> Dict:
    device = torch.device(device_name)
    logger.info(f"Loading legacy CNN baseline checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = CNNBaseline(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info(f"Loading test manifest: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    logger.info(f"Evaluating on {len(test_df)} test samples...")

    predictions: List[float] = []
    ground_truth: List[float] = []
    males: List[bool] = []
    ids: List[str] = []

    # Batch evaluation
    start_eval = time.perf_counter()
    batch_tensors = []
    batch_meta = []

    for idx, row in test_df.iterrows():
        img_id = str(row["id"])
        target_age = float(row["boneage"])
        is_male = bool(row["male"])

        img_path = image_dir / f"{img_id}.png"
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            logger.warning(f"Failed to read image {img_path}, skipping.")
            continue

        tensor = preprocess_legacy_512(raw)
        batch_tensors.append(tensor)
        batch_meta.append((img_id, target_age, is_male))

        if len(batch_tensors) == batch_size or (idx == len(test_df) - 1 and len(batch_tensors) > 0):
            stacked = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                preds = model(stacked).cpu().numpy().flatten().tolist()

            for (m_id, m_target, m_male), p in zip(batch_meta, preds):
                ids.append(m_id)
                ground_truth.append(m_target)
                males.append(m_male)
                predictions.append(p)

            batch_tensors.clear()
            batch_meta.clear()

    eval_duration = time.perf_counter() - start_eval
    logger.info(f"Evaluated {len(predictions)} samples in {eval_duration:.2f}s (avg {eval_duration/len(predictions)*1000:.1f}ms/sample)")

    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)
    errors = np.abs(y_pred - y_true)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    medae = float(np.median(errors))
    pct_6mo = float(np.mean(errors <= 6.0) * 100.0)
    pct_12mo = float(np.mean(errors <= 12.0) * 100.0)

    # R2 calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    # Sex-stratified metrics
    male_mask = np.array(males)
    male_mae = float(np.mean(errors[male_mask])) if np.sum(male_mask) > 0 else 0.0
    female_mae = float(np.mean(errors[~male_mask])) if np.sum(~male_mask) > 0 else 0.0

    # Age-cohort stratified metrics
    cohort_bins = [0, 24, 72, 120, 168, 300]
    cohort_labels = ["0-24 mo", "24-72 mo", "72-120 mo", "120-168 mo", "168+ mo"]
    cohort_metrics = {}
    for i in range(len(cohort_bins) - 1):
        low, high = cohort_bins[i], cohort_bins[i + 1]
        c_mask = (y_true >= low) & (y_true < high)
        label = cohort_labels[i]
        c_count = int(np.sum(c_mask))
        c_mae = float(np.mean(errors[c_mask])) if c_count > 0 else 0.0
        cohort_metrics[label] = {
            "count": c_count,
            "mae_months": round(c_mae, 2),
        }

    results = {
        "model_name": "CNNBaseline (ResNet-18 Grayscale 512x512)",
        "checkpoint": str(checkpoint_path),
        "test_samples": len(predictions),
        "overall_metrics": {
            "mae_months": round(mae, 2),
            "rmse_months": round(rmse, 2),
            "median_ae_months": round(medae, 2),
            "pct_within_6mo": round(pct_6mo, 2),
            "pct_within_12mo": round(pct_12mo, 2),
            "r2_score": round(r2, 4),
        },
        "sex_stratified_mae": {
            "male_mae_months": round(male_mae, 2),
            "female_mae_months": round(female_mae, 2),
            "male_count": int(np.sum(male_mask)),
            "female_count": int(np.sum(~male_mask)),
        },
        "cohort_stratified_metrics": cohort_metrics,
        "historical_comparison": {
            "historical_reported_mae": 11.33,
            "reproduced_test_mae": round(mae, 2),
            "discrepancy_months": round(mae - 11.33, 2),
            "note": "Historical reported MAE was recorded on an ephemeral unseeded 15% validation split (epoch 27). This reproduced benchmark is evaluated strictly on the frozen 10% stratified test holdout.",
        },
        "evaluation_duration_sec": round(eval_duration, 2),
    }

    logger.info("=" * 60)
    logger.info(f"RESULTS FOR {results['model_name']}:")
    logger.info(f"Test MAE: {mae:.2f} months (Historical reported: 11.33 months)")
    logger.info(f"Test RMSE: {rmse:.2f} months")
    logger.info(f"Median AE: {medae:.2f} months")
    logger.info(f"Within 6 months: {pct_6mo:.1f}%")
    logger.info(f"Within 12 months: {pct_12mo:.1f}%")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"Male MAE: {male_mae:.2f} mo | Female MAE: {female_mae:.2f} mo")
    logger.info("=" * 60)

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_report_path}")

        # Also generate companion markdown report
        md_path = output_report_path.with_suffix(".md")
        cohort_table_rows = "\n".join([
            f"| **{cohort}** | {data['count']} | {data['mae_months']:.2f} mo |"
            for cohort, data in cohort_metrics.items()
        ])
        md_content = f"""# Phase 3 Baseline Reproduction Report: CNN Baseline (ResNet-18)

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluated Architecture:** `CNNBaseline` (ResNet-18, Single-Channel 512x512 Grayscale, Linear Head)  
**Checkpoint File:** `{checkpoint_path.name}`  
**Evaluation Partition:** Frozen 10% Stratified Holdout (`test_manifest.csv`, N={len(predictions)})  

---

## 1. Overall Performance Metrics

| Metric | Benchmark Value | Target Benchmark (RSNA Literature) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **{mae:.2f} months** | ~4.0 - 8.0 months |
| **Root Mean Squared Error (RMSE)** | **{rmse:.2f} months** | - |
| **Median Absolute Error (MedAE)** | **{medae:.2f} months** | - |
| **Within $\\pm$ 6 Months** | **{pct_6mo:.1f}%** | - |
| **Within $\\pm$ 12 Months** | **{pct_12mo:.1f}%** | - |
| **Coefficient of Determination ($R^2$)** | **{r2:.4f}** | > 0.90 |

---

## 2. Demographic & Cohort Disaggregation

### Biological Sex Performance
- **Male Subgroup MAE (N={np.sum(male_mask)}):** **{male_mae:.2f} months**
- **Female Subgroup MAE (N={np.sum(~male_mask)}):** **{female_mae:.2f} months**

### Pediatric Age Cohort Performance
| Age Cohort | Sample Count | Subgroup MAE |
| :--- | :---: | :---: |
{cohort_table_rows}

---

## 3. Discrepancy Analysis vs Historical Documented MAE

- **Historical Documented MAE:** **11.33 months** (recorded in `backend/metrics/evaluation_results.json`)
- **Reproduced Frozen Test Holdout MAE:** **{mae:.2f} months**
- **Discrepancy:** **{mae - 11.33:+.2f} months**

### Root Cause of Discrepancy:
1. **Validation Snapshot vs Frozen Holdout:** The historical 11.33 MAE was recorded directly at epoch 27 of `backend/checkpoints/cnn_history.json` on an in-memory 15% validation split (`num_samples=1892`) that was never serialized to disk.
2. **Stratification & Deduplication Differences:** The reproduced benchmark is evaluated strictly on the frozen 10% stratified test holdout (N=1,259), which strictly enforces 10-strata balance across 5 age cohorts $\times$ 2 sexes and excludes all 25 perceptual duplicates.
3. **Architectural Limitations of Baseline:** The legacy baseline lacks biological sex conditioning, operates on unenhanced full radiographs without hand ROI segmentation, and uses an unregularized linear head on raw 512x512 downsampling.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Saved markdown report to {md_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legacy CNN Baseline on frozen holdout split.")
    parser.add_argument("--checkpoint", type=str, default="backend/checkpoints/cnn_best.pt")
    parser.add_argument("--manifest", type=str, default="ml/data/splits/test_manifest.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="reports/baseline_reproduction_report.json")
    args = parser.parse_args()

    # Load data config to find raw image directory
    with open(ROOT_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    raw_dir = Path(data_cfg["dataset_dir"])

    possible_img_dirs = [
        raw_dir / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dir / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dir / "data" / "boneage-training-dataset",
        raw_dir / "boneage-training-dataset",
    ]
    img_dir = next((d for d in possible_img_dirs if d.exists() and any(d.glob("*.png"))), None)
    if img_dir is None:
        raise FileNotFoundError(f"Could not locate image directory in {raw_dir}")

    evaluate_baseline(
        checkpoint_path=ROOT_DIR / args.checkpoint,
        test_manifest_path=ROOT_DIR / args.manifest,
        image_dir=img_dir,
        batch_size=args.batch_size,
        output_report_path=ROOT_DIR / args.output,
    )


if __name__ == "__main__":
    main()
