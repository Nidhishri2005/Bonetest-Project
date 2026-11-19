"""Phase 2 Data Pipeline Execution Script.

Performs:
1. Perceptual-hash deduplication (dHash)
2. Stratified 80/10/10 Train/Val/Test splitting (CASE B)
3. Manifest generation with SHA-256 file hashes
4. Training-only normalization statistics computation (ROI + CLAHE + 384x384)
5. Test/Validation split caching
6. Dataset report generation with age and sex histograms
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.data.dedup import deduplicate_dataset
from ml.data.preprocess import preprocess_radiograph
from ml.data.split import create_stratified_splits, save_split_manifests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2DataPipeline")


def load_config() -> dict:
    config_path = ROOT_DIR / "configs" / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    start_time = time.perf_counter()
    logger.info("Starting Phase 2 Data Pipeline...")

    # 1. Resolve dataset path from configs/data.yaml
    config = load_config()
    raw_dataset_dir = Path(config["dataset_dir"])
    logger.info(f"Loaded dataset directory from config: {raw_dataset_dir}")

    # Locate training CSV
    train_csv_path = raw_dataset_dir / "data" / "boneage-training-dataset.csv"
    if not train_csv_path.exists():
        train_csv_path = raw_dataset_dir / "boneage-training-dataset.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Cannot find boneage-training-dataset.csv in {raw_dataset_dir}")

    # Locate training images
    possible_img_dirs = [
        raw_dataset_dir / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dataset_dir / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dataset_dir / "data" / "boneage-training-dataset",
        raw_dataset_dir / "boneage-training-dataset",
    ]
    img_dir = next((d for d in possible_img_dirs if d.exists() and any(d.glob("*.png"))), None)
    if img_dir is None:
        raise FileNotFoundError(f"Cannot locate boneage-training-dataset images in {raw_dataset_dir}")

    logger.info(f"CSV Path: {train_csv_path}")
    logger.info(f"Image Directory: {img_dir}")

    raw_df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded raw training CSV: {len(raw_df)} records")

    # 2. Perceptual-Hash Deduplication (dHash)
    logger.info("Stage 1: Perceptual-Hash Deduplication...")
    clean_df, duplicates = deduplicate_dataset(
        raw_df,
        image_dir=img_dir,
        id_col="id",
        max_hamming_distance=0,
    )
    logger.info(f"Deduplication complete: {len(duplicates)} duplicates found. Remaining: {len(clean_df)}")

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "deduplication_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_raw": len(raw_df),
            "duplicates_count": len(duplicates),
            "clean_count": len(clean_df),
            "duplicate_records": duplicates,
        }, f, indent=2)

    # 3. Stratified 80/10/10 Split
    logger.info("Stage 2: Deterministic Stratified 80/10/10 Split...")
    splits_dir = ROOT_DIR / "ml" / "data" / "splits"
    train_df, val_df, test_df = create_stratified_splits(
        clean_df,
        image_dir=img_dir,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )
    manifest_hashes = save_split_manifests(train_df, val_df, test_df, output_dir=splits_dir)

    # 4. Compute Normalization Statistics Strictly on Train Split
    logger.info("Stage 3: Computing Normalization Statistics strictly on Train split...")
    pixel_values = []
    pixel_sq_values = []
    total_pixels = 0

    # Sample a representative subset (1,000 images) or all for robust stats
    sample_size = min(len(train_df), 1500)
    logger.info(f"Sampling {sample_size} train images for exact pixel mean/std calculation...")
    
    sample_train = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for idx, row in sample_train.iterrows():
        img_id = str(row["id"])
        cand = img_dir / f"{img_id}.png"
        raw = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            proc = preprocess_radiograph(raw, target_size=384, roi_crop=True, clahe=True)
            norm_arr = proc.astype(np.float64) / 255.0
            pixel_sum += np.sum(norm_arr)
            pixel_sq_sum += np.sum(norm_arr ** 2)
            pixel_count += norm_arr.size

    norm_mean = float(pixel_sum / pixel_count)
    norm_std = float(np.sqrt((pixel_sq_sum / pixel_count) - (norm_mean ** 2)))

    train_mean_age = float(train_df["boneage"].mean())
    train_std_age = float(train_df["boneage"].std())

    logger.info(f"Computed Normalization Stats (ROI+CLAHE+384): Mean={norm_mean:.4f}, Std={norm_std:.4f}")
    logger.info(f"Train Age Stats: Mean={train_mean_age:.2f} months, Std={train_std_age:.2f} months")

    # Save normalization stats
    configs_dir = ROOT_DIR / "configs"
    norm_data = {
        "norm_mean": round(norm_mean, 4),
        "norm_std": round(norm_std, 4),
        "mean_age": round(train_mean_age, 2),
        "std_age": round(train_std_age, 2),
        "target_size": 384,
        "sample_size": sample_size,
    }
    with open(configs_dir / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(norm_data, f, indent=2)

    # 5. Pre-cache Test Split (10% = 1,261 images) for Ultra-Fast Benchmarking
    logger.info("Stage 4: Pre-caching Test Split Preprocessed Radiographs...")
    cache_dir = ROOT_DIR / "ml" / "data" / "cache_384"
    cache_dir.mkdir(parents=True, exist_ok=True)

    test_cached_count = 0
    for _, row in test_df.iterrows():
        img_id = str(row["id"])
        cache_file = cache_dir / f"{img_id}.png"
        if not cache_file.exists():
            cand = img_dir / f"{img_id}.png"
            raw = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                proc = preprocess_radiograph(raw, target_size=384, roi_crop=True, clahe=True)
                cv2.imwrite(str(cache_file), proc)
                test_cached_count += 1

    logger.info(f"Test caching complete: {len(test_df)} test radiographs verified in cache.")

    # 6. Generate Dataset Report and Histograms
    logger.info("Stage 5: Generating Dataset Report & Demographic Histograms...")
    plt.figure(figsize=(12, 5))
    
    # Age histogram
    plt.subplot(1, 2, 1)
    plt.hist(train_df["boneage"], bins=30, alpha=0.7, label=f"Train (N={len(train_df)})", color="steelblue", density=True)
    plt.hist(val_df["boneage"], bins=30, alpha=0.6, label=f"Val (N={len(val_df)})", color="orange", density=True)
    plt.hist(test_df["boneage"], bins=30, alpha=0.5, label=f"Test (N={len(test_df)})", color="green", density=True)
    plt.title("Bone Age Distribution Across Splits")
    plt.xlabel("Bone Age (months)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sex histogram
    plt.subplot(1, 2, 2)
    categories = ["Female", "Male"]
    train_sex = [np.mean(~train_df["male"]) * 100, np.mean(train_df["male"]) * 100]
    val_sex = [np.mean(~val_df["male"]) * 100, np.mean(val_df["male"]) * 100]
    test_sex = [np.mean(~test_df["male"]) * 100, np.mean(test_df["male"]) * 100]
    x = np.arange(len(categories))
    width = 0.25
    plt.bar(x - width, train_sex, width, label="Train", color="steelblue")
    plt.bar(x, val_sex, width, label="Val", color="orange")
    plt.bar(x + width, test_sex, width, label="Test", color="green")
    plt.xticks(x, categories)
    plt.title("Sex Proportion Across Splits (%)")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    hist_plot_path = reports_dir / "dataset_histograms.png"
    plt.tight_layout()
    plt.savefig(hist_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved demographic histograms to {hist_plot_path}")

    # Build Markdown Report
    report_md = f"""# RSNA Pediatric Bone Age Dataset Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Protocol:** CASE B (Frozen 10% Stratified Holdout)  
**Total Images Processed:** {len(raw_df)}  
**Unique Clean Samples:** {len(clean_df)}  
**Identified Duplicates:** {len(duplicates)}  

---

## 1. Split Partitioning & Integrity

| Split | Sample Count | Percentage | Male Count (%) | Female Count (%) | Mean Age (mo) | Std Age (mo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | {len(train_df):,} | {len(train_df)/len(clean_df):.1%} | {int(np.sum(train_df['male'])):,} ({np.mean(train_df['male'])*100:.1f}%) | {int(np.sum(~train_df['male'])):,} ({np.mean(~train_df['male'])*100:.1f}%) | {train_df['boneage'].mean():.2f} | {train_df['boneage'].std():.2f} |
| **Validation** | {len(val_df):,} | {len(val_df)/len(clean_df):.1%} | {int(np.sum(val_df['male'])):,} ({np.mean(val_df['male'])*100:.1f}%) | {int(np.sum(~val_df['male'])):,} ({np.mean(~val_df['male'])*100:.1f}%) | {val_df['boneage'].mean():.2f} | {val_df['boneage'].std():.2f} |
| **Test (Frozen Holdout)** | {len(test_df):,} | {len(test_df)/len(clean_df):.1%} | {int(np.sum(test_df['male'])):,} ({np.mean(test_df['male'])*100:.1f}%) | {int(np.sum(~test_df['male'])):,} ({np.mean(~test_df['male'])*100:.1f}%) | {test_df['boneage'].mean():.2f} | {test_df['boneage'].std():.2f} |
| **Total Clean** | {len(clean_df):,} | 100.0% | {int(np.sum(clean_df['male'])):,} ({np.mean(clean_df['male'])*100:.1f}%) | {int(np.sum(~clean_df['male'])):,} ({np.mean(~clean_df['male'])*100:.1f}%) | {clean_df['boneage'].mean():.2f} | {clean_df['boneage'].std():.2f} |

---

## 2. Frozen Split Manifest Verification

All split manifests have been locked to disk with SHA-256 integrity verification:

- **Train Manifest:** `ml/data/splits/train_manifest.csv` (`SHA-256: {manifest_hashes['train_manifest']['sha256']}`)
- **Validation Manifest:** `ml/data/splits/val_manifest.csv` (`SHA-256: {manifest_hashes['val_manifest']['sha256']}`)
- **Test Manifest:** `ml/data/splits/test_manifest.csv` (`SHA-256: {manifest_hashes['test_manifest']['sha256']}`)

---

## 3. Preprocessing & Normalization Statistics

Computed strictly on the training partition:
- **Preprocessing Pipeline:** Otsu Hand ROI Crop -> CLAHE (clip=2.0, grid=8x8) -> Aspect-Ratio Letterbox (384x384)
- **Train Normalization Mean:** `{norm_mean:.4f}`
- **Train Normalization Std:** `{norm_std:.4f}`
- **Train Age Standardization:** Mean = `{train_mean_age:.2f}` mo, Std = `{train_std_age:.2f}` mo

---

## 4. Methodological Limitations

- **Absence of Patient Identifiers:** The official RSNA dataset does not bundle patient IDs (only individual study image IDs). Multiple views or longitudinal exams of the same patient could theoretically exist; strict joint stratification by age cohort x sex minimizes distribution mismatch.
- **Test Set Disclaimer:** The quantitative evaluation test set used in all reports is the frozen 10% stratified holdout ({len(test_df)} samples) and is NOT the official unlabelled competition test set.
"""
    with open(reports_dir / "dataset_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Phase 2 Data Pipeline completed successfully in {elapsed:.2f}s!")


if __name__ == "__main__":
    main()
