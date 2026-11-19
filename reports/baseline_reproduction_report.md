# Phase 3 Baseline Reproduction Report: CNN Baseline (ResNet-18)

**Generated:** 2026-11-17 22:33:45  
**Evaluated Architecture:** `CNNBaseline` (ResNet-18, Single-Channel 512x512 Grayscale, Linear Head)  
**Checkpoint File:** `cnn_best.pt`  
**Evaluation Partition:** Frozen 10% Stratified Holdout (`test_manifest.csv`, N=1259)  

---

## 1. Overall Performance Metrics

| Metric | Benchmark Value | Target Benchmark (RSNA Literature) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **10.92 months** | ~4.0 - 8.0 months |
| **Root Mean Squared Error (RMSE)** | **14.09 months** | - |
| **Median Absolute Error (MedAE)** | **9.19 months** | - |
| **Within $\pm$ 6 Months** | **34.5%** | - |
| **Within $\pm$ 12 Months** | **63.3%** | - |
| **Coefficient of Determination ($R^2$)** | **0.8801** | > 0.90 |

---

## 2. Demographic & Cohort Disaggregation

### Biological Sex Performance
- **Male Subgroup MAE (N=681):** **11.33 months**
- **Female Subgroup MAE (N=578):** **10.43 months**

### Pediatric Age Cohort Performance
| Age Cohort | Sample Count | Subgroup MAE |
| :--- | :---: | :---: |
| **0-24 mo** | 9 | 24.80 mo |
| **24-72 mo** | 124 | 9.43 mo |
| **72-120 mo** | 295 | 10.48 mo |
| **120-168 mo** | 632 | 10.93 mo |
| **168+ mo** | 199 | 11.84 mo |

---

## 3. Discrepancy Analysis vs Historical Documented MAE

- **Historical Documented MAE:** **11.33 months** (recorded in `backend/metrics/evaluation_results.json`)
- **Reproduced Frozen Test Holdout MAE:** **10.92 months**
- **Discrepancy:** **-0.41 months**

### Root Cause of Discrepancy:
1. **Validation Snapshot vs Frozen Holdout:** The historical 11.33 MAE was recorded directly at epoch 27 of `backend/checkpoints/cnn_history.json` on an in-memory 15% validation split (`num_samples=1892`) that was never serialized to disk.
2. **Stratification & Deduplication Differences:** The reproduced benchmark is evaluated strictly on the frozen 10% stratified test holdout (N=1,259), which strictly enforces 10-strata balance across 5 age cohorts $	imes$ 2 sexes and excludes all 25 perceptual duplicates.
3. **Architectural Limitations of Baseline:** The legacy baseline lacks biological sex conditioning, operates on unenhanced full radiographs without hand ROI segmentation, and uses an unregularized linear head on raw 512x512 downsampling.
