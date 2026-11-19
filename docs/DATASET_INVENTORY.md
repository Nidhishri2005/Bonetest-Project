# Phase 0: RSNA Dataset Inventory & Audit

**Date:** 2026-11-17  
**Auditor:** Senior ML Engineer & Full-Stack Systems Architect  
**Config Path:** `configs/data.yaml`  
**Dataset Location:** `C:\Users\Manya\Desktop\NMAMIT\major project\BoneAgePrediction-1\data`

---

## 1. Overview & File Inventory

The external RSNA Pediatric Bone Age dataset directory was scanned and reconciled.

### CSV Manifests

| CSV Filename | Row Count | Columns | Has Age Label? | Age Column Name |
| :--- | :---: | :--- | :---: | :--- |
| `boneage-training-dataset.csv` | 12,611 | `['id', 'boneage', 'male']` | **Yes** | `boneage` (months) |
| `boneage-test-dataset.csv` | 200 | `['Case ID', 'Sex']` | **No** | None |

### Image Repositories

| Directory Name | Resolved Image Path | Image Count | Extensions Found |
| :--- | :--- | :---: | :--- |
| `boneage-training-dataset` | `.../boneage-training-dataset/boneage-training-dataset` | 12,611 | `.png` |
| `boneage-test-dataset` | `.../boneage-test-dataset/boneage-test-dataset` | 200 | `.png` |

---

## 2. 1-to-1 Image & ID Reconciliation

### Training Set Reconciliation
- **CSV Total Entries:** 12,611
- **Disk Total Images:** 12,611
- **CSV IDs Lacking Images:** **0**
- **Disk Images Lacking CSV IDs:** **0**
- **Integrity Status:** **100% Complete & Reconciled.** Every CSV record maps to an existing image file.

### Test Set Reconciliation
- **Test CSV Total Entries:** 200
- **Disk Total Test Images:** 200
- **Test CSV IDs Lacking Images:** **0**
- **Disk Test Images Lacking CSV IDs:** **0**
- **Integrity Status:** **100% Complete & Reconciled.**

---

## 3. Label Statistics & Demographics

### Training Data (`boneage-training-dataset.csv`)
- **Total Valid Samples:** 12,611
- **Missing / Null Values:** 0
- **Bone Age Range:** 1.0 to 228.0 months (~0.1 to 19.0 years)
- **Mean Bone Age:** 127.32 months (10.61 years)
- **Median Bone Age:** 132.00 months (11.00 years)
- **Standard Deviation:** 41.18 months (3.43 years)
- **Sex Distribution:**
  - Male (`male = True`): 6,833 samples (54.18%)
  - Female (`male = False`): 5,778 samples (45.82%)

### Test Data (`boneage-test-dataset.csv`)
- **Total Samples:** 200
- **Missing / Null Values:** 0
- **Sex Distribution:**
  - Male (`Sex = 'M'`): 100 samples (50.0%)
  - Female (`Sex = 'F'`): 100 samples (50.0%)
- **Ground-Truth Labels:** **Unavailable** (official competition unlabelled evaluation set).

---

## 4. Test Set Strategy Decision: CASE B

Per the evaluation protocol:
- **CASE A:** A second CSV with age labels exists $\rightarrow$ Use as final test set.
- **CASE B:** Only the training CSV is labelled $\rightarrow$ Construct a frozen 10% test set from the labelled training data.

### Determination: **CASE B**
Because `boneage-test-dataset.csv` contains only `Case ID` and `Sex` without ground-truth bone ages, the official test set cannot be used for quantitative MAE/RMSE benchmarking.
Therefore, in **Phase 2 (Data)**:
1. Perceptual-hash deduplication will be applied to the 12,611 labelled images.
2. A deterministic **80% Train / 10% Val / 10% Test** split will be computed, stratified jointly by `age_bin` $\times$ `sex`.
3. The resulting 10% test manifest (`test_manifest.csv`) will be frozen with SHA-256 file hashes.
4. All reports, model cards, and dashboard views will state:
   > *"The evaluation benchmark uses a frozen 10% stratified holdout split from the RSNA training cohort; it is NOT the unlabelled official RSNA competition test set."*

---

## 5. Legacy Model Split Recovery

- **Split Logic in Code:** `backend/ml/dataset.py` contains `train_val_test_split()` with random seed 42 and age bin stratification.
- **Validation Set Size:** Historical evaluation results report `num_samples = 1892`, which corresponds to $\approx 15\%$ of 12,611.
- **Persistent Split Files:** **None exist on disk.** No index manifest was saved alongside the legacy checkpoints.
- **Protocol:** The historical numbers (CNN: 11.33, CNN+RF: 11.04, Multimodal: 11.39) will be marked as **"Split historical in-memory (seed 42, 15% validation); no frozen ID manifest saved on disk; non-comparable to new frozen splits until re-evaluated."**
- In Phase 3, the legacy CNN baseline will be re-evaluated directly on the newly created frozen test split to establish a strictly comparable baseline.
