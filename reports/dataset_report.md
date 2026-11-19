# RSNA Pediatric Bone Age Dataset Report

**Generated:** 2026-11-17 21:18:47  
**Evaluation Protocol:** CASE B (Frozen 10% Stratified Holdout)  
**Total Images Processed:** 12611  
**Unique Clean Samples:** 12586  
**Identified Duplicates:** 25  

---

## 1. Split Partitioning & Integrity

| Split | Sample Count | Percentage | Male Count (%) | Female Count (%) | Mean Age (mo) | Std Age (mo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 10,068 | 80.0% | 5,452 (54.2%) | 4,616 (45.8%) | 127.29 | 41.25 |
| **Validation** | 1,259 | 10.0% | 682 (54.2%) | 577 (45.8%) | 127.42 | 41.11 |
| **Test (Frozen Holdout)** | 1,259 | 10.0% | 681 (54.1%) | 578 (45.9%) | 127.39 | 40.69 |
| **Total Clean** | 12,586 | 100.0% | 6,815 (54.1%) | 5,771 (45.9%) | 127.31 | 41.18 |

---

## 2. Frozen Split Manifest Verification

All split manifests have been locked to disk with SHA-256 integrity verification:

- **Train Manifest:** `ml/data/splits/train_manifest.csv` (`SHA-256: 718bfc88b4b7f68617b5e3696e37f59c54dc162bc354f0fac25ac16a4164e347`)
- **Validation Manifest:** `ml/data/splits/val_manifest.csv` (`SHA-256: c8e3270dd10690cf531c367b04071a4ad4558b3813552e6158457fd7f31effd7`)
- **Test Manifest:** `ml/data/splits/test_manifest.csv` (`SHA-256: 70a26d0db0e7c9ebab855a64d918be1f9299280ce623444af902af4eca4ce823`)

---

## 3. Preprocessing & Normalization Statistics

Computed strictly on the training partition:
- **Preprocessing Pipeline:** Otsu Hand ROI Crop -> CLAHE (clip=2.0, grid=8x8) -> Aspect-Ratio Letterbox (384x384)
- **Train Normalization Mean:** `0.1999`
- **Train Normalization Std:** `0.2229`
- **Train Age Standardization:** Mean = `127.29` mo, Std = `41.25` mo

---

## 4. Methodological Limitations

- **Absence of Patient Identifiers:** The official RSNA dataset does not bundle patient IDs (only individual study image IDs). Multiple views or longitudinal exams of the same patient could theoretically exist; strict joint stratification by age cohort x sex minimizes distribution mismatch.
- **Test Set Disclaimer:** The quantitative evaluation test set used in all reports is the frozen 10% stratified holdout (1259 samples) and is NOT the official unlabelled competition test set.
