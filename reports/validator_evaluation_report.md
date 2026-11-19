# One-Class OOD Hand X-Ray Validation Gatekeeper Evaluation Report

**Method:** One-Class Out-of-Distribution (OOD) k-Nearest Neighbors ($k=5$, cosine distance) on L2-normalized 512-d global-pooled features from a frozen ImageNet-pretrained `ResNet-18` (`torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
**Shared Preprocessing:** `app.models.validator.preprocess_validator_image` (3-channel RGB conversion, aspect-ratio-preserving $224 \times 224$ letterbox resize, ImageNet mean/std normalization).
**Artifact:** [`backend/checkpoints/validator_ood.npz`](file:///c:/bonetest/backend/checkpoints/validator_ood.npz) (`8,672,200` bytes).
**Approved Threshold:** `0.122576` (99th percentile of kNN $k=5$ cosine distance on the 500 stratified held-out RSNA radiographs; configurable via `validator_ood_threshold` in [`backend/app/core/config.py`](file:///c:/bonetest/backend/app/core/config.py)).

---

## 1. Sample Sizes & End-to-End Measured Rates (Stage 1 + Stage 2 + Stage 3)

| Dataset | Sample Size ($N$) | Role / Independence | Accepted | Rejected | Deciding Stage Breakdown |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Stratified RSNA Fit Bank** | `4,138` | Reference bank (`138` in `0-24m`, `1,000` each in `24-72m`, `72-120m`, `120-168m`, `168+m`) | — | — | Used to build `knn_bank` and Ledoit-Wolf shrinkage covariance ($\alpha = 0.005634$) |
| **Stratified RSNA Held-Out Set** | `500` | Threshold calibration (`25` in `0-24m`, `119/119/119/118` in other 4 bins) — **not independent** | **495 / 500 (99.00%)** *(Stage 2)* | **5 / 500 (1.00%)** | `4` in `infant_0_24` (`21/25 = 84.00%`), `1` in `early_child_24_72` (`118/119 = 99.16%`), `1` in `mid_child_72_120` (`118/119 = 99.16%`), `119/119 (100%)` in `120-168m`, `118/118 (100%)` in `168+m` |
| **a) `tests/validator/should_accept`** | `30` | Held-out real RSNA radiographs (excluded from fit & calibration) | **29 / 30 (96.67%)** | **1 / 30 (3.33%)** | `stage2_ood_knn`: `1` (`4244.png`, 18-mo female infant, score `0.128517 > 0.122576`) |
| **b) `tests/validator/should_reject`** | `23` | Real negative images (`8` non-hand X-rays, `12` photos/animals incl. `4` adult-hand photos, `3` screenshots) | **0 / 23 (0.00%)** | **23 / 23 (100.00%)** | `stage1_rules`: `14` (strong-color gate); `stage2_ood_knn`: `9` *(all 23 also score $\ge 0.157954 > 0.122576$ in Stage 2; ROC AUC = `1.000000`)* |
| **c) `200 RSNA Check Images`** | `200` | Excluded RSNA images used to check Stage 1/3 — **not independent** | **199 / 200 (99.50%)** | **1 / 200 (0.50%)** | `stage2_ood_knn`: `0` (`200/200` pass Stage 1 & Stage 2); `stage3_coverage`: `1` (`3566.png`, kNN score `0.040385`, coverage `0.0242 < 0.03`) |

---

## 2. Known Limitations

1. **Infants (`0–24` months) Are Rejected More Often:**
   - Infants account for only `1.53%` (`193 / 12,611`) of the RSNA training CSV and have distinct wrist/hand morphology (unossified carpal centers, higher soft-tissue ratio).
   - Sample sizes in this age bin are small (`138` in the fit bank, `25` in the held-out calibration set, and `4` in `tests/validator/should_accept`).
   - At the 99% threshold (`0.122576`), the accept rate for `0–24` months is **`21/25 (84.00%)`** on the held-out set and **`3/4 (75.00%)`** on `should_accept` (`4244.png`, 18 mo, rejected with kNN score `0.128517`), compared to **`99.16%–100.00%`** across children aged `24+` months.
2. **Small Negative Test Set:**
   - `tests/validator/should_reject` currently contains $N = 23$ images (`8` non-hand X-rays, `8` animal/general photos, `4` optical adult-hand photos, `3` screenshots).
3. **Untested Edge Cases (Adult Hand X-Rays & Photographs of X-Ray Films):**
   - Radiographs of adult hands ($> 20$ years) and camera/smartphone photographs of physical X-ray films on lightboxes or monitors are untested unless added to `tests/validator/should_reject`.
