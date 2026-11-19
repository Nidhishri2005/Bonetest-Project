# Model Card: Pediatric Bone Age Prediction System (`EfficientNetB3SexFusion`)

This Model Card follows the transparency and reporting framework proposed by Mitchell et al. (2019).

---

## 1. Model Details

- **Model Name:** Pediatric Bone Age Prediction Architecture (`EfficientNetB3SexFusion`)
- **Version:** 2.0.0 (Production Modernization & Safety Hardening)
- **Model Type:** Multimodal Deep Convolutional Neural Network (Image Radiography + Biological Sex Embedding Fusion)
- **Backbone Architecture:** `torchvision.models.efficientnet_b3` (ImageNet-1k pretrained weights, 1536-dimensional feature representation)
- **Sex Fusion:** 32-dimensional learned embedding vector via `nn.Embedding(2, 32)` concatenated with pooled visual representations (total: 1568 features)
- **Regression Head:** Dual-stage MLP with Batch Normalization, Dropout ($p=0.3, 0.2$), and ReLU activations mapping to standardized scalar age
- **Loss Function:** Smooth L1 / Huber Loss ($\delta = 1.0$) on standardized age targets ($z = \frac{\text{age} - \mu}{\sigma}$)
- **Inference Runtime Engine:** ONNX Runtime CPU (`CPUExecutionProvider`) with intra-op thread bounding ($\le 2$ cores)
- **License:** MIT License (Research & Academic Evaluation)
- **Developers:** DeepMind Antigravity Engineering Team

---

## 2. Intended Clinical & Research Scope

### Primary Intended Use
Automated, objective estimation of pediatric skeletal maturity (bone age expressed in real months and years) from left-hand and wrist radiographs in pediatric patients aged 0–19 years, aligning with clinical standards (Greulich & Pyle atlas and Tanner-Whitehouse TW3 scoring).

### Target Audience
Pediatric radiologists, pediatric endocrinologists, orthopedic researchers, and clinical decision support system evaluators.

### Out-of-Scope & Prohibited Uses
- **Autonomous Diagnosis:** This software is a **Research Prototype** and must **NOT** be used as a standalone or autonomous diagnostic system without direct oversight by a board-certified radiologist or clinician.
- **Adult Radiography:** Not intended or validated for skeletal age assessment in adults ($> 20$ years).
- **Non-Hand Anatomical Radiographs:** Chest, skull, pelvis, spine, or dental radiographs are strictly prohibited and automatically rejected by the multi-stage gatekeeper.
- **Natural Photography:** Everyday photographs, screenshots, animal imagery, and non-medical images are strictly rejected.

---

## 3. Defense-in-Depth Validation Gatekeeper

To eliminate dangerous false predictions on out-of-distribution or non-hand images, the system enforces a mandatory **Three-Stage Gatekeeper Pipeline** before any inference or explainability algorithm runs:

```
Upload Image
     │
     ▼
[Stage 1: Rule-Based Integrity & Strong-Color Gate]
  ├─ Magic Bytes Signature Check (PNG, JPEG, BMP, TIFF, WebP)
  ├─ OpenCV Byte Decode & Corruption Trap
  ├─ Minimum Spatial Dimension Constraint (min 32x32)
  ├─ Aspect Ratio Bounds (0.20 <= AR <= 5.0)
  ├─ Signal Variance Std Check (std >= 1.0, rejects flat / blank images)
  └─ Strong-Color Gate (rejects mean RGB spread > 25.0 or >20% pixels with RGB spread > 40)
     │ PASSED
     ▼
[Stage 2: One-Class OOD kNN (k=5, Cosine Distance) Gate]
  ├─ Frozen ImageNet-pretrained ResNet-18 (512-d global-pooled L2-normalized features)
  ├─ Shared Preprocessing: preprocess_validator_image (3-ch RGB, 224x224 letterbox, ImageNet norm)
  ├─ Stratified RSNA Reference Bank (backend/checkpoints/validator_ood.npz, N_fit = 4,138)
  │   ├─ infant_0_24: 138 (all remaining eligible)
  │   └─ 24-72, 72-120, 120-168, 168+ mo: 1,000 each
  ├─ Startup Integrity Check: fails loudly if validator_ood.npz is missing or config mismatches
  └─ 99% Calibrated Threshold (on N_holdout = 500 stratified RSNA images):
      s_knn(z) = mean of 5 smallest cosine distances <= 0.122576
     │ PASSED
     ▼
[Stage 3: Tissue-Sensitive Anatomical Mask Coverage Gate (5% Border Ignored)]
  ├─ Central 90%x90% ROI (ignores 5% border on all 4 sides)
  ├─ [P2, P98] percentile contrast stretch + Otsu thresholding & connected components
  └─ Anatomical Frame Coverage Check (0.03 <= coverage <= 0.98)
     │ PASSED
     ▼
Execute Model Inference & Grad-CAM++ Explainability
```

### Measured Validation Performance & Sample Sizes
- **Stratified Fit Set ($N_{\text{fit}} = 4,138$ real RSNA radiographs):** `138` (`0–24` mo), `1,000` (`24–72` mo), `1,000` (`72–120` mo), `1,000` (`120–168` mo), `1,000` (`168+` mo). Excludes all `230` evaluation files in `configs/validator_excluded_eval_files.json`.
- **Stratified Held-Out Calibration Set ($N_{\text{holdout}} = 500$ real RSNA radiographs — used for threshold calibration, not independent evidence):** `25` (`0–24` mo), `119` (`24–72` mo), `119` (`72–120` mo), `119` (`120–168` mo), `118` (`168+` mo).
  - **kNN ($k=5$, cosine) @ 99% Threshold ($\tau = 0.122576$):** **495/500 (99.00%)** accepted overall (`21/25 = 84.00%` on `0–24` mo, `118/119 = 99.16%` on `24–72` mo, `118/119 = 99.16%` on `72–120` mo, `119/119 = 100.00%` on `120–168` mo, `119/118 -> 118/118 = 100.00%` on `168+` mo).
  - **ROC AUC (500 Held-Out RSNA vs 23 `should_reject`):** **`1.000000`** for kNN ($k=5$, cosine), **`0.999913`** for Mahalanobis (Ledoit-Wolf shrinkage $\alpha = 0.005634$).
- **Independent Positive Test Set (`tests/validator/should_accept`, $N = 30$ real RSNA radiographs):**
  - End-to-End (Stage 1 + Stage 2 + Stage 3): **29/30 (96.67%)** accepted, **1/30 (3.33%)** rejected (`4244.png`, 18-mo infant, rejected at Stage 2 with kNN score `0.128517 > 0.122576`).
  - Per-bin breakdown: `3/4 (75.00%)` in `0–24` mo, `7/7 (100.00%)` in `24–72` mo, `6/6 (100.00%)` in `72–120` mo, `7/7 (100.00%)` in `120–168` mo, `6/6 (100.00%)` in `168+` mo.
- **Negative Test Set (`tests/validator/should_reject`, $N = 23$ real negative images):**
  - End-to-End (Stage 1 + Stage 2 + Stage 3): **23/23 (100.00%)** rejected, **0/23 (0.00%)** false accepts (`14` rejected at Stage 1 strong-color gate, `9` rejected at Stage 2 OOD kNN; all 23 also score $\ge 0.157954 > 0.122576$ in Stage 2).
  - Category breakdown: **8/8 (100.00%)** non-hand X-rays rejected, **12/12 (100.00%)** photos and animals rejected (including **4/4** adult-hand optical photos), **3/3 (100.00%)** screenshots and documents rejected.
- **200 RSNA Check Set (`configs/validator_excluded_eval_files.json`, $N = 200$ — used to check Stage 1/3 pass rates, not independent evidence):**
  - End-to-End (Stage 1 + Stage 2 + Stage 3): **199/200 (99.50%)** accepted (`200/200` pass Stage 1 and Stage 2; `1/200` [`3566.png`] rejected at Stage 3 coverage with `cov = 0.0242 < 0.03`).

### Validator Known Limitations
1. **Higher Infant (0–24 Months) Rejection Rate:** Infants represent only `1.53%` (`193 / 12,611`) of the RSNA training CSV and exhibit distinct anatomical morphology (unossified carpals, wider soft-tissue proportions). Because both the fit bank (`138` infant images) and the evaluation samples (`25` held-out, `4` in `should_accept`) are small, infants are rejected more often (`21/25 = 84.00%` accepted on held-out, `3/4 = 75.00%` accepted on `should_accept`) than older age bins (`99.2%–100.0%`).
2. **Small Negative Test Set:** The negative evaluation folder (`tests/validator/should_reject`) currently contains $N = 23$ images (`8` non-hand X-rays, `12` photos/animals including `4` optical adult-hand photos, `3` screenshots).
3. **Untested Edge Cases (Adult Hand X-Rays & Photos of X-Ray Films):** Adult hand radiographs ($> 20$ years) and camera/smartphone photographs of physical X-ray films on lightboxes or monitors are untested unless added to `tests/validator/should_reject`.

---

## 4. Dataset & Partitioning Protocol

- **Dataset:** RSNA Pediatric Bone Age Challenge Dataset (Radiological Society of North America).
- **Raw Sample Count:** 12,611 training radiographs + 200 unlabeled validation radiographs.
- **Protocol:** **CASE B (Frozen Holdout Split)**:
  - Perceptual deduplication using 64-bit difference hashing (`dHash`) and Hamming distance clustering ($\le 2$ bits) identified and purged 25 duplicates ($N = 12,586$ clean samples).
  - Patient-grouped, joint age-cohort $\times$ sex stratified split (80% Train, 10% Validation, 10% Test).
  - **Frozen Test Holdout Split:** $N = 1,259$ samples permanently frozen with SHA-256 checksum manifest (`ml/data/splits/test_manifest.csv`).
- **Demographic Normalization Statistics (Train Partition Only):**
  - Image Mean: `0.1999`, Image Std: `0.2229`
  - Target Age Mean ($\mu$): `127.29` months, Target Age Std ($\sigma$): `41.25` months

---

## 5. Quantitative Evaluation & Comparative Benchmarks

Evaluated on the frozen, deduplicated test holdout partition ($N = 1,259$ radiographs):

| Metric | Legacy CNN Baseline (`cnn_best.pt`) | Target Architecture (`EfficientNetB3SexFusion`) | Improvement ($\Delta$) |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | **10.92 months** [10.27, 11.60] | **6.00 months** [5.68, 6.33] | **-4.92 months** ($p < 0.0001$) |
| **Root Mean Squared Error (RMSE)** | **14.09 months** [13.15, 14.98] | **7.60 months** [7.10, 8.12] | **-6.49 months** |
| **Median Absolute Error (MedAE)** | **9.19 months** [8.43, 9.87] | **4.92 months** [4.51, 5.38] | **-4.27 months** |
| **Predictions Within $\pm 6$ Months** | **34.5%** [31.9%, 37.1%] | **64.2%** [61.5%, 66.8%] | **+29.7%** |
| **Predictions Within $\pm 12$ Months** | **63.3%** [60.6%, 66.0%] | **91.8%** [90.1%, 93.3%] | **+28.5%** |
| **Coefficient of Determination ($R^2$)**| **0.8801** [0.8624, 0.8958] | **0.9658** [0.9592, 0.9715] | **+0.0857** |
| **Bland-Altman Mean Bias** | **+0.22 months** | **-0.08 months** | Unbiased clinical agreement |
| **Bland-Altman 95% Agreement Limits**| **[-27.35, +27.79] months** | **[-14.98, +14.82] months** | ~1.8x narrower limits |
| **90% Conformal Uncertainty Margin** | **$\pm 14.1$ months** | **$\pm 8.2$ months** | Tight, calibrated intervals |

### Demographic & Cohort Subgroup Fairness

| Age Cohort / Subgroup | Age Range (mo) | Baseline MAE | Target MAE | Gain |
|---|---|---|---|---|
| **Infant Cohort** | 0 – 24 mo | 24.80 mo | 7.95 mo | **-16.85 mo** (catastrophic failure resolved) |
| **Early Childhood** | 24 – 72 mo | 9.43 mo | 5.34 mo | **-4.09 mo** |
| **Middle Childhood** | 72 – 120 mo | 10.48 mo | 5.78 mo | **-4.70 mo** |
| **Adolescent** | 120 – 168 mo | 10.93 mo | 6.12 mo | **-4.81 mo** |
| **Post-Pubertal** | 168+ mo | 11.84 mo | 6.81 mo | **-5.03 mo** |
| **Male Patients** ($N=681$) | All | 11.33 mo | 6.11 mo | **-5.22 mo** |
| **Female Patients** ($N=578$) | All | 10.43 mo | 5.87 mo | **-4.56 mo** |

---

## 6. Saliency Verification & Interpretability

- **Algorithm:** Grad-CAM++ with higher-order second ($\alpha_{ij}^k$) and third-order derivative weighting.
- **Anatomical Localization:** Accurately highlights multiple co-occurring ossification centers (metacarpal heads, carpal clusters, distal radial/ulnar epiphyses).
- **Adebayo et al. (2018) Cascading Parameter Randomization Test:**
  - Baseline trained Pearson correlation: **1.0000**
  - Randomize head weights (`fc_head`): $r = 0.9984$ (features intact)
  - Randomize target backbone layer (`backbone_layer4`): $r = 0.0000$, $\text{MAD} = 0.0163$
  - **Verdict:** **PASSED** (saliency maps are strictly dependent on trained visual features and do not act as invariant edge detectors).

---

## 7. ONNX Runtime CPU Serving Specifications

- **Export Tool:** `ml/export/onnx_exporter.py`
- **Dynamic Axes:** Batch dimension axis 0 fully dynamic for both models.
- **Numerical Parity (PyTorch vs ONNX Runtime CPU):**
  - `hand_validator.onnx`: $\max |y_{\text{pt}} - y_{\text{ort}}| = 4.77 \times 10^{-6} < 10^{-4}$ (**PASSED**)
  - `efficientnet_b3.onnx`: $\max |y_{\text{pt}} - y_{\text{ort}}| = 0.000000 < 10^{-4}$ (**PASSED**)
- **Serving Latency (Intel UHD CPU):**
  - Stage 1–3 Validation: ~25–40 ms
  - Bone Age Prediction: ~60–85 ms
  - End-to-End Pipeline: < 150 ms per image

---

## 8. Clinical Limitations & Ethical Considerations

1. **Research Prototype Notice:** Antigravity BoneTest is designed strictly for scientific evaluation and educational demonstration. It is not licensed by regulatory bodies (FDA, CE-MDR) for autonomous clinical decision-making.
2. **Skeletal Dysplasia & Genetic Disorders:** Models were trained on RSNA populations presenting typical pediatric growth distributions. Patients with severe achondroplasia, Turner syndrome, hypopituitarism, or constitutional growth delay may demonstrate atypical ossification trajectories requiring direct specialist evaluation.
3. **Artifact Sensitivity:** Extreme motion artifacts, metallic jewelry, surgical hardware, or cast plaster may impede automatic Otsu ROI segmentation and trigger Stage 3 gatekeeper rejection.
