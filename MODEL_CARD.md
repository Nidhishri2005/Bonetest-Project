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
[Stage 1: Rule-Based Integrity Gate]
  ├─ Magic Bytes Signature Check (PNG, JPEG, BMP, TIFF, WebP)
  ├─ OpenCV Byte Decode & Corruption Trap
  ├─ Minimum Spatial Dimension Constraint (min 32x32)
  ├─ Aspect Ratio Bounds (0.20 <= AR <= 5.0)
  └─ Signal Variance Std Check (rejects flat / blank images)
     │ PASSED
     ▼
[Stage 2: 6-Class Anatomical Classifier & OOD Predictive Entropy Gate]
  ├─ ResNet-18 Transfer Learning Classifier (hand_validator_best.pt / hand_validator.onnx)
  ├─ 6 Clinical Classes:
  │   0: valid_pediatric_hand_xray    3: animal_photograph
  │   1: non_hand_xray                4: document_or_screenshot
  │   2: normal_photograph            5: other_unrelated
  ├─ Normalized Shannon Predictive Entropy: H_norm(p) <= 0.75
  └─ Confidence Gate: p(valid_pediatric_hand_xray) >= 0.80
     │ PASSED
     ▼
[Stage 3: Tissue-Sensitive Anatomical Mask Coverage Gate]
  ├─ Soft-tissue sensitive adaptive Otsu thresholding
  ├─ Connected component labeling & noise artifact filtering
  └─ Anatomical Frame Coverage Check (0.03 <= coverage <= 0.98)
     │ PASSED
     ▼
Execute Model Inference & Grad-CAM++ Explainability
```

### Validation Performance Benchmarks
- **Overall Classifier Accuracy:** **97.22%**
- **Macro Precision / Recall / F1:** **97.46% / 97.22% / 97.16%**
- **Pediatric Hand Radiograph Subgroup:** **100.00% Precision**, **100.00% Recall**, **1.000 F1-Score** (0 false positives, 0 false negatives)

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
