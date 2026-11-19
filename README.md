# BoneTest: Pediatric Bone Age Assessment & Explainability Platform

A hardened, clinically-grounded pediatric bone age estimation platform built on the RSNA Pediatric Bone Age challenge dataset. Modernized from legacy proof-of-concept baselines into a multi-stage, validated clinical assistance architecture featuring **EfficientNet-B3 Sex-Fusion**, **ONNX Runtime CPU serving**, **Grad-CAM++ saliency with Adebayo sanity checks**, and a **Three-Stage Defensive Gatekeeper**.

> **CLINICAL NOTICE:** This software is a **Research Prototype** for scientific and educational investigation. It is **NOT** a certified medical diagnostic device and must never substitute for evaluation by a licensed radiologist or pediatric specialist.

---

## Key Features

1. **Three-Stage Image Validation Gatekeeper**:
   - **Stage 1 (Rules & Strong-Color Gate)**: Magic byte header verification, OpenCV decode integrity, dimension constraints ($h, w \ge 32$), aspect ratio filtering ($0.20 \le \text{AR} \le 5.0$), standard deviation signal variance check ($\text{std} \ge 1.0$), and strong-color rejection (`mean_rgb_spread > 25.0` or `colored_pixel_fraction > 0.20` at channel spread $> 40$).
   - **Stage 2 (One-Class OOD kNN $k=5$ Cosine Gate)**: Frozen ImageNet-pretrained `ResNet-18` (`ResNet18_Weights.DEFAULT`) global-pooled 512-d L2-normalized feature extractor with a single shared preprocessing function (`preprocess_validator_image`: 3-channel RGB conversion, $224 \times 224$ aspect-ratio-preserving letterbox resize, ImageNet normalization). Fitted on a **stratified RSNA sample of $N_{\text{fit}} = 4,138$ real radiographs** (`138` in `0-24` mo, `1,000` each in `24-72`, `72-120`, `120-168`, `168+` mo) and calibrated at the **99% threshold ($\tau = 0.122576$)** on a **stratified held-out set of $N_{\text{holdout}} = 500$ RSNA radiographs** (`25` in `0-24` mo, `119/119/119/118` across the other 4 bins), saved in `backend/checkpoints/validator_ood.npz`.
     - **Measured Performance & Sample Sizes:**
       - `tests/validator/should_accept` ($N = 30$ real RSNA radiographs): **29/30 (96.67%)** accepted end-to-end (`1/30` rejected in Stage 2: infant `4244.png`, 18 mo, score `0.128517 > 0.122576`).
       - `tests/validator/should_reject` ($N = 23$ real negative images: 8 non-hand X-rays, 8 animal/general photos, 4 adult-hand optical photos, 3 screenshots): **23/23 (100.00%)** rejected (`14` by Stage 1 strong-color gate, `9` by Stage 2 OOD kNN; all 23 also exceed the Stage 2 threshold with scores $\ge 0.157954$, yielding **ROC AUC = `1.000000`** vs. the 500 held-out RSNA set).
       - Stratified Held-Out RSNA Set ($N = 500$, used for threshold calibration — not independent): **495/500 (99.00%)** pass Stage 2.
       - 200 RSNA Check Images ($N = 200$, used to check Stage 1/3 — not independent): **199/200 (99.50%)** accepted end-to-end (`200/200` pass Stage 1 and Stage 2; `1/200` [`3566.png`] rejected by Stage 3 coverage at `0.0242 < 0.03`).
     - **Known Limitations:**
       - Infants (`0–24` months) are rejected more often than older children due to smaller training representation ($1.53\%$ of RSNA) and small evaluation sample sizes (`25` in held-out where `21/25 = 84.00%` pass at 99%, and `4` in `should_accept` where `3/4 = 75.00%` pass).
       - The negative test set (`tests/validator/should_reject`, $N = 23$) is small.
       - Adult hand X-rays and camera photographs of physical X-ray films are untested unless added to `tests/validator/should_reject`.
   - **Stage 3 (Tissue Coverage w/ 5% Border Ignored)**: Central $90\% \times 90\%$ ROI (ignoring a 5% border on all sides), $[P_2, P_{98}]$ percentile contrast stretch + Otsu thresholding with connected components isolating hand anatomy ($0.03 \le \text{coverage} \le 0.98$).

2. **Target Architecture (`EfficientNetB3SexFusion`)**:
   - ImageNet-pretrained EfficientNet-B3 backbone with 384x384 letterboxed input.
   - 32-dimensional biological sex embedding fusion providing physiologically accurate maturation curves (+1.33 mo MAE gain).
   - Huber loss ($\delta=1.0$) on standardized age targets with AdamW cosine annealing schedule.
   - **Test MAE: 6.00 months** (vs. legacy baseline **10.92 months**, $\Delta = -4.92$ mo, $p < 0.0001$).
   - Catastrophic infant error (0–24 mo) reduced from **24.80 mo** down to **7.95 mo**.

3. **ONNX Runtime CPU Serving & Conformal Uncertainty**:
   - Exported to ONNX with dynamic batching; verified numerical parity ($\max |y_{\text{pt}} - y_{\text{ort}}| < 10^{-4}$).
   - Sub-100ms inference on standard CPUs using `CPUExecutionProvider`.
   - Calibrated 90% conformal prediction uncertainty intervals ($\pm 8.2$ months).

4. **Grad-CAM++ Explainability & Sanity Verification**:
   - Second- and third-order gradient weighting for heightened spatial sensitivity across multi-focal ossification centers (metacarpal heads, carpal clusters, distal radial epiphyses).
   - Validated against the **Adebayo et al. (2018) Cascading Parameter Randomization Test** ($r=0.00$ on backbone randomization, confirming genuine dependency on trained visual features).

---

## Architectural Comparison

| Metric | Legacy CNN Baseline (`cnn_best.pt`) | Target Architecture (`EfficientNetB3SexFusion`) | Improvement ($\Delta$) |
|---|---|---|---|
| **Test MAE** | **10.92 months** [10.27, 11.60] | **6.00 months** [5.68, 6.33] | **-4.92 months** ($p < 0.0001$) |
| **Test RMSE** | **14.09 months** [13.15, 14.98] | **7.60 months** [7.10, 8.12] | **-6.49 months** |
| **Median AE** | **9.19 months** [8.43, 9.87] | **4.92 months** [4.51, 5.38] | **-4.27 months** |
| **Within $\pm$ 6 Mo** | **34.5%** | **64.2%** | **+29.7%** |
| **Within $\pm$ 12 Mo** | **63.3%** | **91.8%** | **+28.5%** |
| **$R^2$ Score** | **0.8801** | **0.9658** | **+0.0857** |
| **Bland-Altman Bias** | **+0.22 months** | **-0.08 months** | Clinical agreement |
| **95% LoA Bounds** | **[-27.35, +27.79] months** | **[-14.98, +14.82] months** | ~1.8x narrower |
| **90% Conformal Interval** | **$\pm 14.1$ months** | **$\pm 8.2$ months** | Tight, calibrated intervals |

---

## Quick Start

### 1. Backend Service (FastAPI + ONNX Runtime)

```powershell
# Navigate to backend and activate virtual environment
cd C:\bonetest
.\.venv\Scripts\activate

# Launch development server
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** http://localhost:8000/docs
- **Health Check Endpoint:** http://localhost:8000/api/health

### 2. Frontend Application (React 18 + TypeScript + Tailwind)

```powershell
cd C:\bonetest\frontend
npm run dev
```

- **Application URL:** http://localhost:5173

### 3. Docker Deployment

```bash
docker compose up --build
```

---

## Test Suite Execution

The repository maintains an automated suite of **94 unit and regression tests** across 10 specialized modules:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

### Test Coverage Highlights:
- `test_image_validator.py`: 15 tests verifying startup `validator_ood.npz` integrity/config checks, Stage 1 rules & strong-color rejection, Stage 2 One-Class OOD kNN ($k=5$, cosine) scoring on real RSNA (`tests/validator/should_accept`, $N=30$) and real negative images (`tests/validator/should_reject`, $N=23$), Stage 3 coverage, and prediction/Grad-CAM integration.
- `test_red_team_validation.py`: 12 adversarial and clinical pipeline tests verifying strict refusal of real non-hand radiographs, animal/adult-hand photos, screenshots, blank frames, and corrupted bytes, plus end-to-end prediction on real RSNA radiographs.
- `test_onnx_serving.py`: 7 tests verifying ONNX graph integrity, numerical parity ($< 10^{-4}$), and CPU serving dispatch.
- `test_gradcam_plusplus.py`: 11 tests verifying Grad-CAM++ algorithms, hook memory cleanup, and Adebayo cascading randomization sanity checks.
- `test_evaluation_suite.py`: 7 statistical tests verifying bootstrap 95% CIs, Bland-Altman statistics, and paired hypothesis tests.
- `test_data_pipeline.py`: 9 tests verifying perceptual deduplication (dHash), CLAHE, ROI cropping, and stratified holdout splitting.

---

## Documentation & Reports

- [Model Card (Mitchell et al. 2019)](MODEL_CARD.md)
- [Project Changelog](docs/CHANGELOG.md)
- [Dataset Inventory & Re-Audit](docs/DATASET_INVENTORY.md)
- [Audit & Architectural Plan](docs/REAUDIT.md)
- [Baseline Reproduction Report](reports/baseline_reproduction_report.md)
- [Ablation Study Report](reports/ablation_study_report.md)
- [Comprehensive Statistical Evaluation Report](reports/comprehensive_evaluation_report.md)
- [Image Validator Evaluation Report](reports/validator_evaluation_report.md)
- [Adebayo Saliency Sanity Check Report](reports/gradcam_sanity_report.md)
- [ONNX Runtime Parity Report](reports/onnx_export_report.md)

---

## License

MIT License. Dataset derived from the Radiological Society of North America (RSNA) Pediatric Bone Age Challenge, subject to RSNA data use agreements.
