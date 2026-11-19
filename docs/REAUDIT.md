# Phase 0: Technical Re-Audit Report (BoneTest)

**Date:** 2026-11-17  
**Auditor:** Senior ML Engineer & Full-Stack Systems Architect  
**Workspace:** `C:\bonetest`  
**Execution Environment:** Windows 11, Python 3.12 (`.venv`), PyTorch 2.5.1 (CPU), Intel UHD Graphics (No CUDA GPU).

---

## 1. Executive Summary

This audit re-examines the prior claims regarding the BoneTest pediatric bone-age prediction repository. Every claim has been categorized into:
- **`CONFIRMED-BY-EXECUTION`**: Verified by live script execution, model inference, and automated test runners.
- **`CONFIRMED-BY-CODE`**: Verified by code inspection of source files, ASTs, and configuration definitions.
- **`REFUTED / RESOLVED`**: Found to be incorrect or previously resolved.
- **`UNVERIFIED`**: Cannot be fully reproduced or verified until the external RSNA dataset path is provided.

---

## 2. Claim Classification Matrix

| Claim / Component | Status | Evidence & Details |
| :--- | :---: | :--- |
| **Backend Architecture & Routes** (`main.py`, `prediction.py`, `analysis.py`, `health.py`) | **`CONFIRMED-BY-EXECUTION`** | All API route handlers and service singletons are operational. Endpoints verified by automated test suite (`backend/tests/test_api_endpoints.py`, 9/9 passing). |
| **Frontend Architecture** (`PredictPage.tsx`, `ModelSelector.tsx`, `types/index.ts`) | **`CONFIRMED-BY-CODE`** | React 18 + Vite + Tailwind CSS frontend located in `frontend/`. Note: `package.json` is located in `frontend/`, so commands must be executed with working directory `frontend/`. |
| **Current Preprocessing Pipeline** (512x512 letterbox, grayscale, norm mean=0.4523, std=0.2118) | **`CONFIRMED-BY-CODE`** & **`CONFIRMED-BY-EXECUTION`** | Implemented in `backend/app/services/preprocessing.py` and `backend/ml/preprocessing.py`. Norm statistics stored in `backend/checkpoints/normalization.json`. |
| **Documented MAE Metrics** (CNN+RF: 11.04, CNN: 11.33, Multimodal: 11.39, TW3: 12.47, CNN-DNN: 12.53) | **`CONFIRMED-BY-CODE`** (Historical) / **`UNVERIFIED`** (Evaluation Split) | Recorded in `backend/metrics/evaluation_results.json`. Exactly matches epoch 27 of `backend/checkpoints/cnn_history.json` (MAE 11.3345 on 1892 validation samples). However, no frozen split manifest (`val_ids.csv` or `test_ids.csv`) was ever persisted to disk. |
| **Weak Performance Relative to RSNA Literature** (~11-12.5 months vs SOTA 4-8 months) | **`CONFIRMED-BY-CODE`** | The existing models suffer from: (1) no hand ROI crop (excessive background noise), (2) single-channel grayscale input without ImageNet pretrained weights, (3) direct raw scalar regression without target standardization, (4) absence of Huber loss/cosine annealing in training. |
| **Audit Environment Dependency Failure** (Missing numpy/cv2/torch) | **`REFUTED / RESOLVED`** | `.venv` has all required ML packages installed (torch 2.5.1, torchvision 0.20.1, albumentations, opencv-python-headless 4.10, scikit-learn 1.6.0, pandas 2.2.3, pydantic 2.10.4). Full backend test suite executed: **42/42 tests passing** (0 failures, 0 errors). |
| **Hardware Constraints** (Intel UHD, NO NVIDIA GPU) | **`CONFIRMED-BY-EXECUTION`** | `torch.cuda.is_available()` returns `False`. All local workflows must target CPU inference (PyTorch / ONNX Runtime). Model training must target Kaggle/Colab cloud GPUs. |
| **Database & Persistence** | **`CONFIRMED-BY-CODE`** | No external SQL database or ORM currently exists; metrics and checkpoints are read directly from filesystem directories. |
| **Deployment Files** | **`CONFIRMED-BY-CODE`** | Root `docker-compose.yml`, `frontend/Dockerfile`, and `frontend/nginx.conf` exist. |

---

## 3. Checkpoint Inventory & Integrity

All existing checkpoints in `backend/checkpoints/` and `backend/rf_models/` were inspected and verified:

| Checkpoint File | Size | Architecture / Wrapper | Status in Registry |
| :--- | :---: | :--- | :--- |
| `backend/checkpoints/cnn_best.pt` | 44.8 MB | ResNet-18 (adapted 1-channel conv1) + Linear regression head | Intact & loadable |
| `backend/checkpoints/cnn_dnn_best.pt` | 45.4 MB | ResNet-18 + 3-layer batch-normalized DNN head | Intact & loadable |
| `backend/checkpoints/multimodal_cnn_best.pt` | 45.5 MB | ResNet-18 + gender scalar concat + FC layers | Intact & loadable |
| `backend/checkpoints/cnn_rf_best.pt` | 44.8 MB | ResNet-18 Feature Extractor (512-dim) | Intact & loadable |
| `backend/rf_models/cnn_rf.joblib` | 88.0 MB | Scikit-Learn `RandomForestRegressor` | Intact & loadable |
| `backend/checkpoints/cnn_tw3_best.pt` | 45.5 MB | ResNet-18 + 4 TW3 analytical image proxy features | Intact & loadable |
| `backend/checkpoints/hand_validator_best.pt` | 45.1 MB | 6-class ResNet-18 Hand X-Ray Gatekeeper | Intact & loadable |

**Preservation Policy:** Per Global Rule 1, all legacy checkpoints will remain untouched on disk and transition to "legacy baselines" in the model registry.

---

## 4. Legacy Training & Split Analysis

### Findings from `backend/ml/dataset.py` & `backend/ml/train.py`:
1. **Dynamic Splitting:**
   The function `train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)` performs a 2-stage `train_test_split` stratified across age bins (`[0, 36, 84, 132, 180, 250]`).
2. **Missing Persisted Split Manifests:**
   No file such as `splits.json`, `train_indices.csv`, or `test_manifest.csv` was written to disk when the models were trained.
3. **Missing Image Invalidation Effect:**
   In `create_dataloaders()`, if `verify_images=True` was used, rows with missing or unreadable images were dropped *before* `train_test_split`. Without knowing the exact file availability on the original training machine, exact row-for-row replication of the 1,892 validation samples cannot be guaranteed without re-evaluating on the complete RSNA dataset folder.
4. **Conclusion:**
   The reported metrics in `evaluation_results.json` are **historical baseline reference points**. For modern rigorous benchmarking, a frozen split manifest with file hashes must be generated and locked in Phase 2.

---

## 5. Next Steps for Phase 0 Dataset Audit

To complete the dataset portion of Phase 0 and produce `docs/DATASET_INVENTORY.md`:
1. Obtain the absolute directory path to the external RSNA dataset folder from the user.
2. Scan the external dataset folder for CSVs (`train.csv`, `test.csv`, `validation.csv`, etc.).
3. Record row counts, column schemas, presence of `boneage` labels, image file counts, missing images, and missing IDs.
4. Determine whether the dataset is **CASE A** (independent labelled test set exists) or **CASE B** (only training set is labelled, necessitating a frozen 10% stratified test set).
