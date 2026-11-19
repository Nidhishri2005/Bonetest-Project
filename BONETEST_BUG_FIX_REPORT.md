# BoneTest — Bug Fix and Remediation Report

**Repository:** `BoneTest — AI-Based Pediatric Bone Age Prediction`  
**Git Branch:** `bonetest-model-improvements`  
**Date:** November 2026  
**Status:** All Confirmed Bugs Remediated and Verified  

---

## 1. Executive Summary

This report documents the root-cause diagnosis, implementation, and test verification of all confirmed software defects and system inconsistencies identified in the BoneTest project.

Prior to modifications, a clean safety backup of all existing checkpoints, Random Forest models, and evaluation results was established in:
`backend/backups/pre_improvement_backup/`

All fixes maintain backward compatibility, adhere to clean architecture boundaries, and preserve working model checkpoints.

---

## 2. Bug 1: Missing Metrics Plots and 404 Endpoint Failures

### 2.1 Issue Description
The application configured routes in `backend/app/api/routes/analysis.py` to serve evaluation visual artifacts (`/api/metrics/plots/{plot_name}`). However:
1. The target directory `backend/metrics/plots/` was absent from disk.
2. Requesting `/api/metrics/plots/metrics_comparison` or `/api/metrics/plots/scatter_comparison` returned an unhandled HTTP `404 Not Found`.
3. The evaluation plotting pipeline lacked on-demand directory instantiation and automatic fallback plot rendering.

### 2.2 Root Cause
In `backend/app/services/metrics_service.py`, `get_plot_path(plot_name)` only inspected existing file paths without verifying or triggering directory creation. Furthermore, offline training runs in `backend/ml/evaluate.py` saved evaluation results to JSON but omitted automated plot persistence when run without the `--metrics-dir` flag pointing directly to an existing subfolder.

### 2.3 Files Modified
- [`backend/app/services/metrics_service.py`](file:///c:/Bonetest/backend/app/services/metrics_service.py)
- [`backend/app/api/routes/analysis.py`](file:///c:/Bonetest/backend/app/api/routes/analysis.py)
- [`backend/metrics/evaluation_results.json`](file:///c:/Bonetest/backend/metrics/evaluation_results.json)

### 2.4 Fix Implemented
1. **Automated Directory & Plot Generation:**
   Implemented `generate_plots_if_missing()` inside `metrics_service.py`. When an image plot request arrives, the service verifies directory existence, reads verified metrics from `evaluation_results.json`, renders high-resolution plots via Matplotlib/Seaborn, and persists them into `backend/metrics/plots/`.
2. **Defensive Error Handling:**
   Updated `/api/metrics/plots/{plot_name}` in `analysis.py` to validate requested plot keys against allowable identifiers (`metrics_comparison`, `scatter_comparison`), returning an explicit, actionable 404 error if an unsupported plot name is requested.
3. **Data Integrity Alignment:**
   Updated `backend/metrics/evaluation_results.json` to replace mock placeholder metrics with actual verified validation metrics extracted from the trained model checkpoints (`cnn_rf`: 11.04, `cnn`: 11.33, `multimodal_cnn`: 11.39, `cnn_tw3`: 12.47, `cnn_dnn`: 12.53).

### 2.5 Verification Test & Result
- Executed unit test: `test_plot_endpoint_success_and_error` in `test_api_endpoints.py`.
- Verified file existence: `backend/metrics/plots/metrics_comparison.png` (55.8 KB) and `scatter_comparison.png` (49.2 KB).
- **Result:** PASS. `GET /api/metrics/plots/metrics_comparison` returns HTTP 200 with media type `image/png`. Unsupported plot requests return HTTP 404 with allowed plot names enumerated.

---

## 3. Bug 2: Frontend and Backend Model Type Mismatch

### 3.1 Issue Description
A significant schema discrepancy existed between the backend model registry and frontend TypeScript interfaces:
- Backend supported 8 models: `cnn`, `cnn_dnn`, `multimodal_cnn`, `regression_multimodal`, `mask_rcnn`, `ensemble_cnn`, `cnn_tw3`, `cnn_rf`.
- Frontend `frontend/src/types/index.ts` restricted `ModelType` to only 4 models: `"cnn" | "cnn_dnn" | "multimodal_cnn" | "cnn_rf"`.
- Models without checkpoints (e.g., `ensemble_cnn`, `mask_rcnn`) could appear in selector lists without disabling interaction, leading to runtime HTTP 503 errors on prediction.

### 3.2 Root Cause
Frontend types were never updated when experimental models and the CNN-TW3 proxy were added to the backend registry. Furthermore, the frontend `<select>` dropdown did not bind the `disabled` HTML attribute to `m.checkpoint_available`.

### 3.3 Files Modified
- [`backend/app/models/__init__.py`](file:///c:/Bonetest/backend/app/models/__init__.py)
- [`backend/app/schemas/prediction.py`](file:///c:/Bonetest/backend/app/schemas/prediction.py)
- [`backend/app/services/model_registry.py`](file:///c:/Bonetest/backend/app/services/model_registry.py)
- [`frontend/src/types/index.ts`](file:///c:/Bonetest/frontend/src/types/index.ts)
- [`frontend/src/components/ModelSelector.tsx`](file:///c:/Bonetest/frontend/src/components/ModelSelector.tsx)
- [`frontend/src/pages/PredictPage.tsx`](file:///c:/Bonetest/frontend/src/pages/PredictPage.tsx)

### 3.4 Fix Implemented
1. **Unified Schema:**
   Synchronized `ModelType` in `frontend/src/types/index.ts` across all 8 models.
2. **Experimental vs. Production Flagging:**
   Added `is_experimental: bool` to `ModelInfo` schema and `MODEL_REGISTRY`. Production image-only models (`cnn_rf`, `cnn`, `cnn_dnn`, `cnn_tw3`) are marked `is_experimental=False`, while research architectures (`multimodal_cnn`, `ensemble_cnn`, etc.) are marked `is_experimental=True`.
3. **Selector Guarding:**
   Updated `ModelSelector.tsx` so that `<option>` tags enforce `disabled={!m.checkpoint_available}` and label missing weights clearly: `(unavailable - no checkpoint)`. Users cannot trigger predictions on uninstantiated models.
4. **Graceful 503 Handling:**
   Added guard in `/api/predict` verifying `registry.is_checkpoint_available(model_type)` before attempting inference, returning a clear error listing all currently loadable models.

### 3.5 Verification Test & Result
- Executed unit test: `test_model_registry_contains_all_models` and `test_registry_checkpoint_availability`.
- **Result:** PASS. Frontend types and backend registries match 1:1.

---

## 4. Bug 3: Gender Validation Inconsistency & Transition to Image-Only Prediction

### 4.1 Issue Description
1. `/api/predict` enforced that `gender` must be `"male"` or `"female"` for multimodal models (returning 400 if missing).
2. `/api/gradcam` lacked this validation and silently defaulted missing gender to `0.0` (female).
3. The application design conflicted with clinical workflows by presenting demographic radio inputs on primary prediction cards.

### 4.2 Root Cause
Inconsistent input parsing between `app/api/routes/prediction.py` and `app/api/routes/analysis.py`. Furthermore, `MultimodalCNN` was previously set as an equal option alongside image-only models, confusing clinical users who expect an X-ray-only diagnostic tool.

### 4.3 Files Modified
- [`backend/app/api/routes/analysis.py`](file:///c:/Bonetest/backend/app/api/routes/analysis.py)
- [`backend/app/api/routes/prediction.py`](file:///c:/Bonetest/backend/app/api/routes/prediction.py)
- [`backend/app/core/config.py`](file:///c:/Bonetest/backend/app/core/config.py)
- [`frontend/src/components/ModelSelector.tsx`](file:///c:/Bonetest/frontend/src/components/ModelSelector.tsx)
- [`frontend/src/pages/PredictPage.tsx`](file:///c:/Bonetest/frontend/src/pages/PredictPage.tsx)

### 4.4 Fix Implemented
1. **Image-Only Architecture as Standard:**
   Established `cnn_rf` (CNN + Random Forest, MAE 11.04 months) and `cnn` (CNN Baseline, MAE 11.33 months) as the primary, default prediction models. Neither model accepts, requires, or uses demographic sex.
2. **Demographic Input Decoupled:**
   The primary prediction page default workflow is 100% image-only. Demographic inputs are completely hidden unless the user intentionally chooses an experimental research model (`multimodal_cnn`).
3. **Explicit Validation (No Silent Defaults):**
   Both `/api/predict` and `/api/gradcam` now enforce identical validation: if an experimental multimodal model is called without gender, an explicit HTTP 400 is returned directing the user to use standard image-only models.
4. **Grad-CAM Alignment:**
   `GradCamService` and `InferenceService` consume identical tensor inputs, ensuring explainability heatmaps directly correspond to the exact parameters of the prediction pass.

### 4.5 Verification Test & Result
- Executed unit tests: `test_image_only_models_do_not_require_gender`, `test_inference_service_image_only_cnn_rf`, `test_inference_service_image_only_cnn`, and `test_gradcam_service_image_only`.
- **Result:** PASS. Primary workflow operates image-only without silent defaults.

---

## 5. Bug 4: Disconnected Growth Status Functionality

### 5.1 Issue Description
The utility `frontend/src/utils/growthStatus.ts` was present in the codebase calculating difference thresholds (`Under-growth`, `Normal growth`, `Over-growth`). Earlier uncommitted changes attempted to add a chronological age input to the main UI, but were abandoned, leaving the module unused.

### 5.2 Root Cause
A single bone age estimate relative to chronological age is a clinical screening metric, not a definitive diagnostic indicator. Labeling a child as having "Under-growth" or "Over-growth" without height percentiles and clinical history creates risk of misinterpretation.

### 5.3 Files Modified
- [`frontend/src/utils/growthStatus.ts`](file:///c:/Bonetest/frontend/src/utils/growthStatus.ts)

### 5.4 Fix Implemented
1. Maintained the primary UI strictly focused on bone age regression in months and years.
2. Preserved `growthStatus.ts` as a pure research utility.
3. Added comprehensive documentation to the file header explaining why growth diagnostic labels are withheld from the primary clinical UI.

### 5.5 Verification Test & Result
- **Result:** Verified. Primary interface renders bone age in months and equivalent years without unsupported diagnostic classifications.

---

## 6. Bug 5: Test Execution Failures & Path Resolution

### 6.1 Issue Description
1. Running `python -m unittest discover -s backend/tests` failed immediately with:
   `ImportError: Failed to import test module: test_new_models. ModuleNotFoundError: No module named 'app'`
2. Relative paths in `backend/app/core/config.py` (`Path("checkpoints")`) failed when tests were executed from the repository root (`C:\Bonetest`), searching for `C:\Bonetest\checkpoints` instead of `C:\Bonetest\backend\checkpoints`.
3. Unit test functions used pytest-style bare assertions without subclassing `unittest.TestCase`, preventing native Python test runner discovery.

### 6.2 Root Cause
Missing `sys.path` bootstrapping in test files and unanchored relative path definitions in `Settings` when executed across varying current working directories.

### 6.3 Files Modified
- [`backend/app/core/config.py`](file:///c:/Bonetest/backend/app/core/config.py)
- [`backend/tests/test_new_models.py`](file:///c:/Bonetest/backend/tests/test_new_models.py)
- [`backend/tests/test_api_endpoints.py`](file:///c:/Bonetest/backend/tests/test_api_endpoints.py) (New test suite)

### 6.4 Fix Implemented
1. **Dynamic Path Resolution Validator:**
   Updated `Settings` in `config.py` with a Pydantic `@field_validator` that resolves relative paths (`checkpoints`, `rf_models`, `metrics`) against `BASE_DIR = Path(__file__).resolve().parent.parent.parent`, ensuring correct resolution regardless of execution directory.
2. **Zero-Dependency Native Unittest Compatibility:**
   Refactored `test_new_models.py` to subclass `unittest.TestCase` and insert `BACKEND_DIR` into `sys.path`.
3. **Comprehensive End-to-End Test Suite:**
   Created `test_api_endpoints.py` testing health, model enumeration, metrics loading, plot generation, Grad-CAM generation, and inference on all primary image-only architectures using in-memory synthetic bone phantoms.

### 6.5 Verification Test & Result
- Executed: `python -m unittest discover -s backend/tests -v`
- **Result:** PASS. All 18 unit and integration tests passed cleanly in 28.1 seconds.

---

## 7. Summary of Test Results

| Test Class | Test Case | Status | Execution Time |
|:---|:---|:---:|---:|
| `TestAPIEndpointsAndServices` | `test_gradcam_service_cnn_rf` | PASS | 1.1s |
| `TestAPIEndpointsAndServices` | `test_gradcam_service_image_only` | PASS | 1.2s |
| `TestAPIEndpointsAndServices` | `test_health_check` | PASS | 0.01s |
| `TestAPIEndpointsAndServices` | `test_inference_service_image_only_cnn` | PASS | 0.4s |
| `TestAPIEndpointsAndServices` | `test_inference_service_image_only_cnn_rf` | PASS | 0.3s |
| `TestAPIEndpointsAndServices` | `test_inference_service_image_only_cnn_tw3` | PASS | 0.4s |
| `TestAPIEndpointsAndServices` | `test_list_models_contains_default_and_flags` | PASS | 0.02s |
| `TestAPIEndpointsAndServices` | `test_metrics_endpoints` | PASS | 0.02s |
| `TestAPIEndpointsAndServices` | `test_plot_endpoint_success_and_error` | PASS | 0.1s |
| `TestModelArchitectures` | `test_cnn_feature_extractor_dimension` | PASS | 0.2s |
| `TestModelArchitectures` | `test_experimental_multimodal_models_require_gender` | PASS | 0.01s |
| `TestModelArchitectures` | `test_forward_pass_image_only_models` | PASS | 1.8s |
| `TestModelArchitectures` | `test_forward_pass_multimodal_models` | PASS | 0.4s |
| `TestModelArchitectures` | `test_gradcam_target_layers_exist` | PASS | 0.2s |
| `TestModelArchitectures` | `test_image_only_models_do_not_require_gender` | PASS | 0.01s |
| `TestModelArchitectures` | `test_model_registry_contains_all_models` | PASS | 0.01s |
| `TestModelRegistryAndCheckpoints` | `test_registry_checkpoint_availability` | PASS | 0.02s |
| `TestModelRegistryAndCheckpoints` | `test_registry_load_trained_models` | PASS | 21.8s |
| **Total** | **18 Tests Across 3 Suites** | **100% PASS** | **28.1s** |
