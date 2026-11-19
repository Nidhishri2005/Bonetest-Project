# BoneTest — Setup, Training, Evaluation & Execution Guide

**Project:** BoneTest — AI-Based Pediatric Bone Age Prediction Using Hand X-ray Images  
**Repository Branch:** `bonetest-model-improvements`  
**Location:** `C:\Bonetest`  
**Date:** November 2026  

---

## 1. System Requirements & Environment

### 1.1 Hardware Specifications
| Component | Minimum Requirement (Inference Only) | Recommended (Training / Fine-Tuning) |
|:---|:---|:---|
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 cores) | Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores) |
| **System Memory (RAM)** | 8 GB | 16 GB – 32 GB |
| **Graphics Card (GPU)** | Not required (Runs on CPU) | NVIDIA GPU (8 GB+ VRAM, e.g., RTX 3060, T4, A10) |
| **CUDA / cuDNN** | Not required | CUDA 11.8 or 12.1+ with compatible cuDNN |
| **Disk Space** | 2 GB for repository and checkpoints | 25 GB+ (to accommodate raw RSNA dataset) |

### 1.2 Software Prerequisites
- **Python:** Version 3.10, 3.11, or 3.12 (64-bit).
- **Node.js:** Version 18.x or 20.x LTS.
- **Package Managers:** `pip` (Python) and `npm` or `pnpm` (JavaScript/TypeScript).
- **Git:** Version 2.30+.
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS (Apple Silicon supported via MPS).

---

## 2. Repository Setup from Scratch

### 2.1 Clone Repository and Checkout Branch
Open PowerShell (Windows) or Terminal (Linux/macOS):

```bash
git clone https://github.com/manyaa133/Bonetest.git C:\Bonetest
cd C:\Bonetest
git checkout bonetest-model-improvements
```

### 2.2 Python Virtual Environment Setup (Backend)
Create and activate an isolated Python virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Verify backend installation:
```bash
python -c "import torch, cv2, sklearn, fastapi; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

### 2.3 Frontend Dependencies Setup
Open a separate terminal window and navigate to the `frontend/` directory:

```bash
cd C:\Bonetest\frontend
npm install
```

---

## 3. Dataset Setup & Preparation

The project is designed to ingest the **RSNA Pediatric Bone Age Dataset** (or Kaggle Bone Age Challenge dataset).

### 3.1 Downloading the Dataset
You can obtain the dataset from Kaggle via the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c rsna-bone-age
```

Or download directly from Kaggle: [RSNA Pediatric Bone Age Challenge](https://www.kaggle.com/competitions/rsna-bone-age/data).

### 3.2 Expected Directory Layout
Extract the dataset into an accessible directory (e.g., `C:\Bonetest\data\rsna` or an external drive):

```
data/
└── rsna/
    ├── train.csv                  # Metadata containing 'id', 'boneage', 'male'
    └── train/                     # Folder containing radiographs
        ├── 1377.png
        ├── 1378.png
        ├── 1379.png
        └── ...
```

*(Note: The dataset loader automatically supports both `train.csv` and `boneage-training-dataset.csv`, as well as subfolders named `boneage-training-dataset` or `train`.)*

### 3.3 Verifying Dataset Integrity
Run the built-in image verification tool to confirm all image files exist and are readable:

```powershell
cd C:\Bonetest\backend
python -c "
from pathlib import Path
from ml.dataset import create_dataloaders
train_ldr, val_ldr, train_df, val_df = create_dataloaders(Path('../data/rsna'), verify_images=True)
print(f'Train samples: {len(train_df)}, Val samples: {len(val_df)}')
"
```

---

## 4. Execution Commands

### 4.1 Running the Automated Test Suite
To verify backend routing, model registries, tensor shapes, Grad-CAM generation, and evaluation metrics:

```powershell
cd C:\Bonetest
.venv\Scripts\python -m unittest discover -s backend/tests -v
```

**Expected output:**
```
Ran 23 tests in 13.840s
OK
```

---

### 4.2 Training Models

All models can be trained locally (if GPU is present) or on Google Colab / Kaggle.

#### 1. Train CNN Baseline (`cnn`)
```powershell
cd C:\Bonetest\backend
python -m ml.train `
  --model-type cnn `
  --data-dir ../data/rsna `
  --epochs 30 `
  --batch-size 16 `
  --lr 1e-4 `
  --loss smooth_l1 `
  --scheduler plateau `
  --patience 6
```

#### 2. Train Production Champion: CNN + Random Forest (`cnn_rf`)
*(Requires an existing `checkpoints/cnn_best.pt` feature backbone)*
```powershell
cd C:\Bonetest\backend
python -m ml.train `
  --model-type cnn_rf `
  --data-dir ../data/rsna
```
*This extracts 512-dimensional feature embeddings and trains a 200-tree Scikit-Learn Random Forest Regressor in ~2–5 minutes.*

#### 3. Train Multimodal CNN (Experimental — Research Only)
```powershell
cd C:\Bonetest\backend
python -m ml.train `
  --model-type multimodal_cnn `
  --data-dir ../data/rsna `
  --epochs 30 `
  --batch-size 16 `
  --lr 1e-4 `
  --loss smooth_l1
```

---

### 4.3 Evaluating Models & Generating Benchmark Plots
To benchmark all checkpoints against the held-out validation or test set and automatically generate comparison figures:

```powershell
cd C:\Bonetest\backend
python -m ml.evaluate `
  --data-dir ../data/rsna `
  --checkpoints-dir checkpoints `
  --metrics-dir metrics `
  --split val
```

This will automatically create:
- `backend/metrics/evaluation_results.json` (complete multi-metric benchmark).
- `backend/metrics/plots/metrics_comparison.png` (MAE, MSE, RMSE bar charts).
- `backend/metrics/plots/scatter_comparison.png` (Predicted vs Actual scatter plots).
- `backend/metrics/predictions_<model_type>.csv` (per-sample prediction residuals).

---

### 4.4 Running the Application (Full-Stack)

#### Step 1: Launch FastAPI Backend
In Terminal 1:
```powershell
cd C:\Bonetest\backend
..\.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API URL:** `http://localhost:8000`
- **Interactive OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

#### Step 2: Launch Vite Frontend
In Terminal 2:
```powershell
cd C:\Bonetest\frontend
npm run dev
```
- **Frontend UI URL:** `http://localhost:5173`

Open `http://localhost:5173` in your web browser. You can upload any hand X-ray, receive instantaneous image-only bone age estimation via `cnn_rf`, and view the overlaid Grad-CAM explainability heatmap.

---

## 5. Google Colab Training Workflow

For training deep models on free or pro cloud GPUs (NVIDIA T4 / V100 / A100), follow this standard workflow:

### Step 1: Create a New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com).
2. Set Runtime Type: **Runtime > Change runtime type > T4 GPU**.

### Step 2: Paste Setup & Clone Cell
```python
# Cell 1: Mount Drive and Clone
from google.colab import drive
drive.mount('/content/drive')

!git clone -b bonetest-model-improvements https://github.com/manyaa133/Bonetest.git
%cd Bonetest/backend
!pip install -r requirements.txt
```

### Step 3: Download RSNA Dataset via Kaggle
```python
# Cell 2: Download Dataset
import os
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
os.environ['KAGGLE_KEY'] = "your_kaggle_api_key"

!mkdir -p /content/data/rsna
%cd /content/data/rsna
!kaggle competitions download -c rsna-bone-age
!unzip -q rsna-bone-age.zip
%cd /content/Bonetest/backend
```

### Step 4: Run GPU Accelerated Training
```python
# Cell 3: Train CNN Baseline with Mixed Precision & Smooth L1 Loss
!python -m ml.train \
  --model-type cnn \
  --data-dir /content/data/rsna \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --loss smooth_l1 \
  --scheduler plateau \
  --device cuda
```

### Step 5: Train Random Forest Regressor & Save Checkpoints
```python
# Cell 4: Train CNN-RF and Copy Checkpoints to Google Drive
!python -m ml.train \
  --model-type cnn_rf \
  --data-dir /content/data/rsna \
  --device cuda

# Backup trained weights to Google Drive
!cp -r checkpoints /content/drive/MyDrive/bonetest_checkpoints
!cp -r rf_models /content/drive/MyDrive/bonetest_rf_models
print("Checkpoints safely backed up to Google Drive!")
```

---

## 6. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: `FileNotFoundError: Checkpoint not found for 'cnn'`
- **Cause:** Running commands outside the `backend/` directory or missing `cnn_best.pt`.
- **Resolution:** Path resolution is anchored to `backend/checkpoints`. Ensure checkpoints are situated at `C:\Bonetest\backend\checkpoints\cnn_best.pt` (or restored from `backend/backups/pre_improvement_backup/`).

### Q2: Why does the prediction interface not require gender?
- **Design Principle:** The default production model is `cnn_rf` (CNN + Random Forest), which is 100% image-only. Radiographic epiphyseal morphology provides sufficient biological information without demographic priors. Multimodal models that require gender are segregated under the "Experimental" category.

### Q3: `CORS error` when frontend calls backend
- **Cause:** Frontend port mismatch with allowed origins in `app/core/config.py`.
- **Resolution:** By default, `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, and `http://localhost:8080` are whitelisted. If running on another port, add your host URL to `cors_origins` in `backend/app/core/config.py`.

### Q4: Missing metrics comparison plots in the UI
- **Resolution:** The backend now includes an auto-generation safeguard (`generate_plots_if_missing()`). If plot files are deleted, visiting the Dashboard or running `python -m ml.evaluate` regenerates them automatically.
