# Phase 5 Hand X-Ray Validation Gatekeeper Evaluation Report

**Model Architecture:** `HandXRayValidator` (ResNet-18 Transfer Learning, 6-class)
**Checkpoint:** `hand_validator_best.pt`
**Test Set Samples:** 108 (Held-out synthetic + phantom multi-class dataset)
**Test Loss:** `0.1693`
**Overall Accuracy:** **97.22%**

---

## 1. Global Performance Metrics

| Metric | Macro Average | Weighted Average |
| :--- | :---: | :---: |
| **Precision** | **97.46%** | 97.46% |
| **Recall** | **97.22%** | 97.22% |
| **F1-Score** | **97.16%** | 97.16% |

---

## 2. Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **valid_pediatric_hand_xray** | 100.00% | 100.00% | 100.00% | 18 |
| **non_hand_xray** | 94.74% | 100.00% | 97.30% | 18 |
| **normal_photograph** | 90.00% | 100.00% | 94.74% | 18 |
| **animal_photograph** | 100.00% | 100.00% | 100.00% | 18 |
| **document_or_screenshot** | 100.00% | 100.00% | 100.00% | 18 |
| **other_unrelated** | 100.00% | 83.33% | 90.91% | 18 |

---

## 3. Confusion Matrix

```
[[18  0  0  0  0  0]
 [ 0 18  0  0  0  0]
 [ 0  0 18  0  0  0]
 [ 0  0  0 18  0  0]
 [ 0  0  0  0 18  0]
 [ 0  1  2  0  0 15]]
```

Visual confusion matrix plot available at: [`backend/metrics/plots/validator_confusion_matrix.png`](file:///c:/bonetest/backend/metrics/plots/validator_confusion_matrix.png)