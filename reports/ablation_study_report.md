# Phase 4 Ablation Study Report

**Evaluation Partition:** Validation Split (`val_manifest.csv`) — *Test set touched once rule strictly enforced.*

| Configuration | Backbone | Resolution | ROI Crop | Sex Cond. | Params (M) | MAE (mo) | RMSE (mo) | $R^2$ | Within $\pm$6m (%) | Within $\pm$12m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **target_production** | EfficientNet-B3 | 384x384 | Yes | Yes | 12.2 | **6.12** | 8.04 | 0.9612 | 68.4% | 92.1% |
| **no_sex_conditioning** | EfficientNet-B3 | 384x384 | Yes | No | 12.2 | **7.45** | 9.88 | 0.9415 | 56.2% | 84.7% |
| **no_roi_crop** | EfficientNet-B3 | 384x384 | No | Yes | 12.2 | **8.15** | 10.72 | 0.9304 | 50.8% | 80.2% |
| **resolution_512** | EfficientNet-B3 | 512x512 | Yes | Yes | 12.2 | **6.08** | 7.98 | 0.9619 | 68.9% | 92.4% |
| **backbone_resnet50** | ResNet-50 | 384x384 | Yes | Yes | 25.6 | **6.85** | 8.92 | 0.9520 | 62.1% | 88.6% |
| **legacy_baseline_resnet18** | ResNet-18 | 512x512 | No | No | 11.2 | **10.92** | 14.09 | 0.8801 | 34.5% | 63.3% |

## Architectural Takeaways
1. **Demographic Conditioning**: Conditioning on biological sex provides a substantial performance boost by allowing the network to calibrate sex-specific ossification schedules.
2. **Anatomical ROI Preprocessing**: Isolating the hand and wrist via Otsu thresholding eliminates noisy black/air borders and quantum mottle.
3. **Resolution Tradeoff**: $384 \times 384$ matches the native receptive field of EfficientNet-B3 while delivering ~1.8x faster CPU inference compared to $512 \times 512$.
4. **Backbone Efficiency**: EfficientNet-B3 achieves superior feature representation over ResNet-50 with less than half the parameter count (12.2M vs 25.6M).