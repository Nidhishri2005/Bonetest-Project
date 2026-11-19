# Adebayo et al. (2018) Sanity Check for Saliency Methods

**Method:** Grad-CAM++
**Test Status:** PASSED
**Baseline Correlation:** 1.0000
**Final Randomized Correlation:** 0.0000

### Cascading Randomization Steps

| Stage | Layer Randomized | Pearson Correlation | Mean Absolute Diff (MAD) |
|---|---|---|---|
| 1 | `fc_head` | 0.9984 | 0.0012 |
| 2 | `backbone_layer4` | 0.0000 | 0.0163 |

### Clinical & Methodological Interpretation

> PASSED: Explanations are model-sensitive and reflect trained anatomical representations.
