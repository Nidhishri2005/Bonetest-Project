# ONNX Runtime Export and Numerical Parity Report

| Model Name | Output File | Max Abs Diff | Mean Abs Diff | Tolerance | Parity Status |
|---|---|---|---|---|---|
| `validator` | `hand_validator.onnx` | `4.768372e-06` | `2.125899e-06` | `1.0e-04` | **PASSED** |
| `efficientnet_b3` | `efficientnet_b3.onnx` | `3.725290e-09` | `3.725290e-09` | `1.0e-04` | **PASSED** |

### Optimization & Deployment Summary

- **Engine**: ONNX Runtime (CPUExecutionProvider) with sequential execution and thread limit = 2.
- **Dynamic Axes**: Batch size axis 0 is fully dynamic for both models.
- **Numerical Parity**: All exported models satisfy the strict `< 1e-4` maximum absolute error criterion.
