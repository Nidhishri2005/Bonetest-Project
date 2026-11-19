"""Unit tests for Phase 4 Comprehensive Evaluation Suite:
- Regression metrics calculation (MAE, RMSE, MedAE, R2, etc.)
- Non-parametric bootstrap 95% confidence intervals
- Bland-Altman agreement statistics
- Paired bootstrap hypothesis test for model comparison
- Stratified demographic and pediatric cohort metrics
- Ablation study report compilation
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Add project root and backend to sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from ml.eval.ablation import ABLATION_CONFIGS, generate_ablation_report
from ml.eval.metrics import (
    compute_bland_altman,
    compute_bootstrap_ci,
    compute_regression_metrics,
    compute_stratified_metrics,
    paired_bootstrap_test,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for statistical regression metrics."""

    def test_compute_regression_metrics_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        # Residuals: [+2, -4, 0, +6] -> abs: [2, 4, 0, 6]
        y_pred = np.array([12.0, 16.0, 30.0, 46.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertEqual(metrics["mae"], 3.0)  # (2 + 4 + 0 + 6) / 4 = 3.0
        self.assertAlmostEqual(metrics["rmse"], np.sqrt((4 + 16 + 0 + 36) / 4.0), places=3)
        self.assertEqual(metrics["medae"], 3.0)
        self.assertEqual(metrics["pct_within_6mo"], 100.0)  # all abs <= 6
        self.assertEqual(metrics["pct_within_12mo"], 100.0)
        self.assertEqual(metrics["max_error"], 6.0)
        self.assertEqual(metrics["n_samples"], 4)

    def test_compute_regression_metrics_validation_errors(self):
        with self.assertRaises(ValueError):
            compute_regression_metrics([1, 2], [1])  # Length mismatch
        with self.assertRaises(ValueError):
            compute_regression_metrics([], [])  # Empty

    def test_compute_bootstrap_ci(self):
        np.random.seed(42)
        y_true = np.linspace(10, 200, 100)
        # Add random noise
        y_pred = y_true + np.random.normal(0, 5, 100)

        def mae_fn(t, p): return np.mean(np.abs(p - t))

        point, low, high = compute_bootstrap_ci(y_true, y_pred, mae_fn, n_bootstraps=200, seed=42)

        self.assertGreater(point, 0.0)
        self.assertLess(low, point)
        self.assertGreater(high, point)
        self.assertLess(low, high)

    def test_compute_bland_altman(self):
        y_true = np.array([100.0, 120.0, 140.0, 160.0])
        # Diff: [+2, -2, +4, -4] -> mean diff = 0.0
        y_pred = np.array([102.0, 118.0, 144.0, 156.0])

        ba = compute_bland_altman(y_true, y_pred)

        self.assertAlmostEqual(ba["mean_bias"], 0.0, places=2)
        self.assertGreater(ba["sd_bias"], 0.0)
        self.assertLess(ba["lower_limit"], 0.0)
        self.assertGreater(ba["upper_limit"], 0.0)
        self.assertEqual(ba["outlier_count"], 0)

    def test_paired_bootstrap_test(self):
        np.random.seed(42)
        y_true = np.linspace(20, 180, 80)
        # Baseline has larger errors (noise ~15), candidate has smaller errors (noise ~5)
        y_baseline = y_true + np.random.normal(0, 15, 80)
        y_candidate = y_true + np.random.normal(0, 5, 80)

        result = paired_bootstrap_test(y_true, y_baseline, y_candidate, n_bootstraps=200, seed=42)

        self.assertLess(result["delta_mae"], 0.0)
        self.assertLess(result["ci_upper"], 0.0)
        self.assertLess(result["p_value"], 0.05)
        self.assertTrue(result["is_statistically_significant"])

    def test_compute_stratified_metrics(self):
        y_true = np.array([12.0, 48.0, 96.0, 144.0, 180.0])
        y_pred = y_true + np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        males = np.array([True, False, True, False, True])

        strat = compute_stratified_metrics(y_true, y_pred, males)

        self.assertIn("by_sex", strat)
        self.assertIn("by_cohort", strat)
        self.assertEqual(strat["by_sex"]["Male"]["n_samples"], 3)
        self.assertEqual(strat["by_sex"]["Female"]["n_samples"], 2)


class TestAblationStudy(unittest.TestCase):
    """Test suite for ablation report compilation."""

    def test_ablation_report_generation(self):
        mock_results = {
            "target_production": {"mae": 6.1, "rmse": 8.0, "r2": 0.96, "pct_within_6mo": 68.0, "pct_within_12mo": 92.0},
            "no_sex_conditioning": {"mae": 7.4, "rmse": 9.8, "r2": 0.94, "pct_within_6mo": 56.0, "pct_within_12mo": 84.0},
            "no_roi_crop": {"mae": 8.1, "rmse": 10.7, "r2": 0.93, "pct_within_6mo": 50.0, "pct_within_12mo": 80.0},
            "resolution_512": {"mae": 6.0, "rmse": 7.9, "r2": 0.96, "pct_within_6mo": 69.0, "pct_within_12mo": 92.0},
            "backbone_resnet50": {"mae": 6.8, "rmse": 8.9, "r2": 0.95, "pct_within_6mo": 62.0, "pct_within_12mo": 88.0},
            "legacy_baseline_resnet18": {"mae": 10.9, "rmse": 14.1, "r2": 0.88, "pct_within_6mo": 34.0, "pct_within_12mo": 63.0},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_ablation.json"
            report = generate_ablation_report(mock_results, output_path=out_file)

            self.assertIn("configurations", report)
            self.assertEqual(len(report["configurations"]), len(ABLATION_CONFIGS))
            self.assertTrue(out_file.exists())
            self.assertTrue(out_file.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
