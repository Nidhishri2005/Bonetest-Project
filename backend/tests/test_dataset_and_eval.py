"""Unit tests for dataset pipeline, augmentations, and evaluation metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.augmentation import TrainAugmentation
from ml.dataset import train_val_test_split, verify_dataset_images
from ml.evaluate import compute_metrics


class TestDatasetAndMetrics(unittest.TestCase):
    def setUp(self):
        # Create synthetic dataset with 100 samples
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "id": [f"img_{i}" for i in range(n)],
                "boneage": np.random.uniform(10.0, 220.0, size=n),
                "male": np.random.choice([0, 1], size=n),
                "patient_id": [f"patient_{i // 2}" for i in range(n)],  # 2 images per patient
            }
        )

    def test_train_val_test_split_ratios_patient_grouped(self):
        train_df, val_df, test_df = train_val_test_split(
            self.df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            patient_col="patient_id",
        )
        total_len = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_len, len(self.df))

        # Check zero patient data leakage
        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        test_patients = set(test_df["patient_id"])

        self.assertEqual(len(train_patients.intersection(val_patients)), 0)
        self.assertEqual(len(train_patients.intersection(test_patients)), 0)
        self.assertEqual(len(val_patients.intersection(test_patients)), 0)

    def test_train_val_test_split_stratified_age_bins(self):
        # Drop patient_id to test age-bin stratification
        df_no_patient = self.df.drop(columns=["patient_id"])
        train_df, val_df, test_df = train_val_test_split(
            df_no_patient,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df_no_patient))
        self.assertAlmostEqual(len(train_df) / len(df_no_patient), 0.70, delta=0.05)
        self.assertAlmostEqual(len(val_df) / len(df_no_patient), 0.15, delta=0.05)
        self.assertAlmostEqual(len(test_df) / len(df_no_patient), 0.15, delta=0.05)

    def test_train_augmentation_output_shape_and_dtype(self):
        aug = TrainAugmentation()
        dummy_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        augmented = aug(dummy_img)["image"]
        self.assertEqual(augmented.shape, (512, 512))
        self.assertEqual(augmented.dtype, np.uint8)
        self.assertGreaterEqual(augmented.min(), 0)
        self.assertLessEqual(augmented.max(), 255)

    def test_compute_metrics_perfect_predictions(self):
        y_true = np.array([12.0, 36.0, 72.0, 120.0, 180.0])
        y_pred = np.array([12.0, 36.0, 72.0, 120.0, 180.0])
        metrics = compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["mean_error"], 0.0)
        self.assertAlmostEqual(metrics["median_ae"], 0.0)
        self.assertAlmostEqual(metrics["within_6m_pct"], 100.0)
        self.assertAlmostEqual(metrics["within_12m_pct"], 100.0)

    def test_compute_metrics_known_residuals(self):
        y_true = np.array([10.0, 50.0, 100.0, 150.0])
        # Add constant bias of +5 months
        y_pred = np.array([15.0, 55.0, 105.0, 155.0])
        metrics = compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["mae"], 5.0)
        self.assertAlmostEqual(metrics["mse"], 25.0)
        self.assertAlmostEqual(metrics["rmse"], 5.0)
        self.assertAlmostEqual(metrics["mean_error"], 5.0)
        self.assertAlmostEqual(metrics["median_ae"], 5.0)
        self.assertAlmostEqual(metrics["within_6m_pct"], 100.0)
        self.assertAlmostEqual(metrics["within_12m_pct"], 100.0)
        self.assertIn("infancy_0_24", metrics["cohorts"])


if __name__ == "__main__":
    unittest.main()
