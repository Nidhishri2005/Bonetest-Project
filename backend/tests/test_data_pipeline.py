"""Unit tests for Phase 2 Data Pipeline modules:
- Preprocessing (ROI crop, CLAHE, letterbox resize)
- Perceptual hash deduplication (dHash, Hamming distance)
- Stratified splitting & manifest integrity
- RSNABoneAgeDataset loading & tensor shapes
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Add root directory to sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
import pandas as pd
import torch

from ml.data.dedup import compute_dhash, hamming_distance
from ml.data.preprocess import (
    apply_clahe,
    crop_hand_roi,
    preprocess_radiograph,
    resize_letterbox,
)
from ml.data.split import create_stratified_splits
from ml.data.dataset import RSNABoneAgeDataset


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for radiograph preprocessing functions."""

    def setUp(self):
        # Create a synthetic 500x400 image with a bright rectangle (simulating hand) in center
        self.img = np.zeros((500, 400), dtype=np.uint8)
        self.img[100:400, 100:300] = 200  # Bright centered region

    def test_crop_hand_roi_finds_component(self):
        cropped = crop_hand_roi(self.img, margin_ratio=0.05, min_area_ratio=0.10)
        self.assertIsNotNone(cropped)
        # Cropped should be smaller than original image because background is clipped
        self.assertLess(cropped.shape[0], self.img.shape[0])
        self.assertLess(cropped.shape[1], self.img.shape[1])
        # Cropped area should contain bright pixels
        self.assertGreater(cropped.mean(), 100)

    def test_crop_hand_roi_fallback_on_blank_image(self):
        blank = np.zeros((200, 200), dtype=np.uint8)
        fallback = crop_hand_roi(blank, min_area_ratio=0.15)
        # If no component exceeds min_area_ratio, returns original image
        self.assertEqual(fallback.shape, blank.shape)

    def test_apply_clahe(self):
        enhanced = apply_clahe(self.img, clip_limit=2.0, tile_grid_size=(8, 8))
        self.assertEqual(enhanced.shape, self.img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_resize_letterbox_maintains_square_target(self):
        letterboxed = resize_letterbox(self.img, target_size=384)
        self.assertEqual(letterboxed.shape, (384, 384))
        self.assertEqual(letterboxed.dtype, np.uint8)

    def test_preprocess_radiograph_end_to_end(self):
        out = preprocess_radiograph(self.img, target_size=384, roi_crop=True, clahe=True)
        self.assertEqual(out.shape, (384, 384))
        self.assertEqual(out.dtype, np.uint8)


class TestDeduplication(unittest.TestCase):
    """Test suite for dHash perceptual hashing and Hamming distance."""

    def test_dhash_identical_images(self):
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        h1 = compute_dhash(img)
        h2 = compute_dhash(img)
        self.assertEqual(h1, h2)
        self.assertEqual(hamming_distance(h1, h2), 0)

    def test_dhash_distinct_images(self):
        img1 = np.zeros((128, 128), dtype=np.uint8)
        # Gradient pattern
        img2 = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
        h1 = compute_dhash(img1)
        h2 = compute_dhash(img2)
        dist = hamming_distance(h1, h2)
        self.assertGreater(dist, 0)


class TestStratifiedSplits(unittest.TestCase):
    """Test suite for age-bin x sex stratified partitioning."""

    def test_splits_disjoint_and_proportions(self):
        # Construct synthetic cohort guaranteed to have >= 10 items in all 10 strata
        cohorts = [12.0, 48.0, 96.0, 144.0, 192.0]  # Representatives of 5 bins
        records = []
        c = 0
        for age in cohorts:
            for sex in [True, False]:
                for _ in range(20):  # 20 samples per stratum * 10 strata = 200 total
                    records.append({"id": f"img_{c}", "boneage": age, "male": sex})
                    c += 1
        df = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create dummy images
            for img_id in df["id"]:
                dummy = np.zeros((32, 32), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{img_id}.png"), dummy)

            train_df, val_df, test_df = create_stratified_splits(
                df,
                image_dir=tmp_path,
                train_ratio=0.80,
                val_ratio=0.10,
                test_ratio=0.10,
                seed=42,
            )

            # Check sizes
            total_split = len(train_df) + len(val_df) + len(test_df)
            self.assertEqual(total_split, 200)
            self.assertEqual(len(train_df), 160)
            self.assertEqual(len(val_df), 20)
            self.assertEqual(len(test_df), 20)

            # Check disjointness
            train_ids = set(train_df["id"])
            val_ids = set(val_df["id"])
            test_ids = set(test_df["id"])
            self.assertEqual(len(train_ids.intersection(val_ids)), 0)
            self.assertEqual(len(train_ids.intersection(test_ids)), 0)
            self.assertEqual(len(val_ids.intersection(test_ids)), 0)


class TestRSNABoneAgeDataset(unittest.TestCase):
    """Test suite for PyTorch Dataset wrapping."""

    def test_dataset_tensor_shapes_and_types(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.DataFrame({
                "id": [101, 102],
                "boneage": [120.0, 72.0],
                "male": [True, False],
            })

            for img_id in df["id"]:
                dummy = np.full((100, 100), 128, dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{img_id}.png"), dummy)

            ds = RSNABoneAgeDataset(
                manifest=df,
                image_dir=tmp_path,
                target_size=384,
                norm_mean=0.45,
                norm_std=0.22,
                mean_age=120.0,
                std_age=40.0,
            )

            self.assertEqual(len(ds), 2)
            sample = ds[0]

            # Verify keys
            self.assertIn("image", sample)
            self.assertIn("sex", sample)
            self.assertIn("target", sample)
            self.assertIn("boneage_months", sample)
            self.assertIn("id", sample)

            # Verify shapes and types
            self.assertEqual(sample["image"].shape, (3, 384, 384))
            self.assertEqual(sample["image"].dtype, torch.float32)
            self.assertEqual(sample["sex"].dtype, torch.long)
            self.assertEqual(sample["target"].dtype, torch.float32)
            self.assertEqual(sample["sex"].item(), 1)  # male == True -> 1


if __name__ == "__main__":
    unittest.main()
