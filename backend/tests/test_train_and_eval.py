"""Unit tests for Phase 3 training and evaluation modules:
- Legacy 512x512 preprocessing and letterbox resizing
- Cosine schedule with linear warmup
- Deterministic seeding and metadata retrieval
- EfficientNet-B3 Sex-Fusion forward and backward training smoke test
"""

from __future__ import annotations

import sys
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
import torch
import torch.nn as nn
from torch.optim import AdamW

from ml.eval.evaluate_baseline import letterbox_resize_512, preprocess_legacy_512
from ml.models.efficientnet import EfficientNetB3SexFusion
from ml.train.train import (
    get_cosine_schedule_with_warmup,
    get_git_revision_hash,
    set_seed,
)


class TestEvaluateBaselineUtilities(unittest.TestCase):
    """Test suite for legacy evaluation helpers."""

    def test_letterbox_resize_512(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        resized = letterbox_resize_512(img, target_size=512)
        self.assertEqual(resized.shape, (512, 512))
        self.assertEqual(resized.dtype, np.uint8)

    def test_preprocess_legacy_512(self):
        img = np.full((300, 300), 128, dtype=np.uint8)
        tensor = preprocess_legacy_512(img, mean=0.4523, std=0.2118)
        self.assertEqual(tensor.shape, (1, 1, 512, 512))
        self.assertEqual(tensor.dtype, torch.float32)


class TestTrainUtilities(unittest.TestCase):
    """Test suite for training utilities and schedulers."""

    def test_set_seed_reproducibility(self):
        set_seed(42)
        r1 = torch.rand(5)
        set_seed(42)
        r2 = torch.rand(5)
        self.assertTrue(torch.equal(r1, r2))

    def test_git_revision_hash(self):
        commit_hash = get_git_revision_hash()
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)

    def test_cosine_schedule_warmup_and_decay(self):
        dummy_param = nn.Parameter(torch.zeros(1))
        optimizer = AdamW([dummy_param], lr=1e-3)
        warmup_steps = 10
        total_steps = 100

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.01,
        )

        # Step 0: Initial LR should be 0.0
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0, places=5)

        # Step to end of warmup: should reach base lr 1e-3
        for _ in range(warmup_steps):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-3, places=5)

        # Step to total steps: should decay to min_lr_ratio * base_lr (1e-5)
        for _ in range(total_steps - warmup_steps):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-5, places=5)

    def test_efficientnet_training_smoke_step(self):
        """Smoke test: 1-step forward pass, Huber loss calculation, and backward pass."""
        model = EfficientNetB3SexFusion(pretrained=False, sex_embed_dim=16, dropout=0.1)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss(beta=1.0)

        # Dummy batch: 2 samples, 3 channels, 128x128 for quick smoke
        images = torch.randn(2, 3, 128, 128)
        sex = torch.tensor([0, 1], dtype=torch.long)
        targets = torch.tensor([0.2, -0.5], dtype=torch.float32)

        preds = model(images, sex)
        self.assertEqual(preds.shape, (2,))

        loss = criterion(preds, targets)
        self.assertFalse(torch.isnan(loss))

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
