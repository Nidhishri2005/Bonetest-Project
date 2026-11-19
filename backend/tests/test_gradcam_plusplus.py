"""Unit tests for Phase 6 Explainability:
- Grad-CAM and Grad-CAM++ heatmap generation
- Spatial normalization and range bounds [0, 1]
- Adebayo et al. (2018) Cascading Parameter Randomization Test
- Heatmap similarity metrics (Pearson correlation, MAD)
- GradCamService integration with model registry and sex conditioning
"""

from __future__ import annotations

import io
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
from PIL import Image

from app.services.gradcam_service import (
    GradCAM,
    GradCAMPlusPlus,
    GradCamService,
    get_gradcam_service,
)
from ml.eval.gradcam_sanity import (
    compute_heatmap_similarity,
    randomize_layer_weights,
    run_cascading_randomization_test,
)


class DummyConvNet(nn.Module):
    """Small CNN for fast deterministic test execution without loading 50MB checkpoints."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(32, 1)

    def forward(self, x, sex=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_gradcam_target_layer(self):
        return self.conv2


class TestGradCAMCore(unittest.TestCase):
    """Test core GradCAM and GradCAMPlusPlus algorithm implementations."""

    def setUp(self):
        self.model = DummyConvNet().eval()
        self.input_tensor = torch.randn(1, 1, 64, 64)

    def test_gradcam_standard_output(self):
        target_layer = self.model.get_gradcam_target_layer()
        explainer = GradCAM(self.model, target_layer)
        try:
            heatmap = explainer.generate(self.input_tensor)
        finally:
            explainer.remove_hooks()

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_gradcam_plusplus_output(self):
        target_layer = self.model.get_gradcam_target_layer()
        explainer = GradCAMPlusPlus(self.model, target_layer)
        try:
            heatmap = explainer.generate(self.input_tensor)
        finally:
            explainer.remove_hooks()

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_hooks_cleanup(self):
        target_layer = self.model.get_gradcam_target_layer()
        explainer = GradCAMPlusPlus(self.model, target_layer)
        self.assertGreater(len(explainer._hooks), 0)
        explainer.remove_hooks()
        # Ensure model forward/backward works without hooks firing
        out = self.model(self.input_tensor)
        out.backward()
        self.assertIsNotNone(out)


class TestAdebayoSanityChecks(unittest.TestCase):
    """Test Adebayo et al. (2018) Cascading Parameter Randomization sanity check."""

    def test_heatmap_similarity_identity(self):
        h1 = np.ones((32, 32), dtype=np.float32) * 0.5
        h1[10:20, 10:20] = 1.0
        corr, mad = compute_heatmap_similarity(h1, h1)
        self.assertAlmostEqual(corr, 1.0, places=3)
        self.assertAlmostEqual(mad, 0.0, places=3)

    def test_heatmap_similarity_different(self):
        h1 = np.zeros((32, 32), dtype=np.float32)
        h1[:16, :] = 1.0
        h2 = np.zeros((32, 32), dtype=np.float32)
        h2[:, :16] = 1.0
        corr, mad = compute_heatmap_similarity(h1, h2)
        self.assertLess(corr, 0.5)
        self.assertGreater(mad, 0.0)

    def test_randomize_layer_weights(self):
        linear = nn.Linear(10, 10)
        orig_weights = linear.weight.clone()
        randomize_layer_weights(linear, seed=123)
        self.assertFalse(torch.allclose(orig_weights, linear.weight))

    def test_cascading_randomization_execution(self):
        model = DummyConvNet().eval()
        input_tensor = torch.randn(1, 1, 64, 64)

        result = run_cascading_randomization_test(
            model=model,
            input_tensor=input_tensor,
            gradcam_generator_cls=GradCAMPlusPlus,
            seed=42,
        )

        self.assertIn("passed_sanity_check", result)
        self.assertIn("stages", result)
        self.assertGreater(len(result["stages"]), 0)
        # Verify first stage is recorded
        first_stage = result["stages"][0]
        self.assertIn("pearson_correlation", first_stage)
        self.assertIn("mean_absolute_difference", first_stage)
        self.assertIn("clinical_interpretation", result)


class TestGradCamServiceIntegration(unittest.TestCase):
    """Test high-level GradCamService end-to-end integration."""

    @classmethod
    def setUpClass(cls):
        # Create a valid synthetic PNG image
        img = Image.new("L", (128, 128), color=120)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.valid_png_bytes = buf.getvalue()

    def test_service_initialization(self):
        service = get_gradcam_service()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.registry)

    def test_service_generate_gradcam_plusplus(self):
        service = get_gradcam_service()
        # Test on baseline 'cnn' model
        result = service.generate(
            image_bytes=self.valid_png_bytes,
            model_type="cnn",
            method="gradcam++",
        )
        self.assertEqual(result["model_type"], "cnn")
        self.assertEqual(result["method"], "gradcam++")
        self.assertIn("heatmap_base64", result)
        self.assertIn("overlay_base64", result)
        self.assertIn("bone_age_months", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["processing_time_ms"], 0)

    def test_service_generate_gradcam_standard(self):
        service = get_gradcam_service()
        result = service.generate(
            image_bytes=self.valid_png_bytes,
            model_type="cnn",
            method="gradcam",
        )
        self.assertEqual(result["method"], "gradcam")
        self.assertIn("heatmap_base64", result)

    def test_service_unsupported_model_raises_error(self):
        service = get_gradcam_service()
        # 'rf' does not support Grad-CAM
        with self.assertRaises(ValueError):
            service.generate(
                image_bytes=self.valid_png_bytes,
                model_type="rf",
                method="gradcam++",
            )


if __name__ == "__main__":
    unittest.main()
