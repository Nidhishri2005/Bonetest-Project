"""Unit tests for Phase 7 ONNX Export and CPU Serving Integration:
- Export of EfficientNetB3SexFusion and HandXRayValidator models to ONNX
- Numerical parity verification (max absolute difference < 1e-4) between PyTorch and ONNX Runtime CPU
- InferenceService ONNX execution provider dispatch
- Conformal uncertainty intervals and biological sex validation
"""

from __future__ import annotations

import io
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
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES
from app.services.inference_service import InferenceService, get_inference_service
from ml.export.onnx_exporter import (
    ONNXModelSession,
    export_efficientnet_to_onnx,
    export_validator_to_onnx,
    verify_onnx_parity,
)
from ml.models.efficientnet import EfficientNetB3SexFusion


class TestONNXExportAndParity(unittest.TestCase):
    """Test ONNX model export and numerical equivalence with PyTorch."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_validator_to_onnx_and_check(self):
        model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES)).eval()
        onnx_file = self.output_dir / "test_validator.onnx"
        exported_path = export_validator_to_onnx(model, onnx_file)

        self.assertTrue(exported_path.exists())
        self.assertGreater(exported_path.stat().st_size, 1000)

        # Check graph
        loaded = onnx.load(str(exported_path))
        onnx.checker.check_model(loaded)

    def test_validator_numerical_parity(self):
        model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES)).eval()
        onnx_file = self.output_dir / "test_val_parity.onnx"
        export_validator_to_onnx(model, onnx_file)

        dummy_img = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        parity = verify_onnx_parity(
            pytorch_model=model,
            onnx_path=onnx_file,
            dummy_inputs=dummy_img,
            input_names=["image"],
            tolerance=1e-4,
        )

        self.assertTrue(parity["parity_passed"])
        self.assertLess(parity["max_absolute_difference"], 1e-4)

    def test_export_efficientnet_to_onnx_and_check(self):
        model = EfficientNetB3SexFusion(pretrained=False).eval()
        onnx_file = self.output_dir / "test_effnet.onnx"
        exported_path = export_efficientnet_to_onnx(model, onnx_file)

        self.assertTrue(exported_path.exists())
        self.assertGreater(exported_path.stat().st_size, 1000)

        loaded = onnx.load(str(exported_path))
        onnx.checker.check_model(loaded)

    def test_efficientnet_numerical_parity(self):
        model = EfficientNetB3SexFusion(pretrained=False).eval()
        onnx_file = self.output_dir / "test_effnet_parity.onnx"
        export_efficientnet_to_onnx(model, onnx_file)

        dummy_img = torch.randn(2, 1, 384, 384, dtype=torch.float32)
        dummy_sex = torch.tensor([1, 0], dtype=torch.long)
        parity = verify_onnx_parity(
            pytorch_model=model,
            onnx_path=onnx_file,
            dummy_inputs=(dummy_img, dummy_sex),
            input_names=["image", "sex"],
            tolerance=1e-4,
        )

        self.assertTrue(parity["parity_passed"])
        self.assertLess(parity["max_absolute_difference"], 1e-4)


class TestONNXServingIntegration(unittest.TestCase):
    """Test InferenceService ONNX CPU execution provider and conformal intervals."""

    @classmethod
    def setUpClass(cls):
        # Create a valid synthetic image
        img = Image.new("L", (256, 256), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.valid_png_bytes = buf.getvalue()

    def test_inference_service_efficientnet_onnx_serving(self):
        service = get_inference_service()
        # Verify ONNX session is available for efficientnet_b3
        session = service.get_onnx_session("efficientnet_b3")
        self.assertIsNotNone(session, "ONNX session for efficientnet_b3 should be available in checkpoints/")

        result = service.predict(
            image_bytes=self.valid_png_bytes,
            model_type="efficientnet_b3",
            gender="female",
        )

        self.assertEqual(result["model_type"], "efficientnet_b3")
        self.assertEqual(result["gender_used"], "female")
        self.assertEqual(result["inference_engine"], "ONNXRuntime-CPU")
        # efficientnet_b3 does not have a measured interval yet, so interval fields are None
        self.assertIsNone(result["uncertainty_lower_months"])
        self.assertIsNone(result["uncertainty_upper_months"])
        self.assertIsNone(result["uncertainty_interval_months"])
        self.assertIsNone(result["conformal_coverage_pct"])
        self.assertGreaterEqual(result["bone_age_months"], 0.0)
        self.assertLessEqual(result["bone_age_months"], 240.0)

    def test_inference_service_efficientnet_requires_gender(self):
        service = get_inference_service()
        with self.assertRaises(ValueError):
            service.predict(
                image_bytes=self.valid_png_bytes,
                model_type="efficientnet_b3",
                gender=None,
            )

    def test_inference_service_legacy_cnn_fallback(self):
        service = get_inference_service()
        result = service.predict(
            image_bytes=self.valid_png_bytes,
            model_type="cnn",
        )
        self.assertEqual(result["model_type"], "cnn")
        self.assertEqual(result["inference_engine"], "PyTorch-CPU")
        self.assertIn("uncertainty_lower_months", result)
        self.assertIn("uncertainty_upper_months", result)


if __name__ == "__main__":
    unittest.main()
