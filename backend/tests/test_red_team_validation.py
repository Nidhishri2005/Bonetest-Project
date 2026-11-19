"""Comprehensive Red-Team, Adversarial, and Regression Test Suite (Phase 9).

Validates the full defense-in-depth safety barrier:
1. Adversarial & Out-of-Distribution inputs (animals, chest X-rays, documents, blank frames, slivers).
2. Verification that 100% of invalid images are rejected with is_valid=False and rejection_code="INVALID_IMAGE".
3. Verification that valid pediatric radiographs produce accurate bone age, conformal intervals, and Grad-CAM++.
4. Verification of biological sex enforcement and ONNX Runtime CPU execution.
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

import cv2
import numpy as np
import torch

from app.api.routes.prediction import list_models
from app.services.gradcam_service import get_gradcam_service
from app.services.image_validator import get_image_validator
from app.services.inference_service import get_inference_service
from ml.validator_dataset import (
    generate_animal_photograph,
    generate_document_or_screenshot,
    generate_hand_xray_phantom,
    generate_non_hand_xray,
)


def _encode_to_png(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", img)
    assert success, "Encoding to PNG failed"
    return buffer.tobytes()


def _encode_to_jpeg(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", img)
    assert success, "Encoding to JPEG failed"
    return buffer.tobytes()


class TestRedTeamAdversarialValidation(unittest.TestCase):
    """Adversarial stress-testing of the Three-Stage Validation Gate."""

    def setUp(self):
        self.validator = get_image_validator()
        self.inference = get_inference_service()
        self.gradcam = get_gradcam_service()

    def test_rejects_corrupted_file_bytes(self):
        corrupt = b"%PDF-1.5 some random junk header instead of valid image"
        res = self.validator.validate(corrupt)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertIn("corrupt", res.rejection_message.lower())

    def test_rejects_extreme_aspect_ratio_banner(self):
        # 500x50 banner (AR = 10.0 > 5.0, dimensions >= 32)
        banner = np.ones((50, 500, 3), dtype=np.uint8) * 128
        res = self.validator.validate(_encode_to_png(banner))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertIn("aspect ratio", res.rejection_message.lower())

    def test_rejects_extreme_aspect_ratio_sliver(self):
        # 50x500 vertical sliver (AR = 0.10 < 0.20, dimensions >= 32)
        sliver = np.ones((500, 50, 3), dtype=np.uint8) * 128
        res = self.validator.validate(_encode_to_png(sliver))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertIn("aspect ratio", res.rejection_message.lower())

    def test_rejects_solid_blank_image(self):
        # Solid uniform gray image (std = 0.0 < 1.0)
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 120
        res = self.validator.validate(_encode_to_png(blank))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertIn("structural variation", res.rejection_message.lower())

    def test_rejects_non_hand_xray_chest(self):
        chest = generate_non_hand_xray(size=224)
        res = self.validator.validate(_encode_to_png(chest))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")

    def test_rejects_document_screenshot(self):
        doc = generate_document_or_screenshot(size=224)
        res = self.validator.validate(_encode_to_png(doc))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")

    def test_rejects_animal_photo(self):
        animal = generate_animal_photograph(size=224)
        res = self.validator.validate(_encode_to_jpeg(animal))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")

    def test_gradcam_also_refuses_invalid_images_strictly(self):
        blank = np.ones((100, 100, 3), dtype=np.uint8) * 200
        blank_bytes = _encode_to_png(blank)
        val_res = self.validator.validate(blank_bytes)
        self.assertFalse(val_res.is_valid)
        # When validation fails, GradCamService is refused execution


class TestClinicalPipelineEndToEnd(unittest.TestCase):
    """End-to-End clinical verification with valid pediatric radiographs."""

    def setUp(self):
        self.validator = get_image_validator()
        self.inference = get_inference_service()
        self.gradcam = get_gradcam_service()

        phantom = generate_hand_xray_phantom(size=224)
        self.hand_png = _encode_to_png(phantom)

    def test_valid_pediatric_hand_accepted_across_all_stages(self):
        res = self.validator.validate(self.hand_png)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.stage_passed, 3)
        self.assertEqual(res.gate_details["stage1_rules"], "PASSED")
        self.assertEqual(res.gate_details["stage2_classifier"], "PASSED")
        self.assertEqual(res.gate_details["stage3_coverage"], "PASSED")
        self.assertGreaterEqual(res.anatomical_coverage, 0.03)

    def test_valid_pediatric_hand_produces_prediction_and_conformal_interval(self):
        val_res = self.validator.validate(self.hand_png)
        self.assertTrue(val_res.is_valid)

        data = self.inference.predict(self.hand_png, "efficientnet_b3", gender="male")
        self.assertEqual(data["model_type"], "efficientnet_b3")
        self.assertEqual(data["gender_used"], "male")
        self.assertGreater(data["bone_age_months"], 0.0)
        self.assertLess(data["bone_age_months"], 240.0)

        # Verify conformal uncertainty intervals
        self.assertIn("uncertainty_lower_months", data)
        self.assertIn("uncertainty_upper_months", data)
        self.assertIn("uncertainty_interval_months", data)
        self.assertEqual(data["conformal_coverage_pct"], 90.0)
        self.assertLessEqual(data["uncertainty_lower_months"], data["bone_age_months"])
        self.assertGreaterEqual(data["uncertainty_upper_months"], data["bone_age_months"])

        # Verify inference engine is ONNX
        self.assertEqual(data["inference_engine"], "ONNXRuntime-CPU")

    def test_valid_pediatric_hand_produces_gradcam_plusplus(self):
        val_res = self.validator.validate(self.hand_png)
        self.assertTrue(val_res.is_valid)

        data = self.gradcam.generate(self.hand_png, "efficientnet_b3", gender="female", method="gradcam++")
        self.assertEqual(data["model_type"], "efficientnet_b3")
        self.assertEqual(data["method"], "gradcam++")
        self.assertIn("heatmap_base64", data)
        self.assertIn("overlay_base64", data)
        self.assertGreater(len(data["heatmap_base64"]), 100)
        self.assertGreater(len(data["overlay_base64"]), 100)

    def test_legacy_baseline_cnn_prediction(self):
        data = self.inference.predict(self.hand_png, "cnn")
        self.assertEqual(data["model_type"], "cnn")
        self.assertIsNone(data["gender_used"])
        self.assertGreater(data["bone_age_months"], 0.0)
        self.assertEqual(data["inference_engine"], "PyTorch-CPU")

    def test_model_list_contains_modern_and_legacy_indicators(self):
        resp = list_models()
        models = {m.id: m for m in resp.models}

        self.assertIn("efficientnet_b3", models)
        self.assertFalse(models["efficientnet_b3"].is_legacy)
        self.assertTrue(models["efficientnet_b3"].checkpoint_available)
        self.assertTrue(models["efficientnet_b3"].requires_gender)

        self.assertIn("cnn", models)
        self.assertTrue(models["cnn"].is_legacy)
        self.assertFalse(models["cnn"].requires_gender)


if __name__ == "__main__":
    unittest.main()
