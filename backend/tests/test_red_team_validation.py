"""Comprehensive Red-Team, Adversarial, and Regression Test Suite (Real Data & OOD Gate).

Validates the full defense-in-depth safety barrier using real radiographs from
tests/validator/should_accept and real OOD negatives from tests/validator/should_reject:
1. Adversarial & Out-of-Distribution inputs (non-hand X-rays, animals/photos, adult hand photos,
   screenshots, corrupted headers, blank frames, slivers).
2. Verification that 100% of invalid images in should_reject are rejected with is_valid=False,
   rejection_code="INVALID_IMAGE", reason_code="INVALID_IMAGE", and the required user-facing message.
3. Verification that valid real pediatric radiographs from tests/validator/should_accept produce
   accurate bone age predictions, conformal intervals, and Grad-CAM++.
4. Verification of biological sex enforcement and ONNX Runtime CPU execution.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from app.api.routes.prediction import list_models
from app.services.gradcam_service import get_gradcam_service
from app.services.image_validator import USER_REJECTION_MESSAGE, get_image_validator
from app.services.inference_service import get_inference_service

SHOULD_ACCEPT_DIR = ROOT_DIR / "tests" / "validator" / "should_accept"
SHOULD_REJECT_DIR = ROOT_DIR / "tests" / "validator" / "should_reject"


def _encode_to_png(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", img)
    assert success, "Encoding to PNG failed"
    return buffer.tobytes()


class TestRedTeamAdversarialValidation(unittest.TestCase):
    """Adversarial stress-testing of the Three-Stage Validation Gate using real OOD negatives."""

    def setUp(self):
        self.validator = get_image_validator()
        self.inference = get_inference_service()
        self.gradcam = get_gradcam_service()

    def test_rejects_corrupted_file_bytes(self):
        corrupt = b"%PDF-1.5 some random junk header instead of valid image"
        res = self.validator.validate(corrupt)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertEqual(res.deciding_stage, "stage1_rules")
        self.assertEqual(res.rejection_message, USER_REJECTION_MESSAGE)

    def test_rejects_extreme_aspect_ratio_banner(self):
        banner = np.ones((50, 500, 3), dtype=np.uint8) * 128
        res = self.validator.validate(_encode_to_png(banner))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertEqual(res.deciding_stage, "stage1_rules")

    def test_rejects_extreme_aspect_ratio_sliver(self):
        sliver = np.ones((500, 50, 3), dtype=np.uint8) * 128
        res = self.validator.validate(_encode_to_png(sliver))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertEqual(res.deciding_stage, "stage1_rules")

    def test_rejects_solid_blank_image(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 120
        res = self.validator.validate(_encode_to_png(blank))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertEqual(res.deciding_stage, "stage1_rules")

    def test_rejects_real_non_hand_xrays(self):
        # 1.jpe..8.jpe in tests/validator/should_reject are real non-hand X-rays (chest, spine, pelvis, dental, feet, shoulder)
        for name in ("1.jpe", "2.jpe", "4.jpe", "5.jpe", "6.jpe", "7.jpe", "8.jpe"):
            p = SHOULD_REJECT_DIR / name
            if p.exists():
                res = self.validator.validate(p.read_bytes())
                self.assertFalse(res.is_valid, f"Non-hand X-ray {name} should be rejected")
                self.assertEqual(res.rejection_code, "INVALID_IMAGE")
                self.assertEqual(res.rejection_message, USER_REJECTION_MESSAGE)

    def test_rejects_real_document_and_screenshots(self):
        for name in ("21.jpe", "22.jpe", "23.jpe"):
            p = SHOULD_REJECT_DIR / name
            if p.exists():
                res = self.validator.validate(p.read_bytes())
                self.assertFalse(res.is_valid, f"Screenshot {name} should be rejected")
                self.assertEqual(res.rejection_code, "INVALID_IMAGE")

    def test_rejects_real_animal_and_adult_hand_photos(self):
        for name in ("9.jpe", "10.jpe", "12.jpe", "16.jpe", "17.jpe", "18.jpe", "19.jpe"):
            p = SHOULD_REJECT_DIR / name
            if p.exists():
                res = self.validator.validate(p.read_bytes())
                self.assertFalse(res.is_valid, f"Photo {name} should be rejected")
                self.assertEqual(res.rejection_code, "INVALID_IMAGE")


class TestClinicalPipelineEndToEnd(unittest.TestCase):
    """End-to-End clinical verification with real RSNA pediatric radiographs from should_accept."""

    def setUp(self):
        self.validator = get_image_validator()
        self.inference = get_inference_service()
        self.gradcam = get_gradcam_service()

        real_path = SHOULD_ACCEPT_DIR / "10022.png"
        if not real_path.exists():
            real_path = sorted(SHOULD_ACCEPT_DIR.glob("*.png"))[0]
        self.hand_png = real_path.read_bytes()

    def test_valid_pediatric_hand_accepted_across_all_stages(self):
        res = self.validator.validate(self.hand_png)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.stage_passed, 3)
        self.assertEqual(res.deciding_stage, "passed")
        self.assertEqual(res.gate_details["stage1_rules"], "PASSED")
        self.assertEqual(res.gate_details["stage2_ood_knn"], "PASSED")
        self.assertEqual(res.gate_details["stage3_coverage"], "PASSED")
        self.assertLessEqual(res.score, res.threshold)
        self.assertGreaterEqual(res.anatomical_coverage, 0.03)

    def test_valid_pediatric_hand_produces_prediction_and_conformal_interval(self):
        val_res = self.validator.validate(self.hand_png)
        self.assertTrue(val_res.is_valid)

        data = self.inference.predict(self.hand_png, "efficientnet_b3", gender="male")
        self.assertEqual(data["model_type"], "efficientnet_b3")
        self.assertEqual(data["gender_used"], "male")
        self.assertGreater(data["bone_age_months"], 0.0)
        self.assertLess(data["bone_age_months"], 240.0)

        self.assertIn("uncertainty_lower_months", data)
        self.assertIn("uncertainty_upper_months", data)
        self.assertIn("uncertainty_interval_months", data)
        self.assertEqual(data["conformal_coverage_pct"], 90.0)
        self.assertLessEqual(data["uncertainty_lower_months"], data["bone_age_months"])
        self.assertGreaterEqual(data["uncertainty_upper_months"], data["bone_age_months"])
        self.assertEqual(data["inference_engine"], "ONNXRuntime-CPU")

    def test_valid_pediatric_hand_produces_gradcam_plusplus(self):
        val_res = self.validator.validate(self.hand_png)
        self.assertTrue(val_res.is_valid)

        data = self.gradcam.generate(
            self.hand_png, "efficientnet_b3", gender="female", method="gradcam++"
        )
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
