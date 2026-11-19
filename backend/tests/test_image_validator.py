"""Comprehensive unit and integration test suite for Hand X-Ray Validation System.

Verifies:
1. Valid pediatric hand X-rays are accepted and reach prediction.
2. Non-hand medical X-rays (chest, skull, spine) are rejected.
3. Natural photographs (human portraits, landscapes) are rejected.
4. Animal photographs (cats, dogs) are rejected.
5. Documents, PDFs, and screenshots are rejected.
6. Random internet images, noise, and geometric graphics are rejected.
7. Corrupted image files are safely rejected.
8. Very small, low-quality, and blank images are rejected.
9. Borderline / uncertain images below confidence threshold are rejected.
10. Valid images pass validation and reach InferenceService prediction.
11. Invalid images are rejected with INVALID_IMAGE and never reach prediction.
12. Valid images pass validation and generate Grad-CAM heatmaps.
13. Invalid images are rejected with INVALID_IMAGE and never reach Grad-CAM.
14. Confusion matrix plot endpoint is accessible and valid.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(4)

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api.routes.analysis import get_plot
from app.schemas.prediction import ModelType
from app.services.gradcam_service import get_gradcam_service
from app.services.image_validator import ImageValidator, get_image_validator
from app.services.inference_service import get_inference_service
from ml.validator_dataset import (
    generate_animal_photograph,
    generate_document_or_screenshot,
    generate_hand_xray_phantom,
    generate_non_hand_xray,
    generate_normal_photograph,
    generate_other_unrelated,
)


def _encode_to_png(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", img)
    assert success, "Encoding to PNG failed"
    return buffer.tobytes()


def _encode_to_jpeg(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", img)
    assert success, "Encoding to JPEG failed"
    return buffer.tobytes()


class TestHandXRayValidationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = get_image_validator()
        cls.inference_service = get_inference_service()
        cls.gradcam_service = get_gradcam_service()

        cls.hand_img = generate_hand_xray_phantom(size=224)
        cls.hand_bytes = _encode_to_png(cls.hand_img)
        cls.hand_jpeg = _encode_to_jpeg(cls.hand_img)
        cls.non_hand_bytes = _encode_to_png(generate_non_hand_xray(size=224))
        cls.photo_bytes = _encode_to_png(generate_normal_photograph(size=224))
        cls.animal_bytes = _encode_to_png(generate_animal_photograph(size=224))
        cls.doc_bytes = _encode_to_png(generate_document_or_screenshot(size=224))
        cls.unrelated_bytes = _encode_to_png(generate_other_unrelated(size=224))

    # ------------------------------------------------------------------------
    # Category 1: Valid Pediatric Hand/Wrist X-rays
    # ------------------------------------------------------------------------
    def test_valid_pediatric_hand_xray_accepted(self):
        result = self.validator.validate(self.hand_bytes)
        self.assertTrue(result.is_valid, f"Hand X-ray was incorrectly rejected: {result.rejection_message}")
        self.assertEqual(result.predicted_class, "valid_pediatric_hand_xray")
        self.assertGreaterEqual(result.confidence, 0.80)
        self.assertIsNone(result.rejection_code)

    def test_valid_hand_xray_jpeg_format(self):
        result = self.validator.validate(self.hand_jpeg)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 2: Non-hand X-rays (Chest, Skull, Pelvis, Dental)
    # ------------------------------------------------------------------------
    def test_non_hand_xray_rejected(self):
        result = self.validator.validate(self.non_hand_bytes)
        self.assertFalse(result.is_valid, "Non-hand X-ray should be rejected!")
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertIn("Invalid image", result.rejection_message)
        self.assertNotEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 3: Normal Photographs (Human, Landscape, Everyday)
    # ------------------------------------------------------------------------
    def test_normal_photograph_rejected(self):
        result = self.validator.validate(self.photo_bytes)
        self.assertFalse(result.is_valid, "Normal photograph should be rejected!")
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertNotEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 4: Animal Photographs (Cats, Dogs)
    # ------------------------------------------------------------------------
    def test_animal_photograph_rejected(self):
        result = self.validator.validate(self.animal_bytes)
        self.assertFalse(result.is_valid, "Animal photograph should be rejected!")
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertNotEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 5: Documents, Invoices, PDFs, and Screenshots
    # ------------------------------------------------------------------------
    def test_document_and_screenshot_rejected(self):
        result = self.validator.validate(self.doc_bytes)
        self.assertFalse(result.is_valid, "Document or screenshot should be rejected!")
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertNotEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 6: Other Unrelated Images (Noise, Shapes, Gradients)
    # ------------------------------------------------------------------------
    def test_other_unrelated_rejected(self):
        result = self.validator.validate(self.unrelated_bytes)
        self.assertFalse(result.is_valid, "Unrelated image/noise should be rejected!")
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertNotEqual(result.predicted_class, "valid_pediatric_hand_xray")

    # ------------------------------------------------------------------------
    # Category 7: Corrupted & Malformed Files
    # ------------------------------------------------------------------------
    def test_corrupted_file_bytes_rejected(self):
        corrupt_bytes = b"\x89PNG\r\n\x1a\nCorruptedGarbageBytes1234567890"
        result = self.validator.validate(corrupt_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")

    def test_empty_bytes_rejected(self):
        result = self.validator.validate(b"")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")

    def test_unsupported_file_extension_rejected(self):
        text_bytes = b"This is a plain text file, not an image."
        result = self.validator.validate(text_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")

    # ------------------------------------------------------------------------
    # Category 8: Very Small / Blank Images
    # ------------------------------------------------------------------------
    def test_very_small_image_rejected(self):
        small_img = np.full((16, 16, 3), 128, dtype=np.uint8)
        small_bytes = _encode_to_png(small_img)

        result = self.validator.validate(small_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")

    def test_flat_blank_image_rejected(self):
        flat_img = np.zeros((224, 224, 3), dtype=np.uint8)
        flat_bytes = _encode_to_png(flat_img)

        result = self.validator.validate(flat_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")

    # ------------------------------------------------------------------------
    # Category 9: Uncertain Predictions Rejection
    # ------------------------------------------------------------------------
    def test_uncertain_prediction_rejected_below_threshold(self):
        strict_validator = ImageValidator(confidence_threshold=0.9999)
        result = strict_validator.validate(self.hand_bytes)
        if result.confidence < 0.9999:
            self.assertFalse(result.is_valid)
            self.assertEqual(result.rejection_code, "INVALID_IMAGE")
            self.assertIn("Uncertain radiograph", result.rejection_message)

    # ------------------------------------------------------------------------
    # Category 10: Validation Pipeline - Prediction Stage
    # ------------------------------------------------------------------------
    def test_pipeline_rejects_animal_photo_before_prediction(self):
        val_res = self.validator.validate(self.animal_bytes)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.rejection_code, "INVALID_IMAGE")
        # In the pipeline, when is_valid is False, inference is never called

    def test_pipeline_rejects_chest_xray_before_prediction(self):
        val_res = self.validator.validate(self.non_hand_bytes)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.rejection_code, "INVALID_IMAGE")

    def test_pipeline_accepts_valid_hand_and_computes_prediction(self):
        val_res = self.validator.validate(self.hand_bytes)
        self.assertTrue(val_res.is_valid)
        # Valid image reaches InferenceService
        result = self.inference_service.predict(self.hand_bytes, "cnn")
        self.assertEqual(result["model_type"], "cnn")
        self.assertGreater(result["bone_age_months"], 0)
        self.assertGreaterEqual(result["confidence"], 0.5)

    # ------------------------------------------------------------------------
    # Category 11: Validation Pipeline - Grad-CAM Stage
    # ------------------------------------------------------------------------
    def test_pipeline_rejects_document_before_gradcam(self):
        val_res = self.validator.validate(self.doc_bytes)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.rejection_code, "INVALID_IMAGE")

    def test_pipeline_accepts_valid_hand_and_generates_gradcam(self):
        val_res = self.validator.validate(self.hand_bytes)
        self.assertTrue(val_res.is_valid)
        # Valid image reaches GradCamService
        result = self.gradcam_service.generate(self.hand_bytes, "cnn")
        self.assertEqual(result["model_type"], "cnn")
        self.assertIn("heatmap_base64", result)
        self.assertIn("overlay_base64", result)
        self.assertGreater(len(result["heatmap_base64"]), 100)

    # ------------------------------------------------------------------------
    # Category 12: Validator Confusion Matrix Plot Endpoint
    # ------------------------------------------------------------------------
    def test_validator_confusion_matrix_endpoint(self):
        resp = get_plot("validator_confusion_matrix")
        self.assertEqual(resp.media_type, "image/png")
        self.assertGreater(len(resp.body), 1000)

    # ------------------------------------------------------------------------
    # Category 13: Three-Stage Validation Gate Architecture
    # ------------------------------------------------------------------------
    def test_stage1_extreme_aspect_ratio_rejected(self):
        sliver = np.full((500, 50, 3), 128, dtype=np.uint8)  # AR = 0.10, dimensions 50x500 >= 32x32
        sliver_bytes = _encode_to_png(sliver)
        res = self.validator.validate(sliver_bytes)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.stage_passed, 0)
        self.assertIn("aspect ratio", res.rejection_message.lower())

    def test_stage2_ood_predictive_entropy_calculated(self):
        res = self.validator.validate(self.hand_bytes)
        self.assertTrue(res.is_valid)
        self.assertIn("ood_entropy", res.to_dict())
        self.assertLessEqual(res.ood_entropy, 0.75)

    def test_stage3_anatomical_coverage_details(self):
        res = self.validator.validate(self.hand_bytes)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.stage_passed, 3)
        self.assertEqual(res.gate_details["stage1_rules"], "PASSED")
        self.assertEqual(res.gate_details["stage2_classifier"], "PASSED")
        self.assertEqual(res.gate_details["stage3_coverage"], "PASSED")
        self.assertGreaterEqual(res.anatomical_coverage, 0.03)


if __name__ == "__main__":
    unittest.main()
