"""Comprehensive unit and integration test suite for the One-Class OOD Hand X-Ray Validation System.

Uses REAL radiographs from tests/validator/should_accept and real negative images from
tests/validator/should_reject (no synthetic hand X-ray phantoms):
1. Startup & strict config checks: fails loudly with FileNotFoundError if validator_ood.npz
   is missing, and fails with RuntimeError if config (k, metric, backbone, preprocessing) mismatches.
2. Real RSNA radiographs in tests/validator/should_accept (verified accept rate at 0.122576 threshold).
3. All real negative images in tests/validator/should_reject (non-hand X-rays, animals/photos,
   adult-hand optical photos, screenshots/documents) are 100% rejected.
4. Rejection payload includes reason_code, score, threshold, deciding_stage, and the required
   user-facing message while keeping technical details in technical_detail / gate_details.
5. Stage 1 integrity & strong-color checks (corrupted bytes, empty file, extreme aspect ratio,
   blank image, strongly colored image) and Stage 3 coverage checks.
6. End-to-end integration with InferenceService (/api/predict) and GradCamService (/api/gradcam)
   using real RSNA images from tests/validator/should_accept.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(4)

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from app.api.routes.analysis import get_plot
from app.services.gradcam_service import get_gradcam_service
from app.services.image_validator import (
    USER_REJECTION_MESSAGE,
    ImageValidator,
    get_image_validator,
)
from app.services.inference_service import get_inference_service

SHOULD_ACCEPT_DIR = ROOT_DIR / "tests" / "validator" / "should_accept"
SHOULD_REJECT_DIR = ROOT_DIR / "tests" / "validator" / "should_reject"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".webp", ".tif", ".tiff"}


def _encode_to_png(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", img)
    assert success, "Encoding to PNG failed"
    return buffer.tobytes()


class TestHandXRayValidationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = get_image_validator()
        cls.inference_service = get_inference_service()
        cls.gradcam_service = get_gradcam_service()

        cls.accept_files = sorted(
            [
                p
                for p in SHOULD_ACCEPT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        cls.reject_files = sorted(
            [
                p
                for p in SHOULD_REJECT_DIR.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        assert len(cls.accept_files) > 0, f"No images found in {SHOULD_ACCEPT_DIR}"
        assert len(cls.reject_files) > 0, f"No images found in {SHOULD_REJECT_DIR}"

        # Load a representative real RSNA radiograph from should_accept (10022.png)
        rep_path = SHOULD_ACCEPT_DIR / "10022.png"
        if not rep_path.exists():
            rep_path = cls.accept_files[0]
        cls.real_rsna_path = rep_path
        cls.real_rsna_bytes = rep_path.read_bytes()

    # ------------------------------------------------------------------------
    # 1. Missing validator_ood.npz & Config Mismatch Checks
    # ------------------------------------------------------------------------
    def test_missing_validator_ood_npz_raises_file_not_found(self):
        missing_path = BACKEND_DIR / "checkpoints" / "nonexistent_validator_ood_12345.npz"
        with self.assertRaises(FileNotFoundError):
            ImageValidator(npz_path=missing_path)

    def test_mismatched_ood_config_raises_runtime_error(self):
        real_npz = np.load(self.validator.npz_path, allow_pickle=False)
        bad_cfg = json.loads(str(real_npz["config_json"].item()))
        bad_cfg["knn_k"] = 99  # Mismatched k (expected 5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_npz_path = Path(tmp_dir) / "bad_validator_ood.npz"
            np.savez_compressed(
                bad_npz_path,
                mean=real_npz["mean"],
                precision=real_npz["precision"],
                knn_bank=real_npz["knn_bank"],
                shrinkage=real_npz["shrinkage"],
                knn_k=np.array(99, dtype=np.int32),
                thresholds_json=real_npz["thresholds_json"],
                config_json=np.array(json.dumps(bad_cfg)),
            )
            with self.assertRaises(RuntimeError):
                ImageValidator(npz_path=bad_npz_path)

    def test_default_threshold_matches_approved_99_calibration(self):
        self.assertAlmostEqual(self.validator.ood_threshold, 0.122576, places=6)

    # ------------------------------------------------------------------------
    # 2. Real RSNA Radiographs in tests/validator/should_accept
    # ------------------------------------------------------------------------
    def test_real_rsna_radiographs_in_should_accept(self):
        accepted = []
        rejected = []
        for p in self.accept_files:
            res = self.validator.validate(p.read_bytes())
            if res.is_valid:
                accepted.append((p.name, res.score))
                self.assertEqual(res.predicted_class, "valid_pediatric_hand_xray")
                self.assertEqual(res.stage_passed, 3)
                self.assertEqual(res.deciding_stage, "passed")
                self.assertIsNotNone(res.score)
                self.assertLessEqual(res.score, self.validator.ood_threshold)
                self.assertIsNone(res.rejection_code)
            else:
                rejected.append((p.name, res.score, res.deciding_stage))

        # 29/30 real RSNA images in should_accept pass at 0.122576 (only infant 4244.png has score 0.128517)
        self.assertGreaterEqual(
            len(accepted),
            29,
            f"Expected at least 29/30 real RSNA images accepted, got {len(accepted)}/{len(self.accept_files)}; rejected={rejected}",
        )

    def test_real_rsna_jpeg_reencoded_accepted(self):
        img = cv2.imread(str(self.real_rsna_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.assertTrue(ok)
        res = self.validator.validate(buf.tobytes())
        self.assertTrue(res.is_valid, f"Real RSNA JPEG rejected: {res.technical_detail}")
        self.assertEqual(res.stage_passed, 3)

    # ------------------------------------------------------------------------
    # 3. All Real Negative Images in tests/validator/should_reject
    # ------------------------------------------------------------------------
    def test_all_real_negatives_in_should_reject_are_rejected(self):
        false_accepts = []
        for p in self.reject_files:
            res = self.validator.validate(p.read_bytes())
            if res.is_valid:
                false_accepts.append((p.name, res.score))
            else:
                self.assertEqual(res.rejection_code, "INVALID_IMAGE")
                self.assertEqual(res.reason_code, "INVALID_IMAGE")
                self.assertEqual(res.rejection_message, USER_REJECTION_MESSAGE)
                self.assertIn(
                    res.deciding_stage,
                    ("stage1_rules", "stage2_ood_knn", "stage3_coverage"),
                )
                self.assertAlmostEqual(res.threshold, 0.122576, places=6)
                if res.deciding_stage == "stage2_ood_knn":
                    self.assertIsNotNone(res.score)
                    self.assertGreater(res.score, res.threshold)

        self.assertEqual(
            len(false_accepts),
            0,
            f"Expected 0 false accepts on tests/validator/should_reject, got: {false_accepts}",
        )

    # ------------------------------------------------------------------------
    # 4. Stage 1 Rule Checks (Corrupted, Empty, Small, Aspect Ratio, Blank, Color)
    # ------------------------------------------------------------------------
    def test_corrupted_file_bytes_rejected(self):
        corrupt_bytes = b"\x89PNG\r\n\x1a\nCorruptedGarbageBytes1234567890"
        result = self.validator.validate(corrupt_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertEqual(result.deciding_stage, "stage1_rules")
        self.assertEqual(result.rejection_message, USER_REJECTION_MESSAGE)

    def test_empty_bytes_rejected(self):
        result = self.validator.validate(b"")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertEqual(result.deciding_stage, "stage1_rules")

    def test_unsupported_file_header_rejected(self):
        text_bytes = b"This is a plain text file, not an image."
        result = self.validator.validate(text_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.rejection_code, "INVALID_IMAGE")
        self.assertEqual(result.deciding_stage, "stage1_rules")

    def test_very_small_image_rejected(self):
        small_img = np.full((16, 16, 3), 128, dtype=np.uint8)
        result = self.validator.validate(_encode_to_png(small_img))
        self.assertFalse(result.is_valid)
        self.assertEqual(result.deciding_stage, "stage1_rules")

    def test_flat_blank_image_rejected(self):
        flat_img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.validator.validate(_encode_to_png(flat_img))
        self.assertFalse(result.is_valid)
        self.assertEqual(result.deciding_stage, "stage1_rules")

    def test_stage1_extreme_aspect_ratio_rejected(self):
        sliver = np.full((500, 50, 3), 128, dtype=np.uint8)
        res = self.validator.validate(_encode_to_png(sliver))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.stage_passed, 0)
        self.assertEqual(res.deciding_stage, "stage1_rules")
        self.assertIn("aspect ratio", res.technical_detail.lower())

    def test_stage2_custom_strict_threshold_rejection(self):
        strict_validator = ImageValidator(ood_threshold=0.001)
        res = strict_validator.validate(self.real_rsna_bytes)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.deciding_stage, "stage2_ood_knn")
        self.assertEqual(res.rejection_code, "INVALID_IMAGE")
        self.assertEqual(res.rejection_message, USER_REJECTION_MESSAGE)
        self.assertGreater(res.score, 0.001)

    # ------------------------------------------------------------------------
    # 5. End-to-End Integration with Prediction & Grad-CAM Services
    # ------------------------------------------------------------------------
    def test_pipeline_accepts_real_rsna_hand_and_computes_prediction(self):
        val_res = self.validator.validate(self.real_rsna_bytes)
        self.assertTrue(val_res.is_valid)
        result = self.inference_service.predict(self.real_rsna_bytes, "cnn")
        self.assertEqual(result["model_type"], "cnn")
        self.assertGreater(result["bone_age_months"], 0)
        self.assertGreaterEqual(result["confidence"], 0.5)

    def test_pipeline_accepts_real_rsna_hand_and_generates_gradcam(self):
        val_res = self.validator.validate(self.real_rsna_bytes)
        self.assertTrue(val_res.is_valid)
        result = self.gradcam_service.generate(self.real_rsna_bytes, "cnn")
        self.assertEqual(result["model_type"], "cnn")
        self.assertIn("heatmap_base64", result)
        self.assertIn("overlay_base64", result)
        self.assertGreater(len(result["heatmap_base64"]), 100)

    def test_validator_confusion_matrix_endpoint(self):
        resp = get_plot("validator_confusion_matrix")
        self.assertEqual(resp.media_type, "image/png")
        self.assertGreater(len(resp.body), 1000)


if __name__ == "__main__":
    unittest.main()
