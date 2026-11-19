"""Three-Stage Hand X-ray Image Validation Service (2-Class Real-Data Architecture).

Enforces a 3-stage gatekeeper architecture BEFORE any radiograph reaches
the bone-age prediction models or Grad-CAM services:

Stage 1: Rule-Based Integrity & Strong-Color Gate
  - Magic byte signatures for supported image formats (PNG, JPEG, BMP, TIFF, WebP)
  - OpenCV image decoding integrity
  - Minimum dimensional resolution (h >= 32, w >= 32)
  - Aspect ratio validation (0.20 <= AR <= 5.0)
  - Signal variance check (std >= 1.0 to reject flat, blank, solid-color images)
  - Strong-color rejection (rejects strongly colored non-radiographic images)

Stage 2: 2-Class Gatekeeper Classifier (valid_pediatric_hand_xray vs other)
  - ImageNet-pretrained ResNet-18 binary classifier (hand_validator_best.pt)
  - Single shared preprocessing function (preprocess_validator_image)
  - Strict checkpoint loading (raises FileNotFoundError if missing; load_state_dict(strict=True))
  - Confidence threshold gating: p(valid_pediatric_hand_xray) >= confidence_threshold

Stage 3: Suitability & Anatomical Hand-Mask Coverage Gate (5% Border Ignored)
  - Ignores a 5% border on all 4 sides to exclude film frames / scanner edges
  - Otsu thresholding and connected component analysis on central 90%x90% ROI
  - Anatomical coverage ratio check (min_coverage_ratio <= coverage <= max_coverage_ratio)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.models.validator import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    VALIDATOR_CLASSES,
    HandXRayValidator,
    preprocess_validator_image,
)

logger = logging.getLogger(__name__)

# Magic byte signatures for supported image formats
IMAGE_MAGIC_BYTES = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"BM": "BMP",
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
    b"RIFF": "WEBP",
}


@dataclass
class ValidationResult:
    is_valid: bool
    predicted_class: str
    confidence: float
    threshold: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    rejection_code: Optional[str] = None
    rejection_message: Optional[str] = None
    processing_time_ms: float = 0.0
    stage_passed: int = 0
    ood_entropy: float = 0.0
    anatomical_coverage: float = 0.0
    gate_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "threshold": round(self.threshold, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "rejection_code": self.rejection_code,
            "rejection_message": self.rejection_message,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "stage_passed": self.stage_passed,
            "ood_entropy": round(self.ood_entropy, 4),
            "anatomical_coverage": round(self.anatomical_coverage, 4),
            "gate_details": self.gate_details,
        }


class ImageValidator:
    """Production Three-Stage Validation Gatekeeper (2-class real-data model)."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        confidence_threshold: float = 0.50,
        min_coverage_ratio: float = 0.03,
        max_coverage_ratio: float = 0.98,
        max_mean_color_diff: float = 25.0,
        max_colored_pixel_fraction: float = 0.20,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.confidence_threshold = confidence_threshold
        self.min_coverage_ratio = min_coverage_ratio
        self.max_coverage_ratio = max_coverage_ratio
        self.max_mean_color_diff = max_mean_color_diff
        self.max_colored_pixel_fraction = max_colored_pixel_fraction

        self.checkpoint_path = Path(
            checkpoint_path or (self.settings.checkpoint_dir / "hand_validator_best.pt")
        )
        self._device = torch.device(
            self.settings.device
            if torch.cuda.is_available() or self.settings.device == "cpu"
            else "cpu"
        )
        self._model: Optional[HandXRayValidator] = None
        self._classes: List[str] = list(VALIDATOR_CLASSES)
        self._onnx_session: Optional[Any] = None

    @property
    def classes(self) -> List[str]:
        if self._model is None:
            self._load_model()
        return self._classes

    @property
    def model(self) -> HandXRayValidator:
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore

    def _load_model(self) -> None:
        """Load 2-class validator checkpoint strictly. Raises if missing or mismatched."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Validator checkpoint not found at {self.checkpoint_path}. "
                "Refusing to fall back to random weights."
            )

        torch.set_num_threads(2)
        logger.info("Loading 2-class validator weights from %s", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        ckpt_classes = ckpt.get("classes", VALIDATOR_CLASSES)

        if len(ckpt_classes) != len(VALIDATOR_CLASSES):
            raise RuntimeError(
                f"Validator checkpoint at {self.checkpoint_path} has {len(ckpt_classes)} classes "
                f"({ckpt_classes}), expected {len(VALIDATOR_CLASSES)} classes ({VALIDATOR_CLASSES}). "
                "Please replace hand_validator_best.pt with the trained 2-class checkpoint."
            )

        self._classes = list(ckpt_classes)
        model = HandXRayValidator(num_classes=len(self._classes), pretrained=False)
        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()
        self._model = model

    # =========================================================================
    # STAGE 1: Rule-Based Integrity & Strong-Color Gate
    # =========================================================================
    def stage1_rule_checks(
        self, image_bytes: bytes
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """Verify byte headers, decoding, resolution, aspect ratio, variance, and strong color."""
        if not image_bytes or len(image_bytes) == 0:
            return False, None, "Empty file uploaded. Please upload a valid image file."

        if len(image_bytes) > 25 * 1024 * 1024:
            return False, None, "File size exceeds 25 MB limit."

        # Magic byte validation
        has_valid_header = any(image_bytes.startswith(magic) for magic in IMAGE_MAGIC_BYTES)
        if not has_valid_header:
            return (
                False,
                None,
                "Unsupported file format or corrupted header. Allowed formats: PNG, JPEG, BMP.",
            )

        # Image decoding via OpenCV
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return (
                False,
                None,
                "Failed to decode image data. The file appears to be corrupted or invalid.",
            )

        h, w = img.shape[:2]
        if h < 32 or w < 32:
            return (
                False,
                None,
                f"Image dimensions ({w}x{h}) are too small for clinical assessment. Minimum is 32x32.",
            )

        # Extreme aspect ratio check
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.20 or aspect_ratio > 5.0:
            return (
                False,
                None,
                f"Extreme aspect ratio ({aspect_ratio:.2f}). Pediatric hand radiographs require standard framing.",
            )

        # Signal variance check (reject completely blank or solid-color images)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 1.0:
            return (
                False,
                None,
                "Image appears blank or has no structural variation. Please upload a valid radiograph.",
            )

        # Reject strongly colored images (X-rays are grayscale / near-monochromatic)
        if img.ndim == 3 and img.shape[2] >= 3:
            b, g, r = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
            channel_spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            mean_color_diff = float(np.mean(channel_spread))
            colored_pixel_frac = float(np.mean(channel_spread > 40.0))
            if mean_color_diff > self.max_mean_color_diff or colored_pixel_frac > self.max_colored_pixel_fraction:
                return (
                    False,
                    None,
                    f"Image is strongly colored (mean RGB spread={mean_color_diff:.1f}, "
                    f"colored area={colored_pixel_frac * 100:.1f}%). "
                    "Please upload a grayscale pediatric hand/wrist X-ray.",
                )

        return True, img, None

    # =========================================================================
    # STAGE 2: 2-Class Classifier Gate (valid_pediatric_hand_xray vs other)
    # =========================================================================
    def stage2_classifier_and_ood(
        self, img_bgr: np.ndarray
    ) -> Tuple[bool, str, float, float, Dict[str, float], Optional[str]]:
        """Classify radiograph with the 2-class HandXRayValidator (no entropy gate)."""
        # Ensure strict PyTorch checkpoint validation runs first
        pt_model = self.model
        classes = self._classes

        tensor = preprocess_validator_image(img_bgr, target_size=224)
        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            probs_tensor = pt_model.predict_proba(tensor)[0]
            probs = probs_tensor.cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        pred_confidence = float(probs[pred_idx])

        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

        # Diagnostic binary Shannon entropy in [0, 1] (logged only, not used as a rejection gate)
        eps = 1e-12
        shannon_entropy = -float(np.sum(probs * np.log2(probs + eps)))
        max_entropy = math.log2(max(len(classes), 2))
        norm_entropy = float(shannon_entropy / max_entropy)

        hand_xray_class = "valid_pediatric_hand_xray"
        hand_confidence = prob_dict.get(hand_xray_class, float(probs[0]))

        if hand_confidence < self.confidence_threshold:
            if pred_class != hand_xray_class:
                msg = (
                    "Invalid image. Please upload a pediatric hand/wrist X-ray. "
                    f"(Classified as '{pred_class}' with {pred_confidence * 100:.1f}% confidence; "
                    f"hand X-ray probability={hand_confidence * 100:.1f}% < threshold {self.confidence_threshold * 100:.0f}%)"
                )
            else:
                msg = (
                    f"Uncertain radiograph. Hand X-ray probability is {hand_confidence * 100:.1f}% "
                    f"(minimum required: {self.confidence_threshold * 100:.0f}%). "
                    "Please upload a clearer pediatric hand/wrist radiograph."
                )
            return False, pred_class, pred_confidence, norm_entropy, prob_dict, msg

        return True, hand_xray_class, hand_confidence, norm_entropy, prob_dict, None

    # =========================================================================
    # STAGE 3: Suitability & Anatomical Hand-Mask Coverage Gate (5% Border Ignored)
    # =========================================================================
    def stage3_anatomical_coverage(
        self, img_bgr: np.ndarray
    ) -> Tuple[bool, float, Optional[str]]:
        """Verify presence of sufficient radiopaque hand tissue inside central 90%x90% ROI (ignoring 5% border)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Ignore 5% border on all sides to exclude film borders / scanner frames
        margin_y = int(round(h * 0.05))
        margin_x = int(round(w * 0.05))
        if h - 2 * margin_y >= 16 and w - 2 * margin_x >= 16:
            roi = gray[margin_y : h - margin_y, margin_x : w - margin_x]
        else:
            roi = gray

        rh, rw = roi.shape[:2]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        # Percentile clip [p2, p98] before Otsu so tiny 255 lead markers ("L"/"R")
        # and low-contrast backgrounds do not skew Otsu into isolating only the marker
        p2, p98 = np.percentile(blurred, (2, 98))
        if p98 > p2:
            clipped = np.clip(blurred.astype(np.float32), p2, p98)
            stretched = ((clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
        else:
            stretched = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        _, thresh = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If background is brighter than foreground (inverted), invert mask so foreground is the hand
        fg_ratio = float(np.mean(thresh > 0))
        if fg_ratio > 0.85:
            thresh = cv2.bitwise_not(thresh)

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels <= 1:
            return False, 0.0, "No discernible anatomical structure detected in radiograph."

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(np.max(areas))
        coverage_ratio = float(largest_area) / float(rh * rw)

        if coverage_ratio < self.min_coverage_ratio:
            msg = (
                f"Insufficient anatomical coverage (radiopaque hand tissue occupies only "
                f"{coverage_ratio * 100:.1f}% of central frame; minimum required is "
                f"{self.min_coverage_ratio * 100:.0f}%). Please upload a centered hand/wrist radiograph."
            )
            return False, coverage_ratio, msg

        if coverage_ratio > self.max_coverage_ratio:
            msg = (
                f"Excessive image saturation (foreground occupies {coverage_ratio * 100:.1f}% of central frame). "
                "Image lacks clear bone/soft-tissue anatomical delineation."
            )
            return False, coverage_ratio, msg

        return True, coverage_ratio, None

    # =========================================================================
    # Main Validation Pipeline
    # =========================================================================
    def validate(self, image_bytes: bytes) -> ValidationResult:
        """Execute the end-to-end Three-Stage Hand X-ray Image Validation Pipeline."""
        start_time = time.perf_counter()

        # Stage 1: Rule Checks (includes strong-color check)
        ok1, img_bgr, err1 = self.stage1_rule_checks(image_bytes)
        if not ok1 or img_bgr is None:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ValidationResult(
                is_valid=False,
                predicted_class="corrupted_or_invalid_format",
                confidence=0.0,
                threshold=self.confidence_threshold,
                rejection_code="INVALID_IMAGE",
                rejection_message=err1 or "Invalid image file.",
                processing_time_ms=elapsed,
                stage_passed=0,
                gate_details={"stage1_rules": "FAILED", "stage2_classifier": "SKIPPED", "stage3_coverage": "SKIPPED"},
            )

        # Stage 2: 2-Class Classifier Gate
        ok2, pred_class, confidence, entropy, prob_dict, err2 = self.stage2_classifier_and_ood(img_bgr)
        if not ok2:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ValidationResult(
                is_valid=False,
                predicted_class=pred_class,
                confidence=confidence,
                threshold=self.confidence_threshold,
                probabilities=prob_dict,
                rejection_code="INVALID_IMAGE",
                rejection_message=err2 or "Invalid image content.",
                processing_time_ms=elapsed,
                stage_passed=1,
                ood_entropy=entropy,
                gate_details={
                    "stage1_rules": "PASSED",
                    "stage2_classifier": "FAILED",
                    "stage3_coverage": "SKIPPED",
                    "ood_entropy": round(entropy, 4),
                },
            )

        # Stage 3: Anatomical Coverage Gate (5% border ignored)
        ok3, coverage, err3 = self.stage3_anatomical_coverage(img_bgr)
        elapsed = (time.perf_counter() - start_time) * 1000

        if not ok3:
            return ValidationResult(
                is_valid=False,
                predicted_class=pred_class,
                confidence=confidence,
                threshold=self.confidence_threshold,
                probabilities=prob_dict,
                rejection_code="INVALID_IMAGE",
                rejection_message=err3 or "Inadequate anatomical hand structure.",
                processing_time_ms=elapsed,
                stage_passed=2,
                ood_entropy=entropy,
                anatomical_coverage=coverage,
                gate_details={
                    "stage1_rules": "PASSED",
                    "stage2_classifier": "PASSED",
                    "stage3_coverage": "FAILED",
                    "coverage_ratio": round(coverage, 4),
                },
            )

        # All 3 Stages Passed Successfully
        return ValidationResult(
            is_valid=True,
            predicted_class="valid_pediatric_hand_xray",
            confidence=confidence,
            threshold=self.confidence_threshold,
            probabilities=prob_dict,
            processing_time_ms=elapsed,
            stage_passed=3,
            ood_entropy=entropy,
            anatomical_coverage=coverage,
            gate_details={
                "stage1_rules": "PASSED",
                "stage2_classifier": "PASSED",
                "stage3_coverage": "PASSED",
                "confidence": round(confidence, 4),
                "ood_entropy": round(entropy, 4),
                "coverage_ratio": round(coverage, 4),
            },
        )


_image_validator: Optional[ImageValidator] = None


def get_image_validator() -> ImageValidator:
    global _image_validator
    if _image_validator is None:
        _image_validator = ImageValidator()
    return _image_validator
