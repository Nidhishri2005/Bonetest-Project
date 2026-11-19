"""Three-Stage Hand X-ray Image Validation Service (One-Class OOD kNN Architecture).

Enforces a 3-stage gatekeeper architecture BEFORE any radiograph reaches
the bone-age prediction models or Grad-CAM services:

Stage 1: Rule-Based Integrity & Strong-Color Gate
  - Magic byte signatures for supported image formats (PNG, JPEG, BMP, TIFF, WebP)
  - OpenCV image decoding integrity
  - Minimum dimensional resolution (h >= 32, w >= 32)
  - Aspect ratio validation (0.20 <= AR <= 5.0)
  - Signal variance check (std >= 1.0 to reject flat, blank, solid-color images)
  - Strong-color rejection (rejects strongly colored non-radiographic images)

Stage 2: One-Class OOD kNN (k=5, cosine distance) Gate
  - Frozen ImageNet-pretrained ResNet-18 512-d global-pooled feature extractor
  - Single shared preprocessing function (preprocess_validator_image)
  - Fitted on 4,138 stratified real RSNA hand X-rays and calibrated on 500
    stratified held-out RSNA hand X-rays (backend/checkpoints/validator_ood.npz)
  - Strict config validation at startup (raises if validator_ood.npz is missing or
    its k, metric, feature extractor, or preprocessing config does not match)
  - Rejects if kNN (k=5, cosine) score > ood_threshold (default 0.122576, 99% RSNA accept)

Stage 3: Suitability & Anatomical Hand-Mask Coverage Gate (5% Border Ignored)
  - Ignores a 5% border on all 4 sides to exclude film frames / scanner edges
  - Percentile contrast stretch [P2, P98] + Otsu thresholding on central 90%x90% ROI
  - Anatomical coverage ratio check (min_coverage_ratio <= coverage <= max_coverage_ratio)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from app.core.config import Settings, get_settings
from ml.validator_ood import OODValidator, ResNet18FeatureExtractor

logger = logging.getLogger(__name__)

USER_REJECTION_MESSAGE = (
    "Image does not look like a pediatric hand X-ray. Upload a clear hand/wrist radiograph."
)

EXPECTED_OOD_K = 5
EXPECTED_OOD_METRIC = "cosine_distance"
EXPECTED_OOD_BACKBONE = "torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)"
EXPECTED_OOD_FEATURE_DIM = 512
EXPECTED_OOD_PREPROCESS_SUBSTR = "preprocess_validator_image"

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
    score: Optional[float] = None
    reason_code: Optional[str] = None
    deciding_stage: str = "passed"
    probabilities: Dict[str, float] = field(default_factory=dict)
    rejection_code: Optional[str] = None
    rejection_message: Optional[str] = None
    technical_detail: Optional[str] = None
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
            "threshold": round(self.threshold, 6),
            "score": round(self.score, 6) if self.score is not None else None,
            "reason_code": self.reason_code,
            "deciding_stage": self.deciding_stage,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "rejection_code": self.rejection_code,
            "rejection_message": self.rejection_message,
            "technical_detail": self.technical_detail,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "stage_passed": self.stage_passed,
            "ood_entropy": round(self.ood_entropy, 4),
            "anatomical_coverage": round(self.anatomical_coverage, 4),
            "gate_details": self.gate_details,
        }


class ImageValidator:
    """Production Three-Stage Validation Gatekeeper (One-Class OOD kNN k=5 cosine)."""

    def __init__(
        self,
        npz_path: Optional[Path] = None,
        ood_threshold: Optional[float] = None,
        min_coverage_ratio: float = 0.03,
        max_coverage_ratio: float = 0.98,
        max_mean_color_diff: float = 25.0,
        max_colored_pixel_fraction: float = 0.20,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ood_threshold = float(
            ood_threshold if ood_threshold is not None else self.settings.validator_ood_threshold
        )
        # Keep confidence_threshold alias pointing to ood_threshold for compatibility
        self.confidence_threshold = self.ood_threshold
        self.min_coverage_ratio = min_coverage_ratio
        self.max_coverage_ratio = max_coverage_ratio
        self.max_mean_color_diff = max_mean_color_diff
        self.max_colored_pixel_fraction = max_colored_pixel_fraction

        self.npz_path = Path(npz_path or self.settings.validator_ood_npz)
        self._device = torch.device(
            self.settings.device
            if torch.cuda.is_available() or self.settings.device == "cpu"
            else "cpu"
        )
        self._ood_validator: Optional[OODValidator] = None
        self._extractor: Optional[ResNet18FeatureExtractor] = None

        # Load and strictly validate validator_ood.npz at startup
        self._load_ood_validator()

    @property
    def ood_validator(self) -> OODValidator:
        if self._ood_validator is None:
            self._load_ood_validator()
        return self._ood_validator  # type: ignore

    @property
    def extractor(self) -> ResNet18FeatureExtractor:
        if self._extractor is None:
            self._load_ood_validator()
        return self._extractor  # type: ignore

    def _load_ood_validator(self) -> None:
        """Load validator_ood.npz strictly and verify its config matches the code."""
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"OOD validator statistics file not found at {self.npz_path}. "
                "Refusing to start without fitted RSNA OOD statistics."
            )

        logger.info("Loading One-Class OOD validator statistics from %s", self.npz_path)
        ood = OODValidator.load(self.npz_path)
        cfg = ood.config

        # Strictly verify k, metric, feature extractor, and preprocessing match code
        if ood.knn_k != EXPECTED_OOD_K or cfg.get("knn_k") != EXPECTED_OOD_K:
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected knn_k={EXPECTED_OOD_K}, "
                f"got npz.knn_k={ood.knn_k}, config.knn_k={cfg.get('knn_k')}."
            )

        if cfg.get("knn_metric") != EXPECTED_OOD_METRIC:
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected knn_metric='{EXPECTED_OOD_METRIC}', "
                f"got '{cfg.get('knn_metric')}'."
            )

        if cfg.get("backbone") != EXPECTED_OOD_BACKBONE:
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected backbone='{EXPECTED_OOD_BACKBONE}', "
                f"got '{cfg.get('backbone')}'."
            )

        if cfg.get("feature_dim") != EXPECTED_OOD_FEATURE_DIM or ood.knn_bank.shape[1] != EXPECTED_OOD_FEATURE_DIM:
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected feature_dim={EXPECTED_OOD_FEATURE_DIM}, "
                f"got config.feature_dim={cfg.get('feature_dim')}, bank_dim={ood.knn_bank.shape[1]}."
            )

        preprocess_cfg = str(cfg.get("preprocess", ""))
        if EXPECTED_OOD_PREPROCESS_SUBSTR not in preprocess_cfg:
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected preprocess containing "
                f"'{EXPECTED_OOD_PREPROCESS_SUBSTR}', got '{preprocess_cfg}'."
            )

        if not bool(cfg.get("l2_normalized", False)):
            raise RuntimeError(
                f"OOD config mismatch in {self.npz_path}: expected l2_normalized=True."
            )

        # Ensure offline local torch cache is used if present
        torch_cache = self.settings.checkpoint_dir / "torch_cache"
        if torch_cache.exists():
            os.environ.setdefault("TORCH_HOME", str(torch_cache))

        if self._extractor is None:
            self._extractor = ResNet18FeatureExtractor(device=self._device)

        self._ood_validator = ood

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
            b, g, r = (
                img[:, :, 0].astype(np.float32),
                img[:, :, 1].astype(np.float32),
                img[:, :, 2].astype(np.float32),
            )
            channel_spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            mean_color_diff = float(np.mean(channel_spread))
            colored_pixel_frac = float(np.mean(channel_spread > 40.0))
            if (
                mean_color_diff > self.max_mean_color_diff
                or colored_pixel_frac > self.max_colored_pixel_fraction
            ):
                return (
                    False,
                    None,
                    f"Image is strongly colored (mean RGB spread={mean_color_diff:.1f}, "
                    f"colored area={colored_pixel_frac * 100:.1f}%). "
                    "Please upload a grayscale pediatric hand/wrist X-ray.",
                )

        return True, img, None

    # =========================================================================
    # STAGE 2: One-Class OOD kNN (k=5, cosine distance) Gate
    # =========================================================================
    def stage2_ood_knn(
        self, img_bgr: np.ndarray
    ) -> Tuple[bool, str, float, float, Dict[str, float], Optional[str]]:
        """Score image with kNN (k=5, cosine) against stratified RSNA feature bank."""
        feat_1d = self.extractor.extract_single_bgr(img_bgr)
        scores = self.ood_validator.score_single_feature(feat_1d)
        knn_score = float(scores["knn_k5_cosine"])
        mahal_score = float(scores["mahalanobis"])

        # Convert cosine distance [0, 2] to cosine similarity in [0, 1] for display confidence
        cosine_similarity = float(max(0.0, min(1.0, 1.0 - knn_score)))
        score_dict = {
            "knn_k5_cosine": knn_score,
            "mahalanobis": mahal_score,
            "cosine_similarity": cosine_similarity,
        }

        if knn_score > self.ood_threshold:
            tech_detail = (
                f"Stage 2 OOD kNN (k=5, cosine) rejection: score={knn_score:.6f} > "
                f"threshold={self.ood_threshold:.6f} (mahalanobis={mahal_score:.4f})."
            )
            logger.info(tech_detail)
            return False, "out_of_distribution", knn_score, cosine_similarity, score_dict, tech_detail

        return True, "valid_pediatric_hand_xray", knn_score, cosine_similarity, score_dict, None

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

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
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
            logger.info("Stage 1 rule check failed: %s", err1)
            return ValidationResult(
                is_valid=False,
                predicted_class="corrupted_or_invalid_format",
                confidence=0.0,
                threshold=self.ood_threshold,
                score=None,
                reason_code="INVALID_IMAGE",
                deciding_stage="stage1_rules",
                rejection_code="INVALID_IMAGE",
                rejection_message=USER_REJECTION_MESSAGE,
                technical_detail=err1 or "Stage 1 rule check failed.",
                processing_time_ms=elapsed,
                stage_passed=0,
                gate_details={
                    "deciding_stage": "stage1_rules",
                    "reason_code": "INVALID_IMAGE",
                    "technical_reason": err1,
                    "stage1_rules": "FAILED",
                    "stage2_ood_knn": "SKIPPED",
                    "stage3_coverage": "SKIPPED",
                },
            )

        # Stage 2: One-Class OOD kNN (k=5, cosine) Gate
        ok2, pred_class, knn_score, cos_sim, score_dict, err2 = self.stage2_ood_knn(img_bgr)
        if not ok2:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ValidationResult(
                is_valid=False,
                predicted_class=pred_class,
                confidence=cos_sim,
                threshold=self.ood_threshold,
                score=knn_score,
                reason_code="INVALID_IMAGE",
                deciding_stage="stage2_ood_knn",
                probabilities=score_dict,
                rejection_code="INVALID_IMAGE",
                rejection_message=USER_REJECTION_MESSAGE,
                technical_detail=err2,
                processing_time_ms=elapsed,
                stage_passed=1,
                gate_details={
                    "deciding_stage": "stage2_ood_knn",
                    "reason_code": "INVALID_IMAGE",
                    "score": round(knn_score, 6),
                    "threshold": round(self.ood_threshold, 6),
                    "mahalanobis": round(score_dict["mahalanobis"], 4),
                    "technical_reason": err2,
                    "stage1_rules": "PASSED",
                    "stage2_ood_knn": "FAILED",
                    "stage3_coverage": "SKIPPED",
                },
            )

        # Stage 3: Anatomical Coverage Gate (5% border ignored)
        ok3, coverage, err3 = self.stage3_anatomical_coverage(img_bgr)
        elapsed = (time.perf_counter() - start_time) * 1000

        if not ok3:
            logger.info("Stage 3 coverage check failed: %s", err3)
            return ValidationResult(
                is_valid=False,
                predicted_class="insufficient_anatomical_coverage",
                confidence=cos_sim,
                threshold=self.ood_threshold,
                score=knn_score,
                reason_code="INVALID_IMAGE",
                deciding_stage="stage3_coverage",
                probabilities=score_dict,
                rejection_code="INVALID_IMAGE",
                rejection_message=USER_REJECTION_MESSAGE,
                technical_detail=err3,
                processing_time_ms=elapsed,
                stage_passed=2,
                anatomical_coverage=coverage,
                gate_details={
                    "deciding_stage": "stage3_coverage",
                    "reason_code": "INVALID_IMAGE",
                    "score": round(knn_score, 6),
                    "threshold": round(self.ood_threshold, 6),
                    "coverage_ratio": round(coverage, 4),
                    "technical_reason": err3,
                    "stage1_rules": "PASSED",
                    "stage2_ood_knn": "PASSED",
                    "stage3_coverage": "FAILED",
                },
            )

        # All 3 Stages Passed Successfully
        return ValidationResult(
            is_valid=True,
            predicted_class="valid_pediatric_hand_xray",
            confidence=cos_sim,
            threshold=self.ood_threshold,
            score=knn_score,
            reason_code=None,
            deciding_stage="passed",
            probabilities=score_dict,
            processing_time_ms=elapsed,
            stage_passed=3,
            anatomical_coverage=coverage,
            gate_details={
                "deciding_stage": "passed",
                "score": round(knn_score, 6),
                "threshold": round(self.ood_threshold, 6),
                "mahalanobis": round(score_dict["mahalanobis"], 4),
                "coverage_ratio": round(coverage, 4),
                "stage1_rules": "PASSED",
                "stage2_ood_knn": "PASSED",
                "stage3_coverage": "PASSED",
            },
        )


_image_validator: Optional[ImageValidator] = None


def get_image_validator() -> ImageValidator:
    global _image_validator
    if _image_validator is None:
        _image_validator = ImageValidator()
    return _image_validator
