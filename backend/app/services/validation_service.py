"""Hand X-ray validation service for screening incoming images prior to prediction."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.models.validator import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    VALIDATOR_CLASSES,
    HandXRayValidator,
)

logger = logging.getLogger(__name__)

# ImageNet normalization standard for ResNet transfer learning
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Minimum confidence required for accepting an image as a valid pediatric hand X-ray
DEFAULT_VALIDATION_THRESHOLD = 0.70
MIN_IMAGE_DIMENSION = 32


@dataclass
class ValidationResult:
    is_valid: bool
    predicted_class: str
    confidence: float
    threshold: float = DEFAULT_VALIDATION_THRESHOLD
    probabilities: Dict[str, float] = None
    rejection_code: Optional[str] = None
    rejection_message: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "threshold": round(self.threshold, 4),
            "probabilities": {
                k: round(v, 4) for k, v in (self.probabilities or {}).items()
            },
            "rejection_code": self.rejection_code,
            "rejection_message": self.rejection_message,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


def decode_and_verify_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB array with validation checks."""
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image content uploaded.")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode as BGR
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image. Supported formats: PNG, JPEG, BMP, TIFF.")

    h, w = img.shape[:2]
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too small ({w}x{h}). Minimum required resolution is "
            f"{MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
        )

    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


def preprocess_for_validator(image_rgb: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Resize RGB image to 224x224 and normalize using ImageNet parameters."""
    resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose to (C, H, W) and add batch dimension -> (1, 3, 224, 224)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


class ValidationService:
    """Service to validate whether an uploaded image is a pediatric hand/wrist radiograph."""

    def __init__(
        self,
        settings: Settings | None = None,
        confidence_threshold: float = DEFAULT_VALIDATION_THRESHOLD,
    ) -> None:
        self.settings = settings or get_settings()
        self.confidence_threshold = confidence_threshold
        self._model: Optional[HandXRayValidator] = None
        self._device = torch.device(
            self.settings.device
            if torch.cuda.is_available() or self.settings.device == "cpu"
            else "cpu"
        )

    @property
    def validator_checkpoint_path(self) -> Path:
        return self.settings.checkpoint_dir / "hand_validator_best.pt"

    def is_model_available(self) -> bool:
        return self.validator_checkpoint_path.exists()

    def get_model(self) -> HandXRayValidator:
        if self._model is not None:
            return self._model

        model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES))
        ckpt_path = self.validator_checkpoint_path

        if ckpt_path.exists():
            logger.info("Loading HandXRayValidator weights from %s", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                "HandXRayValidator checkpoint not found at %s. Operating with untrained backbone.",
                ckpt_path,
            )

        model.to(self._device)
        model.eval()
        self._model = model
        return self._model

    def validate(
        self,
        image_bytes: bytes,
        threshold: Optional[float] = None,
    ) -> ValidationResult:
        """Validate an image before bone age prediction.
        
        Returns ValidationResult with is_valid=True ONLY if image is classified
        as 'valid_pediatric_hand_xray' with confidence >= threshold.
        """
        import time

        start = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.confidence_threshold

        # 1. Decode & verify basic image integrity
        try:
            image_rgb = decode_and_verify_image(image_bytes)
        except ValueError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ValidationResult(
                is_valid=False,
                predicted_class="corrupted_or_invalid_format",
                confidence=1.0,
                threshold=active_threshold,
                probabilities={cls_name: 0.0 for cls_name in VALIDATOR_CLASSES},
                rejection_code="DECODE_ERROR",
                rejection_message=str(exc),
                processing_time_ms=elapsed,
            )

        # 2. Preprocess for classification model
        tensor = preprocess_for_validator(image_rgb).to(self._device)

        # 3. Model forward pass
        model = self.get_model()
        with torch.no_grad():
            probs_tensor = model.predict_proba(tensor)
            probs = probs_tensor.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_class = IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {
            cls_name: float(probs[i]) for i, cls_name in enumerate(VALIDATOR_CLASSES)
        }

        # 4. Check confidence and class validity
        is_hand_xray = predicted_class == "valid_pediatric_hand_xray"
        has_sufficient_confidence = confidence >= active_threshold

        is_valid = is_hand_xray and has_sufficient_confidence
        rejection_code = None
        rejection_message = None

        if not is_valid:
            if not is_hand_xray:
                rejection_code = f"INVALID_{predicted_class.upper()}"
                category_messages = {
                    "non_hand_xray": "Non-hand medical radiograph detected (e.g. chest, skull, spine, pelvis).",
                    "normal_photograph": "Natural photograph or human portrait detected.",
                    "animal_photograph": "Animal photograph detected.",
                    "document_or_screenshot": "Document, receipt, or screen capture detected.",
                    "other_unrelated": "Unrelated non-clinical image or abstract pattern detected.",
                }
                detail = category_messages.get(predicted_class, f"Detected image type: {predicted_class}.")
                rejection_message = (
                    f"{detail} Please upload a pediatric hand/wrist X-ray."
                )
            else:
                rejection_code = "LOW_CONFIDENCE"
                rejection_message = (
                    f"Pediatric hand X-ray validation confidence ({confidence:.1%}) is below "
                    f"the required threshold ({active_threshold:.1%}). The image may be blurry, "
                    "improperly oriented, or not a clear pediatric hand radiograph."
                )

        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            is_valid=is_valid,
            predicted_class=predicted_class,
            confidence=confidence,
            threshold=active_threshold,
            probabilities=class_probs,
            rejection_code=rejection_code,
            rejection_message=rejection_message,
            processing_time_ms=elapsed,
        )


_validation_service: ValidationService | None = None


def get_validation_service() -> ValidationService:
    global _validation_service
    if _validation_service is None:
        _validation_service = ValidationService()
    return _validation_service
