"""Shared Hand X-ray validator preprocessing (and legacy 2-class classifier definition).

NOTE: Stage 2 of ImageValidator now uses the One-Class OOD kNN (k=5, cosine) validator
in ml/validator_ood.py with preprocess_validator_image below. HandXRayValidator is kept
as a legacy module for backward compatibility / ONNX export tests only.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

VALIDATOR_CLASSES: List[str] = [
    "valid_pediatric_hand_xray",
    "other",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VALIDATOR_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(VALIDATOR_CLASSES)}

# Standard ImageNet normalization for ResNet-18 transfer learning
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_rgb_224(image_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Aspect-ratio preserving resize with zero padding to (target_size, target_size, 3)."""
    h, w = image_rgb.shape[:2]
    scale = target_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_validator_image(image_bgr: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Single shared preprocessing function used by BOTH training and inference.

    1. Convert grayscale/BGRA/BGR to 3-channel RGB uint8.
    2. Letterbox-resize preserving aspect ratio to (target_size, target_size, 3).
    3. Scale to [0, 1] and normalize with ImageNet mean and std.
    4. Return float32 tensor of shape (3, target_size, target_size).
    """
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    letterboxed = letterbox_rgb_224(rgb, target_size=target_size)
    norm = letterboxed.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
    return tensor


class HandXRayValidator(nn.Module):
    """2-class ResNet-18 binary classifier (valid_pediatric_hand_xray vs other)."""

    def __init__(
        self,
        num_classes: int = len(VALIDATOR_CLASSES),
        dropout_rate: float = 0.3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional feature embedding."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape (B, 2)."""
        feats = self.extract_features(x)
        return self.classifier(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probabilities of shape (B, 2)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
