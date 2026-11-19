"""Training-time image augmentation for pediatric hand bone age radiographs.

Applies clinically safe transformations:
- In-plane rotation within +/- 8 degrees (hand placement variability)
- Scaling within 0.95 - 1.05 (focal-film distance variance)
- Brightness and contrast within +/- 10% (mAs and kVp X-ray tube variations)
- Zero vertical flip (hand radiographs are strictly oriented fingers-up)
- Left-hand orientation preserved by default (flip_prob=0.0)
"""

from __future__ import annotations

import cv2
import numpy as np


class TrainAugmentation:
    """Clinically grounded augmentation pipeline for left-hand radiographs."""

    def __init__(
        self,
        flip_prob: float = 0.0,
        rotate_limit: float = 8.0,
        scale_limit: tuple[float, float] = (0.95, 1.05),
        brightness_limit: float = 0.10,
        contrast_limit: float = 0.10,
        noise_std: float = 0.005,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_limit = rotate_limit
        self.scale_min, self.scale_max = scale_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray) -> dict:
        img = image.copy()
        h, w = img.shape[:2]

        # 1. Optional horizontal flip (disabled by default to preserve left-hand anatomy)
        if self.flip_prob > 0.0 and np.random.random() < self.flip_prob:
            img = cv2.flip(img, 1)

        # 2. Mild rotation (+/- 8 degrees)
        if self.rotate_limit > 0:
            angle = float(np.random.uniform(-self.rotate_limit, self.rotate_limit))
            center = (w / 2.0, h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # 3. Mild zoom/scaling (0.95 - 1.05) centered
        scale = float(np.random.uniform(self.scale_min, self.scale_max))
        if not np.isclose(scale, 1.0):
            new_h = int(np.round(h * scale))
            new_w = int(np.round(w * scale))
            scaled = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA,
            )
            canvas = np.zeros_like(img)
            if scale < 1.0:
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                canvas[top : top + new_h, left : left + new_w] = scaled
                img = canvas
            else:
                top = (new_h - h) // 2
                left = (new_w - w) // 2
                img = scaled[top : top + h, left : left + w]

        # 4. Photometric variations: Brightness and Contrast
        img_float = img.astype(np.float32)

        if self.contrast_limit > 0:
            alpha = 1.0 + float(np.random.uniform(-self.contrast_limit, self.contrast_limit))
            img_float = (img_float - 128.0) * alpha + 128.0

        if self.brightness_limit > 0:
            beta = float(np.random.uniform(-self.brightness_limit, self.brightness_limit)) * 255.0
            img_float = img_float + beta

        # 5. Subtle Gaussian noise (sensor noise)
        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std * 255.0, img.shape)
            img_float = img_float + noise

        img = np.clip(img_float, 0.0, 255.0).astype(np.uint8)

        return {"image": img}
