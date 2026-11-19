"""Hand ROI crop, CLAHE contrast enhancement, and radiograph preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def crop_hand_roi(
    image: np.ndarray,
    margin_ratio: float = 0.05,
    min_area_ratio: float = 0.15,
) -> np.ndarray:
    """Isolate hand/wrist region of interest (ROI) using Otsu thresholding and largest component.
    
    Args:
        image: 2D grayscale uint8 image array.
        margin_ratio: Margin to add around bounding box as fraction of dimension.
        min_area_ratio: Minimum fraction of total image area the bbox must occupy to be valid.
        
    Returns:
        Cropped 2D grayscale uint8 image focused on the hand/wrist anatomy.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image passed to crop_hand_roi")

    h, w = image.shape[:2]

    # 1. Mild Gaussian blur to reduce quantum noise in air background
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Otsu thresholding to segment radiopaque tissue from air background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels <= 1:
        # Background only, return original
        return image

    # Ignore label 0 (background); find largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]

    # Validate that largest component is significant
    if largest_area < (h * w * min_area_ratio):
        # Component too small (likely artifact or low contrast), fallback to full image
        return image

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    bw = stats[largest_label, cv2.CC_STAT_WIDTH]
    bh = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # 4. Add margin
    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    # Ensure crop has non-zero size
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return image

    cropped = image[y1:y2, x1:x2]
    return cropped


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances bone trabecular architecture, epiphyses, and carpal ossification visibility.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def resize_letterbox(
    image: np.ndarray,
    target_size: int = 384,
) -> np.ndarray:
    """Resize image preserving aspect ratio with zero padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_radiograph(
    image: np.ndarray,
    target_size: int = 384,
    roi_crop: bool = True,
    clahe: bool = True,
    clahe_clip_limit: float = 2.0,
) -> np.ndarray:
    """Complete 3-step radiograph preprocessing pipeline: ROI crop -> CLAHE -> 384 resize.
    
    Returns:
        Processed 2D uint8 image of shape (target_size, target_size).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Hand ROI Crop
    if roi_crop:
        image = crop_hand_roi(image)

    # Step 2: CLAHE Contrast Enhancement
    if clahe:
        image = apply_clahe(image, clip_limit=clahe_clip_limit)

    # Step 3: Aspect-ratio letterbox resize
    image = resize_letterbox(image, target_size=target_size)
    return image
