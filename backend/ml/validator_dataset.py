"""[SIMULATED / SYNTHETIC FIXTURES] Synthetic OpenCV phantom generator for unit tests only.

WARNING — SYNTHETIC PHANTOMS ONLY:
This module generates synthetic OpenCV geometric drawings (`generate_hand_xray_phantom`,
`generate_non_hand_xray`, etc.). It is marked SIMULATED / legacy and must NEVER be used to
train or evaluate production image validators (which now use real RSNA radiographs via
One-Class OOD kNN in `ml/validator_ood.py`).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.models.validator import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALIDATOR_CLASSES,
    preprocess_validator_image,
)


# ============================================================================
# Synthetic Image Generators for Each Category
# ============================================================================

def generate_hand_xray_phantom(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate a realistic pediatric hand/wrist radiograph phantom.
    
    Includes:
    - Air background (dark, near 0-20 intensity) with quantum noise.
    - Soft tissue envelope (gray 30-70) outlining palm and 5 digits.
    - Metacarpal bones (5 dense shafts, intensity 140-210).
    - Phalanges (proximal, middle, distal) with epiphyses / growth plates.
    - Carpal bone cluster in wrist (ossification centers).
    - Distal radius and ulna with growth plates.
    """
    if rng is None:
        rng = random.Random()

    img = np.zeros((size, size), dtype=np.float32)
    
    # 1. Background dark air + subtle quantum mottle noise
    noise = np.random.normal(12.0, 4.0, (size, size)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    scale = size / 224.0
    center_x = int(112 * scale + rng.uniform(-6, 6) * scale)
    wrist_y = int(185 * scale + rng.uniform(-4, 4) * scale)
    palm_y = int(140 * scale + rng.uniform(-4, 4) * scale)

    # 2. Soft tissue envelope (palm and wrist)
    cv2.ellipse(img, (center_x, palm_y), (int(52 * scale), int(45 * scale)), 0, 0, 360, 50.0, -1)
    cv2.rectangle(img, (int(center_x - 38 * scale), palm_y), (int(center_x + 38 * scale), int(wrist_y + 35 * scale)), 45.0, -1)

    # 3. Soft tissue fingers (5 digits)
    finger_configs = [
        # (offset_x, base_y, length, width, angle)
        (-38, 145, 45, 14, -32),  # Thumb
        (-22, 115, 68, 12, -8),   # Index
        (0, 105, 78, 13, 0),      # Middle
        (22, 115, 70, 12, 6),     # Ring
        (38, 130, 52, 11, 16),    # Little
    ]

    for off_x, base_y, length, width, angle in finger_configs:
        fx = int((center_x + off_x) * scale)
        fy = int(base_y * scale)
        rad = math.radians(angle + rng.uniform(-3, 3))
        tip_x = int(fx + length * scale * math.sin(rad))
        tip_y = int(fy - length * scale * math.cos(rad))
        cv2.line(img, (fx, fy), (tip_x, tip_y), 48.0, int(width * scale), lineType=cv2.LINE_AA)
        cv2.circle(img, (tip_x, tip_y), int(width * scale / 2), 48.0, -1)

    # 4. Dense Radiopaque Bones
    # Forearm bones: Distal Radius (lateral) and Ulna (medial)
    rad_x = int(center_x - 18 * scale)
    ulna_x = int(center_x + 18 * scale)
    shaft_top = int(wrist_y + 5 * scale)
    shaft_bot = int(size)

    # Radius shaft & epiphysis
    cv2.rectangle(img, (int(rad_x - 11 * scale), shaft_top), (int(rad_x + 11 * scale), shaft_bot), 175.0, -1)
    cv2.rectangle(img, (int(rad_x - 14 * scale), shaft_top - int(6 * scale)), (int(rad_x + 14 * scale), shaft_top), 195.0, -1)
    # Radial growth plate radiolucent line
    cv2.line(img, (int(rad_x - 14 * scale), shaft_top + int(4 * scale)), (int(rad_x + 14 * scale), shaft_top + int(4 * scale)), 75.0, 1)

    # Ulna shaft & epiphysis
    cv2.rectangle(img, (int(ulna_x - 8 * scale), shaft_top + int(4 * scale)), (int(ulna_x + 8 * scale), shaft_bot), 165.0, -1)
    cv2.circle(img, (ulna_x, shaft_top + int(1 * scale)), int(6 * scale), 185.0, -1)

    # Carpal bones (wrist ossification centers)
    carpal_positions = [
        (-12, 162, 7),  # Capitate
        (8, 163, 6),    # Hamate
        (-18, 174, 5),  # Scaphoid
        (0, 175, 5),    # Lunate
        (15, 174, 5),   # Triquetrum
    ]
    for cx_off, cy_off, radius in carpal_positions:
        cx = int((center_x + cx_off) * scale)
        cy = int(cy_off * scale)
        r = int(radius * scale + rng.uniform(-1, 1))
        cv2.circle(img, (cx, cy), max(2, r), 180.0 + rng.uniform(-10, 20), -1)

    # Metacarpals (5 bones)
    meta_x_offsets = [-26, -13, 0, 13, 26]
    for i, mx_off in enumerate(meta_x_offsets):
        mx = int((center_x + mx_off) * scale)
        m_top = int(120 * scale + (abs(mx_off) * 0.4) * scale)
        m_bot = int(155 * scale)
        # Metacarpal shaft
        cv2.line(img, (mx, m_bot), (mx, m_top), 185.0, int(6 * scale), lineType=cv2.LINE_AA)
        # Metacarpal head
        cv2.circle(img, (mx, m_top), int(5 * scale), 205.0, -1)
        # Metacarpal growth plate
        cv2.circle(img, (mx, m_top - int(3 * scale)), int(4 * scale), 90.0, 1)

    # Phalanges (Proximal, Middle, Distal) for 5 fingers
    for i, (off_x, base_y, length, width, angle) in enumerate(finger_configs):
        fx = int((center_x + off_x) * scale)
        fy = int(base_y * scale)
        rad = math.radians(angle)
        dir_x = math.sin(rad)
        dir_y = -math.cos(rad)

        num_segments = 2 if i == 0 else 3  # Thumb has 2 phalanges, others have 3
        seg_len = (length * scale) / num_segments

        curr_x = float(fx)
        curr_y = float(fy)

        for s in range(num_segments):
            next_x = curr_x + seg_len * dir_x
            next_y = curr_y + seg_len * dir_y
            bone_thick = max(2, int((width - s * 2.2) * scale * 0.55))
            
            # Phalanx shaft
            cv2.line(img, (int(curr_x), int(curr_y)), (int(next_x), int(next_y)), 190.0, bone_thick, lineType=cv2.LINE_AA)
            # Epiphysis (growth plate) dot
            cv2.circle(img, (int(curr_x), int(curr_y)), bone_thick + 1, 215.0, -1)
            cv2.circle(img, (int(curr_x), int(curr_y)), bone_thick, 85.0, 1)

            curr_x, curr_y = next_x, next_y

    # Smooth soft tissue / bone boundaries slightly with Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    # Radiographic exposure noise
    shot_noise = np.random.normal(0, 3.5, (size, size)).astype(np.float32)
    img = np.clip(img + shot_noise, 0, 255).astype(np.uint8)

    # Convert to 3-channel (radiographs are single-channel displayed as RGB)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def generate_non_hand_xray(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate non-hand medical radiographs (Chest, Skull, Pelvis, Dental)."""
    if rng is None:
        rng = random.Random()

    img = np.zeros((size, size), dtype=np.float32)
    sub_type = rng.choice(["chest", "skull", "spine_pelvis", "dental"])
    scale = size / 224.0

    if sub_type == "chest":
        # Bilateral translucent lungs (dark fields), mediastinum/heart (white), ribs (curved white bands)
        img[:] = 50.0  # Thoracic soft tissue
        # Left and right lung fields
        cv2.ellipse(img, (int(65 * scale), int(110 * scale)), (int(38 * scale), int(65 * scale)), 0, 0, 360, 22.0, -1)
        cv2.ellipse(img, (int(155 * scale), int(115 * scale)), (int(36 * scale), int(62 * scale)), 0, 0, 360, 22.0, -1)
        # Heart shadow (prominent left-sided opacity)
        cv2.ellipse(img, (int(125 * scale), int(135 * scale)), (int(35 * scale), int(42 * scale)), -25, 0, 360, 155.0, -1)
        # Spine vertebrae (central column)
        cv2.rectangle(img, (int(105 * scale), 0), (int(118 * scale), size), 130.0, -1)
        # Rib cage arcs (horizontal curved bone bands)
        for ry in range(40, 200, 22):
            cv2.ellipse(img, (int(112 * scale), int(ry * scale)), (int(85 * scale), int(28 * scale)), 0, 0, 180, 175.0, int(4 * scale))
        # Clavicles (bilateral upper horizontal bones)
        cv2.line(img, (int(30 * scale), int(35 * scale)), (int(105 * scale), int(45 * scale)), 210.0, int(6 * scale))
        cv2.line(img, (int(194 * scale), int(35 * scale)), (int(119 * scale), int(45 * scale)), 210.0, int(6 * scale))

    elif sub_type == "skull":
        # Rounded cranial vault (dense outer cortex) + facial bones
        img[:] = 20.0
        # Cranium
        cv2.ellipse(img, (int(112 * scale), int(105 * scale)), (int(75 * scale), int(85 * scale)), 0, 0, 360, 180.0, int(8 * scale))
        cv2.ellipse(img, (int(112 * scale), int(105 * scale)), (int(67 * scale), int(77 * scale)), 0, 0, 360, 40.0, -1)
        # Cervical spine descending from skull base
        cv2.rectangle(img, (int(100 * scale), int(160 * scale)), (int(124 * scale), size), 160.0, -1)
        # Orbit / eye sockets and sinuses
        cv2.circle(img, (int(75 * scale), int(100 * scale)), int(14 * scale), 20.0, -1)
        cv2.circle(img, (int(149 * scale), int(100 * scale)), int(14 * scale), 20.0, -1)
        # Teeth / jaw
        cv2.rectangle(img, (int(80 * scale), int(145 * scale)), (int(144 * scale), int(165 * scale)), 190.0, -1)

    elif sub_type == "spine_pelvis":
        # Pelvic bowl or knee joint: large femoral shaft and joint space
        img[:] = 40.0
        cv2.ellipse(img, (int(112 * scale), int(80 * scale)), (int(85 * scale), int(45 * scale)), 0, 0, 180, 185.0, int(16 * scale))
        # Left and right femoral shafts
        cv2.line(img, (int(45 * scale), int(90 * scale)), (int(55 * scale), size), 195.0, int(22 * scale))
        cv2.line(img, (int(179 * scale), int(90 * scale)), (int(169 * scale), size), 195.0, int(22 * scale))
        # Lumbar vertebrae column
        for vy in range(10, 85, 18):
            cv2.rectangle(img, (int(98 * scale), int(vy * scale)), (int(126 * scale), int((vy + 12) * scale)), 170.0, -1)

    else:  # dental
        img[:] = 30.0
        # Curved panoramic mandibular arch
        cv2.ellipse(img, (int(112 * scale), int(70 * scale)), (int(90 * scale), int(80 * scale)), 0, 0, 180, 160.0, int(15 * scale))
        # Rows of radiopaque teeth crowns and roots
        for tx in range(35, 190, 10):
            cv2.rectangle(img, (int(tx * scale), int(90 * scale)), (int((tx + 6) * scale), int(115 * scale)), 225.0, -1)
            cv2.line(img, (int((tx + 3) * scale), int(115 * scale)), (int((tx + 3) * scale), int(135 * scale)), 190.0, 2)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    noise = np.random.normal(0, 3.5, (size, size)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def generate_normal_photograph(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate realistic natural/human photographs (faces, portraits, landscapes, indoor objects)."""
    if rng is None:
        rng = random.Random()

    img = np.zeros((size, size, 3), dtype=np.float32)
    sub_type = rng.choice(["portrait", "landscape", "indoor_scene"])
    scale = size / 224.0

    if sub_type == "portrait":
        # Background: warm indoor or bokeh gradient
        bg_color = [rng.uniform(70, 140), rng.uniform(80, 150), rng.uniform(100, 180)]
        img[:, :] = bg_color
        # Human skin tone: BGR (B: 120-160, G: 150-190, R: 200-245)
        skin_bgr = [rng.uniform(120, 160), rng.uniform(150, 190), rng.uniform(205, 245)]
        hair_bgr = [rng.uniform(20, 60), rng.uniform(25, 70), rng.uniform(30, 80)]
        clothing_bgr = [rng.uniform(80, 220), rng.uniform(60, 200), rng.uniform(50, 180)]

        # Shoulders / Clothing
        cv2.ellipse(img, (int(112 * scale), int(230 * scale)), (int(95 * scale), int(70 * scale)), 0, 0, 360, clothing_bgr, -1)
        # Neck
        cv2.rectangle(img, (int(98 * scale), int(140 * scale)), (int(126 * scale), int(185 * scale)), skin_bgr, -1)
        # Head / Face oval
        cv2.ellipse(img, (int(112 * scale), int(105 * scale)), (int(52 * scale), int(68 * scale)), 0, 0, 360, skin_bgr, -1)
        # Hair
        cv2.ellipse(img, (int(112 * scale), int(80 * scale)), (int(56 * scale), int(45 * scale)), 0, 180, 360, hair_bgr, -1)
        # Eyes
        cv2.circle(img, (int(92 * scale), int(100 * scale)), int(6 * scale), [240, 240, 240], -1)
        cv2.circle(img, (int(92 * scale), int(100 * scale)), int(3 * scale), [40, 30, 20], -1)
        cv2.circle(img, (int(132 * scale), int(100 * scale)), int(6 * scale), [240, 240, 240], -1)
        cv2.circle(img, (int(132 * scale), int(100 * scale)), int(3 * scale), [40, 30, 20], -1)
        # Lips / mouth
        cv2.ellipse(img, (int(112 * scale), int(135 * scale)), (int(14 * scale), int(5 * scale)), 0, 0, 360, [110, 110, 195], -1)

    elif sub_type == "landscape":
        # Sky: blue gradient
        for y in range(int(size * 0.55)):
            ratio = y / (size * 0.55)
            b = 230 - 40 * ratio
            g = 180 - 40 * ratio
            r = 100 - 30 * ratio
            img[y, :] = [b, g, r]
        # Sun / clouds
        cv2.circle(img, (int(170 * scale), int(45 * scale)), int(18 * scale), [140, 230, 255], -1)
        # Green hills / ground
        cv2.ellipse(img, (int(60 * scale), int(size * 0.8)), (int(130 * scale), int(75 * scale)), 0, 0, 360, [40, 130, 50], -1)
        cv2.ellipse(img, (int(180 * scale), int(size * 0.85)), (int(120 * scale), int(65 * scale)), 0, 0, 360, [30, 110, 40], -1)
        # Trees
        for tx in [45, 90, 150]:
            cv2.rectangle(img, (int((tx - 3) * scale), int(120 * scale)), (int((tx + 3) * scale), int(160 * scale)), [30, 60, 90], -1)
            cv2.circle(img, (int(tx * scale), int(115 * scale)), int(16 * scale), [25, 90, 35], -1)

    else:  # indoor / everyday scene
        # Colorful room or table with fruit / cup / books
        img[:, :] = [rng.uniform(180, 230), rng.uniform(190, 235), rng.uniform(200, 240)]
        # Table top
        cv2.rectangle(img, (0, int(130 * scale)), (size, size), [60, 90, 140], -1)
        # Cup / mug
        cv2.rectangle(img, (int(60 * scale), int(105 * scale)), (int(95 * scale), int(155 * scale)), [210, 80, 40], -1)
        # Book / laptop
        pts = np.array([[120 * scale, 135 * scale], [195 * scale, 125 * scale], [205 * scale, 165 * scale], [130 * scale, 175 * scale]], np.int32)
        cv2.fillPoly(img, [pts], [50, 160, 220])

    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    img = np.clip(img + np.random.normal(0, 2.0, (size, size, 3)), 0, 255).astype(np.uint8)
    return img


def generate_animal_photograph(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate animal photographs (domestic cats, dogs, wildlife)."""
    if rng is None:
        rng = random.Random()

    img = np.zeros((size, size, 3), dtype=np.float32)
    scale = size / 224.0
    animal_type = rng.choice(["cat", "dog"])

    # Natural background: grass or carpet
    img[:, :] = [rng.uniform(50, 90), rng.uniform(100, 160), rng.uniform(50, 100)]

    # Fur color (golden, brown, black, white, orange)
    fur_bgr = rng.choice([
        [40, 120, 190],   # Ginger / orange
        [30, 65, 110],    # Golden retriever brown
        [35, 35, 40],     # Dark / black fur
        [190, 200, 210],  # White / cream fur
    ])

    cx = int(112 * scale)
    cy = int(118 * scale)

    # Animal body
    cv2.ellipse(img, (cx, int(175 * scale)), (int(65 * scale), int(55 * scale)), 0, 0, 360, fur_bgr, -1)
    # Animal head
    cv2.circle(img, (cx, cy), int(42 * scale), fur_bgr, -1)

    if animal_type == "cat":
        # Pointed triangle cat ears
        ear_l = np.array([[cx - 36 * scale, cy - 25 * scale], [cx - 20 * scale, cy - 65 * scale], [cx - 5 * scale, cy - 35 * scale]], np.int32)
        ear_r = np.array([[cx + 36 * scale, cy - 25 * scale], [cx + 20 * scale, cy - 65 * scale], [cx + 5 * scale, cy - 35 * scale]], np.int32)
        cv2.fillPoly(img, [ear_l, ear_r], fur_bgr)
        # Inner ears (pink)
        cv2.fillPoly(img, [np.array([[cx - 30 * scale, cy - 28 * scale], [cx - 20 * scale, cy - 55 * scale], [cx - 10 * scale, cy - 35 * scale]], np.int32)], [160, 150, 220])
        cv2.fillPoly(img, [np.array([[cx + 30 * scale, cy - 28 * scale], [cx + 20 * scale, cy - 55 * scale], [cx + 10 * scale, cy - 35 * scale]], np.int32)], [160, 150, 220])
        # Green / yellow slit eyes
        cv2.circle(img, (int(cx - 18 * scale), int(cy - 6 * scale)), int(7 * scale), [30, 200, 200], -1)
        cv2.circle(img, (int(cx + 18 * scale), int(cy - 6 * scale)), int(7 * scale), [30, 200, 200], -1)
        cv2.ellipse(img, (int(cx - 18 * scale), int(cy - 6 * scale)), (int(2 * scale), int(6 * scale)), 0, 0, 360, [10, 10, 10], -1)
        cv2.ellipse(img, (int(cx + 18 * scale), int(cy - 6 * scale)), (int(2 * scale), int(6 * scale)), 0, 0, 360, [10, 10, 10], -1)
        # Whiskers
        for dy in [-2, 2, 6]:
            cv2.line(img, (int(cx - 12 * scale), int((cy + 14 + dy) * scale)), (int(cx - 45 * scale), int((cy + 10 + dy * 2) * scale)), [240, 240, 240], 1)
            cv2.line(img, (int(cx + 12 * scale), int((cy + 14 + dy) * scale)), (int(cx + 45 * scale), int((cy + 10 + dy * 2) * scale)), [240, 240, 240], 1)
    else:  # dog
        # Floppy or upright dog ears
        cv2.ellipse(img, (int(cx - 38 * scale), int(cy - 10 * scale)), (int(14 * scale), int(30 * scale)), 25, 0, 360, [b * 0.8 for b in fur_bgr], -1)
        cv2.ellipse(img, (int(cx + 38 * scale), int(cy - 10 * scale)), (int(14 * scale), int(30 * scale)), -25, 0, 360, [b * 0.8 for b in fur_bgr], -1)
        # Snout / muzzle (white / cream or black)
        cv2.ellipse(img, (cx, int(cy + 14 * scale)), (int(22 * scale), int(18 * scale)), 0, 0, 360, [210, 210, 220], -1)
        # Large black nose
        cv2.ellipse(img, (cx, int(cy + 10 * scale)), (int(10 * scale), int(7 * scale)), 0, 0, 360, [20, 20, 20], -1)
        # Brown dog eyes
        cv2.circle(img, (int(cx - 18 * scale), int(cy - 10 * scale)), int(6 * scale), [20, 50, 90], -1)
        cv2.circle(img, (int(cx + 18 * scale), int(cy - 10 * scale)), int(6 * scale), [20, 50, 90], -1)

    # Fur texture noise
    fur_noise = np.random.normal(0, 7.0, (size, size, 3)).astype(np.float32)
    img = np.clip(img + fur_noise, 0, 255).astype(np.uint8)
    return img


def generate_document_or_screenshot(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate documents (PDFs, invoices, receipts) and UI screenshots."""
    if rng is None:
        rng = random.Random()

    img = np.zeros((size, size, 3), dtype=np.float32)
    sub_type = rng.choice(["document", "code_ide", "ui_app"])
    scale = size / 224.0

    if sub_type == "document":
        # White paper background
        img[:, :] = [rng.uniform(245, 255), rng.uniform(245, 255), rng.uniform(245, 255)]
        # Document title / header bar
        cv2.rectangle(img, (int(24 * scale), int(18 * scale)), (int(120 * scale), int(26 * scale)), [30, 30, 30], -1)
        cv2.rectangle(img, (int(24 * scale), int(32 * scale)), (int(200 * scale), int(34 * scale)), [180, 180, 180], -1)
        # Paragraphs of text lines
        y = int(48 * scale)
        for _ in range(12):
            if y >= size - int(20 * scale):
                break
            line_w = int(rng.uniform(120, 176) * scale)
            cv2.rectangle(img, (int(24 * scale), y), (int(24 * scale + line_w), y + int(3 * scale)), [50, 50, 50], -1)
            y += int(rng.uniform(8, 12) * scale)

        # Table grid or barcode box at bottom
        cv2.rectangle(img, (int(24 * scale), int(160 * scale)), (int(200 * scale), int(205 * scale)), [120, 120, 120], 1)
        for gy in range(int(172 * scale), int(205 * scale), int(11 * scale)):
            cv2.line(img, (int(24 * scale), gy), (int(200 * scale), gy), [180, 180, 180], 1)

    elif sub_type == "code_ide":
        # Dark mode code editor (VSCode style)
        img[:, :] = [35, 30, 30]  # #1e1e1e BGR
        # Sidebar
        cv2.rectangle(img, (0, 0), (int(42 * scale), size), [45, 38, 38], -1)
        # Tab bar
        cv2.rectangle(img, (int(42 * scale), 0), (size, int(22 * scale)), [40, 35, 35], -1)
        cv2.rectangle(img, (int(45 * scale), 0), (int(105 * scale), int(22 * scale)), [35, 30, 30], -1)
        # Code lines with syntax highlighting colors (keywords, strings, functions)
        y = int(32 * scale)
        colors = [
            [210, 140, 80],   # Blue keyword
            [100, 190, 215],  # Yellow function
            [100, 180, 140],  # Green string
            [200, 200, 200],  # White variable
        ]
        for _ in range(14):
            x = int(50 * scale)
            indent = int(rng.choice([0, 12, 24]) * scale)
            x += indent
            for _ in range(rng.randint(2, 5)):
                w = int(rng.uniform(15, 35) * scale)
                col = rng.choice(colors)
                cv2.rectangle(img, (x, y), (x + w, y + int(3 * scale)), col, -1)
                x += w + int(4 * scale)
            y += int(11 * scale)

    else:  # UI screenshot
        # Application window with titlebar, controls, buttons
        img[:, :] = [238, 238, 238]
        # Titlebar
        cv2.rectangle(img, (0, 0), (size, int(24 * scale)), [215, 215, 215], -1)
        # Window buttons (red, yellow, green)
        cv2.circle(img, (int(12 * scale), int(12 * scale)), int(4 * scale), [60, 60, 230], -1)
        cv2.circle(img, (int(24 * scale), int(12 * scale)), int(4 * scale), [50, 190, 230], -1)
        cv2.circle(img, (int(36 * scale), int(12 * scale)), int(4 * scale), [70, 210, 80], -1)
        # Navigation header & buttons
        cv2.rectangle(img, (int(15 * scale), int(35 * scale)), (int(75 * scale), int(55 * scale)), [210, 120, 40], -1)
        cv2.rectangle(img, (int(85 * scale), int(35 * scale)), (int(135 * scale), int(55 * scale)), [180, 180, 180], -1)
        # Main content card
        cv2.rectangle(img, (int(15 * scale), int(70 * scale)), (int(209 * scale), int(205 * scale)), [255, 255, 255], -1)

    return img.astype(np.uint8)


def generate_other_unrelated(size: int = 224, rng: Optional[random.Random] = None) -> np.ndarray:
    """Generate abstract unrelated images (noise, solid fields, test patterns, geometric shapes)."""
    if rng is None:
        rng = random.Random()

    sub_type = rng.choice(["pure_noise", "checkerboard", "solid_gradient", "geometric_shapes"])
    scale = size / 224.0

    if sub_type == "pure_noise":
        noise = np.random.uniform(0, 255, (size, size, 3)).astype(np.uint8)
        return noise

    elif sub_type == "checkerboard":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        tile = int(rng.choice([16, 28, 32]) * scale)
        color1 = [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)]
        color2 = [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)]
        for r in range(0, size, tile):
            for c in range(0, size, tile):
                if ((r // tile) + (c // tile)) % 2 == 0:
                    img[r : r + tile, c : c + tile] = color1
                else:
                    img[r : r + tile, c : c + tile] = color2
        return img

    elif sub_type == "solid_gradient":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        c_start = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)])
        c_end = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)])
        for y in range(size):
            alpha = y / float(size)
            img[y, :] = ((1 - alpha) * c_start + alpha * c_end).astype(np.uint8)
        return img

    else:  # geometric shapes
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = [rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60)]
        for _ in range(5):
            shape = rng.choice(["circle", "rect", "line"])
            color = [rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255)]
            if shape == "circle":
                cv2.circle(img, (rng.randint(20, size - 20), rng.randint(20, size - 20)), rng.randint(10, 45), color, -1)
            elif shape == "rect":
                pt1 = (rng.randint(10, size - 50), rng.randint(10, size - 50))
                pt2 = (pt1[0] + rng.randint(20, 60), pt1[1] + rng.randint(20, 60))
                cv2.rectangle(img, pt1, pt2, color, -1)
            else:
                cv2.line(img, (rng.randint(0, size), rng.randint(0, size)), (rng.randint(0, size), rng.randint(0, size)), color, rng.randint(2, 6))
        return img


CATEGORY_GENERATORS = {
    0: generate_hand_xray_phantom,
    1: generate_non_hand_xray,
    2: generate_normal_photograph,
    3: generate_animal_photograph,
    4: generate_document_or_screenshot,
    5: generate_other_unrelated,
}


# ============================================================================
# PyTorch Dataset (Uses single shared preprocess_validator_image from app.models.validator)
# ============================================================================


class RealValidatorDataset(Dataset):
    """Dataset loading real image files from disk for 2-class validator training/evaluation.

    Label 0: valid_pediatric_hand_xray (real RSNA radiographs)
    Label 1: other (real negative images from user-provided folders)
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
        target_size: int = 224,
    ):
        self.samples = [(Path(p), int(lbl)) for p, lbl in samples]
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image file: {img_path}")

        if self.transform is not None:
            image = self.transform(image)

        tensor = preprocess_validator_image(image, self.target_size)
        target = torch.tensor(label, dtype=torch.long)
        return tensor, target


class HandValidationDataset(Dataset):
    """In-memory dataset (kept for unit tests only)."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        transform: Optional[Callable] = None,
        target_size: int = 224,
    ):
        self.samples = samples
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.samples[idx]

        if self.transform is not None:
            image = self.transform(image)

        tensor = preprocess_validator_image(image, self.target_size)
        target = torch.tensor(label, dtype=torch.long)
        return tensor, target


def build_synthetic_validator_dataset(
    samples_per_class: int = 120,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """Synthetic dataset builder (kept strictly for unit tests only; NEVER used for model training)."""
    rng = random.Random(seed)
    np.random.seed(seed)

    train_samples: List[Tuple[np.ndarray, int]] = []
    val_samples: List[Tuple[np.ndarray, int]] = []
    test_samples: List[Tuple[np.ndarray, int]] = []

    n_train = int(samples_per_class * 0.70)
    n_val = int(samples_per_class * 0.15)

    # Map class 0 -> synthetic hand phantom, class 1 -> mix of non-hand synthetic generators
    pos_samples = [(generate_hand_xray_phantom(size=224, rng=rng), 0) for _ in range(samples_per_class)]
    neg_generators = [
        generate_non_hand_xray,
        generate_normal_photograph,
        generate_animal_photograph,
        generate_document_or_screenshot,
        generate_other_unrelated,
    ]
    neg_samples = [
        (neg_generators[i % len(neg_generators)](size=224, rng=rng), 1)
        for i in range(samples_per_class)
    ]

    for class_samples in (pos_samples, neg_samples):
        rng.shuffle(class_samples)
        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train : n_train + n_val])
        test_samples.extend(class_samples[n_train + n_val :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples
