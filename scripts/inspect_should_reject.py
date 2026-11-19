"""Inspect and score all images in tests/validator/should_reject and generate a labeled contact sheet."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from ml.validator_ood import DEFAULT_OOD_NPZ_PATH, OODValidator, ResNet18FeatureExtractor, extract_features_for_paths

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".webp", ".tif", ".tiff"}
SCRATCH_GRID = Path(r"C:\Users\Manya\.gemini\antigravity\brain\093bb757-419b-4e1d-90bc-edab3b8705bf\scratch\should_reject_grid.png")


def natural_key(p: Path):
    stem = p.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def main() -> None:
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"
    files = sorted(
        [p for p in reject_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    print(f"Found {len(files)} files in {reject_dir}")

    # Build contact sheet (5 cols x 5 rows)
    SCRATCH_GRID.parent.mkdir(parents=True, exist_ok=True)
    cols = 5
    rows = (len(files) + cols - 1) // cols
    cell_w, cell_h = 260, 260
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for idx, p in enumerate(files):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            thumb = rgb.copy()
            thumb.thumbnail((cell_w - 16, cell_h - 36))
            grid.paste(thumb, (x0 + 8, y0 + 28))
        rel = str(p.relative_to(reject_dir))
        draw.text((x0 + 8, y0 + 6), f"{rel} ({w}x{h})", fill=(255, 255, 100))

    grid.save(SCRATCH_GRID)
    print(f"Saved contact sheet to {SCRATCH_GRID}")

    # Score with OODValidator
    ood = OODValidator.load(DEFAULT_OOD_NPZ_PATH)
    extractor = ResNet18FeatureExtractor(device=torch.device("cpu"))
    feats, names = extract_features_for_paths(files, extractor, batch_size=32, num_workers=4, label="should_reject")
    scores = ood.score_features(feats)
    mahal = scores["mahalanobis"]
    knn = scores["knn_k5_cosine"]

    m98, m99 = ood.thresholds["mahalanobis_98"], ood.thresholds["mahalanobis_99"]
    k98, k99 = ood.thresholds["knn_k5_cosine_98"], ood.thresholds["knn_k5_cosine_99"]

    for p, ms, ks in zip(files, mahal, knn):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        b, g, r_ch = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
        spread = np.maximum(np.maximum(r_ch, g), b) - np.minimum(np.minimum(r_ch, g), b)
        mean_col = float(np.mean(spread))
        print(
            f"{p.name:<10} | shape={str(img.shape):<16} | mean_rgb_spread={mean_col:6.2f} | "
            f"Mahal={ms:8.4f} (@98={'ACC' if ms<=m98 else 'REJ'}, @99={'ACC' if ms<=m99 else 'REJ'}) | "
            f"kNN={ks:8.6f} (@98={'ACC' if ks<=k98 else 'REJ'}, @99={'ACC' if ks<=k99 else 'REJ'})"
        )


if __name__ == "__main__":
    main()
