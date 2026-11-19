"""STEP 2: Download real negative images (OxfordIIITPet, Flowers102, DTD) via torchvision.

- Reads output root and validator_negatives_dir from configs/data.yaml
- animals : 300 images from torchvision.datasets.OxfordIIITPet
- photos  : 200 images from torchvision.datasets.Flowers102 + 150 from torchvision.datasets.DTD
- Converts ~33% of images to grayscale (saved as 3-channel RGB) so grayscale is not a shortcut
- Saves as JPEG with seeded random quality in [60, 95]
- Safe to re-run (skips existing valid files)
- Logs total and grayscale counts per folder
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Tuple

import torchvision.datasets as dsets
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_paths() -> Tuple[Path, Path]:
    cfg_path = ROOT_DIR / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    neg_dir = Path(cfg.get("validator_negatives_dir", "C:/bonetest/data/validator_data/negatives"))
    data_root = Path(cfg.get("validator_data_root", str(neg_dir.parent)))
    return data_root, neg_dir


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def save_sampled_images(
    dataset,
    out_dir: Path,
    prefix: str,
    target_count: int,
    seed: int,
) -> Tuple[int, int, int]:
    """Sample target_count images from a torchvision dataset and save as JPEG quality 60-95."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    total_len = len(dataset)
    indices = list(range(total_len))
    rng.shuffle(indices)
    chosen_indices = indices[: min(target_count, total_len)]

    written = 0
    skipped = 0
    gray_count = 0

    for i, ds_idx in enumerate(chosen_indices, 1):
        out_path = out_dir / f"{prefix}_{i:04d}.jpg"
        make_gray = (i % 3 == 0)  # ~33.3% grayscale saved as 3-channel RGB
        quality = rng.randint(60, 95)

        if is_valid_image(out_path):
            skipped += 1
            if make_gray:
                gray_count += 1
            continue

        img, _ = dataset[ds_idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if make_gray:
            img = img.convert("L").convert("RGB")
            gray_count += 1
        else:
            img = img.convert("RGB")

        img.save(out_path, format="JPEG", quality=quality)
        written += 1

    return written, skipped, gray_count


def main() -> None:
    data_root, neg_dir = load_paths()
    raw_dir = data_root / "raw"
    raw_xray_dir = data_root / "raw_xray"

    animals_dir = neg_dir / "animals"
    photos_dir = neg_dir / "photos"
    xray_other_dir = neg_dir / "xray_other"
    documents_dir = neg_dir / "documents"

    for d in (raw_dir, raw_xray_dir, animals_dir, photos_dir, xray_other_dir, documents_dir):
        d.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(data_root)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    print("=" * 90)
    print("STEP 1 & 2: DISK SPACE & TORCHVISION NEGATIVE DATASET BUILDER")
    print("=" * 90)
    print(f"Data root              : {data_root}")
    print(f"Negatives root         : {neg_dir}")
    print(f"Disk space on drive    : {free_gb:.2f} GB free / {total_gb:.2f} GB total (need ~4 GB)")
    if free_gb < 4.0:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB free (< 4.0 GB required)")

    # 1. OxfordIIITPet -> animals (300 images)
    print("\n[1/3] Loading / downloading torchvision.datasets.OxfordIIITPet -> negatives/animals (300 images)...")
    pet_ds = dsets.OxfordIIITPet(root=str(raw_dir), split="trainval", download=True)
    w_pet, s_pet, g_pet = save_sampled_images(pet_ds, animals_dir, prefix="pet", target_count=300, seed=1001)
    print(f"  OxfordIIITPet done: newly_written={w_pet}, skipped_existing={s_pet}, gray_rgb={g_pet}/300")

    # 2. Flowers102 -> photos (200 images)
    print("\n[2/3] Loading / downloading torchvision.datasets.Flowers102 -> negatives/photos (200 images)...")
    flowers_ds = dsets.Flowers102(root=str(raw_dir), split="train", download=True)
    w_flw, s_flw, g_flw = save_sampled_images(flowers_ds, photos_dir, prefix="flower", target_count=200, seed=2002)
    print(f"  Flowers102 done   : newly_written={w_flw}, skipped_existing={s_flw}, gray_rgb={g_flw}/200")

    # 3. DTD -> photos (150 images)
    print("\n[3/3] Loading / downloading torchvision.datasets.DTD -> negatives/photos (150 images)...")
    dtd_ds = dsets.DTD(root=str(raw_dir), split="train", download=True)
    w_dtd, s_dtd, g_dtd = save_sampled_images(dtd_ds, photos_dir, prefix="dtd", target_count=150, seed=3003)
    print(f"  DTD done          : newly_written={w_dtd}, skipped_existing={s_dtd}, gray_rgb={g_dtd}/150")

    print("\n" + "-" * 90)
    print("FOLDER COUNTS AFTER STEP 2:")
    for folder in (animals_dir, photos_dir, xray_other_dir, documents_dir):
        files = [p for p in folder.iterdir() if p.is_file()]
        print(f"  {folder.relative_to(data_root)} : {len(files)} files")
    print("=" * 90)


if __name__ == "__main__":
    main()
