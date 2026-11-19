"""STEP 3: Sample ~300 non-hand X-rays from raw_xray into negatives/xray_other.

- Reads validator_raw_xray_dir and validator_negatives_dir from configs/data.yaml
- Strictly filters out any path containing hand/wrist/finger keywords (including MURA XR_HAND / XR_WRIST)
- Balances sampling across available subfolders (chest, knee, foot, etc.) up to 300 images
- Safe to re-run (skips existing valid files)
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FORBIDDEN_KEYWORDS = ("hand", "wrist", "xr_hand", "xr_wrist", "finger", "boneage")


def is_forbidden_hand_wrist(path: Path) -> bool:
    parts_lower = "/".join(p.lower() for p in path.parts)
    return any(kw in parts_lower for kw in FORBIDDEN_KEYWORDS)


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    cfg_path = ROOT_DIR / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    neg_dir = Path(cfg.get("validator_negatives_dir", "C:/bonetest/data/validator_data/negatives"))
    data_root = Path(cfg.get("validator_data_root", str(neg_dir.parent)))
    raw_xray_dir = Path(cfg.get("validator_raw_xray_dir", str(data_root / "raw_xray")))
    out_dir = neg_dir / "xray_other"

    raw_xray_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: List[Path] = []
    skipped_forbidden = 0
    for p in sorted(raw_xray_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if is_forbidden_hand_wrist(p.relative_to(raw_xray_dir)):
            skipped_forbidden += 1
            continue
        all_candidates.append(p)

    print("=" * 90)
    print("STEP 3: SAMPLE NON-HAND X-RAY NEGATIVES")
    print("=" * 90)
    print(f"Raw X-ray source folder        : {raw_xray_dir}")
    print(f"Target output folder           : {out_dir}")
    print(f"Eligible non-hand X-rays found : {len(all_candidates)}")
    print(f"Skipped hand/wrist paths       : {skipped_forbidden}")

    if not all_candidates:
        print(
            "\nNo raw X-ray files found in raw_xray yet. "
            "Place unpacked Chest, Knee, and Foot X-ray folders inside raw_xray and re-run this script."
        )
        return

    # Group by top-level subfolder under raw_xray (e.g. chest, knee, foot) for balanced sampling
    buckets: Dict[str, List[Path]] = {}
    for p in all_candidates:
        rel = p.relative_to(raw_xray_dir)
        group = rel.parts[0].lower() if len(rel.parts) > 1 else "root"
        buckets.setdefault(group, []).append(p)

    rng = random.Random(42)
    for g in buckets:
        rng.shuffle(buckets[g])

    target_total = 300
    selected: List[Tuple[str, Path]] = []
    groups = sorted(buckets.keys())
    per_group_quota = max(1, target_total // len(groups))

    leftovers: List[Tuple[str, Path]] = []
    for g in groups:
        items = buckets[g]
        take = items[:per_group_quota]
        rest = items[per_group_quota:]
        selected.extend((g, p) for p in take)
        leftovers.extend((g, p) for p in rest)

    rng.shuffle(leftovers)
    if len(selected) < target_total:
        need = target_total - len(selected)
        selected.extend(leftovers[:need])
    selected = selected[:target_total]

    written = 0
    skipped_existing = 0
    group_counts: Dict[str, int] = {}

    for idx, (g, src_path) in enumerate(selected, 1):
        out_path = out_dir / f"xray_{g}_{idx:04d}.jpg"
        group_counts[g] = group_counts.get(g, 0) + 1
        if is_valid_image(out_path):
            skipped_existing += 1
            continue
        try:
            with Image.open(src_path) as im:
                rgb = im.convert("RGB")
                rgb.save(out_path, format="JPEG", quality=rng.randint(75, 95))
            written += 1
        except Exception as e:
            print(f"  Skipping corrupt source {src_path}: {e}")

    total_in_out = len([p for p in out_dir.iterdir() if p.is_file()])
    print(f"Sampled groups breakdown       : {group_counts}")
    print(f"Newly written                  : {written}, Skipped existing: {skipped_existing}")
    print(f"Total files in xray_other      : {total_in_out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
