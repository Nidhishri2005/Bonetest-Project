"""STEP 5 & STEP 6: Move 30 hold-out negatives to tests/validator/should_reject,
update configs/validator_excluded_eval_files.json, verify PIL integrity & SHA-256
disjointness, and package negatives into validator_negatives.zip.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_or_remove_corrupt(folder: Path) -> Tuple[List[Path], int]:
    """Verify every file in folder opens and decodes with PIL; remove corrupt files."""
    if not folder.exists():
        return [], 0
    valid_files: List[Path] = []
    removed = 0
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(p) as im:
                im.load()
            valid_files.append(p)
        except Exception as e:
            print(f"  [CORRUPT REMOVED] {p}: {e}")
            p.unlink(missing_ok=True)
            removed += 1
    return valid_files, removed


def categorize_filename(name: str) -> str:
    lower = name.lower()
    if lower.startswith("pet_"):
        return "animals"
    if lower.startswith(("flower_", "dtd_")):
        return "photos"
    if lower.startswith("xray_"):
        return "xray_other"
    if lower.startswith("doc_"):
        return "documents"
    return "other"


def main() -> None:
    cfg_path = ROOT_DIR / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    neg_dir = Path(cfg.get("validator_negatives_dir", "C:/bonetest/data/validator_data/negatives"))
    data_root = Path(cfg.get("validator_data_root", str(neg_dir.parent)))
    should_reject_dir = Path(cfg.get("validator_test_reject_dir", str(ROOT_DIR / "tests" / "validator" / "should_reject")))
    should_reject_dir.mkdir(parents=True, exist_ok=True)

    categories = ["animals", "photos", "xray_other", "documents"]
    for c in categories:
        (neg_dir / c).mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("STEP 5 & 6: HOLD-OUT MOVE (30 IMAGES), PIL VERIFICATION, SHA-256 CHECK & ZIP PACKAGING")
    print("=" * 95)

    # First verify all files currently in negatives and should_reject
    total_corrupt = 0
    for c in categories:
        _, rem = verify_or_remove_corrupt(neg_dir / c)
        total_corrupt += rem
    existing_reject_files, rem_rej = verify_or_remove_corrupt(should_reject_dir)
    total_corrupt += rem_rej

    # Step 5: Move up to 30 random images (mixed across available categories) into should_reject
    target_holdout = 30
    needed = max(0, target_holdout - len(existing_reject_files))

    if needed > 0:
        rng = random.Random(5005)
        avail_by_cat: Dict[str, List[Path]] = {}
        for c in categories:
            files = sorted([p for p in (neg_dir / c).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
            if files:
                rng.shuffle(files)
                avail_by_cat[c] = files

        non_empty_cats = sorted(avail_by_cat.keys())
        if non_empty_cats:
            to_move: List[Path] = []
            per_cat = needed // len(non_empty_cats)
            leftovers: List[Path] = []
            for c in non_empty_cats:
                items = avail_by_cat[c]
                to_move.extend(items[:per_cat])
                leftovers.extend(items[per_cat:])
            rng.shuffle(leftovers)
            if len(to_move) < needed:
                to_move.extend(leftovers[: needed - len(to_move)])

            for src in to_move[:needed]:
                dst = should_reject_dir / src.name
                shutil.move(str(src), str(dst))
            print(f"Moved {len(to_move[:needed])} hold-out negative images -> {should_reject_dir}")
    else:
        print(f"should_reject already contains {len(existing_reject_files)} images (target={target_holdout}); no additional move needed.")

    # Re-list after move
    reject_files, _ = verify_or_remove_corrupt(should_reject_dir)
    reject_names = sorted([p.name for p in reject_files])
    reject_hashes: Dict[str, str] = {p.name: sha256_file(p) for p in reject_files}
    reject_hash_set = set(reject_hashes.values())

    # Ensure zero moved files remain in negatives (by filename or SHA-256)
    removed_dups = 0
    neg_hashes: Dict[str, str] = {}
    cat_counts: Dict[str, int] = {}
    for c in categories:
        c_dir = neg_dir / c
        c_files, rem = verify_or_remove_corrupt(c_dir)
        total_corrupt += rem
        kept = 0
        for p in c_files:
            h = sha256_file(p)
            if p.name in reject_hashes or h in reject_hash_set:
                print(f"  [DUPLICATE WITH should_reject REMOVED FROM negatives] {p}")
                p.unlink()
                removed_dups += 1
            else:
                neg_hashes[str(p.relative_to(neg_dir))] = h
                kept += 1
        cat_counts[c] = kept

    # Update configs/validator_excluded_eval_files.json (append/merge, preserve existing keys)
    manifest_path = ROOT_DIR / "configs" / "validator_excluded_eval_files.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    old_reject_files = set(manifest.get("should_reject_files", []))
    old_reject_hashes = dict(manifest.get("should_reject_sha256", {}))
    merged_reject_files = sorted(old_reject_files | set(reject_names))
    old_reject_hashes.update(reject_hashes)

    manifest["should_reject_count"] = len(merged_reject_files)
    manifest["should_reject_files"] = merged_reject_files
    manifest["should_reject_sha256"] = old_reject_hashes

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Confirm zero SHA-256 overlap between negatives and should_reject
    overlap_hashes = set(neg_hashes.values()) & reject_hash_set
    overlap_names = {Path(k).name for k in neg_hashes.keys()} & set(reject_names)

    # Breakdown of should_reject by category
    reject_by_cat: Dict[str, int] = {c: 0 for c in categories}
    for p in reject_files:
        cat = categorize_filename(p.name)
        reject_by_cat[cat] = reject_by_cat.get(cat, 0) + 1

    # Package negatives into validator_negatives.zip
    zip_path = data_root / "validator_negatives.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for c in categories:
            c_dir = neg_dir / c
            for p in sorted(c_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    arcname = str(p.relative_to(neg_dir)).replace("\\", "/")
                    zf.write(p, arcname=arcname)

    zip_bytes = zip_path.stat().st_size
    zip_mb = zip_bytes / (1024 ** 2)

    print("-" * 95)
    print(f"Corrupt files removed                : {total_corrupt}")
    print(f"SHA-256 overlap (negatives vs reject): {len(overlap_hashes)} (filename overlap: {len(overlap_names)})")
    print(f"Manifest updated                     : {manifest_path}")
    print(f"  should_accept_count in manifest    : {manifest.get('should_accept_count')}")
    print(f"  check_200_count in manifest        : {manifest.get('check_200_count')}")
    print(f"  should_reject_count in manifest    : {manifest.get('should_reject_count')}")
    print("-" * 95)
    print(f"{'Category':<18} | {'In negatives/ (Training)':<26} | {'In tests/validator/should_reject':<32}")
    print("-" * 95)
    for c in categories:
        print(f"{c:<18} | {cat_counts.get(c, 0):<26d} | {reject_by_cat.get(c, 0):<32d}")
    print("-" * 95)
    print(f"{'TOTAL':<18} | {sum(cat_counts.values()):<26d} | {len(reject_files):<32d}")
    print("-" * 95)
    print(f"Created zip archive                  : {zip_path}")
    print(f"Zip size                             : {zip_bytes:,} bytes ({zip_mb:.2f} MB)")
    print("=" * 95)


if __name__ == "__main__":
    main()
