"""Diagnostic & End-to-End Pipeline Check Script: scripts/validator_check.py

Runs the full 3-stage ImageValidator (Stage 1 rules + color gate, Stage 2 OOD kNN k=5 cosine
at 0.122576 threshold, Stage 3 anatomical coverage with 5% border ignored) end-to-end on:
  a) tests/validator/should_accept
  b) tests/validator/should_reject (all files currently in it, listing each file's score)
  c) the 200 check images from configs/validator_excluded_eval_files.json

NOTE: The 200 check images in (c) were previously used to check Stage 1/3 pass rates, and the
500 stratified held-out RSNA images were used to calibrate the 99% threshold (0.122576), so
they are not independent evidence of the final rate.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from app.services.image_validator import ImageValidator

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_rsna_training_dir() -> Path | None:
    cfg_path = ROOT_DIR / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get("dataset_dir", ""))
    for c in [
        base / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        base / "boneage-training-dataset" / "boneage-training-dataset",
        base / "data" / "boneage-training-dataset",
        base / "boneage-training-dataset",
    ]:
        if c.exists() and any(c.glob("*.png")):
            return c
    return None


def evaluate_image_list(
    validator: ImageValidator,
    image_paths: List[Path],
    section_title: str,
    print_every_image: bool = True,
) -> dict:
    print("\n" + "=" * 116)
    print(f"SECTION: {section_title} (Total N={len(image_paths)})")
    print(
        f"Thresholds: Stage 2 kNN(k=5, cosine) <= {validator.ood_threshold:.6f} | "
        f"Stage 3 coverage in [{validator.min_coverage_ratio:.2f}, {validator.max_coverage_ratio:.2f}] (5% border ignored)"
    )
    print("=" * 116)

    if not image_paths:
        print("  (0 images found)")
        return {"total": 0, "accepted": 0, "rejected": 0, "stage_counts": {}}

    accepted = 0
    rejected = 0
    stage_counts: Counter = Counter()
    rejection_rows = []

    for idx, p in enumerate(image_paths, 1):
        raw_bytes = p.read_bytes()
        res = validator.validate(raw_bytes)
        decision = "ACCEPT" if res.is_valid else "REJECT"

        # Compute kNN score even if rejected in Stage 1 so every file's kNN score is listed
        score_str = f"{res.score:.6f}" if res.score is not None else "None (Stage 1)"
        knn_even_if_s1 = res.score
        if knn_even_if_s1 is None:
            import cv2
            import numpy as np
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                feat = validator.extractor.extract_single_bgr(img)
                knn_even_if_s1 = validator.ood_validator.score_single_feature(feat)["knn_k5_cosine"]
                score_str = f"{knn_even_if_s1:.6f} (S1 rej)"

        if res.is_valid:
            accepted += 1
        else:
            rejected += 1
            stage_counts[res.deciding_stage] += 1
            rejection_rows.append((p.name, res.deciding_stage, score_str, res.technical_detail))

        if print_every_image or not res.is_valid:
            print(
                f"[{idx:03d}] file={p.name:<16} | decision={decision:<6} | "
                f"deciding_stage={res.deciding_stage:<16} | kNN_score={score_str:<19} | "
                f"thr={res.threshold:.6f} | cov={res.anatomical_coverage:.4f}"
            )
            if not res.is_valid and res.technical_detail:
                print(f"      tech_detail={res.technical_detail}")

    total = len(image_paths)
    print("-" * 116)
    print(
        f"SUMMARY [{section_title}]:\n"
        f"  - Accepted : {accepted}/{total} ({accepted / total * 100:.2f}%)\n"
        f"  - Rejected : {rejected}/{total} ({rejected / total * 100:.2f}%)\n"
        f"  - Rejections by Deciding Stage: {dict(stage_counts)}"
    )
    print("-" * 116)
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "stage_counts": dict(stage_counts),
        "rejection_rows": rejection_rows,
    }


def main() -> None:
    validator = ImageValidator()
    print("=" * 116)
    print("ONE-CLASS OOD VALIDATOR END-TO-END PIPELINE CHECK (STAGE 1 + STAGE 2 + STAGE 3)")
    print("=" * 116)
    print(f"Loaded OOD NPZ           : {validator.npz_path} ({validator.npz_path.stat().st_size:,} bytes)")
    print(f"OOD Method               : kNN (k={validator.ood_validator.knn_k}, metric={validator.ood_validator.config.get('knn_metric')})")
    print(f"Approved 99% Threshold   : {validator.ood_threshold:.6f}")

    # (a) tests/validator/should_accept
    accept_dir = ROOT_DIR / "tests" / "validator" / "should_accept"
    accept_files = sorted([p for p in accept_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    evaluate_image_list(validator, accept_files, "a) tests/validator/should_accept", print_every_image=True)

    # (b) tests/validator/should_reject
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"
    reject_files = sorted([p for p in reject_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    evaluate_image_list(validator, reject_files, "b) tests/validator/should_reject", print_every_image=True)

    # (c) 200 check images from configs/validator_excluded_eval_files.json
    excl_path = ROOT_DIR / "configs" / "validator_excluded_eval_files.json"
    rsna_dir = resolve_rsna_training_dir()
    if excl_path.exists() and rsna_dir is not None:
        with open(excl_path, "r", encoding="utf-8") as f:
            excl = json.load(f)
        check_200_names = excl.get("check_200_files", [])
        check_200_paths = [rsna_dir / fn for fn in check_200_names if (rsna_dir / fn).exists()]
        evaluate_image_list(
            validator,
            check_200_paths,
            "c) 200 Check Images (configs/validator_excluded_eval_files.json - used to check Stage 1/3, not independent)",
            print_every_image=False,
        )
    else:
        print(f"\n[WARN] Could not run (c) 200 check images: excl_exists={excl_path.exists()}, rsna_dir={rsna_dir}")


if __name__ == "__main__":
    main()
