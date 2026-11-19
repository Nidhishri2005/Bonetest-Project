"""Diagnostic script: scripts/validator_check.py

Runs the ImageValidator on:
  a) 20 random images from the RSNA training folder configured in configs/data.yaml
  b) every image in tests/validator/should_accept and tests/validator/should_reject

For each image prints:
  - filename
  - stage that decided (file/decode/variance/color/classifier/coverage)
  - class probabilities
  - threshold used
  - final decision
And if the checkpoint on disk is still the legacy 6-class checkpoint (awaiting Colab
training of the 2-class checkpoint), reports the exact strict=True load error plus
Stage 1 + Stage 3 pass rates on all test images.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
import yaml

from app.models.validator import CLASS_TO_IDX, IDX_TO_CLASS, VALIDATOR_CLASSES, HandXRayValidator
from app.services.image_validator import IMAGE_MAGIC_BYTES, ImageValidator


def determine_deciding_stage(validator: ImageValidator, image_bytes: bytes) -> Tuple[str, dict]:
    """Trace exact sub-stage that decides ACCEPT or REJECT."""
    if not image_bytes or len(image_bytes) == 0:
        res = validator.validate(image_bytes)
        return "stage1_file_empty", {"result": res}
    if len(image_bytes) > 25 * 1024 * 1024:
        res = validator.validate(image_bytes)
        return "stage1_file_size", {"result": res}
    if not any(image_bytes.startswith(m) for m in IMAGE_MAGIC_BYTES):
        res = validator.validate(image_bytes)
        return "stage1_file_magic", {"result": res}

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        res = validator.validate(image_bytes)
        return "stage1_decode", {"result": res}

    h, w = img.shape[:2]
    if h < 32 or w < 32:
        res = validator.validate(image_bytes)
        return "stage1_dimensions", {"result": res}

    ar = float(w) / float(h)
    if ar < 0.20 or ar > 5.0:
        res = validator.validate(image_bytes)
        return "stage1_aspect_ratio", {"result": res}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 1.0:
        res = validator.validate(image_bytes)
        return "stage1_variance", {"result": res}

    ok1, _, err1 = validator.stage1_rule_checks(image_bytes)
    if not ok1:
        res = validator.validate(image_bytes)
        return "stage1_strong_color", {"result": res, "err1": err1}

    # Check Stage 3 (5% border ignored) independently as well
    ok3, cov3, err3 = validator.stage3_anatomical_coverage(img)

    try:
        res = validator.validate(image_bytes)
    except Exception as exc:
        return "stage2_checkpoint_strict_load_error", {
            "error": str(exc),
            "stage1_passed": ok1,
            "stage3_passed": ok3,
            "stage3_coverage": cov3,
            "stage3_err": err3,
        }

    gd = res.gate_details
    if gd.get("stage2_classifier") == "FAILED":
        return f"stage2_classifier({res.predicted_class})", {"result": res}
    if gd.get("stage3_coverage") == "FAILED":
        return "stage3_anatomical_coverage", {"result": res}
    return "stage3_accepted_all_gates", {"result": res}


def evaluate_image_list(
    validator: ImageValidator,
    image_paths: List[Path],
    section_title: str,
) -> None:
    print("\n" + "=" * 110)
    print(f"SECTION: {section_title} (count={len(image_paths)})")
    print(
        f"Thresholds: confidence>={validator.confidence_threshold:.2f}, "
        f"coverage in [{validator.min_coverage_ratio:.2f}, {validator.max_coverage_ratio:.2f}] (5% border ignored)"
    )
    print("=" * 110)

    if not image_paths:
        print("  (0 images in folder)")
        return

    accepted = 0
    stage1_ok = 0
    stage3_ok = 0
    strict_ckpt_blocked = 0
    total = len(image_paths)

    for idx, p in enumerate(image_paths, 1):
        raw_bytes = p.read_bytes()
        stage_label, info = determine_deciding_stage(validator, raw_bytes)

        if "result" in info:
            res = info["result"]
            decision = "ACCEPT" if res.is_valid else "REJECT"
            if res.is_valid:
                accepted += 1
            print(
                f"[{idx:02d}] file={p.name:<14} | decision={decision:<6} | "
                f"stage={stage_label:<36} | probs={res.probabilities} | "
                f"thr={res.threshold:.2f} | cov={res.anatomical_coverage:.4f}"
            )
            if not res.is_valid and res.rejection_message:
                print(f"     reason={res.rejection_message}")
        else:
            strict_ckpt_blocked += 1
            if info["stage1_passed"]:
                stage1_ok += 1
            if info["stage3_passed"]:
                stage3_ok += 1
            print(
                f"[{idx:02d}] file={p.name:<14} | stage1_rules={'PASS' if info['stage1_passed'] else 'FAIL'} | "
                f"stage3_cov(5%_border_ignored)={'PASS' if info['stage3_passed'] else 'FAIL'} "
                f"(cov={info['stage3_coverage']:.4f}) | stage2={stage_label}"
            )

    print("-" * 110)
    if strict_ckpt_blocked > 0:
        print(
            f"SUMMARY [{section_title}]:\n"
            f"  - Stage 1 (Rules + Strong-Color Gate) Passed : {stage1_ok}/{total} ({stage1_ok / total * 100:.1f}%)\n"
            f"  - Stage 3 (Coverage w/ 5% Border Ignored)    : {stage3_ok}/{total} ({stage3_ok / total * 100:.1f}%)\n"
            f"  - Stage 2 (2-Class Classifier)               : Blocked by strict=True check ({strict_ckpt_blocked}/{total}) "
            "because hand_validator_best.pt on disk is still the 6-class checkpoint awaiting Colab 2-class training."
        )
    else:
        rate = accepted / total * 100.0
        print(f"SUMMARY [{section_title}]: Accepted {accepted}/{total} ({rate:.1f}%), Rejected {total - accepted}/{total} ({100.0 - rate:.1f}%)")
    print("-" * 110)


def inspect_checkpoint(validator: ImageValidator) -> None:
    print("=" * 110)
    print("CHECKPOINT & 2-CLASS MODEL INSPECTION")
    print("=" * 110)
    pt_path = validator.checkpoint_path
    print(f"Configured PyTorch checkpoint path : {pt_path}")
    print(f"  exists={pt_path.exists()}  size={pt_path.stat().st_size if pt_path.exists() else 'N/A'} bytes")
    print(f"Code VALIDATOR_CLASSES (2-class)   : {VALIDATOR_CLASSES}")

    try:
        _ = validator.model
        print("  _load_model() strict=True status : PASSED (2-class checkpoint loaded)")
    except Exception as exc:
        print(f"  _load_model() strict=True status : RAISED {type(exc).__name__}: {exc}")


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


def main() -> None:
    validator = ImageValidator()
    inspect_checkpoint(validator)

    rsna_dir = resolve_rsna_training_dir()
    if rsna_dir is not None:
        all_pngs = sorted(rsna_dir.glob("*.png"))
        rng = random.Random(42)
        sample_20 = rng.sample(all_pngs, min(20, len(all_pngs)))
        evaluate_image_list(validator, sample_20, f"20 Random Raw RSNA Training X-Rays ({rsna_dir})")

    for sub in ("should_accept", "should_reject"):
        d = ROOT_DIR / "tests" / "validator" / sub
        imgs = sorted([p for p in d.iterdir() if p.is_file()]) if d.exists() else []
        evaluate_image_list(validator, imgs, f"tests/validator/{sub}")


if __name__ == "__main__":
    main()
