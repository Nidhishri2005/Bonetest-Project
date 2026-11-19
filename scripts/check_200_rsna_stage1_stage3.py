"""Evaluate Stage 1 and Stage 3 of ImageValidator on 200 random RSNA images
that are NOT in tests/validator/should_accept.

- Does NOT modify any thresholds.
- Saves the 30 should_accept filenames and 200 check filenames to
  configs/validator_excluded_eval_files.json so training can exclude them.
- Reports Stage 1 pass rate, Stage 3 pass rate, joint pass rate, any failures,
  and the full Stage 3 coverage distribution.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import yaml

from app.services.image_validator import ImageValidator


def resolve_rsna_dir() -> Path:
    with open(ROOT_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["dataset_dir"])
    for c in [
        base / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        base / "boneage-training-dataset" / "boneage-training-dataset",
        base / "data" / "boneage-training-dataset",
        base / "boneage-training-dataset",
    ]:
        if c.exists() and any(c.glob("*.png")):
            return c
    raise FileNotFoundError(f"Could not locate RSNA training PNGs under {base}")


def main() -> None:
    rsna_dir = resolve_rsna_dir()
    should_accept_dir = ROOT_DIR / "tests" / "validator" / "should_accept"
    should_reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"

    should_accept_names = sorted([p.name for p in should_accept_dir.iterdir() if p.is_file()]) if should_accept_dir.exists() else []
    should_reject_names = sorted([p.name for p in should_reject_dir.iterdir() if p.is_file()]) if should_reject_dir.exists() else []
    should_accept_set = set(should_accept_names)

    all_rsna_pngs = sorted([p for p in rsna_dir.glob("*.png") if p.name not in should_accept_set])
    rng = random.Random(101)
    sample_200 = rng.sample(all_rsna_pngs, 200)
    sample_200_names = sorted([p.name for p in sample_200], key=lambda s: int(Path(s).stem) if Path(s).stem.isdigit() else s)

    # Save exclusion manifest for training / Colab
    exclusion_manifest = {
        "seed": 101,
        "should_accept_count": len(should_accept_names),
        "should_accept_files": should_accept_names,
        "check_200_count": len(sample_200_names),
        "check_200_files": sample_200_names,
        "should_reject_count": len(should_reject_names),
        "should_reject_files": should_reject_names,
    }
    manifest_out = ROOT_DIR / "configs" / "validator_excluded_eval_files.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(exclusion_manifest, f, indent=2)

    validator = ImageValidator()
    print("=" * 100)
    print("STAGE 1 & STAGE 3 EVALUATION ON 200 RANDOM RSNA IMAGES (EXCLUDING should_accept)")
    print("=" * 100)
    print(f"RSNA folder                   : {rsna_dir}")
    print(f"Total RSNA PNGs on disk       : {len(all_rsna_pngs) + len(should_accept_set)}")
    print(f"Excluded should_accept files  : {len(should_accept_set)}")
    print(f"Eligible pool sampled from    : {len(all_rsna_pngs)}")
    print(f"Sampled check images          : {len(sample_200)} (seed=101)")
    print(
        f"Active thresholds (unchanged) : min_coverage_ratio={validator.min_coverage_ratio}, "
        f"max_coverage_ratio={validator.max_coverage_ratio}, "
        f"max_mean_color_diff={validator.max_mean_color_diff}, "
        f"max_colored_pixel_fraction={validator.max_colored_pixel_fraction}"
    )
    print("-" * 100)

    stage1_passed = 0
    stage3_passed = 0
    both_passed = 0
    coverages = []
    failures = []

    for p in sample_200:
        raw_bytes = p.read_bytes()
        ok1, img_bgr, err1 = validator.stage1_rule_checks(raw_bytes)
        if ok1 and img_bgr is not None:
            stage1_passed += 1
            ok3, cov, err3 = validator.stage3_anatomical_coverage(img_bgr)
            coverages.append(cov)
            if ok3:
                stage3_passed += 1
                both_passed += 1
            else:
                failures.append((p.name, "Stage 3 (Coverage)", cov, err3))
        else:
            failures.append((p.name, "Stage 1 (Rules)", 0.0, err1))

    cov_arr = np.array(coverages, dtype=np.float64)
    total = len(sample_200)

    print(f"Stage 1 Pass Rate             : {stage1_passed}/{total} ({stage1_passed / total * 100:.2f}%)")
    print(f"Stage 3 Pass Rate             : {stage3_passed}/{total} ({stage3_passed / total * 100:.2f}%)")
    print(f"Joint (Stage 1 & 3) Pass Rate : {both_passed}/{total} ({both_passed / total * 100:.2f}%)")
    print("-" * 100)
    print("STAGE 3 COVERAGE DISTRIBUTION (N = 200):")
    print(f"  Min       : {float(np.min(cov_arr)):.4f} ({float(np.min(cov_arr))*100:.2f}%)")
    print(f"  1st pct   : {float(np.percentile(cov_arr, 1)):.4f} ({float(np.percentile(cov_arr, 1))*100:.2f}%)")
    print(f"  5th pct   : {float(np.percentile(cov_arr, 5)):.4f} ({float(np.percentile(cov_arr, 5))*100:.2f}%)")
    print(f"  10th pct  : {float(np.percentile(cov_arr, 10)):.4f} ({float(np.percentile(cov_arr, 10))*100:.2f}%)")
    print(f"  25th pct  : {float(np.percentile(cov_arr, 25)):.4f} ({float(np.percentile(cov_arr, 25))*100:.2f}%)")
    print(f"  Median    : {float(np.median(cov_arr)):.4f} ({float(np.median(cov_arr))*100:.2f}%)")
    print(f"  Mean      : {float(np.mean(cov_arr)):.4f} ({float(np.mean(cov_arr))*100:.2f}%)")
    print(f"  Std       : {float(np.std(cov_arr)):.4f} ({float(np.std(cov_arr))*100:.2f}%)")
    print(f"  75th pct  : {float(np.percentile(cov_arr, 75)):.4f} ({float(np.percentile(cov_arr, 75))*100:.2f}%)")
    print(f"  90th pct  : {float(np.percentile(cov_arr, 90)):.4f} ({float(np.percentile(cov_arr, 90))*100:.2f}%)")
    print(f"  95th pct  : {float(np.percentile(cov_arr, 95)):.4f} ({float(np.percentile(cov_arr, 95))*100:.2f}%)")
    print(f"  99th pct  : {float(np.percentile(cov_arr, 99)):.4f} ({float(np.percentile(cov_arr, 99))*100:.2f}%)")
    print(f"  Max       : {float(np.max(cov_arr)):.4f} ({float(np.max(cov_arr))*100:.2f}%)")

    # Histogram bins
    bins = [0.0, 0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.98, 1.0001]
    counts, _ = np.histogram(cov_arr, bins=bins)
    print("\nCOVERAGE HISTOGRAM BINS:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], min(1.0, bins[i + 1])
        c = int(counts[i])
        bar = "#" * int(round(c / 2))
        flag = "  <-- OUT OF [0.03, 0.98] BOUNDS" if (hi <= 0.03 or lo >= 0.98) else ""
        print(f"  [{lo:0.2f}, {hi:0.2f}) : {c:3d} ({c / total * 100:5.1f}%)  {bar}{flag}")

    print("-" * 100)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for fname, stg, cov, msg in failures:
            print(f"  - {fname:<14} | {stg:<20} | cov={cov:.4f} | {msg}")
    else:
        print("FAILURES: None (0/200 failed)")
    print("=" * 100)
    print(f"Saved excluded file manifest to: {manifest_out}")


if __name__ == "__main__":
    main()
