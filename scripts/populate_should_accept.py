"""Step 1 helper: Populate tests/validator/should_accept with 30 real RSNA images
(varied ages, both sexes, including low-contrast real radiographs) and create
an empty tests/validator/should_reject directory. No synthetic fixtures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def resolve_rsna_dir() -> Path:
    with open(ROOT_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["dataset_dir"])
    candidates = [
        base / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        base / "boneage-training-dataset" / "boneage-training-dataset",
        base / "data" / "boneage-training-dataset",
        base / "boneage-training-dataset",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.png")):
            return c
    raise FileNotFoundError(f"Could not find RSNA training PNGs under {base}")


def main() -> None:
    rsna_dir = resolve_rsna_dir()
    accept_dir = ROOT_DIR / "tests" / "validator" / "should_accept"
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"

    accept_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    # Clear any pre-existing files in should_accept so we have exactly 30 real RSNA images
    for existing in accept_dir.glob("*"):
        if existing.is_file():
            existing.unlink()

    # Load test_manifest.csv (held-out real RSNA radiographs with age_bin and sex)
    manifest_path = ROOT_DIR / "ml" / "data" / "splits" / "test_manifest.csv"
    df = pd.read_csv(manifest_path)

    # 1) Select 20 stratified images: 2 per (age_bin x male) across all 10 strata (seed=42)
    stratified_rows = []
    for _, group in df.groupby("strata", sort=True):
        sampled = group.sample(n=min(2, len(group)), random_state=42)
        stratified_rows.append(sampled)
    strat_df = pd.concat(stratified_rows, ignore_index=True)
    selected_ids = set(strat_df["id"].astype(int).tolist())

    # 2) Scan candidate real RSNA images to pick 10 genuinely low-contrast radiographs (lowest grayscale std)
    remaining_df = df[~df["id"].isin(selected_ids)].sample(n=120, random_state=42)
    contrast_records = []
    for _, row in remaining_df.iterrows():
        img_id = int(row["id"])
        img_path = rsna_dir / f"{img_id}.png"
        if not img_path.exists():
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        std_val = float(np.std(gray))
        dyn_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        contrast_records.append((dyn_range, std_val, row))

    contrast_records.sort(key=lambda x: x[0])
    low_contrast_rows = [r[2] for r in contrast_records[:10]]
    low_contrast_df = pd.DataFrame(low_contrast_rows)

    final_df = pd.concat([strat_df, low_contrast_df], ignore_index=True)
    assert len(final_df) == 30, f"Expected 30 images, got {len(final_df)}"

    print(f"Copying 30 real RSNA radiographs from {rsna_dir} -> {accept_dir}")
    print("-" * 95)
    print(f"{'#':<4} {'File':<12} {'Age (mo)':<10} {'Sex':<8} {'Age Bin':<24} {'Std':<8} {'P95-P5':<8} {'Subset'}")
    print("-" * 95)

    for idx, (_, row) in enumerate(final_df.iterrows(), 1):
        img_id = int(row["id"])
        src = rsna_dir / f"{img_id}.png"
        dst = accept_dir / f"{img_id}.png"
        shutil.copy2(src, dst)

        gray = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
        std_val = float(np.std(gray))
        p95_p5 = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        sex_str = "male" if bool(row["male"]) else "female"
        subset = "stratified" if idx <= 20 else "low_contrast"
        print(
            f"{idx:<4d} {dst.name:<12} {int(row['boneage']):<10d} {sex_str:<8} "
            f"{str(row['age_bin']):<24} {std_val:<8.2f} {p95_p5:<8.1f} {subset}"
        )

    reject_files = list(reject_dir.iterdir())
    print("-" * 95)
    print(f"tests/validator/should_accept count : {len(list(accept_dir.glob('*.png')))}")
    print(f"tests/validator/should_reject count : {len(reject_files)} (empty as requested)")


if __name__ == "__main__":
    main()
