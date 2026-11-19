"""Deterministic stratified 80/10/10 dataset splitting by age cohort and biological sex."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Standard pediatric age cohorts (in months)
AGE_BINS = [0.0, 24.0, 72.0, 120.0, 168.0, 250.0]
BIN_LABELS = [
    "infant_0_24",
    "early_child_24_72",
    "mid_child_72_120",
    "adolescent_120_168",
    "post_pubertal_168_plus",
]


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_stratified_splits(
    df: pd.DataFrame,
    image_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    id_col: str = "id",
    age_col: str = "boneage",
    sex_col: str = "male",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically partition dataset into 80% Train, 10% Val, and 10% Test.
    
    Stratification is performed on the joint product of age bin and biological sex
    (5 age cohorts x 2 sexes = 10 strata) to guarantee balanced demographic distributions.
    
    Note on Patient IDs:
        The official RSNA dataset provides only image IDs (no patient-level IDs exist).
        This limitation is documented; images are partitioned with fixed stratification.
    """
    df = df.copy().dropna(subset=[age_col]).reset_index(drop=True)
    image_dir = Path(image_dir)

    # 1. Assign age bin
    df["age_bin"] = pd.cut(
        df[age_col],
        bins=AGE_BINS,
        labels=BIN_LABELS,
        include_lowest=True,
    ).astype(str)

    # 2. Joint stratification key: age_bin x sex
    df["strata"] = df["age_bin"] + "_" + df[sex_col].astype(str)

    # Check minimum samples per strata
    strata_counts = df["strata"].value_counts()
    logger.info(f"Strata distribution across {len(strata_counts)} joint groups:\n{strata_counts}")

    # First split: Train (80%) vs Holdout (20%)
    holdout_ratio = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=df["strata"],
    )

    # Second split: Val (50% of holdout = 10% total) vs Test (50% of holdout = 10% total)
    relative_test_ratio = test_ratio / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=holdout_df["strata"],
    )

    # Add file path and SHA-256 for provenance
    def enrich_split(split_df: pd.DataFrame) -> pd.DataFrame:
        split_df = split_df.copy().reset_index(drop=True)
        file_names = []
        hashes = []
        for _, row in split_df.iterrows():
            img_id = str(row[id_col])
            found_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                cand = image_dir / f"{img_id}{ext}"
                if cand.exists():
                    found_path = cand
                    file_names.append(f"{img_id}{ext}")
                    hashes.append(compute_file_sha256(cand))
                    break
            if found_path is None:
                file_names.append(f"{img_id}.png")
                hashes.append("MISSING")

        split_df["file_name"] = file_names
        split_df["sha256"] = hashes
        return split_df

    logger.info("Computing SHA-256 file hashes for split verification...")
    train_df = enrich_split(train_df)
    val_df = enrich_split(val_df)
    test_df = enrich_split(test_df)

    logger.info(
        f"Split complete (seed={seed}): "
        f"Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )
    return train_df, val_df, test_df


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    """Save train, val, and test manifest CSVs and record their SHA-256 hashes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_manifest.csv"
    val_path = output_dir / "val_manifest.csv"
    test_path = output_dir / "test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest_hashes = {
        "train_manifest": {
            "path": str(train_path),
            "sha256": compute_file_sha256(train_path),
            "rows": len(train_df),
        },
        "val_manifest": {
            "path": str(val_path),
            "sha256": compute_file_sha256(val_path),
            "rows": len(val_df),
        },
        "test_manifest": {
            "path": str(test_path),
            "sha256": compute_file_sha256(test_path),
            "rows": len(test_df),
        },
    }

    hash_path = output_dir / "manifest_hashes.json"
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(manifest_hashes, f, indent=2)

    logger.info(f"Saved manifests and hashes to {output_dir}")
    return manifest_hashes
