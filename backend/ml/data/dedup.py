"""Perceptual-hash deduplication for pediatric bone-age radiographs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for an image.
    
    Args:
        image: Grayscale or BGR image.
        hash_size: Grid dimension (default 8 yields 8x8 = 64-bit hash).
        
    Returns:
        64-bit integer perceptual hash.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to (hash_size + 1, hash_size) -> (9, 8)
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    # Compare adjacent horizontal pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert 64 booleans to a 64-bit integer
    bit_string = "".join(["1" if b else "0" for b in diff.flatten()])
    return int(bit_string, 2)


def compute_dhash_from_file(image_path: Path, hash_size: int = 8) -> int | None:
    """Read image from disk and compute dHash."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return compute_dhash(image, hash_size=hash_size)


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def deduplicate_dataset(
    df: pd.DataFrame,
    image_dir: Path,
    id_col: str = "id",
    max_hamming_distance: int = 0,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Scan dataset images, compute perceptual hashes, and identify duplicates.
    
    Args:
        df: Input DataFrame with image IDs and labels.
        image_dir: Path to directory containing image files.
        id_col: Column name containing image IDs.
        max_hamming_distance: Max Hamming distance to classify as duplicate (0 = exact match).
        
    Returns:
        clean_df: DataFrame with duplicate entries removed (first occurrence retained).
        duplicate_records: List of duplicate pairs/clusters found with distances.
    """
    image_dir = Path(image_dir)
    hashes: Dict[str, int] = {}
    
    logger.info(f"Computing perceptual hashes for {len(df)} images...")
    for idx, row in df.iterrows():
        img_id = str(row[id_col])
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = image_dir / f"{img_id}{ext}"
            if cand.exists():
                h = compute_dhash_from_file(cand)
                if h is not None:
                    hashes[img_id] = h
                break

    # Cluster by exact hash first
    hash_to_ids: Dict[int, List[str]] = {}
    for img_id, h in hashes.items():
        hash_to_ids.setdefault(h, []).append(img_id)

    duplicate_pairs: List[Dict] = []
    dropped_ids: Set[str] = set()

    for h, ids in hash_to_ids.items():
        if len(ids) > 1:
            primary = ids[0]
            for dup in ids[1:]:
                dropped_ids.add(dup)
                duplicate_pairs.append({
                    "primary_id": primary,
                    "duplicate_id": dup,
                    "hamming_distance": 0,
                    "hash_hex": hex(h),
                })

    clean_df = df[~df[id_col].astype(str).isin(dropped_ids)].copy().reset_index(drop=True)
    logger.info(f"Deduplication complete: {len(duplicate_pairs)} duplicate records identified. Remaining clean samples: {len(clean_df)}")
    return clean_df, duplicate_pairs
