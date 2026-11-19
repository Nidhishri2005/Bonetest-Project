"""RSNA Pediatric Bone Age dataset loader with 3-way split and verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing import letterbox_resize, normalize_image

logger = logging.getLogger(__name__)


class BoneAgeDataset(Dataset):
    """PyTorch Dataset for RSNA Pediatric Bone Age radiographs."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        norm_mean: float = 0.4523,
        norm_std: float = 0.2118,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = str(row["id"])

        possible_paths = [
            self.image_dir / f"{image_id}.png",
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.jpeg",
        ]

        image = None
        for path in possible_paths:
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    break

        if image is None:
            raise FileNotFoundError(
                f"Image {image_id} not found or corrupted in {self.image_dir}"
            )

        image = letterbox_resize(image, self.image_size)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"] if isinstance(augmented, dict) else augmented

        image = normalize_image(
            image,
            self.norm_mean,
            self.norm_std,
        )

        image = torch.from_numpy(image).unsqueeze(0).float()
        age = torch.tensor(
            float(row["boneage"]),
            dtype=torch.float32,
        )
        male = torch.tensor(
            float(row.get("male", 0)),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "bone_age": age,
            "male": male,
            "id": image_id,
        }


def verify_dataset_images(
    df: pd.DataFrame,
    image_dir: Union[str, Path],
    check_readability: bool = True,
) -> pd.DataFrame:
    """Verify that all images listed in dataframe exist and are readable.

    Args:
        df: DataFrame containing at least an 'id' column.
        image_dir: Path to directory containing X-ray images.
        check_readability: If True, tests cv2.imread to ensure no corrupted files.

    Returns:
        Filtered DataFrame with missing/corrupted image rows removed.
    """
    image_dir = Path(image_dir)
    valid_indices = []
    missing_count = 0
    corrupt_count = 0

    for idx, row in df.iterrows():
        img_id = str(row["id"])
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = image_dir / f"{img_id}{ext}"
            if candidate.exists():
                if check_readability:
                    img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        found = True
                        valid_indices.append(idx)
                        break
                    else:
                        corrupt_count += 1
                else:
                    found = True
                    valid_indices.append(idx)
                    break
        if not found and not check_readability:
            missing_count += 1

    clean_df = df.loc[valid_indices].reset_index(drop=True)
    total_dropped = len(df) - len(clean_df)
    if total_dropped > 0:
        logger.warning(
            f"Image verification dropped {total_dropped} entries "
            f"({missing_count} missing, {corrupt_count} corrupt). "
            f"Remaining valid images: {len(clean_df)}"
        )
    else:
        logger.info(f"Image verification passed: all {len(clean_df)} images verified.")

    return clean_df


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split dataframe into Train (70%), Val (15%), and Test (15%).

    Supports:
      1. Patient-level grouping to prevent data leakage across splits.
      2. Age-stratified binning for balanced chronological/bone age distributions.
    """
    df = df.copy().dropna(subset=["boneage"]).reset_index(drop=True)

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    # Detect patient ID column if not explicitly given
    if patient_col is None:
        candidates = ["patient_id", "Patient ID", "patientId", "subject_id", "PatientId"]
        for c in candidates:
            if c in df.columns:
                patient_col = c
                break

    if patient_col is not None and patient_col in df.columns:
        # Patient-grouped splitting to strictly prevent patient data leakage
        unique_patients = df[patient_col].unique()
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(unique_patients)

        n_total = len(shuffled)
        n_train = int(np.round(n_total * train_ratio))
        n_val = int(np.round(n_total * val_ratio))

        train_patients = set(shuffled[:n_train])
        val_patients = set(shuffled[n_train : n_train + n_val])
        test_patients = set(shuffled[n_train + n_val :])

        train_df = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
        val_df = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
        test_df = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    # Fallback to stratified age binning
    age_bins = [0, 36, 84, 132, 180, 250]
    bin_labels = ["infant_toddler", "early_child", "mid_child", "adolescent", "post_pubertal"]
    df["age_bin"] = pd.cut(
        df["boneage"],
        bins=age_bins,
        labels=bin_labels,
        include_lowest=True,
    )

    # Handle any edge ages outside defined bins
    if df["age_bin"].isna().any():
        df["age_bin"] = df["age_bin"].astype(str).fillna("unclassified")

    # First split: (train + val) vs test
    val_plus_test_ratio = val_ratio + test_ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["age_bin"],
    )

    # Second split: train vs val from train_val_df
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val_df["age_bin"],
    )

    train_df = train_df.drop(columns=["age_bin"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["age_bin"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["age_bin"]).reset_index(drop=True)

    return train_df, val_df, test_df


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible stratified 2-way split."""
    train_ratio = 1.0 - val_ratio
    train_df, val_df, _ = train_val_test_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    return train_df, val_df


def find_image_directory(data_dir: Union[str, Path]) -> Path:
    """Automatically locate the RSNA training image folder."""
    data_dir = Path(data_dir)

    possible_dirs = [
        data_dir / "train",
        data_dir / "boneage-training-dataset" / "boneage-training-dataset",
        data_dir / "boneage-training-dataset",
        data_dir,
    ]

    for d in possible_dirs:
        if d.exists() and d.is_dir():
            has_png = any(d.glob("*.png"))
            has_jpg = any(d.glob("*.jpg"))
            if has_png or has_jpg:
                return d

    raise FileNotFoundError(f"Could not locate image directory in {data_dir}.")


def create_dataloaders(
    data_dir: Union[str, Path],
    csv_name: str = "train.csv",
    batch_size: int = 16,
    image_size: int = 512,
    num_workers: int = 2,
    train_transform: Optional[Callable] = None,
    norm_mean: float = 0.4523,
    norm_std: float = 0.2118,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    return_test: bool = False,
    verify_images: bool = False,
):
    """Create train, validation, and optional test DataLoaders."""
    data_dir = Path(data_dir)

    possible_csv = [
        data_dir / csv_name,
        data_dir / "train.csv",
        data_dir / "boneage-training-dataset.csv",
    ]

    csv_path = None
    for p in possible_csv:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(f"No training CSV found in {data_dir}.")

    df = pd.read_csv(csv_path)
    image_dir = find_image_directory(data_dir)

    if verify_images:
        df = verify_dataset_images(df, image_dir)

    train_df, val_df, test_df = train_val_test_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
    )

    train_ds = BoneAgeDataset(
        train_df,
        image_dir,
        image_size,
        train_transform,
        norm_mean,
        norm_std,
    )
    val_ds = BoneAgeDataset(
        val_df,
        image_dir,
        image_size,
        None,
        norm_mean,
        norm_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if return_test:
        test_ds = BoneAgeDataset(
            test_df,
            image_dir,
            image_size,
            None,
            norm_mean,
            norm_std,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return (
            train_loader,
            val_loader,
            test_loader,
            train_df,
            val_df,
            test_df,
        )

    return (
        train_loader,
        val_loader,
        train_df,
        val_df,
    )
