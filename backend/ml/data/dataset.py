"""PyTorch Dataset and DataLoader for RSNA Pediatric Bone Age radiographs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data.preprocess import preprocess_radiograph

logger = logging.getLogger(__name__)


class RSNABoneAgeDataset(Dataset):
    """Dataset for pediatric bone age prediction with ROI crop, CLAHE, and sex fusion."""

    def __init__(
        self,
        manifest: Union[pd.DataFrame, Path, str],
        image_dir: Union[Path, str],
        cache_dir: Optional[Union[Path, str]] = None,
        target_size: int = 384,
        roi_crop: bool = True,
        clahe: bool = True,
        norm_mean: float = 0.4523,
        norm_std: float = 0.2118,
        mean_age: float = 127.32,
        std_age: float = 41.18,
        transform: Optional[Callable] = None,
        preload_to_ram: bool = False,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy().reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.target_size = target_size
        self.roi_crop = roi_crop
        self.clahe = clahe
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mean_age = mean_age
        self.std_age = std_age
        self.transform = transform
        self.preload_to_ram = preload_to_ram

        self._ram_cache: dict[str, np.ndarray] = {}
        if preload_to_ram:
            logger.info(f"Preloading {len(self.df)} processed images to RAM...")
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                img_id = str(row["id"])
                self._ram_cache[img_id] = self._load_and_preprocess(img_id, row.get("file_name"))

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_preprocess(self, image_id: str, file_name: Optional[str] = None) -> np.ndarray:
        # 1. Check disk cache first
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{image_id}.png"
            if cache_file.exists():
                cached = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE)
                if cached is not None and cached.shape == (self.target_size, self.target_size):
                    return cached

        # 2. Load raw image
        raw_image = None
        candidates = []
        if file_name:
            candidates.append(self.image_dir / file_name)
        candidates.extend([
            self.image_dir / f"{image_id}.png",
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.jpeg",
        ])

        for path in candidates:
            if path.exists():
                raw_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if raw_image is not None and raw_image.size > 0:
                    break

        if raw_image is None:
            raise FileNotFoundError(f"Image {image_id} not found in {self.image_dir}")

        # 3. Apply ROI crop + CLAHE + letterbox resize
        processed = preprocess_radiograph(
            raw_image,
            target_size=self.target_size,
            roi_crop=self.roi_crop,
            clahe=self.clahe,
        )

        # 4. Save to disk cache if enabled
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.cache_dir / f"{image_id}.png"), processed)

        return processed

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        image_id = str(row["id"])

        if self.preload_to_ram and image_id in self._ram_cache:
            img = self._ram_cache[image_id]
        else:
            img = self._load_and_preprocess(image_id, row.get("file_name"))

        # Optional training augmentation
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"] if isinstance(augmented, dict) else augmented

        # Normalize [0, 1] then subtract mean, divide std
        norm_img = img.astype(np.float32) / 255.0
        norm_img = (norm_img - self.norm_mean) / max(self.norm_std, 1e-6)

        # Replicate grayscale to 3 channels: (3, H, W)
        img_tensor = torch.from_numpy(norm_img).unsqueeze(0).repeat(3, 1, 1).float()

        # Labels
        raw_age = float(row["boneage"])
        standardized_age = (raw_age - self.mean_age) / max(self.std_age, 1e-6)

        # Sex: male=1, female=0
        male_val = row.get("male", 0)
        if isinstance(male_val, (bool, np.bool_)):
            sex_int = 1 if male_val else 0
        elif str(male_val).lower() in ("true", "1", "m", "male"):
            sex_int = 1
        else:
            sex_int = 0

        return {
            "image": img_tensor,
            "target": torch.tensor(standardized_age, dtype=torch.float32),
            "standardized_age": torch.tensor(standardized_age, dtype=torch.float32),
            "bone_age": torch.tensor(raw_age, dtype=torch.float32),
            "boneage_months": torch.tensor(raw_age, dtype=torch.float32),
            "sex": torch.tensor(sex_int, dtype=torch.long),
            "id": image_id,
        }


def create_dataloaders(
    train_manifest: Union[Path, str],
    val_manifest: Union[Path, str],
    test_manifest: Optional[Union[Path, str]],
    image_dir: Union[Path, str],
    cache_dir: Optional[Union[Path, str]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: int = 384,
    norm_mean: float = 0.4523,
    norm_std: float = 0.2118,
    mean_age: float = 127.32,
    std_age: float = 41.18,
    train_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build PyTorch DataLoaders for train, validation, and optional test sets."""
    train_ds = RSNABoneAgeDataset(
        manifest=train_manifest,
        image_dir=image_dir,
        cache_dir=cache_dir,
        target_size=target_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        mean_age=mean_age,
        std_age=std_age,
        transform=train_transform,
    )
    val_ds = RSNABoneAgeDataset(
        manifest=val_manifest,
        image_dir=image_dir,
        cache_dir=cache_dir,
        target_size=target_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        mean_age=mean_age,
        std_age=std_age,
        transform=None,
    )
    test_ds = None
    if test_manifest:
        test_ds = RSNABoneAgeDataset(
            manifest=test_manifest,
            image_dir=image_dir,
            cache_dir=cache_dir,
            target_size=target_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
            mean_age=mean_age,
            std_age=std_age,
            transform=None,
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
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_ds is not None
        else None
    )

    return train_loader, val_loader, test_loader
