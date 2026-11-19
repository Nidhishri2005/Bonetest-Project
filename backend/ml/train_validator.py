"""Train the 2-Class Hand X-Ray Validator (valid_pediatric_hand_xray vs other) on REAL data only.

Requirements enforced:
- 2-class model: ["valid_pediatric_hand_xray", "other"]
- ImageNet-pretrained ResNet-18 backbone (models.ResNet18_Weights.DEFAULT)
- Single shared preprocessing function: app.models.validator.preprocess_validator_image
- Trained strictly on REAL images:
    * Positives (label 0): Real RSNA radiographs from dataset_dir in configs/data.yaml
    * Negatives (label 1): Real negative images recursively discovered in validator_negatives_dir
- Saves class list in checkpoint metadata ("classes": VALIDATOR_CLASSES)
- Designed to be executed on Colab GPU (see notebooks/train_validator_colab.ipynb)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
for p in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES
from ml.validator_dataset import RealValidatorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect_image_files(folder: Path) -> List[Path]:
    """Recursively collect all real image files under folder."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def resolve_rsna_positive_dir(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        dataset_dir / "boneage-training-dataset" / "boneage-training-dataset",
        dataset_dir / "data" / "boneage-training-dataset",
        dataset_dir / "boneage-training-dataset",
        dataset_dir,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.png")):
            return c
    raise FileNotFoundError(f"No RSNA training PNG files found under {dataset_dir}")


def build_real_splits(
    rsna_dir: Path,
    negatives_dir: Path,
    max_positives: int = 2000,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Build stratified 70% train / 15% val / 15% test splits from real images on disk."""
    rng = random.Random(seed)

    pos_files = collect_image_files(rsna_dir)
    neg_files = collect_image_files(negatives_dir)

    if not pos_files:
        raise RuntimeError(f"Zero positive RSNA images found in {rsna_dir}")
    if not neg_files:
        raise RuntimeError(
            f"Zero real negative images found in {negatives_dir}. "
            "Synthetic data is prohibited for training; please populate validator_negatives_dir with real negative images."
        )

    rng.shuffle(pos_files)
    if max_positives and len(pos_files) > max_positives:
        pos_files = pos_files[:max_positives]

    rng.shuffle(neg_files)

    pos_labeled = [(p, 0) for p in pos_files]
    neg_labeled = [(p, 1) for p in neg_files]

    def split_list(items: List[Tuple[Path, int]]):
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]

    p_tr, p_val, p_te = split_list(pos_labeled)
    n_tr, n_val, n_te = split_list(neg_labeled)

    train_split = p_tr + n_tr
    val_split = p_val + n_val
    test_split = p_te + n_te

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_split)

    logger.info(
        "Real dataset splits: Positives=%d, Negatives=%d | Train=%d, Val=%d, Test=%d",
        len(pos_labeled),
        len(neg_labeled),
        len(train_split),
        len(val_split),
        len(test_split),
    )
    return train_split, val_split, test_split


def train_validator_real(
    rsna_dir: Path,
    negatives_dir: Path,
    checkpoints_dir: Path = BACKEND_ROOT / "checkpoints",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_positives: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    train_samples, val_samples, test_samples = build_real_splits(
        rsna_dir=rsna_dir,
        negatives_dir=negatives_dir,
        max_positives=max_positives,
        seed=seed,
    )

    train_loader = DataLoader(RealValidatorDataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(RealValidatorDataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=2)

    # ImageNet-pretrained ResNet-18 2-class model
    model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES), pretrained=True).to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"epochs": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * images.size(0)
            tr_correct += (logits.argmax(dim=1) == labels).sum().item()
            tr_total += images.size(0)

        scheduler.step()
        train_loss = tr_loss / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device_obj), labels.to(device_obj)
                logits = model(images)
                loss = criterion(logits, labels)
                vl_loss += loss.item() * images.size(0)
                vl_correct += (logits.argmax(dim=1) == labels).sum().item()
                vl_total += images.size(0)

        val_loss = vl_loss / max(1, vl_total)
        val_acc = vl_correct / max(1, vl_total)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            "Epoch %02d/%02d | Train Loss=%.4f Acc=%.2f%% | Val Loss=%.4f Acc=%.2f%%",
            epoch, epochs, train_loss, train_acc * 100, val_loss, val_acc * 100,
        )

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_path = checkpoints_dir / "hand_validator_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": list(VALIDATOR_CLASSES),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "model_architecture": "ResNet18-2Class-Validator",
                },
                save_path,
            )
            logger.info(">>> Saved best 2-class checkpoint to %s", save_path)

    return {"best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2-class HandXRayValidator on REAL data only.")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-positives", type=int, default=2000)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rsna_dir = resolve_rsna_positive_dir(Path(cfg["dataset_dir"]))
    negatives_dir = Path(cfg["validator_negatives_dir"])
    train_validator_real(
        rsna_dir=rsna_dir,
        negatives_dir=negatives_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_positives=args.max_positives,
    )


if __name__ == "__main__":
    main()
