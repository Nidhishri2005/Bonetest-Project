"""Production training script for EfficientNet-B3 Sex-Fusion bone age regression models.

Key specifications:
- Architecture: EfficientNetB3SexFusion (ImageNet pretrained, 3-channel 384x384 input, 32-d sex embedding)
- Loss: Smooth L1 / Huber Loss (delta=1.0) on standardized age targets
- Optimizer: AdamW with weight decay
- Learning Rate Schedule: Linear warmup + Cosine Annealing
- Checkpoint Metadata: git commit hash, split manifest hash, config dictionary, best validation MAE
- Hardware handling: AMP enabled on CUDA, graceful CPU fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.data.dataset import create_dataloaders
from ml.models.efficientnet import EfficientNetB3SexFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainEfficientNet")


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_revision_hash() -> str:
    """Retrieve current git commit hash if in a git repository."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine learning rate decay with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    use_amp = scaler is not None and device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sex = batch["sex"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                preds = model(images, sex)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, sex)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    mean_age: float,
    std_age: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation/test set.
    
    Returns:
        val_loss: Standardized Huber loss
        val_mae_months: Absolute error in unstandardized real months
    """
    model.eval()
    total_loss = 0.0
    abs_errors = []
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sex = batch["sex"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        true_months = batch["boneage_months"].numpy()

        preds_std = model(images, sex)
        loss = criterion(preds_std, targets)

        # Unstandardize prediction to real months
        preds_months = preds_std.cpu().numpy() * std_age + mean_age
        errors = np.abs(preds_months - true_months)

        abs_errors.extend(errors.tolist())
        total_loss += loss.item()
        num_batches += 1

    val_loss = total_loss / max(1, num_batches)
    val_mae = float(np.mean(abs_errors))
    return val_loss, val_mae


def train(
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    output_dir: Path | None = None,
) -> Path:
    set_seed(seed)
    output_dir = output_dir or (ROOT_DIR / "checkpoints" / "ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Configurations & Normalization
    with open(ROOT_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(ROOT_DIR / "configs" / "normalization.json", "r", encoding="utf-8") as f:
        norm_cfg = json.load(f)
    with open(ROOT_DIR / "ml" / "data" / "splits" / "manifest_hashes.json", "r", encoding="utf-8") as f:
        manifest_meta = json.load(f)

    # Resolve image directory
    raw_dir = Path(data_cfg["dataset_dir"])
    possible_img_dirs = [
        raw_dir / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dir / "boneage-training-dataset" / "boneage-training-dataset",
        raw_dir / "data" / "boneage-training-dataset",
        raw_dir / "boneage-training-dataset",
    ]
    img_dir = next((d for d in possible_img_dirs if d.exists() and any(d.glob("*.png"))), None)

    # 2. Setup DataLoaders
    train_manifest = ROOT_DIR / "ml" / "data" / "splits" / "train_manifest.csv"
    val_manifest = ROOT_DIR / "ml" / "data" / "splits" / "val_manifest.csv"

    logger.info("Initializing DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        image_dir=img_dir,
        batch_size=batch_size,
        num_workers=4 if os.name != "nt" else 0,
        target_size=norm_cfg["target_size"],
        norm_mean=norm_cfg["norm_mean"],
        norm_std=norm_cfg["norm_std"],
        mean_age=norm_cfg["mean_age"],
        std_age=norm_cfg["std_age"],
    )

    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device} (Seed: {seed})")

    model = EfficientNetB3SexFusion(pretrained=True, sex_embed_dim=32, dropout=0.2)
    model.to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss with delta=1.0
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # 4. Training Loop with Early Stopping
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_save_path = output_dir / f"efficientnet_b3_seed_{seed}.pt"

    git_hash = get_git_revision_hash()
    train_manifest_hash = manifest_meta.get("train_manifest", {}).get("sha256", "unknown")

    logger.info(f"Starting training for {epochs} epochs (Early stopping patience: {patience})...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        val_loss, val_mae = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            mean_age=norm_cfg["mean_age"],
            std_age=norm_cfg["std_age"],
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.2f} mo | "
            f"Duration: {epoch_time:.1f}s"
        )

        # Check for improvement
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint with full audit metadata
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": best_val_mae,
                "val_loss": val_loss,
                "git_hash": git_hash,
                "train_manifest_sha256": train_manifest_hash,
                "seed": seed,
                "norm_config": norm_cfg,
                "model_arch": "EfficientNetB3SexFusion",
            }, checkpoint_save_path)
            logger.info(f"[*] New best validation MAE ({best_val_mae:.2f} mo) saved to {checkpoint_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. Best epoch: {best_epoch} (Val MAE: {best_val_mae:.2f} mo)")
                break

    logger.info(f"Training completed for seed {seed}. Best checkpoint: {checkpoint_save_path} (Val MAE: {best_val_mae:.2f} mo)")
    return checkpoint_save_path


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB3SexFusion model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (e.g. 42, 1337, 2026)")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ensemble", help="Output directory")
    args = parser.parse_args()

    train(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        output_dir=ROOT_DIR / args.output_dir,
    )


if __name__ == "__main__":
    main()
