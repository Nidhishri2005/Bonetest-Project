"""Train the Hand X-Ray Validation classifier with feature caching and fine-tuning.

Trains HandXRayValidator using transfer learning from the radiological backbone
and synthetic/benchmark datasets across 6 visual categories.

Procedure:
1. Preload ResNet-18 backbone with weights transferred from cnn_best.pt.
2. Stage 1 (Frozen Backbone with Feature Caching):
   - Precompute 512-d features once through the frozen backbone.
   - Train classification head on cached features for 15 epochs with Cosine Annealing.
3. Stage 2 (End-to-End Fine-Tuning):
   - Unfreeze layer 4 residual blocks and fine-tune for 2 epochs.
4. Save best checkpoint and training history.

Saves:
- Checkpoint: backend/checkpoints/hand_validator_best.pt
- Training History: backend/checkpoints/validator_training_history.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES
from ml.validator_dataset import HandValidationDataset, build_synthetic_validator_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def precompute_features(
    model: HandXRayValidator,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute 512-d features through frozen backbone to accelerate training."""
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.extract_features(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def train_validator(
    stage1_epochs: int = 15,
    stage2_epochs: int = 2,
    batch_size: int = 32,
    lr_head: float = 2e-3,
    lr_finetune: float = 1e-4,
    samples_per_class: int = 120,
    checkpoints_dir: Path = BACKEND_ROOT / "checkpoints",
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.set_num_threads(8)
    torch.manual_seed(seed)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    logger.info("Building dataset: %d samples per class across %d classes...", samples_per_class, len(VALIDATOR_CLASSES))
    train_data, val_data, test_data = build_synthetic_validator_dataset(
        samples_per_class=samples_per_class, seed=seed
    )
    logger.info("Dataset partitions: Train=%d, Val=%d, Test=%d", len(train_data), len(val_data), len(test_data))

    raw_train_loader = DataLoader(
        HandValidationDataset(train_data),
        batch_size=batch_size,
        shuffle=False,
    )
    raw_val_loader = DataLoader(
        HandValidationDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model
    model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES))

    # Transfer radiological weights from existing cnn_best.pt
    cnn_ckpt_path = checkpoints_dir / "cnn_best.pt"
    if cnn_ckpt_path.exists():
        logger.info("Transferring pretrained radiological features from %s...", cnn_ckpt_path.name)
        ckpt = torch.load(cnn_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Adapt single-channel conv1 to 3-channel input by repeating across channels
        if "backbone.conv1.weight" in state_dict and state_dict["backbone.conv1.weight"].shape[1] == 1:
            w_1ch = state_dict["backbone.conv1.weight"]
            model.backbone.conv1.weight.data.copy_(w_1ch.repeat(1, 3, 1, 1) / 3.0)

        # Transfer intermediate residual blocks
        loaded_layers = 0
        for k, v in state_dict.items():
            if k.startswith("backbone.") and not k.startswith("backbone.fc") and not k.startswith("backbone.conv1"):
                backbone_key = k.replace("backbone.", "")
                if hasattr(model.backbone, backbone_key.split(".")[0]):
                    try:
                        dict_to_load = {k: v}
                        model.load_state_dict(dict_to_load, strict=False)
                        loaded_layers += 1
                    except Exception:
                        pass
        logger.info("Transferred %d pretrained residual layers from radiological backbone", loaded_layers)

    model.to(device_obj)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "stage": [],
    }

    start_time = time.perf_counter()

    # Precompute features through frozen backbone
    logger.info("Precomputing 512-d feature embeddings for train and validation splits...")
    t_feat = time.perf_counter()
    train_feats, train_targets = precompute_features(model, raw_train_loader, device_obj)
    val_feats, val_targets = precompute_features(model, raw_val_loader, device_obj)
    logger.info("Feature precomputation completed in %.2fs (Train: %s, Val: %s)", time.perf_counter() - t_feat, train_feats.shape, val_feats.shape)

    cached_train_loader = DataLoader(
        TensorDataset(train_feats, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )

    # =========================================================================
    # STAGE 1: Train Classification Head on Cached Features (15 epochs)
    # =========================================================================
    logger.info("Stage 1: Training Classification Head on cached features (%d epochs)...", stage1_epochs)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-5)

    current_epoch = 0
    for epoch in range(1, stage1_epochs + 1):
        current_epoch += 1
        model.classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for feats_batch, labels_batch in cached_train_loader:
            feats_batch = feats_batch.to(device_obj)
            labels_batch = labels_batch.to(device_obj)

            optimizer.zero_grad()
            outputs = model.classifier(feats_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += feats_batch.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation on cached features
        model.classifier.eval()
        with torch.no_grad():
            vf = val_feats.to(device_obj)
            vt = val_targets.to(device_obj)
            val_out = model.classifier(vf)
            val_loss = float(criterion(val_out, vt).item())
            val_preds = val_out.argmax(dim=1)
            val_acc = float((val_preds == vt).sum().item() / vt.size(0))

        history["epochs"].append(current_epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["stage"].append("Stage 1 (Head)")

        logger.info(
            "[Stage 1] Epoch %02d/%02d - Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch, stage1_epochs, train_loss, train_acc * 100, val_loss, val_acc * 100
        )
        sys.stdout.flush()

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            _save_checkpoint(model, checkpoints_dir, val_acc, val_loss, current_epoch)

    # =========================================================================
    # STAGE 2: End-to-End Fine-Tuning (Layer 4 + Head) for 2 epochs
    # =========================================================================
    if stage2_epochs > 0:
        logger.info("Stage 2: End-to-End Fine-Tuning (Layer 4 + Head) for %d epochs...", stage2_epochs)
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

        finetune_params = [
            {"params": model.backbone.layer4.parameters(), "lr": lr_finetune * 0.5},
            {"params": model.classifier.parameters(), "lr": lr_finetune},
        ]
        optimizer = torch.optim.AdamW(finetune_params, weight_decay=1e-4)

        raw_train_loader_shuffle = DataLoader(
            HandValidationDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(1, stage2_epochs + 1):
            current_epoch += 1
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in raw_train_loader_shuffle:
                images = images.to(device_obj)
                labels = labels.to(device_obj)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # Validation
            model.eval()
            val_loss, val_acc = _evaluate_raw(model, raw_val_loader, criterion, device_obj)

            history["epochs"].append(current_epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 4))
            history["stage"].append("Stage 2 (Fine-tune)")

            logger.info(
                "[Stage 2] Epoch %02d/%02d - Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
                epoch, stage2_epochs, train_loss, train_acc * 100, val_loss, val_acc * 100
            )
            sys.stdout.flush()

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                _save_checkpoint(model, checkpoints_dir, val_acc, val_loss, current_epoch)

    elapsed = time.perf_counter() - start_time
    logger.info("Training complete in %.2f seconds. Best Val Accuracy: %.2f%%", elapsed, best_val_acc * 100)

    # Save training history
    history_path = checkpoints_dir / "validator_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "history": history,
        "training_time_s": elapsed,
    }


def _evaluate_raw(model, loader, criterion, device):
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

    return val_running_loss / val_total, val_correct / val_total


def _save_checkpoint(model, checkpoints_dir, val_acc, val_loss, epoch):
    save_path = checkpoints_dir / "hand_validator_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": VALIDATOR_CLASSES,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch,
            "model_architecture": "ResNet18-Validator",
        },
        save_path,
    )
    logger.info(">>> Checkpoint saved to %s (Val Acc: %.2f%%, Val Loss: %.4f)", save_path.name, val_acc * 100, val_loss)


if __name__ == "__main__":
    train_validator()
