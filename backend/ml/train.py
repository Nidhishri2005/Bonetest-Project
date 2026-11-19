"""Training script for all bone age models with configurable loss and scheduling.

Usage:
    cd backend
    python -m ml.train --model-type cnn --data-dir /path/to/rsna --epochs 30 --loss smooth_l1 --scheduler plateau
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY
from app.models.cnn_rf import CNNFeatureExtractor
from app.models.new_models import MultiscaleCNNFeatureExtractor
from ml.augmentation import TrainAugmentation
from ml.dataset import create_dataloaders
from ml.preprocessing import compute_dataset_statistics, save_normalization_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bone age regression models")
    parser.add_argument(
        "--model-type",
        choices=[
            "cnn",
            "cnn_dnn",
            "multimodal_cnn",
            "regression_multimodal",
            "ensemble_cnn",
            "cnn_tw3",
            "cnn_rf",
            "multiscale_cnn_rf",
        ],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "mae", "l1", "mse", "huber"],
        default="smooth_l1",
        help="Regression loss function (smooth_l1 provides outlier resistance)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine", "step", "none"],
        default="plateau",
        help="Learning rate schedule policy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verify-images", action="store_true", help="Pre-scan and drop corrupt images")
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints"
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=None,
        help="Path to a pretrained CNN checkpoint",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _model_requires_gender(model_type: str) -> bool:
    meta = MODEL_REGISTRY.get(model_type, {})
    return meta.get("requires_gender", False)


def train_epoch(model, loader, optimizer, criterion, device, model_type, scaler):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"Training batch {batch_idx+1}/{num_batches}")

        images = batch["image"].to(device)
        targets = batch["bone_age"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            if _model_requires_gender(model_type):
                gender = batch["male"].unsqueeze(-1).to(device)
                outputs = model(images, gender)
            else:
                outputs = model(images)

            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    for batch in loader:
        images = batch["image"].to(device)
        y = batch["bone_age"].to(device)

        if _model_requires_gender(model_type):
            gender = batch["male"].unsqueeze(-1).to(device)
            outputs = model(images, gender)
        else:
            outputs = model(images)

        loss = criterion(outputs, y)
        total_loss += loss.item() * images.size(0)
        preds.extend(outputs.cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    rmse = float(np.sqrt(mse))
    return total_loss / len(loader.dataset), mae, mse, rmse


def train_random_forest(
    data_dir: Path,
    checkpoints_dir: Path,
    device: torch.device,
    image_size: int,
    norm_mean: float,
    norm_std: float,
    model_type: str,
    verify_images: bool = False,
) -> None:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from ml.dataset import (
        BoneAgeDataset,
        train_val_test_split,
        find_image_directory,
        verify_dataset_images,
    )

    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        csv_path = data_dir / "boneage-training-dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found in {data_dir}.")

    df = pd.read_csv(csv_path)
    image_dir = find_image_directory(data_dir)
    print(f"\nUsing image directory: {image_dir}\n")

    if verify_images:
        df = verify_dataset_images(df, image_dir)

    train_df, val_df, _ = train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    if model_type == "cnn_rf":
        cnn_checkpoint = checkpoints_dir / "cnn_best.pt"
        if not cnn_checkpoint.exists():
            raise FileNotFoundError(f"Cannot find pretrained CNN: {cnn_checkpoint}")

        print(f"Loading pretrained CNN backbone from {cnn_checkpoint}")
        cnn = CNNFeatureExtractor(pretrained=False)
        checkpoint = torch.load(cnn_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cnn.load_state_dict(state_dict, strict=False)
    else:
        print("Initializing deterministic multiscale feature extractor")
        torch.manual_seed(42)
        cnn = MultiscaleCNNFeatureExtractor(pretrained=False)

    cnn.to(device)
    cnn.eval()

    def extract_features(subset_df):
        dataset = BoneAgeDataset(
            subset_df,
            image_dir,
            image_size,
            None,
            norm_mean,
            norm_std,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        features = []
        ages = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i % 50 == 0:
                    print(f"Extracting features: batch {i+1}/{len(loader)}")
                images = batch["image"].to(device)
                feats = cnn(images).cpu().numpy()
                features.append(feats)
                ages.extend(batch["bone_age"].numpy().tolist())

        return np.vstack(features), np.array(ages)

    print("\nExtracting training features...")
    X_train, y_train = extract_features(train_df)

    print("\nExtracting validation features...")
    X_val, y_val = extract_features(val_df)

    print("\nTraining Random Forest Regressor (200 trees)...")
    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    duration = time.time() - start
    print(f"Random Forest training completed in {duration/60:.2f} minutes.")

    preds = rf.predict(X_val)
    mae = float(np.mean(np.abs(preds - y_val)))
    mse = float(np.mean((preds - y_val) ** 2))
    rmse = float(np.sqrt(mse))

    print(f"\nRandom Forest Validation:")
    print(f"  MAE : {mae:.2f} months")
    print(f"  RMSE: {rmse:.2f} months")

    rf_dir = BACKEND_ROOT / "rf_models"
    rf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, rf_dir / f"{model_type}.joblib")

    torch.save(
        {
            "model_state_dict": cnn.state_dict(),
            "model_type": model_type,
            "val_mae": mae,
            "val_mse": mse,
            "val_rmse": rmse,
        },
        checkpoints_dir / f"{model_type}_best.pt",
    )
    print("Random Forest model and feature extractor checkpoint saved successfully.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    norm_mean = 0.4523
    norm_std = 0.2118
    stats = {"mean": norm_mean, "std": norm_std}
    save_normalization_stats(stats, args.checkpoints_dir / "normalization.json")

    if args.model_type in {"cnn_rf", "multiscale_cnn_rf"}:
        train_random_forest(
            args.data_dir,
            args.checkpoints_dir,
            device,
            args.image_size,
            norm_mean,
            norm_std,
            args.model_type,
            verify_images=args.verify_images,
        )
        return

    augment = TrainAugmentation()
    train_loader, val_loader, _, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_transform=augment,
        norm_mean=norm_mean,
        norm_std=norm_std,
        verify_images=args.verify_images,
    )

    meta = MODEL_REGISTRY[args.model_type]
    model = meta["class"](pretrained=args.pretrained).to(device)

    # Optional transfer learning from pretrained CNN checkpoint
    if args.model_type in [
        "multimodal_cnn",
        "regression_multimodal",
        "cnn_dnn",
        "cnn_tw3",
    ] and args.cnn_checkpoint is not None:
        print(f"\nLoading pretrained CNN from {args.cnn_checkpoint}")
        checkpoint = torch.load(args.cnn_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "image_backbone.", 1)
            elif key.startswith("image_backbone."):
                new_key = key
            else:
                continue
            backbone_state[new_key] = value

        model.load_state_dict(backbone_state, strict=False)
        print("Pretrained CNN backbone loaded successfully.")

        if args.model_type in {"multimodal_cnn", "cnn_tw3"}:
            for p in model.image_backbone.parameters():
                p.requires_grad = False
        elif args.model_type == "cnn_dnn":
            for p in model.backbone.parameters():
                p.requires_grad = False
        print("Backbone frozen; training regression head.")

    # Select loss function
    if args.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss(beta=1.0)
    elif args.loss in {"mae", "l1"}:
        criterion = nn.L1Loss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Select LR scheduler
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:
        scheduler = None

    scaler = GradScaler(enabled=device.type == "cuda")

    best_mae = float("inf")
    patience_counter = 0
    history = []

    print(
        f"Starting training for {args.model_type} ({args.epochs} epochs, "
        f"loss={args.loss}, scheduler={args.scheduler}, lr={args.lr})..."
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.model_type,
            scaler,
        )

        val_loss, mae, mse, rmse = validate(
            model,
            val_loader,
            criterion,
            device,
            args.model_type,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mae)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {mae:.2f} mo | "
            f"Val RMSE: {rmse:.2f} mo | "
            f"LR: {current_lr:.2e}"
        )

        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            ckpt_name = f"{args.model_type}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                    "loss_func": args.loss,
                    "epoch": epoch,
                },
                args.checkpoints_dir / ckpt_name,
            )
            print(f"  -> Best model saved with MAE: {mae:.2f} months")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs with no MAE improvement.")
                break

    history_path = args.checkpoints_dir / f"{args.model_type}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
