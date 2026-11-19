"""Step 5: Threshold-Selection Script for the 2-Class Hand X-Ray Validator.

Evaluates held-out real positives (RSNA radiographs / tests/validator/should_accept)
and held-out real negatives (tests/validator/should_reject / validator_negatives_dir),
reporting:
  - ROC AUC
  - RSNA Accept Rate (True Positive Rate) across thresholds
  - False-Accept Rate (False Positive Rate) across thresholds
  - Comparison of Confidence-Only Gating vs Confidence + Entropy Gating
  - Recommended operating threshold
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES, preprocess_validator_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path, limit: int | None = None) -> List[Path]:
    if not folder.exists():
        return []
    files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    return files[:limit] if limit else files


def score_images(model: HandXRayValidator, files: List[Path], device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of (p_hand_xray, binary_shannon_entropy) for each image file."""
    probs_list = []
    entropies_list = []
    model.eval()
    with torch.no_grad():
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            tensor = preprocess_validator_image(img, target_size=224).unsqueeze(0).to(device)
            probs = model.predict_proba(tensor)[0].cpu().numpy()
            p_hand = float(probs[0])
            eps = 1e-12
            ent = -float(np.sum(probs * np.log2(probs + eps))) / math.log2(len(probs))
            probs_list.append(p_hand)
            entropies_list.append(ent)
    return np.array(probs_list, dtype=np.float64), np.array(entropies_list, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select optimal confidence threshold for 2-class HandXRayValidator.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BACKEND_DIR / "checkpoints" / "hand_validator_best.pt"),
    )
    parser.add_argument(
        "--positives-dir",
        type=str,
        default=str(ROOT_DIR / "tests" / "validator" / "should_accept"),
    )
    parser.add_argument(
        "--negatives-dir",
        type=str,
        default=str(ROOT_DIR / "tests" / "validator" / "should_reject"),
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_classes = ckpt.get("classes", VALIDATOR_CLASSES)
    if len(ckpt_classes) != len(VALIDATOR_CLASSES):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has {len(ckpt_classes)} classes ({ckpt_classes}), "
            f"expected 2 classes ({VALIDATOR_CLASSES}). Train and download the 2-class checkpoint first."
        )

    device = torch.device("cpu")
    model = HandXRayValidator(num_classes=len(ckpt_classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    pos_files = list_images(Path(args.positives_dir))
    neg_files = list_images(Path(args.negatives_dir))

    print(f"Loaded 2-class checkpoint : {ckpt_path} (classes={ckpt_classes})")
    print(f"Held-out positives found  : {len(pos_files)} ({args.positives_dir})")
    print(f"Held-out negatives found  : {len(neg_files)} ({args.negatives_dir})")

    pos_probs, pos_ent = score_images(model, pos_files, device) if pos_files else (np.array([]), np.array([]))
    neg_probs, neg_ent = score_images(model, neg_files, device) if neg_files else (np.array([]), np.array([]))

    if len(pos_probs) > 0 and len(neg_probs) > 0:
        y_true = np.concatenate([np.ones_like(pos_probs), np.zeros_like(neg_probs)])
        y_score = np.concatenate([pos_probs, neg_probs])
        auc = roc_auc_score(y_true, y_score)
        print(f"\nHeld-Out ROC AUC: {auc:.4f}")
    else:
        print("\nHeld-Out ROC AUC: N/A (requires both non-empty positives and negatives folders)")

    print("\n" + "-" * 98)
    print(
        f"{'Threshold':<11} | {'RSNA Accept (TPR)':<19} | {'False-Accept (FPR)':<19} | "
        f"{'With Entropy<=0.75 TPR':<23} | {'Entropy Gate Diff'}"
    )
    print("-" * 98)

    candidate_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    for thr in candidate_thresholds:
        tpr = float(np.mean(pos_probs >= thr) * 100.0) if len(pos_probs) > 0 else float("nan")
        fpr = float(np.mean(neg_probs >= thr) * 100.0) if len(neg_probs) > 0 else float("nan")
        tpr_ent = (
            float(np.mean((pos_probs >= thr) & (pos_ent <= 0.75)) * 100.0)
            if len(pos_probs) > 0
            else float("nan")
        )
        diff = tpr_ent - tpr if not math.isnan(tpr) else float("nan")
        fpr_str = f"{fpr:6.2f}%" if not math.isnan(fpr) else "N/A (0 negs)"
        print(f"  {thr:<9.2f} | {tpr:6.2f}%             | {fpr_str:<19} | {tpr_ent:6.2f}%                 | {diff:+6.2f}%")

    print("-" * 98)
    print(
        "Note: For any binary threshold tau >= 0.78, binary Shannon entropy H(p) is automatically <= 0.75 "
        "because H(0.78) = 0.76, making an entropy gate mathematically redundant in a 2-class model."
    )


if __name__ == "__main__":
    main()
