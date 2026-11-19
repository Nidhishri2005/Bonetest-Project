"""One-Class Out-of-Distribution (OOD) Validator for Pediatric Hand Radiographs.

Uses frozen ImageNet-pretrained ResNet-18 (global-pooled 512-d features) and the
single shared preprocess_validator_image function to fit one-class OOD density /
distance estimators on a STRATIFIED sample of real RSNA training radiographs:
  - ~1,000 images per age bin for 24-72, 72-120, 120-168, 168+ months
  - All remaining available eligible images (138) in the 0-24 month infant bin
  - Separate stratified held-out set of 500 RSNA images (25 infant_0_24 + 119/119/119/118
    across the other 4 age bins, stratified by sex)
  - Strictly excludes all 230 files in configs/validator_excluded_eval_files.json

Implements two OOD scoring methods on L2-normalized 512-d features:
1. Mahalanobis distance with Ledoit-Wolf shrinkage covariance:
     d_mahal(z) = sqrt((z - mu)^T @ Sigma_shrunk^{-1} @ (z - mu))
2. k-Nearest Neighbors (k=5, cosine distance):
     s_knn(z) = mean of 5 smallest cosine distances (1 - z . z_train_i)

Persists fitted statistics, per-bin metadata, and 98% / 99% held-out RSNA thresholds to:
  backend/checkpoints/validator_ood.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _resolve_repo_root() -> Path:
    curr = Path(__file__).resolve().parent
    while curr != curr.parent:
        if (curr / "configs").is_dir() and (curr / "backend").is_dir():
            return curr
        curr = curr.parent
    return Path(__file__).resolve().parent.parent


ROOT_DIR = _resolve_repo_root()
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import torch
import torch.nn as nn
import torchvision.models as models
import yaml

from app.models.validator import preprocess_validator_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ValidatorOOD")

DEFAULT_OOD_NPZ_PATH = BACKEND_DIR / "checkpoints" / "validator_ood.npz"
TORCH_CACHE_DIR = BACKEND_DIR / "checkpoints" / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

AGE_BINS = [0.0, 24.0, 72.0, 120.0, 168.0, 250.0]
BIN_LABELS = [
    "infant_0_24",
    "early_child_24_72",
    "mid_child_72_120",
    "adolescent_120_168",
    "post_pubertal_168_plus",
]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of a 2D array (N, D) or a 1D vector (D,)."""
    if x.ndim == 1:
        norm = max(float(np.linalg.norm(x)), eps)
        return (x / norm).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)


class ResNet18FeatureExtractor(nn.Module):
    """Frozen ImageNet-pretrained ResNet-18 returning global-pooled 512-d features."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        self.backbone = backbone.to(self.device)

    def train(self, mode: bool = True) -> "ResNet18FeatureExtractor":
        return super().train(False)

    @torch.inference_mode()
    def extract_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        feats = self.backbone(batch_tensor.to(self.device))
        return feats.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_single_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess_validator_image(img_bgr, target_size=224).unsqueeze(0)
        return self.extract_batch(tensor)[0]


def _load_and_preprocess_path(path: Path) -> Optional[torch.Tensor]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return preprocess_validator_image(img, target_size=224)


def extract_features_for_paths(
    paths: List[Path],
    extractor: ResNet18FeatureExtractor,
    batch_size: int = 64,
    num_workers: int = 8,
    label: str = "images",
) -> Tuple[np.ndarray, List[str]]:
    """Extract (N, 512) features for a list of image paths using threaded I/O + batched CPU forward."""
    n_total = len(paths)
    all_feats: List[np.ndarray] = []
    valid_names: List[str] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in range(0, n_total, batch_size):
            chunk = paths[start : start + batch_size]
            tensors = list(pool.map(_load_and_preprocess_path, chunk))

            batch_list = []
            for p, t in zip(chunk, tensors):
                if t is not None:
                    batch_list.append(t)
                    valid_names.append(p.name)

            if batch_list:
                batch_tensor = torch.stack(batch_list, dim=0)
                feats = extractor.extract_batch(batch_tensor)
                all_feats.append(feats)

            done = min(start + batch_size, n_total)
            if done % 500 == 0 or done == n_total:
                elapsed = time.perf_counter() - t0
                rate = done / max(elapsed, 1e-6)
                logger.info(
                    "[%s] Extracted %d/%d (%.1f img/s, elapsed=%.1fs)",
                    label, done, n_total, rate, elapsed,
                )

    return np.vstack(all_feats).astype(np.float32), valid_names


class OODValidator:
    """One-class OOD validator supporting Mahalanobis (shrinkage) and kNN (k=5, cosine) scoring."""

    def __init__(
        self,
        mean: np.ndarray,
        precision: np.ndarray,
        knn_bank: np.ndarray,
        shrinkage: float,
        knn_k: int,
        thresholds: Dict[str, float],
        config: Dict[str, Any],
    ) -> None:
        self.mean = mean.astype(np.float64)
        self.precision = precision.astype(np.float64)
        self.knn_bank = l2_normalize(knn_bank.astype(np.float32))
        self.shrinkage = float(shrinkage)
        self.knn_k = int(knn_k)
        self.thresholds = thresholds
        self.config = config

    @classmethod
    def fit_from_features(
        cls,
        fit_features: np.ndarray,
        holdout_features: np.ndarray,
        knn_k: int = 5,
        seed: int = 42,
        fit_filenames: Optional[List[str]] = None,
        holdout_filenames: Optional[List[str]] = None,
        fit_bin_counts: Optional[Dict[str, int]] = None,
        holdout_bin_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple["OODValidator", Dict[str, np.ndarray]]:
        """Fit Ledoit-Wolf shrinkage Mahalanobis and kNN index on fit_features, calibrate on holdout_features."""
        z_fit = l2_normalize(fit_features).astype(np.float64)

        logger.info("Fitting Ledoit-Wolf shrinkage covariance on %d x %d L2-normalized features...", *z_fit.shape)
        lw = LedoitWolf(assume_centered=False)
        lw.fit(z_fit)
        mean = lw.location_.astype(np.float64)
        precision = lw.precision_.astype(np.float64)
        shrinkage = float(lw.shrinkage_)
        logger.info("Ledoit-Wolf shrinkage coefficient alpha = %.6f", shrinkage)

        temp_instance = cls(
            mean=mean,
            precision=precision,
            knn_bank=z_fit.astype(np.float32),
            shrinkage=shrinkage,
            knn_k=knn_k,
            thresholds={},
            config={},
        )

        holdout_scores = temp_instance.score_features(holdout_features)
        mahal_hold = holdout_scores["mahalanobis"]
        knn_hold = holdout_scores["knn_k5_cosine"]

        thresholds = {
            "mahalanobis_98": float(np.percentile(mahal_hold, 98)),
            "mahalanobis_99": float(np.percentile(mahal_hold, 99)),
            "knn_k5_cosine_98": float(np.percentile(knn_hold, 98)),
            "knn_k5_cosine_99": float(np.percentile(knn_hold, 99)),
        }

        config = {
            "backbone": "torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)",
            "feature_dim": 512,
            "l2_normalized": True,
            "sampling_strategy": "stratified_by_age_bin_and_sex",
            "preprocess": "app.models.validator.preprocess_validator_image (letterbox 224x224 RGB + ImageNet norm)",
            "fit_count": int(fit_features.shape[0]),
            "holdout_count": int(holdout_features.shape[0]),
            "fit_bin_counts": fit_bin_counts or {},
            "holdout_bin_counts": holdout_bin_counts or {},
            "knn_k": int(knn_k),
            "knn_metric": "cosine_distance",
            "covariance_estimator": "LedoitWolf",
            "shrinkage": shrinkage,
            "seed": int(seed),
            "thresholds": thresholds,
            "fit_filenames_sample": (fit_filenames or [])[:20],
            "holdout_filenames_sample": (holdout_filenames or [])[:20],
        }

        temp_instance.thresholds = thresholds
        temp_instance.config = config
        return temp_instance, holdout_scores

    def score_features(self, raw_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Mahalanobis distance and kNN (k=5, cosine) distance for (N, 512) or (512,) features."""
        single = raw_features.ndim == 1
        feats_2d = raw_features[None, :] if single else raw_features
        z = l2_normalize(feats_2d)

        diff = z.astype(np.float64) - self.mean[None, :]
        left = diff @ self.precision
        mahal_sq = np.maximum(np.sum(left * diff, axis=1), 0.0)
        mahal = np.sqrt(mahal_sq).astype(np.float64)

        sim_matrix = z.astype(np.float32) @ self.knn_bank.T
        cos_dist = np.maximum(1.0 - sim_matrix, 0.0).astype(np.float64)
        k = min(self.knn_k, cos_dist.shape[1])
        k_smallest = np.partition(cos_dist, kth=k - 1, axis=1)[:, :k]
        knn_score = np.mean(k_smallest, axis=1).astype(np.float64)

        return {
            "mahalanobis": mahal,
            "knn_k5_cosine": knn_score,
        }

    def score_single_feature(self, raw_feature_1d: np.ndarray) -> Dict[str, float]:
        res = self.score_features(raw_feature_1d)
        return {
            "mahalanobis": float(res["mahalanobis"][0]),
            "knn_k5_cosine": float(res["knn_k5_cosine"][0]),
        }

    @classmethod
    def load(cls, npz_path: Path = DEFAULT_OOD_NPZ_PATH) -> "OODValidator":
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"OOD validator statistics file not found at {npz_path}. "
                "Refusing to run without fitted RSNA OOD statistics."
            )
        data = np.load(npz_path, allow_pickle=False)
        thresholds = json.loads(str(data["thresholds_json"].item()))
        config = json.loads(str(data["config_json"].item()))
        return cls(
            mean=data["mean"],
            precision=data["precision"],
            knn_bank=data["knn_bank"],
            shrinkage=float(data["shrinkage"].item()),
            knn_k=int(data["knn_k"].item()),
            thresholds=thresholds,
            config=config,
        )


def resolve_rsna_paths() -> Tuple[Path, Path]:
    with open(ROOT_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["dataset_dir"])

    img_candidates = [
        base / "data" / "boneage-training-dataset" / "boneage-training-dataset",
        base / "boneage-training-dataset" / "boneage-training-dataset",
        base / "data" / "boneage-training-dataset",
        base / "boneage-training-dataset",
    ]
    img_dir = next((c for c in img_candidates if c.exists() and any(c.glob("*.png"))), None)
    if img_dir is None:
        raise FileNotFoundError(f"Could not locate RSNA training PNGs under {base}")

    csv_candidates = [
        base / "data" / "boneage-training-dataset.csv",
        base / "boneage-training-dataset.csv",
    ]
    csv_path = next((c for c in csv_candidates if c.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"Could not locate boneage-training-dataset.csv under {base}")

    return img_dir, csv_path


def load_full_rsna_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(int)
    df["file_name"] = df["id"].astype(str) + ".png"
    df["age_bin"] = pd.cut(
        df["boneage"],
        bins=AGE_BINS,
        labels=BIN_LABELS,
        include_lowest=True,
    ).astype(str)
    return df


def build_stratified_fit_and_holdout(
    df: pd.DataFrame,
    rsna_dir: Path,
    excluded_files: set,
    fit_per_bin: int = 1000,
    holdout_total: int = 500,
    infant_holdout: int = 25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select a stratified 500-image held-out set and a stratified ~1,000-per-bin fit set (all remaining for 0-24m)."""
    elig = df[~df["file_name"].isin(excluded_files)].copy()
    elig = elig[elig["file_name"].apply(lambda fn: (rsna_dir / fn).exists())].reset_index(drop=True)

    # Holdout allocation across the 5 bins summing to 500:
    # infant_0_24 gets 25 (~15.3% of its 163 eligible images), remaining 475 split across the other 4 bins (119, 119, 119, 118)
    other_bins = [b for b in BIN_LABELS if b != "infant_0_24"]
    rem_holdout = holdout_total - infant_holdout
    holdout_quotas = {"infant_0_24": infant_holdout}
    for i, b in enumerate(other_bins):
        holdout_quotas[b] = rem_holdout // len(other_bins) + (1 if i < (rem_holdout % len(other_bins)) else 0)

    fit_dfs = []
    holdout_dfs = []

    for b in BIN_LABELS:
        b_df = elig[elig["age_bin"] == b].copy()
        # Sort deterministically then shuffle with fixed seed stratified by sex
        b_male = b_df[b_df["male"] == True].sample(frac=1.0, random_state=seed)
        b_female = b_df[b_df["male"] == False].sample(frac=1.0, random_state=seed)

        # Interleave male and female rows so any slice preserves the exact sex ratio of the age bin
        interleaved_idx = []
        m_idx = b_male.index.tolist()
        f_idx = b_female.index.tolist()
        max_len = max(len(m_idx), len(f_idx))
        for i in range(max_len):
            if i < len(m_idx):
                interleaved_idx.append(m_idx[i])
            if i < len(f_idx):
                interleaved_idx.append(f_idx[i])
        b_shuffled = b_df.loc[interleaved_idx].reset_index(drop=True)

        h_quota = min(holdout_quotas[b], len(b_shuffled) // 2)
        h_part = b_shuffled.iloc[:h_quota].copy()
        remaining = b_shuffled.iloc[h_quota:].copy()
        f_part = remaining.iloc[: min(fit_per_bin, len(remaining))].copy()

        holdout_dfs.append(h_part)
        fit_dfs.append(f_part)

    fit_final = pd.concat(fit_dfs, ignore_index=True)
    holdout_final = pd.concat(holdout_dfs, ignore_index=True)
    return fit_final, holdout_final


def print_distribution_summary(name: str, scores: np.ndarray, thr_98: float, thr_99: float) -> None:
    print(f"\n  Method: {name}")
    print(f"    Count       : {len(scores)}")
    print(f"    Min         : {float(np.min(scores)):.6f}")
    print(f"    5th pct     : {float(np.percentile(scores, 5)):.6f}")
    print(f"    10th pct    : {float(np.percentile(scores, 10)):.6f}")
    print(f"    25th pct    : {float(np.percentile(scores, 25)):.6f}")
    print(f"    Median (50%): {float(np.median(scores)):.6f}")
    print(f"    Mean        : {float(np.mean(scores)):.6f}")
    print(f"    Std         : {float(np.std(scores)):.6f}")
    print(f"    75th pct    : {float(np.percentile(scores, 75)):.6f}")
    print(f"    90th pct    : {float(np.percentile(scores, 90)):.6f}")
    print(f"    95th pct    : {float(np.percentile(scores, 95)):.6f}")
    print(f"    98th pct    : {thr_98:.6f}  <-- 98% RSNA Accept Threshold (accepts {np.mean(scores <= thr_98)*100:.2f}% of holdout)")
    print(f"    99th pct    : {thr_99:.6f}  <-- 99% RSNA Accept Threshold (accepts {np.mean(scores <= thr_99)*100:.2f}% of holdout)")
    print(f"    Max         : {float(np.max(scores)):.6f}")


def print_per_bin_accept_table(
    title: str,
    age_bins: List[str],
    mahal_scores: np.ndarray,
    knn_scores: np.ndarray,
    thresholds: Dict[str, float],
) -> None:
    m98, m99 = thresholds["mahalanobis_98"], thresholds["mahalanobis_99"]
    k98, k99 = thresholds["knn_k5_cosine_98"], thresholds["knn_k5_cosine_99"]
    bins_arr = np.array(age_bins)

    print("\n" + "-" * 118)
    print(f"PER-AGE-BIN ACCEPT RATES: {title} (Total N={len(age_bins)})")
    print("-" * 118)
    print(
        f"{'Age Bin':<26} | {'Count':<6} | {'Mahal @98%':<18} | {'Mahal @99%':<18} | "
        f"{'kNN(k=5) @98%':<18} | {'kNN(k=5) @99%':<18}"
    )
    print("-" * 118)
    for b in BIN_LABELS:
        mask = bins_arr == b
        n = int(np.sum(mask))
        if n == 0:
            continue
        ms = mahal_scores[mask]
        ks = knn_scores[mask]
        ma98 = int(np.sum(ms <= m98))
        ma99 = int(np.sum(ms <= m99))
        ka98 = int(np.sum(ks <= k98))
        ka99 = int(np.sum(ks <= k99))
        print(
            f"{b:<26} | {n:<6d} | {ma98:3d}/{n:<3d} ({ma98/n*100:6.2f}%) | "
            f"{ma99:3d}/{n:<3d} ({ma99/n*100:6.2f}%) | "
            f"{ka98:3d}/{n:<3d} ({ka98/n*100:6.2f}%) | "
            f"{ka99:3d}/{n:<3d} ({ka99/n*100:6.2f}%)"
        )
    n_all = len(age_bins)
    ma98_all = int(np.sum(mahal_scores <= m98))
    ma99_all = int(np.sum(mahal_scores <= m99))
    ka98_all = int(np.sum(knn_scores <= k98))
    ka99_all = int(np.sum(knn_scores <= k99))
    print("-" * 118)
    print(
        f"{'TOTAL':<26} | {n_all:<6d} | {ma98_all:3d}/{n_all:<3d} ({ma98_all/n_all*100:6.2f}%) | "
        f"{ma99_all:3d}/{n_all:<3d} ({ma99_all/n_all*100:6.2f}%) | "
        f"{ka98_all:3d}/{n_all:<3d} ({ka98_all/n_all*100:6.2f}%) | "
        f"{ka99_all:3d}/{n_all:<3d} ({ka99_all/n_all*100:6.2f}%)"
    )
    print("-" * 118)


def classify_negative_subcategory(path: Path) -> str:
    """Categorize negative test images into 'xray_other' (chest/knee/foot/etc.) vs 'photos' vs 'screenshots_docs'."""
    s = "/".join(p.lower() for p in path.parts)
    if any(k in s for k in ("xray", "chest", "knee", "foot", "skull", "pelvis", "spine", "mura", "chex")):
        return "xray_other (chest/knee/foot)"
    if any(k in s for k in ("screen", "doc", "table", "code", "pdf", "text", "ui", "capture")):
        return "screenshots_and_documents"
    return "photos_and_animals"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate Stratified One-Class OOD Validator on RSNA radiographs.")
    parser.add_argument("--fit-per-bin", type=int, default=1000)
    parser.add_argument("--holdout-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_OOD_NPZ_PATH))
    parser.add_argument("--refit", action="store_true", help="Force re-extraction and stratified refit")
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)
    npz_path = Path(args.npz_path)
    rsna_dir, csv_path = resolve_rsna_paths()
    full_df = load_full_rsna_dataframe(csv_path)
    id_to_row = {int(r["id"]): r for _, r in full_df.iterrows()}

    # 1. Print full RSNA CSV age bin shares
    print("=" * 118)
    print("ITEM 1: RSNA TRAINING CSV AGE-BIN DISTRIBUTION (boneage-training-dataset.csv, N = 12,611)")
    print("=" * 118)
    bin_counts = full_df["age_bin"].value_counts()
    print(f"{'Age Bin':<26} | {'Age Range (mo)':<16} | {'Count':<8} | {'Share (%)'}")
    print("-" * 75)
    ranges = {
        "infant_0_24": "0 - 24",
        "early_child_24_72": "24 - 72",
        "mid_child_72_120": "72 - 120",
        "adolescent_120_168": "120 - 168",
        "post_pubertal_168_plus": "168+",
    }
    for b in BIN_LABELS:
        c = int(bin_counts.get(b, 0))
        print(f"{b:<26} | {ranges[b]:<16} | {c:<8d} | {c / len(full_df) * 100:6.2f}%")
    print("-" * 75)
    print(f"{'TOTAL':<26} | {'0 - 228':<16} | {len(full_df):<8d} | 100.00%")

    extractor = ResNet18FeatureExtractor(device=torch.device("cpu"))

    if not npz_path.exists() or args.refit:
        excl_path = ROOT_DIR / "configs" / "validator_excluded_eval_files.json"
        excluded_names = set()
        if excl_path.exists():
            with open(excl_path, "r", encoding="utf-8") as f:
                excl_data = json.load(f)
            excluded_names.update(excl_data.get("should_accept_files", []))
            excluded_names.update(excl_data.get("check_200_files", []))
            excluded_names.update(excl_data.get("should_reject_files", []))

        fit_df, holdout_df = build_stratified_fit_and_holdout(
            df=full_df,
            rsna_dir=rsna_dir,
            excluded_files=excluded_names,
            fit_per_bin=args.fit_per_bin,
            holdout_total=args.holdout_count,
            infant_holdout=25,
            seed=args.seed,
        )

        fit_bin_counts = {b: int((fit_df["age_bin"] == b).sum()) for b in BIN_LABELS}
        holdout_bin_counts = {b: int((holdout_df["age_bin"] == b).sum()) for b in BIN_LABELS}

        print("\n" + "=" * 118)
        print("ITEM 2: STRATIFIED FIT & HELD-OUT SAMPLE ALLOCATION (EXCLUDING 230 EVAL FILES)")
        print("=" * 118)
        print(f"{'Age Bin':<26} | {'Fit Count':<12} | {'Held-Out Count':<16}")
        print("-" * 65)
        for b in BIN_LABELS:
            print(f"{b:<26} | {fit_bin_counts[b]:<12d} | {holdout_bin_counts[b]:<16d}")
        print("-" * 65)
        print(f"{'TOTAL':<26} | {len(fit_df):<12d} | {len(holdout_df):<16d}")

        fit_paths = [rsna_dir / fn for fn in fit_df["file_name"].tolist()]
        holdout_paths = [rsna_dir / fn for fn in holdout_df["file_name"].tolist()]

        fit_feats, fit_names = extract_features_for_paths(
            fit_paths, extractor, batch_size=args.batch_size, label="Stratified-Fit"
        )
        hold_feats, hold_names = extract_features_for_paths(
            holdout_paths, extractor, batch_size=args.batch_size, label="Stratified-Holdout-500"
        )
        hold_bins = holdout_df["age_bin"].tolist()

        ood_model, holdout_scores = OODValidator.fit_from_features(
            fit_features=fit_feats,
            holdout_features=hold_feats,
            knn_k=5,
            seed=args.seed,
            fit_filenames=fit_names,
            holdout_filenames=hold_names,
            fit_bin_counts=fit_bin_counts,
            holdout_bin_counts=holdout_bin_counts,
        )

        np.savez_compressed(
            npz_path,
            mean=ood_model.mean.astype(np.float32),
            precision=ood_model.precision.astype(np.float32),
            knn_bank=ood_model.knn_bank.astype(np.float32),
            shrinkage=np.array(ood_model.shrinkage, dtype=np.float64),
            knn_k=np.array(ood_model.knn_k, dtype=np.int32),
            thresholds_json=np.array(json.dumps(ood_model.thresholds)),
            config_json=np.array(json.dumps(ood_model.config)),
            holdout_mahalanobis=holdout_scores["mahalanobis"].astype(np.float32),
            holdout_knn_k5_cosine=holdout_scores["knn_k5_cosine"].astype(np.float32),
            holdout_age_bins=np.array(hold_bins),
        )
        logger.info("Saved stratified OOD validator statistics to %s (%d bytes)", npz_path, npz_path.stat().st_size)
    else:
        ood_model = OODValidator.load(npz_path)
        raw_npz = np.load(npz_path, allow_pickle=False)
        holdout_scores = {
            "mahalanobis": raw_npz["holdout_mahalanobis"].astype(np.float64),
            "knn_k5_cosine": raw_npz["holdout_knn_k5_cosine"].astype(np.float64),
        }
        hold_bins = [str(x) for x in raw_npz["holdout_age_bins"].tolist()]

    print("\n" + "=" * 118)
    print("STRATIFIED HELD-OUT 500 RSNA SCORE DISTRIBUTION & 98% / 99% THRESHOLDS")
    print("=" * 118)
    print(f"Artifact path            : {npz_path} (size={npz_path.stat().st_size:,} bytes)")
    print(f"Fit RSNA count           : {ood_model.config['fit_count']} ({ood_model.config.get('fit_bin_counts')})")
    print(f"Held-out RSNA count      : {ood_model.config['holdout_count']} ({ood_model.config.get('holdout_bin_counts')})")
    print(f"Ledoit-Wolf shrinkage    : {ood_model.shrinkage:.6f}")

    print_distribution_summary(
        "Mahalanobis Distance (Ledoit-Wolf Shrinkage Covariance)",
        holdout_scores["mahalanobis"],
        ood_model.thresholds["mahalanobis_98"],
        ood_model.thresholds["mahalanobis_99"],
    )
    print_distribution_summary(
        "kNN Distance (k=5, Cosine Distance)",
        holdout_scores["knn_k5_cosine"],
        ood_model.thresholds["knn_k5_cosine_98"],
        ood_model.thresholds["knn_k5_cosine_99"],
    )

    # Per-age-bin accept rates on the 500 stratified held-out set
    print_per_bin_accept_table(
        "500 Stratified Held-Out RSNA Images",
        hold_bins,
        holdout_scores["mahalanobis"],
        holdout_scores["knn_k5_cosine"],
        ood_model.thresholds,
    )

    # Evaluate tests/validator/should_accept (with age_bin, boneage, sex per image + per-bin accept rates)
    accept_dir = ROOT_DIR / "tests" / "validator" / "should_accept"
    accept_files = sorted([p for p in accept_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    acc_feats, acc_names = extract_features_for_paths(accept_files, extractor, batch_size=32, num_workers=4, label="should_accept")
    acc_scores = ood_model.score_features(acc_feats)
    acc_mahal = acc_scores["mahalanobis"]
    acc_knn = acc_scores["knn_k5_cosine"]

    m98, m99 = ood_model.thresholds["mahalanobis_98"], ood_model.thresholds["mahalanobis_99"]
    k98, k99 = ood_model.thresholds["knn_k5_cosine_98"], ood_model.thresholds["knn_k5_cosine_99"]

    print("\n" + "=" * 126)
    print("PER-IMAGE EVALUATION & AGE-BIN MAPPING: tests/validator/should_accept (N = 30)")
    print("=" * 126)
    print(
        f"{'#':<3} {'File':<11} {'Age(m)':<7} {'Sex':<7} {'Age Bin':<23} | "
        f"{'Mahal':<9} {'@98%':<7} {'@99%':<7} | {'kNN(k=5)':<10} {'@98%':<7} {'@99%':<7}"
    )
    print("-" * 126)

    acc_bins: List[str] = []
    for i, (fname, ms, ks) in enumerate(zip(acc_names, acc_mahal, acc_knn), 1):
        img_id = int(Path(fname).stem)
        row = id_to_row.get(img_id)
        age_m = int(row["boneage"]) if row is not None else -1
        sex_s = ("male" if bool(row["male"]) else "female") if row is not None else "n/a"
        abin = str(row["age_bin"]) if row is not None else "unknown"
        acc_bins.append(abin)

        m_d98 = "ACCEPT" if ms <= m98 else "REJECT"
        m_d99 = "ACCEPT" if ms <= m99 else "REJECT"
        k_d98 = "ACCEPT" if ks <= k98 else "REJECT"
        k_d99 = "ACCEPT" if ks <= k99 else "REJECT"
        print(
            f"{i:<3d} {fname:<11} {age_m:<7d} {sex_s:<7} {abin:<23} | "
            f"{ms:<9.4f} {m_d98:<7} {m_d99:<7} | {ks:<10.6f} {k_d98:<7} {k_d99:<7}"
        )

    print_per_bin_accept_table(
        "tests/validator/should_accept (30 Real RSNA Images)",
        acc_bins,
        acc_mahal,
        acc_knn,
        ood_model.thresholds,
    )

    # Evaluate tests/validator/should_reject if populated
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"
    reject_files = sorted([p for p in reject_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]) if reject_dir.exists() else []
    print("\n" + "=" * 126)
    print(f"EVALUATION: tests/validator/should_reject ({reject_dir}) (count={len(reject_files)})")
    print("=" * 126)
    if not reject_files:
        print("  Folder is currently EMPTY (0 image files).")
    else:
        rej_feats, rej_names = extract_features_for_paths(reject_files, extractor, batch_size=32, num_workers=4, label="should_reject")
        rej_scores = ood_model.score_features(rej_feats)
        rej_mahal = rej_scores["mahalanobis"]
        rej_knn = rej_scores["knn_k5_cosine"]
        rej_subcats = [classify_negative_subcategory(p.relative_to(reject_dir)) for p in reject_files]

        print(
            f"{'#':<4} {'File':<28} {'Subcategory':<28} | "
            f"{'Mahal':<9} {'@98%':<7} {'@99%':<7} | {'kNN(k=5)':<10} {'@98%':<7} {'@99%':<7}"
        )
        print("-" * 126)
        for i, (p, sc, ms, ks) in enumerate(zip(reject_files, rej_subcats, rej_mahal, rej_knn), 1):
            rel_name = str(p.relative_to(reject_dir))
            m_d98 = "ACCEPT" if ms <= m98 else "REJECT"
            m_d99 = "ACCEPT" if ms <= m99 else "REJECT"
            k_d98 = "ACCEPT" if ks <= k98 else "REJECT"
            k_d99 = "ACCEPT" if ks <= k99 else "REJECT"
            print(
                f"{i:<4d} {rel_name[:27]:<28} {sc:<28} | "
                f"{ms:<9.4f} {m_d98:<7} {m_d99:<7} | {ks:<10.6f} {k_d98:<7} {k_d99:<7}"
            )

        n_rej = len(reject_files)
        print("-" * 126)
        print(f"OVERALL REJECT & FALSE-ACCEPT RATES ON should_reject (N={n_rej}):")
        for label_m, arr, thr in [
            ("Mahalanobis @98%", rej_mahal, m98),
            ("Mahalanobis @99%", rej_mahal, m99),
            ("kNN(k=5)    @98%", rej_knn, k98),
            ("kNN(k=5)    @99%", rej_knn, k99),
        ]:
            rej_c = int(np.sum(arr > thr))
            fa_c = n_rej - rej_c
            print(f"  - {label_m} (thr={thr:.6f}) : Rejected {rej_c}/{n_rej} ({rej_c/n_rej*100:6.2f}%) | False-Accept {fa_c}/{n_rej} ({fa_c/n_rej*100:6.2f}%)")

        # Subcategory breakdown (xray_other vs photos vs screenshots_docs)
        print("\nSUBCATEGORY FALSE-ACCEPT BREAKDOWN:")
        sub_arr = np.array(rej_subcats)
        for sc in sorted(set(rej_subcats)):
            mask = sub_arr == sc
            ns = int(np.sum(mask))
            ms_s = rej_mahal[mask]
            ks_s = rej_knn[mask]
            print(
                f"  [{sc:<28}] (N={ns:3d}) | "
                f"False-Accept: Mahal@98%={int(np.sum(ms_s<=m98))}/{ns} ({np.mean(ms_s<=m98)*100:5.1f}%), "
                f"Mahal@99%={int(np.sum(ms_s<=m99))}/{ns} ({np.mean(ms_s<=m99)*100:5.1f}%), "
                f"kNN@98%={int(np.sum(ks_s<=k98))}/{ns} ({np.mean(ks_s<=k98)*100:5.1f}%), "
                f"kNN@99%={int(np.sum(ks_s<=k99))}/{ns} ({np.mean(ks_s<=k99)*100:5.1f}%)"
            )
        print("=" * 126)


if __name__ == "__main__":
    main()
