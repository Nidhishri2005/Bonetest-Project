"""Comprehensive Step 3 Evaluation of Stratified One-Class OOD Validator on:
  - 500 Stratified Held-Out RSNA Images (from backend/checkpoints/validator_ood.npz)
  - 30 Real RSNA Images in tests/validator/should_accept
  - 23 Real Negative Images in tests/validator/should_reject

Computes and prints:
1. Per-image scores and decisions at 98% and 99% thresholds for both Mahalanobis and kNN (k=5, cosine).
2. Overall and per-subcategory reject rates and false-accept rates:
     - non-hand X-rays (1.jpe - 8.jpe, N=8)
     - photos and animals (9.jpe - 15.jpe, 16.jpe - 19.jpe, 24.jpe; also split into animals/general photos vs adult-hand optical photos 16.jpe-19.jpe)
     - screenshots and documents (21.jpe - 23.jpe, plus 15.jpe note)
     - adult-hand subfolder check
3. Every wrongly accepted negative image, its exact score, and how far below the 98% and 99% thresholds it lies.
4. ROC AUC for 500 held-out RSNA (positives) vs should_reject (negatives) for both methods,
   and the exact threshold that gives ZERO false accepts (and its corresponding RSNA accept rate).
5. The 3 lowest-scoring negatives and 3 highest-scoring RSNA images (held-out 500 and should_accept 30) with exact scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from ml.validator_ood import DEFAULT_OOD_NPZ_PATH, OODValidator, ResNet18FeatureExtractor, extract_features_for_paths

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".webp", ".tif", ".tiff"}

# Exact visual inventory of the 23 files in tests/validator/should_reject from should_reject_grid.png
FILE_METADATA = {
    "1.jpe":  ("non-hand X-rays",           "Chest PA/AP X-ray (blue-tinted)"),
    "2.jpe":  ("non-hand X-rays",           "Lateral cervical spine / skull base X-ray"),
    "3.jpe":  ("non-hand X-rays",           "Full-body skeleton radiograph illustration"),
    "4.jpe":  ("non-hand X-rays",           "Pelvis / bilateral hip AP X-ray"),
    "5.jpe":  ("non-hand X-rays",           "Dental panoramic orthopantomogram X-ray"),
    "6.jpe":  ("non-hand X-rays",           "Bilateral feet AP X-ray (blue-tinted)"),
    "7.jpe":  ("non-hand X-rays",           "Lateral foot and ankle X-ray"),
    "8.jpe":  ("non-hand X-rays",           "Right shoulder / clavicle AP X-ray"),
    "9.jpe":  ("photos and animals",        "Deer herd in grass (color animal photo)"),
    "10.jpe": ("photos and animals",        "Leopard portrait (grayscale animal photo)"),
    "11.jpe": ("photos and animals",        "Giraffe on dark background (animal photo)"),
    "12.jpe": ("photos and animals",        "Colobus monkey in tree (color animal photo)"),
    "13.jpe": ("photos and animals",        "Giant panda walking (color animal photo)"),
    "14.jpe": ("photos and animals",        "Butterfly on leaf (grayscale insect photo)"),
    "15.jpe": ("photos and animals",        "Hands holding open book (color photo)"),
    "16.jpe": ("photos and animals",        "Adult hand optical photo (palmar, dark skin)"),
    "17.jpe": ("photos and animals",        "Adult hand optical photo (dorsal, white bg)"),
    "18.jpe": ("photos and animals",        "Two adult hands optical photo (palmar/dorsal)"),
    "19.jpe": ("photos and animals",        "Two adult hands optical photo (spread fingers)"),
    "21.jpe": ("screenshots and documents", "Android smartphone home screen screenshot"),
    "22.jpe": ("screenshots and documents", "Visual Studio C# IDE code screenshot"),
    "23.jpe": ("screenshots and documents", "Mobile payment app UI screenshot"),
    "24.jpe": ("photos and animals",        "Medical AI stock composite photo/graphic"),
}


def natural_key(p: Path):
    stem = p.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def main() -> None:
    npz_path = DEFAULT_OOD_NPZ_PATH
    ood = OODValidator.load(npz_path)
    raw_npz = np.load(npz_path, allow_pickle=False)
    hold_mahal = raw_npz["holdout_mahalanobis"].astype(np.float64)
    hold_knn = raw_npz["holdout_knn_k5_cosine"].astype(np.float64)
    hold_bins = [str(x) for x in raw_npz["holdout_age_bins"].tolist()]

    # Also reconstruct held-out filenames so we can name the top-3 highest-scoring held-out RSNA images
    from ml.validator_ood import build_stratified_fit_and_holdout, load_full_rsna_dataframe, resolve_rsna_paths
    import json
    try:
        rsna_dir, csv_path = resolve_rsna_paths()
        full_df = load_full_rsna_dataframe(csv_path)
        with open(ROOT_DIR / "configs" / "validator_excluded_eval_files.json", "r", encoding="utf-8") as f:
            excl_data = json.load(f)
        excluded_names = (
            set(excl_data.get("should_accept_files", []))
            | set(excl_data.get("check_200_files", []))
            | set(excl_data.get("should_reject_files", []))
        )
        _, holdout_df = build_stratified_fit_and_holdout(
            df=full_df, rsna_dir=rsna_dir, excluded_files=excluded_names, fit_per_bin=1000, holdout_total=500, infant_holdout=25, seed=42
        )
        hold_filenames = holdout_df["file_name"].tolist()
        hold_ages = holdout_df["boneage"].astype(int).tolist()
    except Exception:
        hold_filenames = [f"holdout_{i:03d}.png" for i in range(len(hold_mahal))]
        hold_ages = [-1] * len(hold_mahal)

    extractor = ResNet18FeatureExtractor(device=torch.device("cpu"))

    # Score should_accept (30 files)
    accept_dir = ROOT_DIR / "tests" / "validator" / "should_accept"
    accept_files = sorted([p for p in accept_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=natural_key)
    acc_feats, acc_names = extract_features_for_paths(accept_files, extractor, batch_size=32, num_workers=4, label="should_accept")
    acc_scores = ood.score_features(acc_feats)
    acc_mahal = acc_scores["mahalanobis"]
    acc_knn = acc_scores["knn_k5_cosine"]

    # Score should_reject (23 files)
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"
    subdirs = [d for d in reject_dir.iterdir() if d.is_dir()]
    reject_files = sorted([p for p in reject_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=natural_key)
    rej_feats, rej_names = extract_features_for_paths(reject_files, extractor, batch_size=32, num_workers=4, label="should_reject")
    rej_scores = ood.score_features(rej_feats)
    rej_mahal = rej_scores["mahalanobis"]
    rej_knn = rej_scores["knn_k5_cosine"]

    m98 = ood.thresholds["mahalanobis_98"]
    m99 = ood.thresholds["mahalanobis_99"]
    k98 = ood.thresholds["knn_k5_cosine_98"]
    k99 = ood.thresholds["knn_k5_cosine_99"]

    print("\n" + "=" * 135)
    print(f"PER-IMAGE EVALUATION ON tests/validator/should_reject (N = {len(reject_files)}, subdirs={ [d.name for d in subdirs] })")
    print(f"Active Thresholds | Mahalanobis: @98%={m98:.6f}, @99%={m99:.6f} | kNN(k=5,cosine): @98%={k98:.6f}, @99%={k99:.6f}")
    print("=" * 135)
    print(
        f"{'#':<3} {'File':<8} {'Category':<26} {'Visual Content':<44} | "
        f"{'Mahal':<9} {'@98%':<7} {'@99%':<7} | {'kNN(k=5)':<10} {'@98%':<7} {'@99%':<7}"
    )
    print("-" * 135)

    cats = []
    for i, (p, ms, ks) in enumerate(zip(reject_files, rej_mahal, rej_knn), 1):
        cat, desc = FILE_METADATA.get(p.name, ("other", p.name))
        cats.append(cat)
        m_d98 = "ACCEPT" if ms <= m98 else "REJECT"
        m_d99 = "ACCEPT" if ms <= m99 else "REJECT"
        k_d98 = "ACCEPT" if ks <= k98 else "REJECT"
        k_d99 = "ACCEPT" if ks <= k99 else "REJECT"
        print(
            f"{i:<3d} {p.name:<8} {cat:<26} {desc:<44} | "
            f"{ms:<9.4f} {m_d98:<7} {m_d99:<7} | {ks:<10.6f} {k_d98:<7} {k_d99:<7}"
        )

    # 1. Reject rates overall and per category
    print("\n" + "=" * 135)
    print("ITEM 1: REJECT RATE & FALSE-ACCEPT RATE OVERALL AND BY SUBCATEGORY")
    print("=" * 135)
    cats_arr = np.array(cats)

    def report_group(label: str, mask: np.ndarray) -> None:
        n = int(np.sum(mask))
        if n == 0:
            print(f"{label:<42} | N=0 (none present)")
            return
        ms = rej_mahal[mask]
        ks = rej_knn[mask]
        mr98, mr99 = int(np.sum(ms > m98)), int(np.sum(ms > m99))
        kr98, kr99 = int(np.sum(ks > k98)), int(np.sum(ks > k99))
        print(
            f"{label:<42} | N={n:<2d} | "
            f"Mahal@98%: Rej {mr98:2d}/{n:<2d} ({mr98/n*100:6.2f}%) [FA {n-mr98:2d}] | "
            f"Mahal@99%: Rej {mr99:2d}/{n:<2d} ({mr99/n*100:6.2f}%) [FA {n-mr99:2d}] | "
            f"kNN@98%: Rej {kr98:2d}/{n:<2d} ({kr98/n*100:6.2f}%) [FA {n-kr98:2d}] | "
            f"kNN@99%: Rej {kr99:2d}/{n:<2d} ({kr99/n*100:6.2f}%) [FA {n-kr99:2d}]"
        )

    report_group("ALL should_reject (Overall)", np.ones(len(reject_files), dtype=bool))
    report_group("1. non-hand X-rays (1.jpe - 8.jpe)", cats_arr == "non-hand X-rays")
    report_group("2. photos and animals (all 12 images)", cats_arr == "photos and animals")
    adult_hand_mask = np.array([p.name in {"16.jpe", "17.jpe", "18.jpe", "19.jpe"} for p in reject_files])
    animals_other_photos_mask = (cats_arr == "photos and animals") & (~adult_hand_mask)
    report_group("   2a. animals & general photos (8 imgs)", animals_other_photos_mask)
    report_group("   2b. adult-hand optical photos (16-19.jpe)", adult_hand_mask)
    report_group("3. screenshots and documents (21-23.jpe)", cats_arr == "screenshots and documents")
    print(f"4. adult-hand subfolder                    | Subfolders in should_reject: {[d.name for d in subdirs]} (No 'adult-hand' subfolder exists; 16.jpe-19.jpe are 4 optical photos of hands in root)")

    # 2. Every wrongly accepted image and how far below the threshold it is
    print("\n" + "=" * 135)
    print("ITEM 2: EVERY WRONGLY ACCEPTED NEGATIVE IMAGE & MARGIN BELOW THRESHOLD (threshold - score)")
    print("=" * 135)
    for method_label, scores_arr, thr_val in [
        ("Mahalanobis @ 98% (thr=44.477459)", rej_mahal, m98),
        ("Mahalanobis @ 99% (thr=46.983149)", rej_mahal, m99),
        ("kNN (k=5, cosine) @ 98% (thr=0.112936)", rej_knn, k98),
        ("kNN (k=5, cosine) @ 99% (thr=0.122576)", rej_knn, k99),
    ]:
        wrong_idx = [idx for idx, s in enumerate(scores_arr) if s <= thr_val]
        print(f"\n  [{method_label}] -> Wrongly Accepted: {len(wrong_idx)}/{len(scores_arr)}")
        if not wrong_idx:
            print("    None (0 false accepts!)")
        else:
            for idx in wrong_idx:
                p = reject_files[idx]
                s = float(scores_arr[idx])
                diff = thr_val - s
                pct_below = (diff / thr_val) * 100.0
                cat, desc = FILE_METADATA.get(p.name, ("other", p.name))
                print(
                    f"    - {p.name:<8} ({cat}: {desc}) | score={s:.6f} | "
                    f"below_threshold_by={diff:.6f} ({pct_below:.2f}% below threshold {thr_val:.6f})"
                )

    # 3. ROC AUC for RSNA held-out (positives, N=500) vs should_reject (negatives, N=23), and zero-false-accept threshold
    print("\n" + "=" * 135)
    print("ITEM 3: ROC AUC (500 RSNA Held-Out Positives vs 23 should_reject Negatives) & ZERO-FALSE-ACCEPT THRESHOLD")
    print("=" * 135)
    # y_true = 1 for RSNA positive, 0 for should_reject negative; anomaly score is higher for negatives, so -score is positive score
    y_true = np.concatenate([np.ones(len(hold_mahal), dtype=int), np.zeros(len(rej_mahal), dtype=int)])

    for m_name, pos_s, neg_s, acc30_s in [
        ("Mahalanobis (Ledoit-Wolf Shrinkage)", hold_mahal, rej_mahal, acc_mahal),
        ("kNN (k=5, Cosine Distance)", hold_knn, rej_knn, acc_knn),
    ]:
        all_scores_pos_dir = np.concatenate([-pos_s, -neg_s])
        auc = float(roc_auc_score(y_true, all_scores_pos_dir))
        min_neg = float(np.min(neg_s))
        min_neg_file = reject_files[int(np.argmin(neg_s))].name
        # Any threshold strictly below min_neg gives 0/23 false accepts; max RSNA score < min_neg is the tightest data-achieved threshold
        pos_below_min_neg = pos_s[pos_s < min_neg]
        max_pos_below = float(np.max(pos_below_min_neg)) if len(pos_below_min_neg) > 0 else min_neg * 0.999
        holdout_accept_at_zero_fa = int(np.sum(pos_s < min_neg))
        holdout_rate_at_zero_fa = (holdout_accept_at_zero_fa / len(pos_s)) * 100.0
        acc30_at_zero_fa = int(np.sum(acc30_s < min_neg))
        acc30_rate_at_zero_fa = (acc30_at_zero_fa / len(acc30_s)) * 100.0

        print(f"\n  Method: {m_name}")
        print(f"    ROC AUC (500 Held-Out RSNA vs 23 should_reject) : {auc:.6f}")
        print(f"    Lowest negative score in should_reject          : {min_neg:.6f} (file: {min_neg_file})")
        print(f"    Zero-False-Accept Threshold (thr < {min_neg:.6f}, e.g. {max_pos_below:.6f}):")
        print(f"      - False Accepts on should_reject (N=23)       : 0/23 (0.00%) [100.00% Reject Rate]")
        print(f"      - Held-Out RSNA Accept Rate (N=500)           : {holdout_accept_at_zero_fa}/500 ({holdout_rate_at_zero_fa:.2f}%)")
        print(f"      - should_accept RSNA Accept Rate (N=30)       : {acc30_at_zero_fa}/30 ({acc30_rate_at_zero_fa:.2f}%)")

    # 4. Three lowest-scoring negatives and three highest-scoring RSNA images
    print("\n" + "=" * 135)
    print("ITEM 4: THREE LOWEST-SCORING NEGATIVES & THREE HIGHEST-SCORING RSNA IMAGES")
    print("=" * 135)

    for m_name, pos_s, neg_s, acc30_s in [
        ("Mahalanobis Distance", hold_mahal, rej_mahal, acc_mahal),
        ("kNN (k=5, Cosine Distance)", hold_knn, rej_knn, acc_knn),
    ]:
        print(f"\n  === {m_name} ===")
        neg_order = np.argsort(neg_s)[:3]
        print("  3 Lowest-Scoring Negatives (should_reject):")
        for rank, idx in enumerate(neg_order, 1):
            p = reject_files[idx]
            cat, desc = FILE_METADATA.get(p.name, ("other", p.name))
            print(f"    {rank}. {p.name:<8} | score={float(neg_s[idx]):.6f} | {cat} ({desc})")

        hold_order = np.argsort(pos_s)[::-1][:3]
        print("  3 Highest-Scoring RSNA Images (500 Stratified Held-Out Set):")
        for rank, idx in enumerate(hold_order, 1):
            print(
                f"    {rank}. {hold_filenames[idx]:<12} | score={float(pos_s[idx]):.6f} | "
                f"age_bin={hold_bins[idx]} (age={hold_ages[idx]} mo)"
            )

        acc_order = np.argsort(acc30_s)[::-1][:3]
        print("  3 Highest-Scoring RSNA Images (30 tests/validator/should_accept):")
        for rank, idx in enumerate(acc_order, 1):
            print(f"    {rank}. {acc_names[idx]:<12} | score={float(acc30_s[idx]):.6f}")


if __name__ == "__main__":
    main()
