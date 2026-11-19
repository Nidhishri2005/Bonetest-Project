"""Statistical evaluation engine for pediatric bone-age prediction.

Provides:
- Comprehensive regression metrics (MAE, RMSE, MedAE, % within 6/12mo, R2, MAPE, MaxAE)
- Non-parametric empirical bootstrap 95% confidence intervals
- Bland-Altman agreement analysis (mean bias, limits of agreement, outlier rate)
- Paired bootstrap hypothesis testing for model comparisons
- Stratified demographic subgroup metrics (by sex and pediatric age cohorts)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Standard pediatric age cohorts (in months)
PEDIATRIC_COHORT_BINS = [0.0, 24.0, 72.0, 120.0, 168.0, 300.0]
PEDIATRIC_COHORT_LABELS = [
    "0-24 mo (Infant)",
    "24-72 mo (Early Child)",
    "72-120 mo (Mid Child)",
    "120-168 mo (Adolescent)",
    "168+ mo (Post-Pubertal)",
]


def compute_regression_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """Compute primary bone age regression metrics.
    
    Args:
        y_true: Ground truth bone age array in months.
        y_pred: Predicted bone age array in months.
        
    Returns:
        Dictionary of statistical regression metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Array length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    medae = float(np.median(abs_errors))
    pct_6mo = float(np.mean(abs_errors <= 6.0) * 100.0)
    pct_12mo = float(np.mean(abs_errors <= 12.0) * 100.0)
    max_error = float(np.max(abs_errors))

    # MAPE (filter zero/near-zero to avoid division by zero)
    valid_mask = y_true > 1e-4
    mape = float(np.mean(abs_errors[valid_mask] / y_true[valid_mask]) * 100.0) if np.any(valid_mask) else 0.0

    # R2 Score
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "medae": round(medae, 4),
        "pct_within_6mo": round(pct_6mo, 2),
        "pct_within_12mo": round(pct_12mo, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "max_error": round(max_error, 2),
        "n_samples": len(y_true),
    }


def compute_bootstrap_ci(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical non-parametric bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth array.
        y_pred: Prediction array.
        metric_fn: Function taking (y_true, y_pred) and returning a scalar float.
        n_bootstraps: Number of bootstrap iterations.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)

    point_estimate = float(metric_fn(y_true, y_pred))

    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        bootstrap_estimates[i] = metric_fn(y_true[indices], y_pred[indices])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_estimates, alpha * 100.0))
    ci_upper = float(np.percentile(bootstrap_estimates, (1.0 - alpha) * 100.0))

    return round(point_estimate, 4), round(ci_lower, 4), round(ci_upper, 4)


def compute_bland_altman(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """Compute Bland-Altman limits of agreement.
    
    Differences defined as: diff = y_pred - y_true
    Means defined as: mean = (y_pred + y_true) / 2
    
    Returns:
        Dictionary with mean bias, SD of bias, lower/upper limits of agreement (1.96 * SD),
        and percentage of outliers outside limits.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diffs = y_pred - y_true
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    lower_limit = mean_bias - 1.96 * sd_diff
    upper_limit = mean_bias + 1.96 * sd_diff

    outliers = (diffs < lower_limit) | (diffs > upper_limit)
    outlier_pct = float(np.mean(outliers) * 100.0)

    return {
        "mean_bias": round(mean_bias, 3),
        "sd_bias": round(sd_diff, 3),
        "lower_limit": round(lower_limit, 3),
        "upper_limit": round(upper_limit, 3),
        "outlier_count": int(np.sum(outliers)),
        "outlier_pct": round(outlier_pct, 2),
    }


def paired_bootstrap_test(
    y_true: Union[np.ndarray, List[float]],
    y_pred_baseline: Union[np.ndarray, List[float]],
    y_pred_candidate: Union[np.ndarray, List[float]],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Union[float, bool]]:
    """Paired bootstrap hypothesis test for statistical significance between two models.
    
    Hypothesis:
        H0: MAE(candidate) == MAE(baseline)
        H1: MAE(candidate) < MAE(baseline) (candidate improves upon baseline)
        
    Returns:
        Difference in MAE, 95% CI of delta, p-value, and whether improvement is significant.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_base = np.asarray(y_pred_baseline, dtype=np.float64)
    y_cand = np.asarray(y_pred_candidate, dtype=np.float64)
    n = len(y_true)

    base_errors = np.abs(y_base - y_true)
    cand_errors = np.abs(y_cand - y_true)

    delta_mae = float(np.mean(cand_errors) - np.mean(base_errors))

    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        bootstrap_deltas[i] = np.mean(cand_errors[idx]) - np.mean(base_errors[idx])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_deltas, alpha * 100.0))
    ci_upper = float(np.percentile(bootstrap_deltas, (1.0 - alpha) * 100.0))

    # One-sided p-value: fraction of bootstrap samples where delta >= 0
    p_value = float(np.mean(bootstrap_deltas >= 0.0))
    is_significant = bool(ci_upper < 0.0 and p_value < 0.05)

    return {
        "delta_mae": round(delta_mae, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "is_statistically_significant": is_significant,
    }


def compute_stratified_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    males: Union[np.ndarray, List[bool]],
) -> Dict[str, Dict]:
    """Compute disaggregated regression metrics across sex and age cohort strata."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    males = np.asarray(males, dtype=bool)

    # 1. Biological Sex Strata
    sex_results = {}
    for sex_label, mask in [("Male", males), ("Female", ~males)]:
        if np.sum(mask) > 0:
            sex_results[sex_label] = compute_regression_metrics(y_true[mask], y_pred[mask])
        else:
            sex_results[sex_label] = {"n_samples": 0}

    # 2. Pediatric Age Cohort Strata
    cohort_results = {}
    for i in range(len(PEDIATRIC_COHORT_BINS) - 1):
        low, high = PEDIATRIC_COHORT_BINS[i], PEDIATRIC_COHORT_BINS[i + 1]
        label = PEDIATRIC_COHORT_LABELS[i]
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            cohort_results[label] = compute_regression_metrics(y_true[mask], y_pred[mask])
        else:
            cohort_results[label] = {"n_samples": 0}

    return {
        "by_sex": sex_results,
        "by_cohort": cohort_results,
    }
