"""Adebayo et al. (2018) Sanity Checks for Saliency Methods (Grad-CAM++).

Implements the Cascading Parameter Randomization Test:
- Verifies that saliency explanations are genuinely faithful to learned model parameters
  and do not function merely as parameter-invariant edge detectors.
- Sequentially randomizes weights from the top regression head down to lower feature layers
  and measures the degradation in spatial heatmap correlation.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_heatmap_similarity(h1: np.ndarray, h2: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation and Mean Absolute Difference (MAD) between two heatmaps."""
    f1 = h1.flatten().astype(np.float64)
    f2 = h2.flatten().astype(np.float64)

    # Mean Absolute Difference
    mad = float(np.mean(np.abs(f1 - f2)))

    # Pearson Correlation
    std1 = float(np.std(f1))
    std2 = float(np.std(f2))

    if std1 < 1e-6 or std2 < 1e-6:
        corr = 0.0
    else:
        norm1 = (f1 - np.mean(f1)) / std1
        norm2 = (f2 - np.mean(f2)) / std2
        corr = float(np.mean(norm1 * norm2))

    return round(corr, 4), round(mad, 4)


def randomize_layer_weights(module: nn.Module, seed: int = 42) -> None:
    """Re-initialize weights of a PyTorch module using Kaiming normal distribution."""
    torch.manual_seed(seed)
    for p in module.parameters():
        if p.dim() >= 2:
            nn.init.kaiming_normal_(p)
        else:
            nn.init.zeros_(p)


def run_cascading_randomization_test(
    model: nn.Module,
    input_tensor: torch.Tensor,
    gradcam_generator_cls,
    gender_tensor: torch.Tensor | None = None,
    seed: int = 42,
) -> Dict:
    """Execute the Adebayo Cascading Parameter Randomization Test on a model.
    
    Args:
        model: Trained PyTorch regression model.
        input_tensor: Standardized input image tensor (1, C, H, W).
        gradcam_generator_cls: GradCAMPlusPlus or GradCAM class with generate() and remove_hooks().
        gender_tensor: Optional demographic metadata tensor.
        seed: Random seed for parameter randomization.
        
    Returns:
        Structured dictionary containing correlation drops and test verdict.
    """
    # 1. Compute baseline trained heatmap
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    target_layer = model_copy.get_gradcam_target_layer()

    explainer = gradcam_generator_cls(model_copy, target_layer)
    try:
        baseline_heatmap = explainer.generate(input_tensor, gender_tensor)
    finally:
        explainer.remove_hooks()

    # 2. Identify randomizable candidate layers (top-down)
    randomization_steps = []

    # Head / classifier layers (top)
    head_layers = []
    if hasattr(model_copy, "head"):
        head_layers.append(("regression_head", model_copy.head))
    elif hasattr(model_copy, "backbone") and hasattr(model_copy.backbone, "fc"):
        head_layers.append(("fc_head", model_copy.backbone.fc))

    # Feature backbone layers (target and prior)
    backbone_layers = []
    if hasattr(model_copy, "features"):
        backbone_layers.append(("features_last_block", model_copy.features[-1]))
    elif hasattr(model_copy, "backbone") and hasattr(model_copy.backbone, "layer4"):
        backbone_layers.append(("backbone_layer4", model_copy.backbone.layer4))

    all_stages = head_layers + backbone_layers

    # 3. Cascading randomization
    for stage_idx, (stage_name, layer_module) in enumerate(all_stages):
        randomize_layer_weights(layer_module, seed=seed + stage_idx)
        target_layer = model_copy.get_gradcam_target_layer()

        step_explainer = gradcam_generator_cls(model_copy, target_layer)
        try:
            randomized_heatmap = step_explainer.generate(input_tensor, gender_tensor)
        finally:
            step_explainer.remove_hooks()

        corr, mad = compute_heatmap_similarity(baseline_heatmap, randomized_heatmap)
        randomization_steps.append({
            "stage": stage_idx + 1,
            "layer_randomized": stage_name,
            "pearson_correlation": corr,
            "mean_absolute_difference": mad,
        })

    # 4. Evaluation verdict: Correlation must drop significantly (<= 0.65 after top layers)
    final_corr = randomization_steps[-1]["pearson_correlation"] if randomization_steps else 1.0
    passed = bool(final_corr < 0.65)

    return {
        "test_name": "Adebayo et al. (2018) Cascading Parameter Randomization Test",
        "method": "Grad-CAM++",
        "passed_sanity_check": passed,
        "baseline_correlation": 1.0,
        "final_randomized_correlation": final_corr,
        "stages": randomization_steps,
        "clinical_interpretation": (
            "PASSED: Explanations are model-sensitive and reflect trained anatomical representations."
            if passed else
            "FAILED: Explanations show invariance to model weight randomization (potential edge-detector artifact)."
        ),
    }


def main():
    import json
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    backend_dir = repo_root / "backend"
    for p in [str(repo_root), str(backend_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import cv2
    from app.services.gradcam_service import GradCAMPlusPlus
    from app.services.model_registry import get_registry
    from app.services.preprocessing import preprocess_for_inference

    repo_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Use a sample cached radiograph or synthetic if not found
    sample_img_path = repo_root / "ml" / "data" / "cache_384" / "10000.png"
    if sample_img_path.exists():
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
    else:
        # Fallback synthetic image
        img = (np.ones((384, 384), dtype=np.uint8) * 128)
        _, buf = cv2.imencode(".png", img)
        img_bytes = buf.tobytes()

    tensor, _ = preprocess_for_inference(img_bytes)
    registry = get_registry()
    model = registry.get_pytorch_module("cnn")

    print("[Phase 6] Running Adebayo et al. (2018) Cascading Randomization Test on cnn baseline...")
    report = run_cascading_randomization_test(
        model=model,
        input_tensor=tensor,
        gradcam_generator_cls=GradCAMPlusPlus,
        seed=42,
    )

    json_path = reports_dir / "gradcam_sanity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = reports_dir / "gradcam_sanity_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Adebayo et al. (2018) Sanity Check for Saliency Methods\n\n")
        f.write(f"**Method:** {report['method']}\n")
        f.write(f"**Test Status:** {'PASSED' if report['passed_sanity_check'] else 'FAILED'}\n")
        f.write(f"**Baseline Correlation:** {report['baseline_correlation']:.4f}\n")
        f.write(f"**Final Randomized Correlation:** {report['final_randomized_correlation']:.4f}\n\n")
        f.write("### Cascading Randomization Steps\n\n")
        f.write("| Stage | Layer Randomized | Pearson Correlation | Mean Absolute Diff (MAD) |\n")
        f.write("|---|---|---|---|\n")
        for s in report["stages"]:
            f.write(f"| {s['stage']} | `{s['layer_randomized']}` | {s['pearson_correlation']:.4f} | {s['mean_absolute_difference']:.4f} |\n")
        f.write(f"\n### Clinical & Methodological Interpretation\n\n")
        f.write(f"> {report['clinical_interpretation']}\n")

    print(f"[Phase 6] Sanity check completed. Passed: {report['passed_sanity_check']}. Final corr: {report['final_randomized_correlation']}")
    print(f"[Phase 6] Reports saved to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
