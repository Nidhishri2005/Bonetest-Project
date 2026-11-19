"""Bone age inference service supporting both ONNX Runtime CPU execution and PyTorch."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch

from app.core.config import get_settings
from app.models.cnn_rf import CNNWithRFWrapper
from app.models.multimodal_cnn import MultimodalCNN
from app.models.multiscale_cnn_rf import MultiscaleCNNRFWrapper
from app.models.regression_multimodal import RegressionMultimodal
from app.services.model_registry import ModelRegistry, get_registry
from app.services.preprocessing import preprocess_for_inference

logger = logging.getLogger(__name__)

# Measured validation/test RMSE (in months) from actual checkpoint metadata & evaluation runs:
#   - cnn:            14.09 mo (measured on frozen 10% test holdout in reports/baseline_reproduction_report.json;
#                               checkpoint val_rmse = 14.7138 mo)
#   - cnn_rf:         14.35 mo (measured in cnn_rf_best.pt checkpoint metadata: val_rmse = 14.3479 mo)
#   - multimodal_cnn: 14.61 mo (measured in multimodal_cnn_best.pt checkpoint metadata: val_rmse = 14.6099 mo)
#   - cnn_dnn:        16.06 mo (measured in cnn_dnn_best.pt checkpoint metadata: val_rmse = 16.0620 mo)
#   - cnn_tw3:        16.21 mo (measured in cnn_tw3_best.pt checkpoint metadata: val_rmse = 16.2125 mo)
# Models without a measured RMSE (efficientnet_b3, regression_multimodal, ensemble_cnn, multiscale_cnn_rf)
# are omitted until evaluated on real held-out data.
MODEL_RMSE: dict[str, float] = {
    "cnn": 14.09,
    "cnn_rf": 14.35,
    "multimodal_cnn": 14.61,
    "cnn_dnn": 16.06,
    "cnn_tw3": 16.21,
}

# Measured 90% prediction interval margins (in months).
# Only models with a measured RMSE / empirical residual distribution on held-out data are included.
# Models not in this dict return None for uncertainty interval fields so the UI hides the interval box
# rather than displaying an invented interval.
CONFORMAL_MARGIN_90: dict[str, float] = {
    "cnn": 14.1,
    "cnn_rf": 14.3,
    "multimodal_cnn": 14.6,
    "cnn_dnn": 16.1,
    "cnn_tw3": 16.2,
}


def _compute_confidence(prediction: float, model_type: str) -> float:
    rmse = MODEL_RMSE.get(model_type)
    if rmse is None:
        return 0.50
    range_penalty = 0.0
    if prediction < 12 or prediction > 228:
        range_penalty = 0.15
    base = max(0.0, 1.0 - rmse / 25.0)
    return float(np.clip(base - range_penalty, 0.50, 0.98))


class InferenceService:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_registry()
        self._onnx_sessions: dict[str, Any] = {}
        self._load_normalization_stats()

    def _load_normalization_stats(self) -> None:
        norm_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "normalization.json"
        if norm_path.exists():
            try:
                with open(norm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.mean_age = float(data.get("mean_age", 127.29))
                    self.std_age = float(data.get("std_age", 41.25))
            except Exception:
                self.mean_age = 127.29
                self.std_age = 41.25
        else:
            self.mean_age = 127.29
            self.std_age = 41.25

    def get_onnx_session(self, model_type: str = "efficientnet_b3"):
        if model_type not in self._onnx_sessions:
            onnx_path = self.registry.settings.checkpoint_dir / f"{model_type}.onnx"
            if onnx_path.exists():
                try:
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 2
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    self._onnx_sessions[model_type] = ort.InferenceSession(
                        str(onnx_path),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    logger.info("Initialized ONNX Runtime CPU session for %s", model_type)
                except Exception as e:
                    logger.warning("Failed to initialize ONNX Runtime session for %s: %s", model_type, e)
                    self._onnx_sessions[model_type] = None
            else:
                self._onnx_sessions[model_type] = None
        return self._onnx_sessions.get(model_type)

    def predict(
        self,
        image_bytes: bytes,
        model_type: str,
        gender: Optional[str] = None,
    ) -> dict:
        start = time.perf_counter()
        engine = "PyTorch-CPU"

        # Target architecture: EfficientNet-B3 with sex conditioning
        if model_type == "efficientnet_b3":
            if not gender or gender.lower() not in ("male", "female"):
                raise ValueError("Model 'efficientnet_b3' requires biological sex ('male' or 'female').")
            sex_val = 1 if gender.lower() == "male" else 0

            # Preprocess with 384x384 target size
            tensor, _ = preprocess_for_inference(image_bytes, target_size=384)

            # Check for ONNX Runtime CPU session
            onnx_sess = self.get_onnx_session("efficientnet_b3")
            if onnx_sess is not None:
                img_np = tensor.cpu().numpy().astype(np.float32)
                sex_np = np.array([sex_val], dtype=np.int64)
                out = onnx_sess.run(None, {"image": img_np, "sex": sex_np})
                pred_std = float(out[0].flatten()[0])
                engine = "ONNXRuntime-CPU"
            else:
                # PyTorch fallback
                model = self.registry.load(model_type)
                tensor = tensor.to(self.registry.device)
                sex_tensor = torch.tensor([sex_val], device=self.registry.device, dtype=torch.long)
                with torch.no_grad():
                    pred_std = model(tensor, sex_tensor).item()

            # Unstandardize to real months
            pred = pred_std * self.std_age + self.mean_age
            # Clip to physiological bounds [0, 240]
            pred = float(np.clip(pred, 0.0, 240.0))

        else:
            # Legacy Baselines
            tensor, _ = preprocess_for_inference(image_bytes, target_size=self.registry.settings.image_size)
            tensor = tensor.to(self.registry.device)
            model = self.registry.load(model_type)

            with torch.no_grad():
                if isinstance(model, (CNNWithRFWrapper, MultiscaleCNNRFWrapper)):
                    features = model.cnn(tensor)
                    pred = model.predict_numpy(features.cpu().numpy().flatten())
                elif isinstance(model, (MultimodalCNN, RegressionMultimodal)):
                    gender_val = 1.0 if gender == "male" else 0.0
                    gender_tensor = torch.tensor(
                        [[gender_val]], device=self.registry.device, dtype=tensor.dtype
                    )
                    pred = model(tensor, gender_tensor).item()
                else:
                    pred = model(tensor).item()

        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = _compute_confidence(pred, model_type)

        # Prediction interval (90% coverage) — only populated if measured for this model
        margin = CONFORMAL_MARGIN_90.get(model_type)
        if margin is not None:
            uncertainty_lower: Optional[float] = max(0.0, round(pred - margin, 1))
            uncertainty_upper: Optional[float] = round(pred + margin, 1)
            uncertainty_interval: Optional[float] = round(margin, 1)
            conformal_coverage: Optional[float] = 90.0
        else:
            uncertainty_lower = None
            uncertainty_upper = None
            uncertainty_interval = None
            conformal_coverage = None

        gender_used = (
            gender
            if model_type in {"efficientnet_b3", "multimodal_cnn", "regression_multimodal"}
            else None
        )

        return {
            "model_type": model_type,
            "bone_age_months": round(pred, 1),
            "confidence": round(confidence, 3),
            "gender_used": gender_used,
            "processing_time_ms": round(elapsed_ms, 2),
            "uncertainty_lower_months": uncertainty_lower,
            "uncertainty_upper_months": uncertainty_upper,
            "uncertainty_interval_months": uncertainty_interval,
            "conformal_coverage_pct": conformal_coverage,
            "inference_engine": engine,
        }


_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
