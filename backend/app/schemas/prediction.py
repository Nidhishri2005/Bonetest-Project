from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    efficientnet_b3 = "efficientnet_b3"
    cnn = "cnn"
    cnn_dnn = "cnn_dnn"
    multimodal_cnn = "multimodal_cnn"
    regression_multimodal = "regression_multimodal"
    mask_rcnn = "mask_rcnn"
    ensemble_cnn = "ensemble_cnn"
    cnn_tw3 = "cnn_tw3"
    cnn_rf = "cnn_rf"


class Gender(str, Enum):
    male = "male"
    female = "female"


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]


class ModelInfo(BaseModel):
    id: str
    display_name: str
    description: str
    requires_gender: bool
    supports_gradcam: bool
    checkpoint_available: bool
    is_experimental: bool = False
    is_legacy: bool = False


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str


class ValidationDetails(BaseModel):
    is_valid: bool
    predicted_class: str
    confidence: float
    threshold: float
    probabilities: Optional[Dict[str, float]] = None
    rejection_code: Optional[str] = None
    rejection_message: Optional[str] = None
    processing_time_ms: float = 0.0


class InvalidImageResponse(BaseModel):
    success: bool = False
    error: str = "INVALID_IMAGE"
    message: str = "Invalid image. Please upload a pediatric hand/wrist X-ray."
    validation: Optional[ValidationDetails] = None


class PredictionResponse(BaseModel):
    success: bool = True
    model_type: str
    bone_age_months: float
    confidence: float = Field(ge=0.0, le=1.0)
    gender_used: Optional[str] = None
    processing_time_ms: float
    uncertainty_lower_months: Optional[float] = None
    uncertainty_upper_months: Optional[float] = None
    uncertainty_interval_months: Optional[float] = None
    conformal_coverage_pct: Optional[float] = None
    inference_engine: Optional[str] = None
    validation: Optional[ValidationDetails] = None


class GradCamResponse(BaseModel):
    success: bool = True
    model_type: str
    bone_age_months: float
    confidence: float
    method: Optional[str] = "gradcam++"
    heatmap_base64: str
    overlay_base64: str
    processing_time_ms: float
    validation: Optional[ValidationDetails] = None


class MetricValues(BaseModel):
    mae: float
    mse: float
    rmse: float


class ModelMetrics(BaseModel):
    model_type: str
    display_name: str
    metrics: MetricValues
    num_samples: int


class ComparisonResponse(BaseModel):
    models: List[ModelMetrics]
    best_model: str
    generated_at: str


class EvaluationResponse(BaseModel):
    comparison: ComparisonResponse
    plots: Dict[str, str]
    details: Optional[Dict] = None


class ErrorResponse(BaseModel):
    detail: str
