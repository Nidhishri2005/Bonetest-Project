from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.models import SUPPORTED_MODELS
from app.schemas.prediction import (
    ModelInfo,
    ModelType,
    ModelsListResponse,
    PredictionResponse,
)
from app.services.image_validator import USER_REJECTION_MESSAGE, get_image_validator
from app.services.inference_service import get_inference_service
from app.services.model_registry import get_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    settings = get_settings()
    registry = get_registry()
    models = [ModelInfo(**m) for m in registry.list_models()]
    return ModelsListResponse(models=models, default_model=settings.default_model)


prediction_router = APIRouter(prefix="/predict", tags=["prediction"])


@prediction_router.post("", response_model=PredictionResponse)
async def predict_bone_age(
    file: UploadFile = File(..., description="Hand/wrist X-ray image"),
    model_type: ModelType = Form(ModelType.cnn),
    gender: str | None = Form(None, description="Optional demographic metadata; only used for experimental multimodal models"),
) -> PredictionResponse:
    if model_type.value not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_type}")

    registry = get_registry()
    meta = registry.list_models()
    model_info = next((m for m in meta if m["id"] == model_type.value), None)

    if not registry.is_checkpoint_available(model_type.value):
        available = [m["id"] for m in meta if m["checkpoint_available"]]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{model_type.value}' is not ready for inference because its checkpoint "
                f"was not found. Currently available models with trained weights: {available}."
            ),
        )

    if model_info and model_info["requires_gender"] and gender not in ("male", "female"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_type.value}' is an experimental multimodal model that "
                "requires the 'gender' parameter ('male' or 'female'). "
                "For standard image-only prediction, select an image-only model."
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Perform Hand X-ray Image Validation before inference
    validator = get_image_validator()
    val_result = validator.validate(content)
    if not val_result.is_valid:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": val_result.rejection_code or "INVALID_IMAGE",
                "reason_code": val_result.reason_code or "INVALID_IMAGE",
                "message": USER_REJECTION_MESSAGE,
                "score": round(val_result.score, 6) if val_result.score is not None else None,
                "threshold": round(val_result.threshold, 6),
                "stage": val_result.deciding_stage,
                "detail": val_result.technical_detail or USER_REJECTION_MESSAGE,
                "validation": val_result.to_dict(),
            },
        )

    try:
        result = get_inference_service().predict(
            content, model_type.value, gender
        )
        result["validation"] = val_result.to_dict()
        result["success"] = True
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)
