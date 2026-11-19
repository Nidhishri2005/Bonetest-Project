from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.models import SUPPORTED_MODELS
from app.schemas.prediction import ComparisonResponse, EvaluationResponse, GradCamResponse, ModelType
from app.services.gradcam_service import get_gradcam_service
from app.services.image_validator import USER_REJECTION_MESSAGE, get_image_validator
from app.services.metrics_service import get_plot_path, load_evaluation_results
from app.services.model_registry import get_registry

router = APIRouter(tags=["gradcam", "metrics"])


@router.post("/gradcam", response_model=GradCamResponse)
async def generate_gradcam(
    file: UploadFile = File(...),
    model_type: ModelType = Form(ModelType.cnn),
    gender: str | None = Form(None),
) -> GradCamResponse:
    if model_type.value not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_type}")

    meta = get_registry().list_models()
    model_info = next((m for m in meta if m["id"] == model_type.value), None)
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

    # Perform Hand X-ray Image Validation before Grad-CAM computation
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
        result = get_gradcam_service().generate(
            content, model_type.value, gender
        )
        result["validation"] = val_result.to_dict()
        result["success"] = True
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}") from exc

    return GradCamResponse(**result)


@router.get("/metrics", response_model=EvaluationResponse)
def get_metrics() -> EvaluationResponse:
    data = load_evaluation_results()
    return EvaluationResponse(**data)


@router.get("/metrics/comparison", response_model=ComparisonResponse)
def get_comparison() -> ComparisonResponse:
    data = load_evaluation_results()
    return ComparisonResponse(**data["comparison"])


@router.get("/metrics/plots/{plot_name}")
def get_plot(plot_name: str) -> Response:
    path = get_plot_path(plot_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Plot '{plot_name}' not found. "
                "Available plots: 'metrics_comparison', 'scatter_comparison', 'validator_confusion_matrix'"
            ),
        )
    return Response(content=path.read_bytes(), media_type="image/png")
