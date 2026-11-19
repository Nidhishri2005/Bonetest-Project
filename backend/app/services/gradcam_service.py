"""Grad-CAM and Grad-CAM++ explainability service for pediatric bone-age prediction."""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.models.cnn_rf import CNNWithRFWrapper
from app.models.multimodal_cnn import MultimodalCNN
from app.models.new_models import RegressionMultimodal
from app.services.inference_service import (
    InferenceService,
    get_inference_service,
)
from app.services.model_registry import get_registry
from app.services.preprocessing import (
    encode_image_base64,
    overlay_heatmap,
    preprocess_for_inference,
    tensor_to_display_image,
)

logger = logging.getLogger(__name__)


class GradCAM:
    """Standard Grad-CAM (first-order global average pooling)."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = []

        def forward_hook(_module, _input, output) -> None:
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output) -> None:
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        gender_tensor: torch.Tensor | None = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().requires_grad_(True)

        if isinstance(self.model, (MultimodalCNN, RegressionMultimodal)):
            output = self.model(input_tensor, gender_tensor)
        elif hasattr(self.model, "sex_embedding"):
            # EfficientNetB3SexFusion
            gender_val = gender_tensor if gender_tensor is not None else torch.tensor([1], device=input_tensor.device)
            output = self.model(input_tensor, gender_val)
        else:
            output = self.model(input_tensor)

        if output.dim() == 0:
            output = output.unsqueeze(0)

        self.model.zero_grad(set_to_none=True)
        output.backward(torch.ones_like(output), retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        else:
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


class GradCAMPlusPlus:
    """Grad-CAM++ explainability with second- and third-order gradient weighting.
    
    Provides enhanced spatial sensitivity for multiple co-occurring ossification
    centers (carpal bones, metacarpal heads, phalanges) compared to standard Grad-CAM.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = []

        def forward_hook(_module, _input, output) -> None:
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output) -> None:
            self.gradients = grad_output[0]

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        gender_tensor: torch.Tensor | None = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().requires_grad_(True)

        if isinstance(self.model, (MultimodalCNN, RegressionMultimodal)):
            output = self.model(input_tensor, gender_tensor)
        elif hasattr(self.model, "sex_embedding"):
            gender_val = gender_tensor if gender_tensor is not None else torch.tensor([1], device=input_tensor.device)
            output = self.model(input_tensor, gender_val)
        else:
            output = self.model(input_tensor)

        if output.dim() == 0:
            output = output.unsqueeze(0)

        self.model.zero_grad(set_to_none=True)
        output.backward(torch.ones_like(output), retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM++ hooks did not capture activations")

        grads = self.gradients
        acts = self.activations

        # Grad-CAM++ weighting coefficients
        # alpha_ijk = (grads^2) / (2 * grads^2 + sum(acts * grads^3) + eps)
        grads_2 = grads ** 2
        grads_3 = grads_2 * grads

        sum_acts_grads3 = (acts * grads_3).sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grads_2 + sum_acts_grads3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * 1e-7)

        aij = grads_2 / denom
        w = (aij * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (w * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.ndim == 0:
            cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        else:
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


class GradCamService:
    def __init__(
        self,
        inference: InferenceService | None = None,
    ) -> None:
        self.inference = inference or get_inference_service()
        self.registry = get_registry()

    def generate(
        self,
        image_bytes: bytes,
        model_type: str,
        gender: Optional[str] = None,
        method: str = "gradcam++",
    ) -> dict:
        start = time.perf_counter()
        meta = self.registry.list_models()

        model_meta = next((m for m in meta if m["id"] == model_type), None)
        if model_meta and not model_meta["supports_gradcam"]:
            raise ValueError(f"Model '{model_type}' does not support Grad-CAM/Grad-CAM++")

        # Preprocess input tensor (512x512 letterbox legacy or 384x384 target)
        tensor, _display = preprocess_for_inference(image_bytes)
        tensor = tensor.to(self.registry.device)

        pytorch_module = self.registry.get_pytorch_module(model_type)
        target_layer = pytorch_module.get_gradcam_target_layer()

        gender_tensor = None
        if isinstance(pytorch_module, (MultimodalCNN, RegressionMultimodal)):
            gender_val = 1.0 if gender == "male" else 0.0
            gender_tensor = torch.tensor([[gender_val]], device=self.registry.device, dtype=tensor.dtype)
        elif hasattr(pytorch_module, "sex_embedding"):
            gender_val = 1 if gender == "male" else 0
            gender_tensor = torch.tensor([gender_val], device=self.registry.device, dtype=torch.long)

        pytorch_module.eval()

        # Select explainer engine: Grad-CAM++ (default) or Grad-CAM
        explainer_cls = GradCAMPlusPlus if method.lower() in ("gradcam++", "gradcamplusplus", "plusplus") else GradCAM
        explainer = explainer_cls(pytorch_module, target_layer)

        try:
            heatmap = explainer.generate(tensor, gender_tensor)
        finally:
            explainer.remove_hooks()

        # Get normal model prediction
        prediction = self.inference.predict(image_bytes, model_type, gender)

        # Spatially aligned display image
        display_image = tensor_to_display_image(tensor)
        overlay = overlay_heatmap(display_image, heatmap)

        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "model_type": model_type,
            "bone_age_months": prediction["bone_age_months"],
            "confidence": prediction["confidence"],
            "method": method,
            "heatmap_base64": encode_image_base64(heatmap_color),
            "overlay_base64": encode_image_base64(overlay),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def run_sanity_check(
        self,
        image_bytes: bytes,
        model_type: str = "cnn",
        method: str = "gradcam++",
    ) -> Dict:
        """Run the Adebayo et al. (2018) Cascading Parameter Randomization Test."""
        tensor, _ = preprocess_for_inference(image_bytes)
        tensor = tensor.to(self.registry.device)

        pytorch_module = self.registry.get_pytorch_module(model_type)
        explainer_cls = GradCAMPlusPlus if method.lower() in ("gradcam++", "gradcamplusplus", "plusplus") else GradCAM

        from ml.eval.gradcam_sanity import run_cascading_randomization_test
        return run_cascading_randomization_test(
            model=pytorch_module,
            input_tensor=tensor,
            gradcam_generator_cls=explainer_cls,
        )


_gradcam_service: GradCamService | None = None


def get_gradcam_service() -> GradCamService:
    global _gradcam_service
    if _gradcam_service is None:
        _gradcam_service = GradCamService()
    return _gradcam_service
