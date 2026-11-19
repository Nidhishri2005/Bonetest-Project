import sys
from pathlib import Path

# Add root directory and backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.cnn_tw3 import CNNTW3
from app.models.ensemble_cnn import EnsembleCNN
from app.models.mask_rcnn import MaskRCNNRegression
from app.models.multimodal_cnn import MultimodalCNN
from app.models.regression_multimodal import RegressionMultimodal
from ml.models.efficientnet import EfficientNetB3SexFusion

MODEL_REGISTRY = {
    # Production Target Architecture (Phase 1-8)
    "efficientnet_b3": {
        "class": EfficientNetB3SexFusion,
        "display_name": "EfficientNet-B3 (Target Architecture)",
        "description": "ImageNet-pretrained EfficientNet-B3 with sex embedding fusion, ROI preprocessing, and conformal uncertainty",
        "requires_gender": True,
        "supports_gradcam": True,
        "is_experimental": False,
        "is_legacy": False,
    },
    # Primary Legacy Baselines (Maintained for comparative benchmarking)
    "cnn_rf": {
        "class": CNNFeatureExtractor,
        "display_name": "CNN + Random Forest [Legacy Baseline]",
        "description": "ResNet-18 feature extractor paired with Scikit-Learn Random Forest regressor (Image-Only)",
        "requires_gender": False,
        "supports_gradcam": True,
        "wrapper": CNNWithRFWrapper,
        "is_experimental": False,
        "is_legacy": True,
    },
    "cnn": {
        "class": CNNBaseline,
        "display_name": "CNN Baseline [Legacy Baseline]",
        "description": "Single-channel ResNet-18 backbone with linear regression head (Image-Only)",
        "requires_gender": False,
        "supports_gradcam": True,
        "is_experimental": False,
        "is_legacy": True,
    },
    # Legacy Architectures
    "cnn_dnn": {
        "class": CNNDNN,
        "display_name": "CNN + DNN [Legacy Baseline]",
        "description": "ResNet-18 backbone with 3-layer batch-normalized deep regression head (Image-Only)",
        "requires_gender": False,
        "supports_gradcam": True,
        "is_experimental": False,
        "is_legacy": True,
    },
    "cnn_tw3": {
        "class": CNNTW3,
        "display_name": "CNN + TW3 Proxy [Legacy Baseline]",
        "description": "CNN features fused with 4 analytical image-derived proxy features (Image-Only)",
        "requires_gender": False,
        "supports_gradcam": True,
        "is_experimental": False,
        "is_legacy": True,
    },
    "multimodal_cnn": {
        "class": MultimodalCNN,
        "display_name": "Multimodal CNN [Legacy Baseline]",
        "description": "Experimental research model combining image features with biological sex metadata",
        "requires_gender": True,
        "supports_gradcam": True,
        "is_experimental": True,
        "is_legacy": True,
    },
    "regression_multimodal": {
        "class": RegressionMultimodal,
        "display_name": "Regression Multimodal [Legacy Baseline]",
        "description": "Experimental multimodal deep learning network with 64-d metadata projection",
        "requires_gender": True,
        "supports_gradcam": True,
        "is_experimental": True,
        "is_legacy": True,
    },
    "mask_rcnn": {
        "class": MaskRCNNRegression,
        "display_name": "Mask R-CNN Proxy [Legacy Baseline]",
        "description": "Experimental region-guided pseudo-attention mask regressor",
        "requires_gender": False,
        "supports_gradcam": True,
        "is_experimental": True,
        "is_legacy": True,
    },
    "ensemble_cnn": {
        "class": EnsembleCNN,
        "display_name": "Ensemble CNN [Legacy Baseline]",
        "description": "Experimental multi-branch ensemble of 3 ResNet-18 regressors",
        "requires_gender": False,
        "supports_gradcam": True,
        "is_experimental": True,
        "is_legacy": True,
    },
}

SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())
