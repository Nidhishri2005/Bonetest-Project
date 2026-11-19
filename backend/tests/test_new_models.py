import math
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from app.core.config import Settings
from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.cnn_tw3 import CNNTW3
from app.models.ensemble_cnn import EnsembleCNN
from app.models.multimodal_cnn import MultimodalCNN
from app.models.multiscale_cnn_rf import (
    MultiscaleCNNFeatureExtractor,
    MultiscaleCNNRFWrapper,
)
from app.models.regression_multimodal import RegressionMultimodal
from app.services.inference_service import InferenceService
from app.services.model_registry import ModelRegistry, get_registry


def _create_dummy_xray_png() -> bytes:
    img = np.full((256, 256), 110, dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 180, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestModelArchitectures(unittest.TestCase):
    def test_model_registry_contains_all_models(self):
        expected_models = [
            "cnn_rf",
            "cnn",
            "cnn_dnn",
            "cnn_tw3",
            "multimodal_cnn",
            "regression_multimodal",
            "ensemble_cnn",
            "multiscale_cnn_rf",
        ]
        for m in expected_models:
            self.assertIn(m, MODEL_REGISTRY, f"Expected {m} in MODEL_REGISTRY")
            self.assertIn(m, SUPPORTED_MODELS, f"Expected {m} in SUPPORTED_MODELS")

    def test_image_only_models_do_not_require_gender(self):
        image_only = [
            "cnn",
            "cnn_rf",
            "cnn_dnn",
            "cnn_tw3",
            "ensemble_cnn",
            "multiscale_cnn_rf",
        ]
        for m in image_only:
            self.assertFalse(
                MODEL_REGISTRY[m]["requires_gender"],
                f"Model {m} must be image-only and not require gender",
            )

    def test_experimental_multimodal_models_require_gender(self):
        multimodal = ["multimodal_cnn", "regression_multimodal"]
        for m in multimodal:
            self.assertTrue(
                MODEL_REGISTRY[m]["requires_gender"],
                f"Model {m} must declare requires_gender=True",
            )
            self.assertTrue(
                MODEL_REGISTRY[m]["is_experimental"],
                f"Model {m} must be marked is_experimental=True",
            )

    def test_forward_pass_image_only_models(self):
        batch = torch.randn(2, 1, 64, 64)

        cnn = CNNBaseline(pretrained=False)
        cnn_dnn = CNNDNN(pretrained=False)
        cnn_tw3 = CNNTW3(pretrained=False)
        ensemble = EnsembleCNN(pretrained=False, num_branches=2)

        self.assertEqual(cnn(batch).shape, torch.Size([2]))
        self.assertEqual(cnn_dnn(batch).shape, torch.Size([2]))
        self.assertEqual(cnn_tw3(batch).shape, torch.Size([2]))
        self.assertEqual(ensemble(batch).shape, torch.Size([2]))

    def test_forward_pass_multimodal_models(self):
        batch = torch.randn(2, 1, 64, 64)
        gender = torch.tensor([[1.0], [0.0]])

        multi_cnn = MultimodalCNN(pretrained=False)
        reg_multi = RegressionMultimodal(pretrained=False)

        self.assertEqual(multi_cnn(batch, gender).shape, torch.Size([2]))
        self.assertEqual(reg_multi(batch, gender).shape, torch.Size([2]))

    def test_cnn_feature_extractor_dimension(self):
        batch = torch.randn(2, 1, 64, 64)
        extractor = CNNFeatureExtractor(pretrained=False)
        features = extractor(batch)
        self.assertEqual(features.shape, torch.Size([2, 512]))

    def test_multiscale_cnn_feature_extractor_dimension(self):
        batch = torch.randn(2, 1, 64, 64)
        extractor = MultiscaleCNNFeatureExtractor(pretrained=False).eval()
        with torch.no_grad():
            features = extractor(batch)
        # 128 (layer1) + 128 (layer2) + 256 (layer4) = 512
        self.assertEqual(features.shape, torch.Size([2, 512]))
        self.assertTrue(torch.isfinite(features).all())

    def test_multiscale_rf_receives_cnn_features_and_predicts_finite_months(self):
        extractor = MultiscaleCNNFeatureExtractor(pretrained=False).eval()
        received_shapes = []

        class SpyRF:
            def predict(self, X):
                received_shapes.append(X.shape)
                return np.array([134.5], dtype=np.float64)

        wrapper = MultiscaleCNNRFWrapper(cnn=extractor, rf_model=SpyRF())
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            feats = wrapper.cnn(x).cpu().numpy().flatten()
        pred_months = wrapper.predict_numpy(feats)

        self.assertEqual(received_shapes, [(1, 512)])
        self.assertIsInstance(pred_months, float)
        self.assertTrue(math.isfinite(pred_months))
        self.assertGreater(pred_months, 0.0)
        self.assertLessEqual(pred_months, 240.0)

    def test_multiscale_cnn_rf_end_to_end_inference_service_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "checkpoints"
            rf_dir = tmp_path / "rf_models"
            ckpt_dir.mkdir()
            rf_dir.mkdir()

            ext = MultiscaleCNNFeatureExtractor(pretrained=False).eval()
            torch.save({"model_state_dict": ext.state_dict()}, ckpt_dir / "multiscale_cnn_rf_best.pt")

            rng = np.random.RandomState(42)
            X_dummy = rng.randn(20, 512).astype(np.float32)
            y_dummy = rng.uniform(24.0, 180.0, size=20).astype(np.float32)
            rf = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42)
            rf.fit(X_dummy, y_dummy)
            joblib.dump(rf, rf_dir / "multiscale_cnn_rf.joblib")

            custom_settings = Settings(checkpoint_dir=ckpt_dir, rf_models_dir=rf_dir)
            custom_registry = ModelRegistry(settings=custom_settings)
            self.assertTrue(custom_registry.is_checkpoint_available("multiscale_cnn_rf"))

            service = InferenceService(registry=custom_registry)
            out = service.predict(_create_dummy_xray_png(), "multiscale_cnn_rf")
            self.assertEqual(out["model_type"], "multiscale_cnn_rf")
            self.assertTrue(math.isfinite(out["bone_age_months"]))
            self.assertGreaterEqual(out["bone_age_months"], 24.0)
            self.assertLessEqual(out["bone_age_months"], 180.0)
            self.assertIsNone(out["gender_used"])

    def test_registry_load_fails_loudly_on_state_dict_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "checkpoints"
            rf_dir = tmp_path / "rf_models"
            ckpt_dir.mkdir()
            rf_dir.mkdir()

            # Save a mismatched state_dict (e.g. CNNBaseline saved under multiscale_cnn_rf_best.pt)
            wrong_model = CNNBaseline(pretrained=False)
            torch.save(
                {"model_state_dict": wrong_model.state_dict()},
                ckpt_dir / "multiscale_cnn_rf_best.pt",
            )

            custom_settings = Settings(checkpoint_dir=ckpt_dir, rf_models_dir=rf_dir)
            custom_registry = ModelRegistry(settings=custom_settings)
            with self.assertRaises(RuntimeError):
                custom_registry.load("multiscale_cnn_rf")

    def test_gradcam_target_layers_exist(self):
        models_to_test = [
            CNNBaseline(pretrained=False),
            CNNDNN(pretrained=False),
            CNNTW3(pretrained=False),
            MultimodalCNN(pretrained=False),
            RegressionMultimodal(pretrained=False),
            EnsembleCNN(pretrained=False, num_branches=2),
            CNNFeatureExtractor(pretrained=False),
            MultiscaleCNNFeatureExtractor(pretrained=False),
        ]
        for m in models_to_test:
            target = m.get_gradcam_target_layer()
            self.assertIsInstance(target, torch.nn.Module)


class TestModelRegistryAndCheckpoints(unittest.TestCase):
    def setUp(self):
        self.registry = get_registry()

    def test_registry_checkpoint_availability(self):
        # Known trained checkpoints on disk
        trained_models = ["cnn_rf", "cnn", "cnn_dnn", "cnn_tw3", "multimodal_cnn"]
        for m in trained_models:
            self.assertTrue(
                self.registry.is_checkpoint_available(m),
                f"Expected checkpoint to be available for trained model '{m}'",
            )
        # Untrained models must report checkpoint_available == False until trained
        for m in ["multiscale_cnn_rf", "ensemble_cnn", "regression_multimodal"]:
            self.assertFalse(
                self.registry.is_checkpoint_available(m),
                f"Expected checkpoint_available=False for untrained model '{m}'",
            )

    def test_registry_load_trained_models(self):
        for m in ["cnn", "cnn_rf"]:
            loaded = self.registry.load(m)
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
