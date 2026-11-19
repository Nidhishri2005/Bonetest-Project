import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.cnn_tw3 import CNNTW3
from app.models.ensemble_cnn import EnsembleCNN
from app.models.mask_rcnn import MaskRCNNRegression
from app.models.multimodal_cnn import MultimodalCNN
from app.models.regression_multimodal import RegressionMultimodal
from app.services.model_registry import ModelRegistry, get_registry


class TestModelArchitectures(unittest.TestCase):
    def test_model_registry_contains_all_models(self):
        expected_models = [
            "cnn_rf",
            "cnn",
            "cnn_dnn",
            "cnn_tw3",
            "multimodal_cnn",
            "regression_multimodal",
            "mask_rcnn",
            "ensemble_cnn",
        ]
        for m in expected_models:
            self.assertIn(m, MODEL_REGISTRY, f"Expected {m} in MODEL_REGISTRY")
            self.assertIn(m, SUPPORTED_MODELS, f"Expected {m} in SUPPORTED_MODELS")

    def test_image_only_models_do_not_require_gender(self):
        image_only = ["cnn", "cnn_rf", "cnn_dnn", "cnn_tw3", "ensemble_cnn", "mask_rcnn"]
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
        mask_rcnn = MaskRCNNRegression(pretrained=False)

        self.assertEqual(cnn(batch).shape, torch.Size([2]))
        self.assertEqual(cnn_dnn(batch).shape, torch.Size([2]))
        self.assertEqual(cnn_tw3(batch).shape, torch.Size([2]))
        self.assertEqual(ensemble(batch).shape, torch.Size([2]))
        self.assertEqual(mask_rcnn(batch).shape, torch.Size([2]))

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

    def test_gradcam_target_layers_exist(self):
        models_to_test = [
            CNNBaseline(pretrained=False),
            CNNDNN(pretrained=False),
            CNNTW3(pretrained=False),
            MultimodalCNN(pretrained=False),
            RegressionMultimodal(pretrained=False),
            EnsembleCNN(pretrained=False, num_branches=2),
            MaskRCNNRegression(pretrained=False),
            CNNFeatureExtractor(pretrained=False),
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

    def test_registry_load_trained_models(self):
        for m in ["cnn", "cnn_rf"]:
            loaded = self.registry.load(m)
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
