"""Unit tests for Phase 1 restructure, model registry, and target architecture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.services.model_registry import get_registry
from ml.models.efficientnet import EfficientNetB3SexFusion


class TestPhase1Restructure(unittest.TestCase):
    def test_root_directories_exist(self):
        self.assertTrue((ROOT_DIR / "ml").is_dir(), "ml/ directory missing")
        self.assertTrue((ROOT_DIR / "ml" / "data").is_dir(), "ml/data/ directory missing")
        self.assertTrue((ROOT_DIR / "ml" / "models").is_dir(), "ml/models/ directory missing")
        self.assertTrue((ROOT_DIR / "ml" / "train").is_dir(), "ml/train/ directory missing")
        self.assertTrue((ROOT_DIR / "ml" / "eval").is_dir(), "ml/eval/ directory missing")
        self.assertTrue((ROOT_DIR / "ml" / "export").is_dir(), "ml/export/ directory missing")
        self.assertTrue((ROOT_DIR / "configs").is_dir(), "configs/ directory missing")
        self.assertTrue((ROOT_DIR / "reports").is_dir(), "reports/ directory missing")
        self.assertTrue((ROOT_DIR / "configs" / "data.yaml").is_file(), "configs/data.yaml missing")
        self.assertTrue((ROOT_DIR / "configs" / "model.yaml").is_file(), "configs/model.yaml missing")

    def test_model_registry_classification(self):
        registry = get_registry()
        models = registry.list_models()
        model_map = {m["id"]: m for m in models}

        # Check target architecture presence
        self.assertIn("efficientnet_b3", model_map)
        eff_meta = model_map["efficientnet_b3"]
        self.assertFalse(eff_meta["is_legacy"], "EfficientNet-B3 should be production/target, not legacy")
        self.assertTrue(eff_meta["requires_gender"], "EfficientNet-B3 requires sex fusion input")

        # Check legacy models marked appropriately
        legacy_models = [
            "cnn_dnn",
            "cnn_tw3",
            "multimodal_cnn",
            "regression_multimodal",
            "mask_rcnn",
            "ensemble_cnn",
            "cnn",
            "cnn_rf",
        ]
        for m_id in legacy_models:
            self.assertIn(m_id, model_map, f"Missing model: {m_id}")
            self.assertTrue(model_map[m_id]["is_legacy"], f"Model {m_id} should be marked is_legacy=True")

    def test_efficientnet_b3_architecture_forward_pass(self):
        torch.set_num_threads(2)
        model = EfficientNetB3SexFusion(pretrained=False, sex_embedding_dim=32)
        model.eval()

        batch_size = 2
        # Test 1-channel grayscale input (auto-replicated to 3 channels)
        x_1ch = torch.randn(batch_size, 1, 384, 384)
        sex = torch.tensor([1, 0])  # male=1, female=0

        with torch.no_grad():
            out = model(x_1ch, sex)

        self.assertEqual(out.shape, (batch_size,))
        self.assertFalse(torch.isnan(out).any())

        # Test Grad-CAM target layer
        target_layer = model.get_gradcam_target_layer()
        self.assertIsInstance(target_layer, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
