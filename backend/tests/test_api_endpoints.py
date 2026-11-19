import io
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
from app.api.routes.analysis import get_comparison, get_metrics, get_plot
from app.api.routes.health import health_check
from app.api.routes.prediction import list_models
from app.services.gradcam_service import get_gradcam_service
from app.services.inference_service import get_inference_service
from app.services.metrics_service import get_plot_path
from fastapi import HTTPException


def _create_dummy_png_bytes(width=256, height=256) -> bytes:
    """Create a synthetic hand-shaped bone phantom X-ray in memory."""
    img = np.zeros((height, width), dtype=np.uint8)
    # Background soft tissue
    cv2.circle(img, (width // 2, height // 2), 70, 60, -1)
    # Metacarpal bones (dense radiopaque structures)
    for x in [80, 110, 140, 170]:
        cv2.rectangle(img, (x, 70), (x + 15, 180), 200, -1)
    # Growth plates (epiphyses)
    for x in [80, 110, 140, 170]:
        cv2.circle(img, (x + 7, 60), 8, 230, -1)
    success, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


class TestAPIEndpointsAndServices(unittest.TestCase):
    def setUp(self):
        self.dummy_bytes = _create_dummy_png_bytes()

    def test_health_check(self):
        resp = health_check()
        self.assertEqual(resp.status, "ok")
        self.assertIsInstance(resp.version, str)

    def test_list_models_contains_default_and_flags(self):
        resp = list_models()
        self.assertIn(resp.default_model, ["cnn", "cnn_rf"])
        model_ids = [m.id for m in resp.models]
        self.assertIn("cnn_rf", model_ids)
        self.assertIn("cnn", model_ids)
        self.assertIn("cnn_tw3", model_ids)
        self.assertIn("multimodal_cnn", model_ids)

    def test_metrics_endpoints(self):
        metrics_resp = get_metrics()
        self.assertIn("comparison", metrics_resp.model_dump())
        self.assertGreater(len(metrics_resp.comparison.models), 0)

        comp_resp = get_comparison()
        self.assertIn(comp_resp.best_model, ["efficientnet_b3", "cnn_rf"])

    def test_plot_endpoint_success_and_error(self):
        # Valid plots
        valid_path = get_plot_path("metrics_comparison")
        self.assertIsNotNone(valid_path)
        self.assertTrue(valid_path.exists())

        resp = get_plot("metrics_comparison")
        self.assertEqual(resp.media_type, "image/png")
        self.assertGreater(len(resp.body), 0)

        # Invalid plot returns 404
        with self.assertRaises(HTTPException) as cm:
            get_plot("nonexistent_plot_xyz")
        self.assertEqual(cm.exception.status_code, 404)

    def test_inference_service_image_only_cnn_rf(self):
        service = get_inference_service()
        result = service.predict(self.dummy_bytes, "cnn_rf", gender=None)
        self.assertEqual(result["model_type"], "cnn_rf")
        self.assertIsInstance(result["bone_age_months"], float)
        self.assertGreater(result["bone_age_months"], 0)
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertIsNone(result["gender_used"])

    def test_inference_service_image_only_cnn(self):
        service = get_inference_service()
        result = service.predict(self.dummy_bytes, "cnn", gender=None)
        self.assertEqual(result["model_type"], "cnn")
        self.assertIsInstance(result["bone_age_months"], float)
        self.assertIsNone(result["gender_used"])

    def test_inference_service_image_only_cnn_tw3(self):
        service = get_inference_service()
        result = service.predict(self.dummy_bytes, "cnn_tw3", gender=None)
        self.assertEqual(result["model_type"], "cnn_tw3")
        self.assertIsInstance(result["bone_age_months"], float)
        self.assertIsNone(result["gender_used"])

    def test_gradcam_service_image_only(self):
        service = get_gradcam_service()
        result = service.generate(self.dummy_bytes, "cnn", gender=None)
        self.assertEqual(result["model_type"], "cnn")
        self.assertIn("heatmap_base64", result)
        self.assertIn("overlay_base64", result)
        self.assertGreater(len(result["heatmap_base64"]), 100)
        self.assertGreater(len(result["overlay_base64"]), 100)

    def test_gradcam_service_cnn_rf(self):
        service = get_gradcam_service()
        result = service.generate(self.dummy_bytes, "cnn_rf", gender=None)
        self.assertEqual(result["model_type"], "cnn_rf")
        self.assertIn("overlay_base64", result)


if __name__ == "__main__":
    unittest.main()
