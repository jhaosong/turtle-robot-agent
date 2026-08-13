from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_agent.perception import ColorBlobDetector, VlmDetector, YoloDetector, build_detector


class DetectorContractTest(unittest.TestCase):
    def test_color_blob_validates_query_before_motion(self):
        detector = ColorBlobDetector()
        detector.validate_query(color="blue", label=None)

        with self.assertRaisesRegex(ValueError, "requires one of"):
            detector.validate_query(color=None, label="chair")

    def test_vlm_backend_fails_fast_while_unimplemented(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            build_detector("vlm")

        with self.assertRaises(NotImplementedError):
            VlmDetector().validate_query(color=None, label="chair")

    def test_unknown_backend_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported detector backend"):
            build_detector("unknown")

    def test_yolo_uses_nano_model_and_bounded_input_by_default(self):
        created_models = []

        class FakeYolo:
            def __init__(self, model_name):
                self.model_name = model_name
                self.predict_calls = []
                created_models.append(self)

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                box = SimpleNamespace(
                    cls=[0],
                    conf=[0.82],
                    xyxy=[[100.0, 50.0, 300.0, 250.0]],
                )
                return [
                    SimpleNamespace(
                        names={0: "chair"},
                        boxes=[box],
                        orig_shape=(400, 800),
                    )
                ]

        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=FakeYolo)}):
            detector = YoloDetector()
            detections = detector.detect(object(), label="chair")

        self.assertEqual(created_models[0].model_name, "yolov8n.pt")
        self.assertEqual(created_models[0].predict_calls[0]["imgsz"], 640)
        self.assertEqual(detections[0].label, "chair")
        self.assertEqual(
            detections[0].image_position.to_dict(),
            {
                "x_px": 200.0,
                "y_px": 150.0,
                "x_normalized": 0.25,
                "y_normalized": 0.375,
            },
        )


if __name__ == "__main__":
    unittest.main()
