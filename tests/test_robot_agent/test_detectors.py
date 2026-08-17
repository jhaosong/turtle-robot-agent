from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_agent.config import RobotAgentSettings
from robot_agent.perception import (
    ColorBlobDetector,
    VlmDetector,
    YoloDetector,
    YoloeDetector,
    build_detector,
)


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
        self.assertEqual(created_models[0].predict_calls[0]["conf"], 0.1)
        self.assertEqual(detections[0].label, "chair")
        self.assertEqual(
            detections[0].image_position.to_dict(),
            {
                "x_px": 200.0,
                "y_px": 150.0,
                "x_normalized": 0.25,
                "y_normalized": 0.375,
                "width_normalized": 0.25,
                "height_normalized": 0.5,
            },
        )

    def test_yoloe_uses_text_prompt_and_preserves_bounding_box(self):
        created_models = []

        class FakeYoloe:
            def __init__(self, model_name):
                self.model_name = model_name
                self.classes = []
                self.predict_calls = []
                created_models.append(self)

            def set_classes(self, classes):
                self.classes.append(classes)

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                box = SimpleNamespace(
                    cls=[0], conf=[0.91], xyxy=[[80.0, 40.0, 240.0, 280.0]]
                )
                return [
                    SimpleNamespace(
                        names={0: "fire extinguisher"},
                        boxes=[box],
                        orig_shape=(400, 800),
                    )
                ]

        with patch.dict(
            "sys.modules", {"ultralytics": SimpleNamespace(YOLOE=FakeYoloe)}
        ):
            detector = YoloeDetector()
            detections = detector.detect(
                object(), color="red", label="fire extinguisher"
            )
            detector.detect(object(), label="fire extinguisher")

        self.assertEqual(created_models[0].model_name, "yoloe-26s-seg.pt")
        self.assertEqual(created_models[0].classes, [["fire extinguisher"]])
        self.assertEqual(len(created_models[0].predict_calls), 2)
        self.assertEqual(created_models[0].predict_calls[0]["imgsz"], 640)
        self.assertEqual(created_models[0].predict_calls[0]["conf"], 0.1)
        self.assertIsNone(detections[0].color)
        self.assertEqual(
            detections[0].image_position.to_dict(),
            {
                "x_px": 160.0,
                "y_px": 160.0,
                "x_normalized": 0.2,
                "y_normalized": 0.4,
                "width_normalized": 0.2,
                "height_normalized": 0.6,
            },
        )

    def test_yoloe_backend_fails_fast_without_capable_ultralytics(self):
        with patch.dict("sys.modules", {"ultralytics": None}):
            with self.assertRaisesRegex(RuntimeError, "YOLOE support"):
                build_detector("yoloe")

    def test_default_settings_select_yoloe(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = RobotAgentSettings(
                location_file=root / "locations.yaml",
                run_directory=root / "runs",
            )
        self.assertEqual(settings.detector_backend, "yoloe")
        self.assertEqual(settings.annotated_camera_topic, "/camera/yoloe_annotated")
        self.assertEqual(settings.detection_interval_sec, 0.2)
        self.assertEqual(settings.detection_box_threshold, 0.05)
        self.assertEqual(settings.detection_confidence_threshold, 0.05)
        self.assertEqual(settings.detection_tracking_confidence_threshold, 0.01)
        self.assertEqual(settings.detection_tracking_max_center_jump, 0.25)
        self.assertEqual(settings.detection_confirmation_frames, 2)
        self.assertEqual(settings.image_center_tolerance, 0.03)
        self.assertEqual(settings.centering_min_angular_speed, 0.025)
        self.assertEqual(settings.centering_min_linear_speed, 0.02)
        self.assertEqual(settings.centering_stable_frames, 3)
        self.assertEqual(settings.centering_timeout_sec, 30.0)
        self.assertEqual(settings.target_box_size_normalized, 0.6)
        self.assertEqual(settings.centering_detection_hold_sec, 1.0)
        self.assertEqual(settings.post_cancel_settle_sec, 0.5)

    def test_box_threshold_cannot_exceed_navigation_stop_threshold(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                RobotAgentSettings(
                    location_file=root / "locations.yaml",
                    run_directory=root / "runs",
                    detection_box_threshold=0.2,
                    detection_confidence_threshold=0.1,
                )

    def test_tracking_threshold_cannot_exceed_acquisition_threshold(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "tracking_confidence_threshold"):
                RobotAgentSettings(
                    location_file=root / "locations.yaml",
                    run_directory=root / "runs",
                    detection_confidence_threshold=0.05,
                    detection_tracking_confidence_threshold=0.06,
                )

    def test_color_blob_uses_keypoint_extent_as_box_size(self):
        point = SimpleNamespace(pt=(200.0, 100.0), size=80.0)
        detector_instance = SimpleNamespace(detect=lambda mask: [point])
        fake_cv2 = SimpleNamespace(
            COLOR_BGR2HSV=1,
            cvtColor=lambda image, code: object(),
            inRange=lambda hsv, lower, upper: object(),
            SimpleBlobDetector_Params=lambda: SimpleNamespace(),
            SimpleBlobDetector_create=lambda params: detector_instance,
        )
        image = SimpleNamespace(shape=(400, 800, 3))

        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            detection = ColorBlobDetector().detect(image, color="red")[0]

        self.assertEqual(detection.image_position.width_normalized, 0.1)
        self.assertEqual(detection.image_position.height_normalized, 0.2)


if __name__ == "__main__":
    unittest.main()
