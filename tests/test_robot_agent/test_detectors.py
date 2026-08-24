from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_agent.config import RobotAgentSettings
from robot_agent.perception import (
    YoloeDetector,
)


class DetectorContractTest(unittest.TestCase):

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
                YoloeDetector()

    def test_yoloe_initializes_and_reuses_visual_prompt(self):
        created_models = []

        class FakePredictor:
            pass

        class FakeYoloe:
            def __init__(self, model_name):
                self.model_name = model_name
                self.predict_calls = []
                created_models.append(self)

            def set_classes(self, classes):
                raise AssertionError("visual prompt should be used")

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                box = SimpleNamespace(
                    cls=[0], conf=[0.87], xyxy=[[20.0, 10.0, 180.0, 390.0]]
                )
                return [
                    SimpleNamespace(
                        names={0: "object0"},
                        boxes=[box],
                        orig_shape=(406, 196),
                    )
                ]

        with TemporaryDirectory() as temporary_directory:
            prompt_root = Path(temporary_directory)
            (prompt_root / "reference.png").touch()
            catalog = prompt_root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "fire extinguisher": {
                            "descriptions": ["fire extinguisher", "red cylinder"],
                            "reference_image": "reference.png",
                            "reference_bbox": [10, 5, 190, 400],
                        }
                    }
                ),
                encoding="utf-8",
            )
            modules = {
                "ultralytics": SimpleNamespace(YOLOE=FakeYoloe),
                "ultralytics.models": SimpleNamespace(),
                "ultralytics.models.yolo": SimpleNamespace(),
                "ultralytics.models.yolo.yoloe": SimpleNamespace(
                    YOLOEVPSegPredictor=FakePredictor
                ),
            }
            with patch.dict(sys.modules, modules):
                detector = YoloeDetector(prompt_catalog_path=catalog)
                detections = detector.detect(
                    object(), label="fire extinguisher"
                )
                detector.detect(object(), label="fire extinguisher")

        self.assertEqual(detector.prompt_mode, "visual")
        self.assertEqual(detections[0].label, "fire extinguisher")
        self.assertEqual(len(created_models[0].predict_calls), 3)
        initialization = created_models[0].predict_calls[0]
        self.assertEqual(initialization["refer_image"], str(prompt_root / "reference.png"))
        self.assertIs(initialization["predictor"], FakePredictor)

    def test_default_settings_select_yoloe(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = RobotAgentSettings(
                location_file=root / "locations.yaml",
                run_directory=root / "runs",
            )
        self.assertEqual(settings.annotated_camera_topic, "/camera/yoloe_annotated")
        self.assertEqual(settings.detection_interval_sec, 0.2)
        self.assertEqual(settings.detection_box_threshold, 0.05)
        self.assertEqual(settings.detection_confidence_threshold, 0.05)
        self.assertEqual(settings.detection_tracking_confidence_threshold, 0.01)
        self.assertEqual(settings.detection_tracking_max_center_jump, 0.25)
        self.assertEqual(settings.detection_confirmation_frames, 1)
        self.assertEqual(settings.image_center_tolerance, 0.10)
        self.assertEqual(settings.centering_max_angular_speed, 0.25)
        self.assertEqual(settings.centering_min_angular_speed, 0.10)
        self.assertEqual(settings.centering_gain, 0.8)
        self.assertEqual(settings.centering_min_linear_speed, 0.08)
        self.assertEqual(settings.centering_max_linear_speed, 0.25)
        self.assertEqual(settings.centering_linear_gain, 1.0)
        self.assertEqual(settings.centering_stable_frames, 3)
        self.assertEqual(settings.centering_timeout_sec, 30.0)
        self.assertEqual(settings.target_box_size_normalized, 0.4)
        self.assertEqual(settings.centering_detection_hold_sec, 1.0)
        self.assertEqual(settings.post_cancel_settle_sec, 0.5)

    def test_environment_settings_preserve_numeric_and_boolean_parsing(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.dict(
                os.environ,
                {
                    "ROBOT_AGENT_TRACE": "off",
                    "ROBOT_AGENT_MAX_TOOL_CALLS": "7",
                    "ROBOT_AGENT_DETECTION_INTERVAL_SEC": "0.4",
                    "ROBOT_AGENT_CENTER_ON_DETECTION": "yes",
                    "ROBOT_AGENT_INSPECTION_RADIUS_M": "1.5",
                    "ROBOT_AGENT_INSPECTION_MIN_RADIUS_M": "1.2",
                    "ROBOT_AGENT_INSPECTION_MAX_RADIUS_M": "2.0",
                },
                clear=True,
            ):
                settings = RobotAgentSettings.from_env(root)

        self.assertFalse(settings.trace)
        self.assertEqual(settings.max_tool_calls, 7)
        self.assertEqual(settings.detection_interval_sec, 0.4)
        self.assertTrue(settings.center_on_detection)
        self.assertEqual(settings.inspection_radius_m, 1.5)

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

if __name__ == "__main__":
    unittest.main()
