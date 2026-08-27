from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from robot_agent.config import RobotAgentSettings
from robot_agent.perception.detector import Detector
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime.runtime import RobotAgentRuntime
from robot_agent.state import Detection, ImagePosition, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class CropCameraAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.frame[40:60, 60:140] = 255
        self.overlay = []

    def get_camera_frame(self):
        return self.frame.copy()

    def update_detection_overlay(self, detections):
        self.overlay = detections

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        del pose
        return ToolResult(status=ToolStatus.SUCCESS)

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)


class CropDetector(Detector):
    def validate_query(self, *, color=None, label=None):
        del color, label

    def detect(self, image, *, color=None, label=None):
        del image
        return [
            Detection(
                label=label or "object",
                color=color,
                confidence=0.9,
                image_position=ImagePosition(
                    x_px=100.0,
                    y_px=50.0,
                    x_normalized=0.5,
                    y_normalized=0.5,
                    width_normalized=0.4,
                    height_normalized=0.2,
                ),
            )
        ]


class CaptureObjectCropToolTest(unittest.TestCase):
    def test_tool_detects_current_frame_and_exports_only_bbox_crop(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            locations = root / "locations.yaml"
            locations.write_text("inspection_start: [0, 0, 0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
            )
            runtime = RobotAgentRuntime(settings, "zoom in on the extinguisher")
            adapter = CropCameraAdapter()
            registry = RobotToolRegistry(
                runtime,
                adapter,
                bt_skill=object(),
                detector=CropDetector(),
            )
            tools = {tool.name: tool for tool in registry.build()}

            result = tools["capture_object_crop"].invoke(
                {
                    "label": "fire extinguisher",
                    "padding_ratio": 0.0,
                }
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(result["data"]["crop"]["crop_size_px"], {"width": 80, "height": 20})
            self.assertTrue(Path(result["data"]["image_path"]).is_file())
            self.assertEqual(len(adapter.overlay), 1)
            self.assertIn("capture_object_crop", tools)


if __name__ == "__main__":
    unittest.main()
