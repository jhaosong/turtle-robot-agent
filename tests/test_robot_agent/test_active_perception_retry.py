from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.ros import adapter as adapter_module
from robot_agent.ros.adapter import Ros2CliAdapter, RclpyRos2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Detection, ImagePosition, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class RecoveringPerceptionAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.detect_calls = 0
        self.align_calls = 0

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise AssertionError

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        raise AssertionError

    def cancel_navigation(self) -> ToolResult:
        raise AssertionError

    def detect_color(self, color: str) -> ToolResult:
        self.detect_calls += 1
        detections = [] if self.detect_calls == 1 else [Detection(
            label="colored_object",
            color=color,
            confidence=0.8,
            image_position=ImagePosition(320.0, 180.0, 0.5, 0.5, 0.2, 0.35),
        ).to_dict()]
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": detections})

    def align_to_detection(self, on_tick, **kwargs) -> ToolResult:
        self.align_calls += 1
        detection = on_tick()
        if detection is None:
            return ToolResult(status=ToolStatus.FAILED, error="target lost")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"operation": "align_to_detection", "found": detection.to_dict()},
        )

class ActivePerceptionRetryTest(unittest.TestCase):
    def test_empty_detection_aligns_from_bbox_then_retries(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=False,
                execute_ros2=True,
                active_perception_retry_enabled=True,
                detector_backend="color_blob",
            )
            runtime = RobotAgentRuntime(settings, "find blue")
            ros = RecoveringPerceptionAdapter()
            tool = {
                tool.name: tool
                for tool in RobotToolRegistry(runtime, ros, bt_skill=object()).build()
            }["inspect_for_color"]

            result = tool.invoke({"color": "blue"})

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(ros.detect_calls, 2)
            self.assertEqual(ros.align_calls, 1)
            self.assertEqual(len(result["data"]["matches"]), 1)
            self.assertEqual(result["data"]["attempts"], 2)

    def test_legacy_perception_motion_paths_are_not_reintroduced(self):
        source = inspect.getsource(adapter_module)
        self.assertNotIn("def adjust_for_perception", source)
        self.assertNotIn("def center_target_in_view", source)
        for adapter_type in (Ros2Adapter, Ros2CliAdapter, RclpyRos2Adapter):
            self.assertFalse(hasattr(adapter_type, "adjust_for_perception"))
            self.assertFalse(hasattr(adapter_type, "center_target_in_view"))


if __name__ == "__main__":
    unittest.main()
