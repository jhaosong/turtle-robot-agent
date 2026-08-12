from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class RecoveringPerceptionAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.detect_calls = 0
        self.adjust_calls = 0

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
        detections = [] if self.detect_calls == 1 else [
            {"label": "colored_object", "color": color, "confidence": 0.8, "position": None}
        ]
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": detections})

    def adjust_for_perception(self, linear_x: float, duration_sec: float) -> ToolResult:
        self.adjust_calls += 1
        return ToolResult(status=ToolStatus.SUCCESS, data={"linear_x": linear_x, "duration_sec": duration_sec})


class ActivePerceptionRetryTest(unittest.TestCase):
    def test_empty_detection_backs_off_once_then_retries(self):
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
            self.assertEqual(ros.adjust_calls, 1)
            self.assertEqual(len(result["data"]["matches"]), 1)
            self.assertEqual(result["data"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
