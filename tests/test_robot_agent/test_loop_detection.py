from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class UnusedRosAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise AssertionError("unexpected navigation")

    def stop_robot(self) -> ToolResult:
        raise AssertionError("unexpected stop")

    def get_pose(self) -> ToolResult:
        raise AssertionError("unexpected pose read")

    def cancel_navigation(self) -> ToolResult:
        raise AssertionError("unexpected cancellation")

    def detect_color(self, color: str) -> ToolResult:
        raise AssertionError("unexpected perception")


class LoopDetectionTest(unittest.TestCase):
    def test_warns_on_third_identical_call_and_blocks_fifth(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=False,
                loop_warn_threshold=3,
                repeated_tool_limit=5,
                max_no_progress_continuations=10,
            )
            runtime = RobotAgentRuntime(settings, "list locations repeatedly")
            registry = RobotToolRegistry(runtime, UnusedRosAdapter(), bt_skill=object())
            tool = {tool.name: tool for tool in registry.build()}["get_known_locations"]

            results = [tool.invoke({}) for _ in range(5)]

            self.assertEqual(results[0]["status"], ToolStatus.SUCCESS.value)
            self.assertNotIn("loop_warning", results[0]["data"])
            self.assertNotIn("loop_warning", results[1]["data"])
            self.assertEqual(results[2]["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(results[2]["data"]["loop_warning"]["count"], 3)
            self.assertEqual(results[3]["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(results[4]["status"], ToolStatus.FAILED.value)
            self.assertIn("loop", results[4]["error"].lower())


if __name__ == "__main__":
    unittest.main()
