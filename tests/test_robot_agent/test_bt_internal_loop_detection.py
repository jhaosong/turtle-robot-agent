from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class CountingRosAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.navigation_calls = 0

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self.navigation_calls += 1
        return ToolResult(status=ToolStatus.FAILED, error="retryable", retryable=True)

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.FAILED)

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class RepeatingBtSkill:
    def run(self, goal, *, navigate, stop, wait, abort, on_node_started):
        first = navigate("kitchen", 1)
        second = navigate("kitchen", 1)
        return ToolResult(
            status=second.status,
            data={"first": first.to_dict(), "second": second.to_dict()},
            error=second.error,
        )


class BtInternalLoopDetectionTest(unittest.TestCase):
    def test_internal_navigation_retry_uses_same_loop_detector(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=False,
                loop_warn_threshold=1,
                repeated_tool_limit=2,
                max_no_progress_continuations=10,
            )
            runtime = RobotAgentRuntime(settings, "retry navigation")
            ros = CountingRosAdapter()
            registry = RobotToolRegistry(runtime, ros, RepeatingBtSkill())
            tool = {tool.name: tool for tool in registry.build()}["run_behavior_tree"]

            result = tool.invoke({"goal": "visit kitchen"})

            self.assertEqual(result["status"], ToolStatus.FAILED.value)
            self.assertEqual(ros.navigation_calls, 1)
            internal = [entry for entry in runtime.state.tool_history if entry["tool"] == "run_behavior_tree.GoToPose"]
            self.assertEqual(len(internal), 2)
            self.assertIn("loop", internal[-1]["result"]["error"].lower())


if __name__ == "__main__":
    unittest.main()
