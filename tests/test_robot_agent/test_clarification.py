from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.goal_monitor import GoalMonitor
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import GoalBlocker, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class UnusedRosAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise AssertionError

    def stop_robot(self) -> ToolResult:
        raise AssertionError

    def get_pose(self) -> ToolResult:
        raise AssertionError

    def cancel_navigation(self) -> ToolResult:
        raise AssertionError


class ClarificationTest(unittest.TestCase):
    def test_structured_clarification_surfaces_as_goal_blocker(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                ),
                "go over there",
            )
            registry = RobotToolRegistry(runtime, UnusedRosAdapter(), bt_skill=object())
            tool = {tool.name: tool for tool in registry.build()}["request_clarification"]

            result = tool.invoke(
                {
                    "question": "Which known location should I navigate to?",
                    "reason": "The requested destination is ambiguous.",
                }
            )
            evaluation = GoalMonitor().evaluate(runtime.state)

            self.assertEqual(result["status"], ToolStatus.NEEDS_INPUT.value)
            self.assertEqual(evaluation.blocker, GoalBlocker.NEEDS_USER_INPUT)
            self.assertEqual(evaluation.evidence["question"], "Which known location should I navigate to?")


if __name__ == "__main__":
    unittest.main()
