from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class UnusedRosAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise AssertionError("navigation is not expected")

    def stop_robot(self) -> ToolResult:
        raise AssertionError("stop is not expected")

    def get_pose(self) -> ToolResult:
        raise AssertionError("pose is not expected")

    def cancel_navigation(self) -> ToolResult:
        raise AssertionError("cancellation is not expected")

    def detect_color(self, color: str) -> ToolResult:
        raise AssertionError("perception is not expected")


class WaitCallingBehaviorTree:
    def run(self, goal, *, navigate, stop, wait, abort, on_node_started):
        result = wait(2.5, 0)
        return ToolResult(
            status=result.status,
            data={"node_results": [{"result": result.to_dict()}]},
        )


class WaitDeduplicationTest(unittest.TestCase):
    def _registry(self, root: Path, bt_skill) -> RobotToolRegistry:
        location_file = root / "locations.yaml"
        location_file.write_text("location1: [0.0, 0.0, 0.0]\n", encoding="utf-8")
        runtime = RobotAgentRuntime(
            RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=False,
                max_no_progress_continuations=5,
            ),
            "wait",
        )
        return RobotToolRegistry(runtime, UnusedRosAdapter(), bt_skill)

    def test_top_level_wait_delegates_to_wait_for(self):
        with TemporaryDirectory() as temporary_directory:
            registry = self._registry(Path(temporary_directory), object())
            registry._wait_for = Mock(
                return_value=ToolResult(
                    status=ToolStatus.PLANNED,
                    data={"operation": "wait", "seconds": 1.5},
                )
            )
            tool = {item.name: item for item in registry.build()}["wait_seconds"]

            tool.invoke({"seconds": 1.5})

            registry._wait_for.assert_called_once_with(1.5)

    def test_behavior_tree_wait_delegates_to_wait_for(self):
        with TemporaryDirectory() as temporary_directory:
            registry = self._registry(
                Path(temporary_directory),
                WaitCallingBehaviorTree(),
            )
            registry._wait_for = Mock(
                return_value=ToolResult(
                    status=ToolStatus.PLANNED,
                    data={"operation": "wait", "seconds": 2.5},
                )
            )
            tool = {item.name: item for item in registry.build()}["run_behavior_tree"]

            tool.invoke({"goal": "wait once"})

            registry._wait_for.assert_called_once_with(2.5)


if __name__ == "__main__":
    unittest.main()
