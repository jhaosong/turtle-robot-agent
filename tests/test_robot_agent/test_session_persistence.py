from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter, build_ros2_adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.skills import BehaviorTreeSkill
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class SuccessfulRosAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self._pose = pose
        return ToolResult(status=ToolStatus.SUCCESS, data={"target_pose": pose.to_dict()})

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"pose": self._pose.to_dict()})

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class SessionPersistenceTest(unittest.TestCase):
    def _settings(self, root: Path, *, execute_ros2: bool) -> RobotAgentSettings:
        location_file = root / "locations.yaml"
        location_file.write_text("kitchen: [1.5, -0.5, 0.25]\n", encoding="utf-8")
        return RobotAgentSettings(
            location_file=location_file,
            run_directory=root / "runs",
            execute_ros2=execute_ros2,
            trace=False,
            max_no_progress_continuations=5,
        )

    def _navigate_and_restart(
        self,
        settings: RobotAgentSettings,
        adapter: Ros2Adapter,
    ) -> RobotAgentRuntime:
        runtime = RobotAgentRuntime(settings, "go to the kitchen")
        registry = RobotToolRegistry(
            runtime,
            adapter,
            BehaviorTreeSkill(None, {"kitchen"}, runtime.run_path / "bt"),
        )
        navigate = next(tool for tool in registry.build() if tool.name == "navigate_to")
        navigate.invoke({"location": "kitchen"})
        runtime.finish("succeeded")
        return RobotAgentRuntime(settings, "now find something")

    def test_dry_run_persists_planned_pose_without_claiming_arrival(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root, execute_ros2=False)
            restarted = self._navigate_and_restart(settings, build_ros2_adapter(settings),)

            expected = {"x": 1.5, "y": -0.5, "yaw": 0.25, "frame_id": "map"}
            self.assertEqual(restarted.state.robot_state.navigation_status, "planned")
            self.assertEqual(restarted.state.robot_state.last_planned_pose.to_dict(), expected)
            self.assertIsNone(restarted.state.robot_state.pose)
            self.assertEqual(restarted.state.visited_locations, [])

    def test_successful_navigation_persists_confirmed_pose_and_visit(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root, execute_ros2=True)
            restarted = self._navigate_and_restart(settings, SuccessfulRosAdapter())

            expected = {"x": 1.5, "y": -0.5, "yaw": 0.25, "frame_id": "map"}
            self.assertEqual(restarted.state.robot_state.navigation_status, "succeeded")
            self.assertIsNone(restarted.state.robot_state.last_planned_pose)
            self.assertEqual(restarted.state.robot_state.pose.to_dict(), expected)
            self.assertEqual(restarted.state.visited_locations, ["kitchen"])


if __name__ == "__main__":
    unittest.main()
