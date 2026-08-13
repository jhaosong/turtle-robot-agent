from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter, build_ros2_adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class SuccessfulPoseAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self.actual_pose = Pose2D(
            x=pose.x + 0.03,
            y=pose.y - 0.42,
            yaw=pose.yaw - 0.05,
            frame_id=pose.frame_id,
        )
        return ToolResult(status=ToolStatus.SUCCESS, data={"target_pose": pose.to_dict()})

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"pose": self.actual_pose.to_dict()})

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class NavigateToPoseTest(unittest.TestCase):
    def _runtime(self, root: Path, *, execute_ros2: bool) -> RobotAgentRuntime:
        location_file = root / "locations.yaml"
        location_file.write_text("location1: [0.0, 0.0, 0.0]\n", encoding="utf-8")
        settings = RobotAgentSettings(
            location_file=location_file,
            run_directory=root / "runs",
            execute_ros2=execute_ros2,
            trace=False,
            max_no_progress_continuations=5,
        )
        return RobotAgentRuntime(settings, "navigate to x=5, y=3")

    @staticmethod
    def _tool(runtime: RobotAgentRuntime, adapter: Ros2Adapter):
        tools = RobotToolRegistry(runtime, adapter, bt_skill=object()).build()
        return {tool.name: tool for tool in tools}["navigate_to_pose"]

    def test_success_updates_confirmed_pose_without_named_visit(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory), execute_ros2=True)
            result = self._tool(runtime, SuccessfulPoseAdapter()).invoke(
                {"x": 5.0, "y": 3.0, "yaw": 1.57}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(
                runtime.state.robot_state.pose.to_dict(),
                {"x": 5.03, "y": 2.58, "yaw": 1.52, "frame_id": "map"},
            )
            self.assertAlmostEqual(result["data"]["position_error_m"], 0.421070, places=5)
            self.assertEqual(result["data"]["actual_pose"], runtime.state.robot_state.pose.to_dict())
            self.assertEqual(runtime.state.visited_locations, [])

    def test_success_without_observed_pose_is_not_confirmed(self):
        class UnobservablePoseAdapter(SuccessfulPoseAdapter):
            def get_pose(self) -> ToolResult:
                return ToolResult(status=ToolStatus.FAILED, error="TF unavailable")

        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory), execute_ros2=True)
            result = self._tool(runtime, UnobservablePoseAdapter()).invoke(
                {"x": 5.0, "y": 3.0}
            )

            self.assertEqual(result["status"], ToolStatus.FAILED.value)
            self.assertEqual(runtime.state.robot_state.navigation_status, "needs_verification")
            self.assertIsNone(runtime.state.robot_state.pose)

    def test_dry_run_records_planned_pose_only(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory), execute_ros2=False)
            result = self._tool(runtime, build_ros2_adapter(runtime.settings)).invoke(
                {"x": 2.0, "y": -1.0}
            )

            self.assertEqual(result["status"], ToolStatus.PLANNED.value)
            self.assertIsNone(runtime.state.robot_state.pose)
            self.assertEqual(runtime.state.robot_state.last_planned_pose.x, 2.0)

    def test_workspace_guard_rejects_out_of_bounds_coordinate(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory), execute_ros2=True)
            result = self._tool(runtime, SuccessfulPoseAdapter()).invoke(
                {"x": 50.0, "y": 3.0}
            )

            self.assertEqual(result["status"], ToolStatus.FAILED.value)
            self.assertIn("outside the configured safety workspace", result["error"])
            self.assertIsNone(runtime.state.robot_state.pose)


if __name__ == "__main__":
    unittest.main()
