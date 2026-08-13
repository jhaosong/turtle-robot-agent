from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class RelativeMotionAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.pose = Pose2D(2.0, 3.0, math.pi / 2.0)
        self.sent_pose = None

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self.sent_pose = pose
        self.pose = pose
        return ToolResult(status=ToolStatus.SUCCESS, data={"target_pose": pose.to_dict()})

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"pose": self.pose.to_dict()})

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class MoveRelativeTest(unittest.TestCase):
    def test_uses_live_pose_and_heading_to_compute_map_target(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("location1: [0, 0, 0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    execute_ros2=True,
                    trace=False,
                    max_no_progress_continuations=5,
                ),
                "move forward one meter",
            )
            adapter = RelativeMotionAdapter()
            tool = {
                tool.name: tool
                for tool in RobotToolRegistry(runtime, adapter, bt_skill=object()).build()
            }["move_relative"]

            result = tool.invoke({"distance_m": 1.0})

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertAlmostEqual(adapter.sent_pose.x, 2.0)
            self.assertAlmostEqual(adapter.sent_pose.y, 4.0)
            self.assertAlmostEqual(adapter.sent_pose.yaw, math.pi / 2.0)
            self.assertEqual(result["data"]["start_pose"]["x"], 2.0)
            self.assertEqual(result["data"]["computed_target_pose"]["y"], 4.0)

    def test_rejects_distance_outside_bounded_relative_motion(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("location1: [0, 0, 0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                ),
                "move forward",
            )
            tool = {
                tool.name: tool
                for tool in RobotToolRegistry(
                    runtime,
                    RelativeMotionAdapter(),
                    bt_skill=object(),
                ).build()
            }["move_relative"]

            with self.assertRaises(Exception):
                tool.invoke({"distance_m": 3.0})


if __name__ == "__main__":
    unittest.main()
