from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import robot_agent.harness as harness_module
from robot_agent.agents.lead_agent.planner import TaskPlan
from robot_agent.config import RobotAgentSettings
from robot_agent.ros import Ros2Adapter
from robot_agent.state import Pose2D, ToolResult, ToolStatus


class FakeRosAdapter(Ros2Adapter):
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


class SessionStateTest(unittest.TestCase):
    def test_semantic_state_survives_two_harness_invocations(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.5, -0.5, 0.25]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                execute_ros2=True,
                trace=False,
            )
            invocation_count = 0

            class FakeAgent:
                def __init__(self, tools):
                    self.tools = {tool.name: tool for tool in tools}

                def invoke(self, payload, config):
                    nonlocal invocation_count
                    invocation_count += 1
                    if invocation_count == 1:
                        self.tools["navigate_to"].invoke({"location": "kitchen"})
                        self.tools["inspect_for_color"].invoke({"color": "blue"})
                    return {"messages": [AIMessage(content="ok")]}

            def fake_make_lead_agent(**kwargs):
                return SimpleNamespace(
                    agent=FakeAgent(kwargs["tools"]),
                    tools=kwargs["tools"],
                    metadata={"test": True},
                    system_prompt="test",
                )

            with (
                patch.object(harness_module, "build_ros2_adapter", lambda settings, backend: FakeRosAdapter()),
                patch.object(harness_module, "make_lead_agent", fake_make_lead_agent),
                patch.object(
                    harness_module.LeadTaskPlanner,
                    "plan",
                    lambda self, goal, robot_state, available_capabilities, known_locations=None: TaskPlan(objective=goal),
                ),
            ):
                harness = harness_module.RoboticsAgentHarness(settings, model=object())
                first = harness.invoke("go to kitchen")
                # A new harness instance simulates a process-level restart; the
                # second run must reload semantic state from the session file.
                restarted_harness = harness_module.RoboticsAgentHarness(settings, model=object())
                second = restarted_harness.invoke("now find the blue box")

            expected_pose = {"x": 1.5, "y": -0.5, "yaw": 0.25, "frame_id": "map"}
            self.assertEqual(first["agent_state"]["robot_state"]["pose"], expected_pose)
            self.assertEqual(second["agent_state"]["robot_state"]["pose"], expected_pose)
            self.assertEqual(second["agent_state"]["robot_state"]["navigation_status"], "succeeded")
            self.assertIsNotNone(second["agent_state"]["robot_state"]["last_perception_at"])
            self.assertEqual(second["agent_state"]["visited_locations"], ["kitchen"])
            self.assertTrue((settings.run_directory / "sessions" / "default.json").exists())


if __name__ == "__main__":
    unittest.main()
