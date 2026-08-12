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


class NoopRosAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={})

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class NoProgressTest(unittest.TestCase):
    def test_unchanged_successes_trigger_no_progress_blocker(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                execute_ros2=True,
                trace=False,
                max_tool_calls=12,
                max_no_progress_continuations=3,
            )

            class RepeatingAgent:
                def __init__(self, tools):
                    self.tool = {tool.name: tool for tool in tools}["get_known_locations"]

                def invoke(self, payload, config):
                    for _ in range(settings.max_no_progress_continuations):
                        self.tool.invoke({})
                    return {"messages": [AIMessage(content="done")]} 

            def fake_make_lead_agent(**kwargs):
                return SimpleNamespace(
                    agent=RepeatingAgent(kwargs["tools"]),
                    tools=kwargs["tools"],
                    metadata={"test": True},
                    system_prompt="test",
                )

            with (
                patch.object(harness_module, "build_ros2_adapter", lambda settings, backend: NoopRosAdapter()),
                patch.object(harness_module, "make_lead_agent", fake_make_lead_agent),
                patch.object(
                    harness_module.LeadTaskPlanner,
                    "plan",
                    lambda self, goal, robot_state, available_capabilities, known_locations=None: TaskPlan(objective=goal),
                ),
            ):
                result = harness_module.RoboticsAgentHarness(settings, model=object()).invoke("inspect known locations")

            self.assertEqual(len(result["tool_history"]), 3)
            self.assertEqual(result["run_status"], "failed")
            self.assertEqual(result["agent_state"]["goal_evaluation"]["blocker"], "no_progress")
            self.assertLess(len(result["tool_history"]), settings.max_tool_calls)


if __name__ == "__main__":
    unittest.main()
