from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import robot_agent.harness as harness_module
from robot_agent.agents.lead_agent.planner import PlannedStep, TaskPlan
from robot_agent.config import RobotAgentSettings
from robot_agent.middlewares.plan_completion import PlanCompletionMiddleware
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus


class SequencedToolModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "sequenced-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class PlanTestRosAdapter(Ros2Adapter):
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
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"detections": [{"label": "colored_object", "color": color, "confidence": 0.9}]},
        )


class PlanCompletionTest(unittest.TestCase):
    def test_empty_perception_does_not_complete_find_step(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("location1: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                    max_no_progress_continuations=5,
                ),
                "find green in the room",
            )
            runtime.state.goal_requirements = {
                "requires_perception": True,
                "requested_colors": ["green"],
            }
            runtime.state.plan = [
                {
                    "description": "search for green",
                    "preferred_capability": "perception",
                    "status": "pending",
                }
            ]

            runtime.record_tool_result(
                "inspect_for_color",
                {"color": "green"},
                ToolResult(status=ToolStatus.SUCCESS, data={"matches": []}),
            )
            self.assertEqual(runtime.state.plan[0]["status"], "pending")

            runtime.state.robot_state.visible_objects = [
                Detection("colored_object", 0.9, color="green")
            ]
            runtime.record_tool_result(
                "search_for_object",
                {"route": ["location1"], "color": "green"},
                ToolResult(status=ToolStatus.SUCCESS, data={"found": {}}),
            )
            self.assertEqual(runtime.state.plan[0]["status"], "completed")

    def test_planned_external_action_does_not_force_continuation(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("location1: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                    max_no_progress_continuations=5,
                ),
                "navigate to location1",
            )
            runtime.state.plan = [
                {"description": "navigate", "preferred_capability": "navigation", "status": "pending"}
            ]
            runtime.record_tool_result("navigate_to", {}, ToolResult(status=ToolStatus.PLANNED))

            update = PlanCompletionMiddleware(runtime).after_model(
                {"messages": [AIMessage(content="Navigation command was planned.")]},
                runtime=None,
            )

            self.assertIsNone(update)

    def test_two_reminders_then_returns_clean_terminal_response(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                execute_ros2=True,
                trace=False,
                max_no_progress_continuations=6,
            )
            model = SequencedToolModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "navigate_to",
                                "args": {"location": "kitchen"},
                                "id": "nav",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="first premature answer"),
                    AIMessage(content="second premature answer"),
                    AIMessage(content="terminal answer after reminder budget"),
                ]
            )
            task_plan = TaskPlan(
                objective="three steps",
                steps=[
                    PlannedStep(description="navigate", preferred_capability="navigation"),
                    PlannedStep(description="inspect", preferred_capability="perception"),
                    PlannedStep(description="stop", preferred_capability="control"),
                ],
            )
            with (
                patch.object(harness_module, "build_ros2_adapter", lambda settings, backend: PlanTestRosAdapter()),
                patch.object(
                    harness_module.LeadTaskPlanner,
                    "plan",
                    lambda self, goal, robot_state, available_capabilities, known_locations=None: task_plan,
                ),
            ):
                result = harness_module.RoboticsAgentHarness(settings, model=model).invoke("start only")

            self.assertEqual(model._index, 4)
            self.assertEqual(result["model_response"], "terminal answer after reminder budget")
            self.assertNotIn("advisory task plan", result["final_response"])
            self.assertEqual(sum(step["status"] == "completed" for step in result["agent_state"]["plan"]), 1)

    def test_real_agent_graph_continues_after_premature_final_answer(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                execute_ros2=True,
                trace=False,
                max_no_progress_continuations=6,
            )
            model = SequencedToolModel(
                [
                    AIMessage(content="", tool_calls=[{"name": "navigate_to", "args": {"location": "kitchen"}, "id": "nav", "type": "tool_call"}]),
                    AIMessage(content="premature final"),
                    AIMessage(content="", tool_calls=[{"name": "stop_robot", "args": {}, "id": "stop", "type": "tool_call"}]),
                    AIMessage(content="still premature"),
                    AIMessage(content="", tool_calls=[{"name": "inspect_for_color", "args": {"color": "blue"}, "id": "inspect", "type": "tool_call"}]),
                    AIMessage(content="all steps complete"),
                ]
            )
            task_plan = TaskPlan(
                objective="navigate, stop, inspect",
                requires_perception=True,
                requested_colors=["blue"],
                steps=[
                    PlannedStep(description="navigate to kitchen", preferred_capability="navigation"),
                    PlannedStep(description="stop safely", preferred_capability="control"),
                    PlannedStep(description="inspect blue target", preferred_capability="perception"),
                ],
            )
            with (
                patch.object(harness_module, "build_ros2_adapter", lambda settings, backend: PlanTestRosAdapter()),
                patch.object(
                    harness_module.LeadTaskPlanner,
                    "plan",
                    lambda self, goal, robot_state, available_capabilities, known_locations=None: task_plan,
                ),
            ):
                result = harness_module.RoboticsAgentHarness(settings, model=model).invoke("run all three steps")

            self.assertEqual(model._index, 6)
            self.assertEqual(result["run_status"], "succeeded")
            self.assertTrue(all(step["status"] == "completed" for step in result["agent_state"]["plan"]))

    def test_incomplete_plan_forces_hidden_continuation_turn(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                    max_no_progress_continuations=10,
                ),
                "navigate, inspect, then stop",
            )
            runtime.state.plan = [
                {"description": "navigate", "preferred_capability": "navigation", "status": "pending"},
                {"description": "inspect", "preferred_capability": "perception", "status": "pending"},
                {"description": "stop", "preferred_capability": "control", "status": "pending"},
            ]
            runtime.record_tool_result("navigate_to", {}, ToolResult(status=ToolStatus.SUCCESS))
            self.assertEqual(runtime.state.plan[0]["status"], "completed")
            self.assertEqual(runtime.state.plan[1]["status"], "pending")

            middleware = PlanCompletionMiddleware(runtime)
            state = {"messages": [AIMessage(content="I am done")]} 
            update = middleware.after_model(state, runtime=None)
            self.assertEqual(update, {"jump_to": "model"})

            captured_messages = []

            def handler(request):
                captured_messages.extend(request.messages)
                return ModelResponse(result=[AIMessage(content="continuing")])

            request = ModelRequest(model=object(), messages=[HumanMessage(content="goal")])
            middleware.wrap_model_call(request, handler)

            reminder = captured_messages[-1]
            self.assertIsInstance(reminder, HumanMessage)
            self.assertTrue(reminder.additional_kwargs["hide_from_ui"])
            self.assertIn("inspect", reminder.content)
            self.assertIn("stop", reminder.content)


if __name__ == "__main__":
    unittest.main()
