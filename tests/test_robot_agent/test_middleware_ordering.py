from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from robot_agent.agents.lead_agent import make_lead_agent
from robot_agent.config import RobotAgentSettings
from robot_agent.harness import build_lead_agent_middleware
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import ToolResult, ToolStatus


class OrderingModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "middleware-ordering-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class MiddlewareOrderingTest(unittest.TestCase):
    def test_termination_repair_runs_before_plan_completion_in_real_graph(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
            runtime = RobotAgentRuntime(
                RobotAgentSettings(
                    location_file=location_file,
                    run_directory=root / "runs",
                    trace=False,
                    max_no_progress_continuations=5,
                ),
                "stop safely",
            )
            runtime.state.plan = [
                {"description": "stop safely", "preferred_capability": "control", "status": "pending"}
            ]

            @tool
            def stop_robot() -> dict:
                """Stop the robot."""
                result = ToolResult(status=ToolStatus.SUCCESS)
                runtime.record_tool_result("stop_robot", {}, result)
                return result.to_dict()

            model = OrderingModel(
                [
                    AIMessage(content="truncated", response_metadata={"finish_reason": "length"}),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "stop_robot", "args": {}, "id": "stop", "type": "tool_call"}],
                    ),
                    AIMessage(content="complete"),
                ]
            )
            middleware = build_lead_agent_middleware(runtime)
            assembly = make_lead_agent(
                model=model,
                tools=[stop_robot],
                known_locations=["kitchen"],
                max_tool_calls=5,
                middleware=middleware,
            )

            assembly.agent.invoke(
                {"messages": [HumanMessage(content="stop safely")]},
                config={"recursion_limit": 12},
            )

            events = [
                json.loads(line)["type"]
                for line in runtime.journal.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertLess(events.index("model_length_capped"), events.index("plan_completion_reminder"))
            self.assertEqual(model._index, 3)
            self.assertEqual(runtime.state.plan[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
