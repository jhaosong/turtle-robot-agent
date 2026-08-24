from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from robot_agent.agents.lead_agent import make_lead_agent
from robot_agent.config import RobotAgentSettings
from robot_agent.middlewares import SequentialToolCallMiddleware
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.state import ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry


class NoopAdapter(Ros2Adapter):
    def navigate_to_pose(self, pose):
        return ToolResult(status=ToolStatus.SUCCESS)

    def stop_robot(self):
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self):
        return ToolResult(status=ToolStatus.SUCCESS, data={})

    def cancel_navigation(self):
        return ToolResult(status=ToolStatus.SUCCESS)


class TwoCallModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "two-call-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class SequentialToolCallsTest(unittest.TestCase):
    def _runtime(self, root: Path) -> RobotAgentRuntime:
        locations = root / "locations.yaml"
        locations.write_text("location1: [0.0, 0.0, 0.0]\n", encoding="utf-8")
        return RobotAgentRuntime(
            RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
                max_no_progress_continuations=10,
            ),
            "test sequential tools",
        )

    def test_middleware_keeps_only_first_tool_call(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory))
            middleware = SequentialToolCallMiddleware(runtime)
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "navigate_to",
                        "args": {"location": "location1"},
                        "id": "first",
                        "type": "tool_call",
                    },
                    {
                        "name": "navigate_to",
                        "args": {"location": "location2"},
                        "id": "second",
                        "type": "tool_call",
                    },
                ],
            )

            update = middleware.after_model({"messages": [message]}, runtime=None)
            cleaned = update["messages"][0]

            self.assertEqual(len(cleaned.tool_calls), 1)
            self.assertEqual(cleaned.tool_calls[0]["id"], "first")

    def test_real_agent_graph_executes_only_first_call_from_batch(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory))
            executed: list[str] = []

            @tool
            def navigate_to(location: str) -> dict:
                """Navigate to one named location."""
                executed.append(location)
                return ToolResult(status=ToolStatus.SUCCESS).to_dict()

            model = TwoCallModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "navigate_to",
                                "args": {"location": "first"},
                                "id": "first-call",
                                "type": "tool_call",
                            },
                            {
                                "name": "navigate_to",
                                "args": {"location": "second"},
                                "id": "second-call",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="complete"),
                ]
            )
            assembly = make_lead_agent(
                model=model,
                tools=[navigate_to],
                known_locations=["first", "second"],
                max_tool_calls=4,
                middleware=[SequentialToolCallMiddleware(runtime)],
            )

            assembly.agent.invoke(
                {"messages": [HumanMessage(content="move safely")]},
                config={"recursion_limit": 12},
            )

            self.assertEqual(executed, ["first"])

    def test_registry_serializes_concurrent_tool_operations(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory))
            registry = RobotToolRegistry(
                runtime,
                NoopAdapter(),
                bt_skill=object(),
            )
            state_lock = threading.Lock()
            active = 0
            max_active = 0

            def operation() -> ToolResult:
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return ToolResult(status=ToolStatus.SUCCESS)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        registry._execute,
                        f"test_tool_{index}",
                        {"index": index},
                        operation,
                    )
                    for index in range(4)
                ]
                for future in futures:
                    self.assertEqual(future.result().status, ToolStatus.SUCCESS)

            self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
