"""Force one physical robot action decision per model turn."""

from __future__ import annotations

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from robot_agent.runtime.runtime import RobotAgentRuntime


class SequentialToolCallMiddleware(AgentMiddleware[AgentState]):
    """Keep only the first tool call so results are observed before replanning."""

    def __init__(self, robot_runtime: RobotAgentRuntime) -> None:
        super().__init__()
        self.robot_runtime = robot_runtime

    def after_model(self, state: AgentState, runtime) -> dict | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        tool_calls = list(message.tool_calls or [])
        if len(tool_calls) <= 1:
            return None

        additional = dict(message.additional_kwargs or {})
        additional.pop("tool_calls", None)
        additional.pop("function_call", None)
        retained = tool_calls[0]
        self.robot_runtime.emit(
            "parallel_tool_calls_suppressed",
            {
                "retained_tool": retained.get("name"),
                "suppressed_tools": [item.get("name") for item in tool_calls[1:]],
            },
            category="control",
        )
        return {
            "messages": [
                message.model_copy(
                    update={
                        "tool_calls": [retained],
                        "additional_kwargs": additional,
                    }
                )
            ]
        }
