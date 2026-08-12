"""Repair provider-capped model responses before they can dispatch ROS tools."""

from __future__ import annotations

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from robot_agent.runtime.runtime import RobotAgentRuntime


_LENGTH_REASONS = {"length", "max_tokens", "max_output_tokens", "model_length"}
_SAFETY_REASONS = {"content_filter", "content_filtered", "safety", "refusal", "blocked"}


def _finish_reason(message: AIMessage) -> str | None:
    metadata = message.response_metadata or {}
    additional = message.additional_kwargs or {}
    for key in ("finish_reason", "stop_reason"):
        value = metadata.get(key, additional.get(key))
        if value is not None:
            return str(value).strip().lower()
    return None


def _has_raw_tool_intent(message: AIMessage) -> bool:
    additional = message.additional_kwargs or {}
    return bool(
        message.tool_calls
        or getattr(message, "invalid_tool_calls", None)
        or additional.get("tool_calls")
        or additional.get("function_call")
    )


def _append_explanation(content, explanation: str):
    if not content:
        return explanation
    if isinstance(content, list):
        return [*content, {"type": "text", "text": f"\n\n{explanation}"}]
    return f"{content}\n\n{explanation}"


class ModelTerminationMiddleware(AgentMiddleware[AgentState]):
    """Mark length caps and suppress safety/partial tool calls deterministically."""

    def __init__(self, robot_runtime: RobotAgentRuntime) -> None:
        super().__init__()
        self.robot_runtime = robot_runtime

    def _clean_tool_calls(self, message: AIMessage, reason: str) -> AIMessage:
        additional = dict(message.additional_kwargs or {})
        additional.pop("tool_calls", None)
        additional.pop("function_call", None)
        additional["model_termination"] = {
            "reason": reason,
            "suppressed_tool_calls": len(message.tool_calls or []),
        }
        explanation = (
            f"The model provider stopped this response ({reason}). "
            "Any incomplete tool calls were suppressed and were not executed."
        )
        return message.model_copy(
            update={
                "content": _append_explanation(message.content, explanation),
                "tool_calls": [],
                "invalid_tool_calls": [],
                "additional_kwargs": additional,
            }
        )

    def after_model(self, state: AgentState, runtime) -> dict | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        reason = _finish_reason(message)
        if reason in _LENGTH_REASONS:
            self.robot_runtime.state.model_stop_reason = "model_length_capped"
            self.robot_runtime.emit(
                "model_length_capped",
                {"finish_reason": reason},
                category="model",
            )
            self.robot_runtime.save_checkpoint()
            if _has_raw_tool_intent(message):
                return {"messages": [self._clean_tool_calls(message, reason)]}
            return None
        if reason in _SAFETY_REASONS:
            self.robot_runtime.state.model_stop_reason = "model_safety_capped"
            self.robot_runtime.emit(
                "model_safety_capped",
                {"finish_reason": reason, "tool_calls_suppressed": _has_raw_tool_intent(message)},
                category="model",
            )
            self.robot_runtime.save_checkpoint()
            if _has_raw_tool_intent(message) or not message.content:
                return {"messages": [self._clean_tool_calls(message, reason)]}
        return None
