"""Track advisory plan completion and prevent clean premature agent exits."""

from __future__ import annotations

from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from robot_agent.runtime.runtime import RobotAgentRuntime
from robot_agent.state import ToolStatus


def _has_tool_intent(message: AIMessage) -> bool:
    return bool(
        message.tool_calls
        or getattr(message, "invalid_tool_calls", None)
        or (message.additional_kwargs or {}).get("tool_calls")
        or (message.additional_kwargs or {}).get("function_call")
    )


class PlanCompletionMiddleware(AgentMiddleware[AgentState]):
    """Give an incomplete plan two hidden continuation turns before exit."""

    def __init__(self, robot_runtime: RobotAgentRuntime, max_reminders: int = 2) -> None:
        super().__init__()
        self.robot_runtime = robot_runtime
        self.max_reminders = max_reminders
        self._reminder_count = 0
        self._pending_reminder: str | None = None

    def _pending_steps(self) -> list[dict]:
        return [step for step in self.robot_runtime.state.plan if step.get("status") != "completed"]

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime) -> dict | None:
        messages = state.get("messages") or []
        last_ai = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
        if last_ai is None or _has_tool_intent(last_ai):
            return None
        last_result = self.robot_runtime.state.last_tool_result
        if last_result is not None and last_result.status in {
            ToolStatus.NEEDS_INPUT,
            ToolStatus.PLANNED,
        }:
            return None
        pending = self._pending_steps()
        if not pending:
            return None
        run_state = self.robot_runtime.state
        if self._reminder_count >= self.max_reminders or run_state.continuation_count >= run_state.max_continuations:
            return None
        lines = "\n".join(f"- {step.get('description', 'unnamed step')}" for step in pending)
        self._pending_reminder = (
            "The advisory task plan still has incomplete steps. Continue with a safe valid tool call, "
            "or report the concrete blocker if no valid action exists:\n" + lines
        )
        self._reminder_count += 1
        self.robot_runtime.emit(
            "plan_completion_reminder",
            {"pending_steps": len(pending), "reminder_count": self._reminder_count},
            category="control",
        )
        return {"jump_to": "model"}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        if self._pending_reminder is None:
            return handler(request)
        reminder = self._pending_reminder
        self._pending_reminder = None
        augmented = request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=reminder,
                    name="plan_completion_reminder",
                    additional_kwargs={"hide_from_ui": True},
                ),
            ]
        )
        return handler(augmented)
