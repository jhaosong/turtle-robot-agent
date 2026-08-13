"""Own the lifecycle of one goal execution.

This is deliberately separate from the agent factory, following DeerFlow's
agent/runtime split. The runtime is responsible for state and persistence;
the lead agent only reasons and selects tools.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.state import RunState, SemanticSessionState, ToolResult, ToolStatus

from .checkpoint import JsonCheckpointStore
from .events import RuntimeEvent
from .journal import RunJournal


class RobotAgentRuntime:
    _TOOL_CAPABILITIES = {
        "navigate_to": "navigation",
        "navigate_to_pose": "navigation",
        "move_relative": "navigation",
        "inspect_for_color": "perception",
        "search_for_object": "perception",
        "find_object": "perception",
        "run_behavior_tree": "behavior_tree",
        "stop_robot": "control",
        "wait_seconds": "control",
    }
    def __init__(self, settings: RobotAgentSettings, goal: str) -> None:
        self.settings = settings
        run_id = uuid4().hex[:12]
        self.run_path = Path(settings.run_directory) / run_id
        self.session_checkpoint = JsonCheckpointStore(
            Path(settings.run_directory) / "sessions" / f"{settings.session_id}.json"
        )
        session_payload = self.session_checkpoint.load()
        semantic_state = (
            SemanticSessionState.from_snapshot(session_payload)
            if session_payload is not None
            else SemanticSessionState()
        )
        self.state = RunState(
            run_id=run_id,
            goal=goal,
            robot_state=semantic_state.robot_state,
            visited_locations=semantic_state.visited_locations,
            max_continuations=settings.max_continuations,
            max_no_progress_continuations=settings.max_no_progress_continuations,
        )
        self._last_progress_signature = self.state.progress_signature()
        self._no_progress_event_emitted = False
        self.journal = RunJournal(self.run_path / "events.jsonl", run_id)
        self.checkpoint = JsonCheckpointStore(self.run_path / "checkpoint.json")
        self.emit("run_started", {"goal": goal, "execute_ros2": settings.execute_ros2})
        self.save_checkpoint()

    def emit(self, event_type: str, payload: dict, category: str = "lifecycle") -> None:
        event = RuntimeEvent(run_id=self.state.run_id, type=event_type, payload=payload, category=category)
        self.journal.append(event)
        if self.settings.trace:
            print(f"[ROBOT AGENT] {event_type}: {payload}")

    def record_tool_result(self, tool_name: str, arguments: dict, result: ToolResult) -> None:
        self.state.continuation_count += 1
        current_signature = self.state.progress_signature()
        made_progress = result.status == ToolStatus.SUCCESS and current_signature != self._last_progress_signature
        if made_progress:
            self.state.no_progress_count = 0
        else:
            self.state.no_progress_count += 1
        self._last_progress_signature = current_signature
        if self.state.no_progress_count >= self.state.max_no_progress_continuations:
            self.state.status = "no_progress"
        self._update_plan(tool_name, result)
        entry = {"tool": tool_name, "arguments": arguments, "result": result.to_dict()}
        self.state.tool_history.append(entry)
        self.state.last_tool_result = result
        if result.status.value in {"failed", "timeout", "canceled"} and result.error:
            self.state.failures.append(result.error)
        self.emit("tool_completed", entry, category="tool")
        if self.state.status == "no_progress" and not self._no_progress_event_emitted:
            self.emit(
                "no_progress_limit_reached",
                {
                    "count": self.state.no_progress_count,
                    "limit": self.state.max_no_progress_continuations,
                },
                category="control",
            )
            self._no_progress_event_emitted = True
        self.save_checkpoint()

    def _update_plan(self, tool_name: str, result: ToolResult) -> None:
        """Update one closest capability match at the single tool-result choke point."""
        capability = self._TOOL_CAPABILITIES.get(tool_name)
        if capability is None:
            return
        step = next(
            (
                candidate
                for candidate in self.state.plan
                if candidate.get("status", "pending") != "completed"
                and candidate.get("preferred_capability") == capability
            ),
            None,
        )
        if step is None:
            return
        step["last_tool"] = tool_name
        step["last_result_status"] = result.status.value
        completed = result.status == ToolStatus.SUCCESS
        if (
            completed
            and capability == "perception"
            and self.state.goal_requirements.get("requires_perception", False)
        ):
            requested_colors = set(
                self.state.goal_requirements.get("requested_colors") or []
            )
            completed = any(
                not requested_colors or detection.color in requested_colors
                for detection in self.state.robot_state.visible_objects
            )
        if completed:
            step["status"] = "completed"
            step["completed_at"] = result.timestamp
        else:
            step["status"] = "pending"
        self.emit(
            "plan_step_updated",
            {"description": step.get("description"), "status": step["status"], "tool": tool_name},
            category="plan",
        )

    @property
    def no_progress_exhausted(self) -> bool:
        return self.state.no_progress_count >= self.state.max_no_progress_continuations

    def save_checkpoint(self) -> None:
        self.checkpoint.save(self.state.to_snapshot())
        self.session_checkpoint.save(self.state.to_semantic_session_state().to_snapshot())

    def finish(self, status: str) -> None:
        self.state.status = status
        self.save_checkpoint()
        self.emit("run_finished", {"status": status, "tool_calls": len(self.state.tool_history)})
