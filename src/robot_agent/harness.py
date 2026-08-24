"""Top-level orchestration: create a run, assemble dependencies, invoke agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from robot_agent.agents.lead_agent import make_lead_agent
from robot_agent.agents.lead_agent import LeadTaskPlanner
from robot_agent.config.settings import RobotAgentSettings
from robot_agent.goal_monitor import GoalMonitor
from robot_agent.models import load_chat_model
from robot_agent.middlewares import (
    ModelTerminationMiddleware,
    PlanCompletionMiddleware,
    SequentialToolCallMiddleware,
)
from robot_agent.ros import RclpyRos2Adapter
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.runtime.serialization import sanitize_text
from robot_agent.skills import BehaviorTreeSkill
from robot_agent.state import GoalBlocker, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry, load_locations


def build_bounded_goal(goal: str, plan: list[dict[str, Any]], robot_state: Any) -> str:
    """Delimit user text and serialize model context without Python repr syntax."""
    return (
        f"--- BEGIN USER GOAL ---\n{goal}\n--- END USER GOAL ---\n\n"
        "Structured task plan (advisory, must stay within tool policy):\n"
        f"<task_plan_json>\n{json.dumps(plan, ensure_ascii=False)}\n</task_plan_json>\n\n"
        "Current semantic state:\n"
        f"<robot_state_json>\n{json.dumps(robot_state.to_agent_context(), ensure_ascii=False)}"
        "\n</robot_state_json>"
    )


def build_lead_agent_middleware(runtime: RobotAgentRuntime) -> list[Any]:
    """Return registration order whose reverse after-model order repairs output first."""
    perception_steps = sum(
        step.get("preferred_capability") == "perception"
        for step in runtime.state.plan
    )
    reminder_budget = perception_steps if perception_steps >= 4 else 2
    return [
        PlanCompletionMiddleware(
            runtime,
            max_reminders=min(reminder_budget, runtime.settings.max_tool_calls),
        ),
        SequentialToolCallMiddleware(runtime),
        ModelTerminationMiddleware(runtime),
    ]


def build_verified_final_response(runtime: RobotAgentRuntime, run_status: str) -> str:
    """Render the user-facing result from execution evidence, not model prose."""
    evaluation = runtime.state.goal_evaluation
    if evaluation is None:
        return "Execution ended without a goal evaluation."

    successful_tools = [
        entry["tool"]
        for entry in runtime.state.tool_history
        if entry["result"]["status"] == "success"
    ]
    lines = [f"Status: {run_status}.", f"Verification: {evaluation.reason}."]
    if successful_tools:
        lines.append(f"Completed tools: {' -> '.join(successful_tools)}.")

    successful_search = next(
        (
            entry["result"]["data"]
            for entry in reversed(runtime.state.tool_history)
            if entry["tool"] == "search_for_object"
            and entry["result"]["status"] == ToolStatus.SUCCESS.value
            and isinstance(entry["result"]["data"].get("found"), dict)
        ),
        None,
    )
    if successful_search is not None:
        found = successful_search["found"]
        lines.append(
            "Detected object: "
            f"label={found['label']}, confidence={float(found['confidence']):.3f}."
        )
        image_position = successful_search.get("image_position")
        if isinstance(image_position, dict):
            lines.append(
                "Image position: "
                f"x={float(image_position['x_px']):.1f}px, "
                f"y={float(image_position['y_px']):.1f}px, "
                f"normalized=({float(image_position['x_normalized']):.3f}, "
                f"{float(image_position['y_normalized']):.3f})."
            )
        observation_pose = successful_search.get("observation_pose")
        if isinstance(observation_pose, dict):
            lines.append(
                "Robot pose at detection: "
                f"x={observation_pose['x']}, y={observation_pose['y']}, "
                f"yaw={observation_pose['yaw']}, frame={observation_pose['frame_id']}."
            )

    successful_inspection = next(
        (
            entry["result"]["data"]
            for entry in reversed(runtime.state.tool_history)
            if entry["tool"] == "circle_object_for_inspection"
            and entry["result"]["status"] == ToolStatus.SUCCESS.value
        ),
        None,
    )
    if successful_inspection is not None:
        object_position = successful_inspection.get("object_position")
        if isinstance(object_position, dict):
            lines.append(
                "Triangulated object position: "
                f"x={float(object_position['x']):.3f}, "
                f"y={float(object_position['y']):.3f}, "
                f"frame={object_position['frame_id']}."
            )
        captures = successful_inspection.get("captures") or []
        image_paths = [
            item.get("image_path")
            for item in captures
            if isinstance(item, dict) and item.get("image_path")
        ]
        lines.append(
            f"Verified inspection viewpoints: {len(captures)}; "
            f"saved images: {len(image_paths)}."
        )

    pose = evaluation.evidence.get("pose")
    if isinstance(pose, dict):
        lines.append(
            "Confirmed pose: "
            f"x={pose['x']}, y={pose['y']}, yaw={pose['yaw']}, frame={pose['frame_id']}."
        )
    question = evaluation.evidence.get("question")
    if question:
        lines.append(f"Clarification needed: {question}")
    return "\n".join(lines)


class RoboticsAgentHarness:
    """DeerFlow-style harness: agent assembly remains separate from run lifecycle."""

    def __init__(self, settings: RobotAgentSettings, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model

    def invoke(self, goal: str) -> dict[str, Any]:
        # Some remote terminals can decode malformed input as lone surrogates,
        # which are invalid in UTF-8 and unsafe to send to model providers.
        goal = sanitize_text(goal)
        runtime = RobotAgentRuntime(self.settings, goal)
        locations = load_locations(self.settings.location_file)
        model = self.model or load_chat_model(streaming=False)
        ros = RclpyRos2Adapter(self.settings)
        bt_skill = BehaviorTreeSkill(
            model=model,
            known_locations=set(locations),
            output_directory=runtime.run_path / "behavior_tree",
            navigation_retry_count=self.settings.bt_navigation_retries,
        )
        registry = RobotToolRegistry(runtime, ros, bt_skill)
        try:
            task_plan = LeadTaskPlanner(model).plan(
                goal,
                runtime.state.robot_state.to_agent_context(),
                available_capabilities={"navigation", "perception", "behavior_tree", "control"},
                known_locations=sorted(locations),
            )
            runtime.state.plan = [
                {**step.model_dump(), "status": "pending"}
                for step in task_plan.steps
            ]
            runtime.state.goal_requirements = {
                "requires_perception": task_plan.requires_perception,
                "requested_colors": task_plan.requested_colors,
                "requested_labels": task_plan.requested_labels,
            }
            runtime.emit("task_planned", task_plan.model_dump())
        except Exception as exc:
            # Planning is an aid, not a prerequisite for a safe tool call.
            runtime.emit("task_planning_unavailable", {"error_type": type(exc).__name__})
        assembly = make_lead_agent(
            model=model,
            tools=registry.build(),
            known_locations=sorted(locations),
            max_tool_calls=self.settings.max_tool_calls,
            # LangChain runs after_model hooks in reverse registration order;
            # termination repair must see raw provider output before plan logic.
            middleware=build_lead_agent_middleware(runtime),
        )
        runtime.emit("agent_assembled", assembly.metadata)
        bounded_goal = build_bounded_goal(goal, runtime.state.plan, runtime.state.robot_state)
        try:
            output = assembly.agent.invoke(
                {"messages": [HumanMessage(content=bounded_goal)]},
                # A ReAct turn is model -> tool -> model. Keep one extra turn
                # for the terminal response while enforcing the configured cap.
                config={
                    "metadata": assembly.metadata,
                    # Each model/tool cycle traverses three after-model guards.
                    # Tool-call count remains bounded independently by runtime.
                    "recursion_limit": 3 * self.settings.max_tool_calls + 4,
                },
            )
        except Exception as exc:
            runtime.emit("agent_failed", {"error_type": type(exc).__name__})
            try:
                stop_result = ros.stop_robot()
                runtime.emit(
                    "agent_failure_stop",
                    {"result": stop_result.to_dict()},
                    category="control",
                )
            except Exception as stop_exc:
                runtime.emit(
                    "agent_failure_stop_failed",
                    {"error_type": type(stop_exc).__name__},
                    category="control",
                )
            runtime.finish("failed")
            raise RuntimeError(f"Lead agent invocation failed: {type(exc).__name__}") from exc
        finally:
            ros.close()

        messages = output.get("messages", []) if isinstance(output, dict) else []
        model_response = messages[-1].content if messages else "Agent produced no final response."
        evaluation = GoalMonitor().evaluate(runtime.state)
        runtime.state.goal_evaluation = evaluation
        runtime.emit("goal_evaluated", evaluation.to_dict(), category="verification")
        if runtime.state.model_stop_reason == "model_length_capped":
            run_status = "length_capped"
        elif runtime.state.model_stop_reason == "model_safety_capped":
            run_status = "safety_capped"
        elif evaluation.satisfied:
            run_status = "succeeded"
        elif evaluation.blocker in {GoalBlocker.RUN_FAILED, GoalBlocker.NO_PROGRESS}:
            run_status = "failed"
        elif evaluation.blocker == GoalBlocker.NEEDS_USER_INPUT:
            run_status = "needs_input"
        else:
            run_status = "needs_verification"
        final = build_verified_final_response(runtime, run_status)
        runtime.finish(run_status)
        return {
            "run_id": runtime.state.run_id,
            "run_directory": str(runtime.run_path),
            "final_response": final,
            "model_response": model_response,
            "tool_history": runtime.state.tool_history,
            "agent_state": runtime.state.to_agent_context(),
            "run_status": run_status,
            "clarification_question": evaluation.evidence.get("question")
            if evaluation.blocker == GoalBlocker.NEEDS_USER_INPUT
            else None,
        }
