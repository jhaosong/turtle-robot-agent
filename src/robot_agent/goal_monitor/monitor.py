"""Deterministic goal verification inspired by DeerFlow's goal evaluator.

This monitor deliberately evaluates execution evidence instead of trusting the
lead agent's natural-language final response. It is conservative: unavailable
observations and dry-run commands can never be marked successful.
"""

from __future__ import annotations

from robot_agent.state import (
    GoalBlocker,
    GoalEvaluation,
    RunState,
    ToolStatus,
    goal_requirements_satisfied,
    matching_goal_detections,
)


class GoalMonitor:
    def evaluate(self, state: RunState) -> GoalEvaluation:
        if not state.tool_history:
            return GoalEvaluation(False, GoalBlocker.MISSING_EVIDENCE, "No tool was executed for the goal")

        if state.no_progress_count >= state.max_no_progress_continuations:
            return GoalEvaluation(
                False,
                GoalBlocker.NO_PROGRESS,
                "Execution stopped after repeated tool calls produced no semantic progress",
                {"count": state.no_progress_count, "limit": state.max_no_progress_continuations},
            )

        clarification = next(
            (
                entry
                for entry in reversed(state.tool_history)
                if entry["result"]["status"] == ToolStatus.NEEDS_INPUT.value
            ),
            None,
        )
        if clarification is not None:
            return GoalEvaluation(
                False,
                GoalBlocker.NEEDS_USER_INPUT,
                clarification["result"]["data"].get("reason", "User clarification is required"),
                {"question": clarification["result"]["data"].get("question")},
            )

        if any(entry["result"]["status"] in {ToolStatus.FAILED.value, ToolStatus.TIMEOUT.value, ToolStatus.CANCELED.value} for entry in state.tool_history):
            return GoalEvaluation(False, GoalBlocker.RUN_FAILED, "At least one executed tool failed", {"failures": state.failures})

        if any(entry["result"]["status"] == ToolStatus.PLANNED.value for entry in state.tool_history):
            return GoalEvaluation(False, GoalBlocker.EXTERNAL_WAIT, "ROS2 commands were planned but not executed")

        pending_steps = [step for step in state.plan if step.get("status", "pending") != "completed"]
        if pending_steps:
            return GoalEvaluation(
                False,
                GoalBlocker.MISSING_EVIDENCE,
                "Execution ended with incomplete planned steps",
                {"pending_steps": pending_steps},
            )

        if state.goal_requirements.get("requires_perception", False):
            if not state.robot_state.last_perception_at:
                return GoalEvaluation(False, GoalBlocker.MISSING_EVIDENCE, "Goal requires perception but no semantic observation exists")
            if not goal_requirements_satisfied(
                state.goal_requirements,
                state.robot_state.visible_objects,
            ):
                return GoalEvaluation(False, GoalBlocker.GOAL_NOT_MET_YET, "Perception ran but did not verify the requested object")
            matches = matching_goal_detections(
                state.goal_requirements,
                state.robot_state.visible_objects,
            )
            return GoalEvaluation(
                True,
                GoalBlocker.NONE,
                "Perception verified the requested semantic object evidence",
                {"matches": [item.to_dict() for item in matches]},
            )

        last_result = state.last_tool_result
        if last_result is None or last_result.status != ToolStatus.SUCCESS:
            return GoalEvaluation(False, GoalBlocker.MISSING_EVIDENCE, "No successful terminal tool result exists")

        if state.robot_state.navigation_status == "succeeded" and state.robot_state.pose is not None:
            return GoalEvaluation(
                True,
                GoalBlocker.NONE,
                "Navigation completed with a semantic pose update",
                {"pose": state.robot_state.pose.to_dict()},
            )
        if state.tool_history[-1]["tool"] == "run_behavior_tree":
            node_results = last_result.data.get("node_results", [])
            if node_results and all(item["result"]["status"] == ToolStatus.SUCCESS.value for item in node_results):
                return GoalEvaluation(True, GoalBlocker.NONE, "All behavior tree nodes completed successfully")

        return GoalEvaluation(False, GoalBlocker.MISSING_EVIDENCE, "Execution finished without independently verifiable goal evidence")
