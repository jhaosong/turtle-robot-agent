from __future__ import annotations

from datetime import datetime, timezone
import unittest

import robot_agent.goal_monitor.monitor as monitor_module
import robot_agent.runtime.runtime as runtime_module
import robot_agent.state as state_module
from robot_agent.agents.lead_agent.planner import TaskPlan
from robot_agent.goal_monitor import GoalMonitor
from robot_agent.state import (
    Detection,
    GoalBlocker,
    RunState,
    ToolResult,
    ToolStatus,
    goal_requirements_satisfied,
)


class StructuredGoalRequirementsTest(unittest.TestCase):
    def test_label_only_goal_rejects_unrelated_detection(self):
        requirements = {
            "requires_perception": True,
            "requested_labels": ["fire_extinguisher"],
        }
        visible = [Detection(label="trash_can", confidence=0.9)]

        self.assertFalse(goal_requirements_satisfied(requirements, visible))

        state = RunState(run_id="test", goal="find the fire extinguisher")
        state.goal_requirements = requirements
        state.robot_state.last_perception_at = datetime.now(timezone.utc).isoformat()
        state.robot_state.visible_objects = visible
        successful_result = ToolResult(status=ToolStatus.SUCCESS)
        state.tool_history = [
            {
                "tool": "search_for_object",
                "arguments": {"label": "fire_extinguisher"},
                "result": successful_result.to_dict(),
            }
        ]
        state.last_tool_result = successful_result

        evaluation = GoalMonitor().evaluate(state)
        self.assertFalse(evaluation.satisfied)
        self.assertEqual(evaluation.blocker, GoalBlocker.GOAL_NOT_MET_YET)

    def test_label_only_goal_accepts_matching_detection(self):
        requirements = {
            "requires_perception": True,
            "requested_labels": ["fire_extinguisher"],
        }
        visible = [Detection(label="fire_extinguisher", confidence=0.9)]

        self.assertTrue(goal_requirements_satisfied(requirements, visible))

    def test_color_only_goal_behavior_is_preserved(self):
        requirements = {"requires_perception": True, "requested_colors": ["blue"]}

        self.assertFalse(
            goal_requirements_satisfied(
                requirements,
                [Detection(label="colored_object", color="red", confidence=0.9)],
            )
        )
        self.assertTrue(
            goal_requirements_satisfied(
                requirements,
                [Detection(label="colored_object", color="blue", confidence=0.9)],
            )
        )

    def test_combined_color_and_label_goal_requires_both(self):
        requirements = {
            "requires_perception": True,
            "requested_colors": ["red"],
            "requested_labels": ["fire_extinguisher"],
        }

        self.assertFalse(
            goal_requirements_satisfied(
                requirements,
                [Detection(label="fire_extinguisher", color="blue", confidence=0.9)],
            )
        )
        self.assertFalse(
            goal_requirements_satisfied(
                requirements,
                [Detection(label="trash_can", color="red", confidence=0.9)],
            )
        )
        self.assertTrue(
            goal_requirements_satisfied(
                requirements,
                [Detection(label="fire_extinguisher", color="red", confidence=0.9)],
            )
        )

    def test_unspecified_perception_target_accepts_any_detection(self):
        self.assertTrue(
            goal_requirements_satisfied(
                {"requires_perception": True},
                [Detection(label="anything", confidence=0.1)],
            )
        )
        self.assertFalse(
            goal_requirements_satisfied({"requires_perception": True}, [])
        )

    def test_runtime_and_monitor_share_one_requirement_implementation(self):
        self.assertIs(
            runtime_module.goal_requirements_satisfied,
            state_module.goal_requirements_satisfied,
        )
        self.assertIs(
            monitor_module.goal_requirements_satisfied,
            state_module.goal_requirements_satisfied,
        )

    def test_planner_requires_perception_for_label_requirements(self):
        with self.assertRaisesRegex(ValueError, "requires_perception=true"):
            TaskPlan(objective="find extinguisher", requested_labels=["fire extinguisher"])

        plan = TaskPlan(
            objective="find extinguisher",
            requires_perception=True,
            requested_labels=["fire extinguisher"],
        )
        self.assertEqual(plan.requested_labels, ["fire extinguisher"])

    def test_monitor_uses_structured_perception_requirement_not_keywords(self):
        state = RunState(run_id="test", goal="find documentation")
        state.goal_requirements = {"requires_perception": False}
        state.tool_history = [
            {"tool": "wait_seconds", "arguments": {}, "result": ToolResult(status=ToolStatus.SUCCESS).to_dict()}
        ]
        state.last_tool_result = ToolResult(status=ToolStatus.SUCCESS)
        evaluation = GoalMonitor().evaluate(state)
        self.assertNotEqual(evaluation.reason, "Goal requires perception but no semantic observation exists")

        state.goal = "寻找蓝色目标"
        state.goal_requirements = {"requires_perception": True}
        evaluation = GoalMonitor().evaluate(state)
        self.assertEqual(evaluation.blocker, GoalBlocker.MISSING_EVIDENCE)
        self.assertEqual(evaluation.reason, "Goal requires perception but no semantic observation exists")

        state.robot_state.last_perception_at = datetime.now(timezone.utc).isoformat()
        state.robot_state.visible_objects = [
            Detection(label="colored_object", color="red", confidence=0.95)
        ]
        state.goal_requirements = {"requires_perception": True, "requested_colors": ["blue"]}
        evaluation = GoalMonitor().evaluate(state)
        self.assertEqual(evaluation.blocker, GoalBlocker.GOAL_NOT_MET_YET)

        state.robot_state.visible_objects.append(
            Detection(label="colored_object", color="blue", confidence=0.9)
        )
        evaluation = GoalMonitor().evaluate(state)
        self.assertTrue(evaluation.satisfied)
        self.assertEqual(evaluation.evidence["matches"][0]["color"], "blue")


if __name__ == "__main__":
    unittest.main()
