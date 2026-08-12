from __future__ import annotations

from datetime import datetime, timezone
import unittest

from robot_agent.goal_monitor import GoalMonitor
from robot_agent.state import Detection, GoalBlocker, RunState, ToolResult, ToolStatus


class StructuredGoalRequirementsTest(unittest.TestCase):
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
