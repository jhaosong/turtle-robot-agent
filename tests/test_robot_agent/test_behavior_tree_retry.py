from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from robot_agent.skills.behavior_tree import (
    BehaviorTreeNode,
    BehaviorTreePlan,
    BehaviorTreeSkill,
    behavior_tree_to_xml,
)
from robot_agent.state import ToolResult, ToolStatus


class BehaviorTreeRetryTest(unittest.TestCase):
    def test_behavior_tree_plan_rejects_more_than_twelve_nodes(self):
        with self.assertRaises(ValidationError):
            BehaviorTreePlan(
                goal="too large",
                nodes=[BehaviorTreeNode(type="Wait", seconds=1.0) for _ in range(12)]
                + [BehaviorTreeNode(type="Stop")],
            )

    def test_skill_validation_rejects_constructed_thirteen_node_plan(self):
        skill = BehaviorTreeSkill(
            model=None,
            known_locations=set(),
            output_directory=Path("unused"),
        )
        plan = BehaviorTreePlan.model_construct(
            goal="too large",
            nodes=[BehaviorTreeNode(type="Wait", seconds=1.0) for _ in range(12)]
            + [BehaviorTreeNode(type="Stop")],
        )

        with self.assertRaisesRegex(ValueError, "at most 12"):
            skill._validate(plan)

    def test_retryable_navigation_failure_retries_once_and_completes(self):
        with TemporaryDirectory() as temporary_directory:
            skill = BehaviorTreeSkill(
                model=None,
                known_locations={"kitchen"},
                output_directory=Path(temporary_directory),
                navigation_retry_count=1,
            )
            plan = BehaviorTreePlan(
                goal="visit kitchen",
                nodes=[
                    BehaviorTreeNode(type="GoToPose", location="kitchen"),
                    BehaviorTreeNode(type="Stop"),
                ],
            )
            skill.generate = lambda goal: ToolResult(  # type: ignore[method-assign]
                status=ToolStatus.SUCCESS,
                data={
                    "goal": plan.goal,
                    "nodes": [node.model_dump() for node in plan.nodes],
                    "json_path": "unused",
                    "xml_path": "unused",
                },
            )
            navigation_attempts = 0
            abort_calls = 0

            def navigate(location: str, index: int) -> ToolResult:
                nonlocal navigation_attempts
                navigation_attempts += 1
                if navigation_attempts == 1:
                    return ToolResult(status=ToolStatus.FAILED, error="temporary Nav2 failure", retryable=True)
                return ToolResult(status=ToolStatus.SUCCESS)

            def abort() -> ToolResult:
                nonlocal abort_calls
                abort_calls += 1
                return ToolResult(status=ToolStatus.SUCCESS)

            result = skill.run(
                plan.goal,
                navigate=navigate,
                stop=lambda index: ToolResult(status=ToolStatus.SUCCESS),
                wait=lambda seconds, index: ToolResult(status=ToolStatus.SUCCESS),
                abort=abort,
            )

            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertEqual(navigation_attempts, 2)
            self.assertEqual(abort_calls, 0)
            self.assertEqual(len(result.data["node_results"][0]["attempts"]), 2)

            xml = behavior_tree_to_xml(plan, navigation_retry_count=1)
            self.assertIn("<Fallback", xml)
            self.assertEqual(xml.count("<GoToPose"), 2)


if __name__ == "__main__":
    unittest.main()
