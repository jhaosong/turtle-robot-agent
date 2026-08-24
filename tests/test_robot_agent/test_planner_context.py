from __future__ import annotations

import unittest

from robot_agent.agents.lead_agent.prompt import build_lead_agent_prompt
from robot_agent.agents.lead_agent.planner import PLANNER_PROMPT, LeadTaskPlanner, TaskPlan


class CapturingStructuredModel:
    def __init__(self) -> None:
        self.messages = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return TaskPlan(objective="navigate")


class PlannerContextTest(unittest.TestCase):
    def test_planner_keeps_moving_search_as_one_perception_step(self):
        self.assertIn("one perception step", PLANNER_PROMPT)
        self.assertIn("continuous Nav2 route", PLANNER_PROMPT)
        self.assertIn('"find/search in the\nroom" as moving search', PLANNER_PROMPT)
        self.assertIn("Populate requested_labels", PLANNER_PROMPT)
        self.assertIn('"fire extinguisher"', PLANNER_PROMPT)
        self.assertIn("exactly two ordered perception steps", PLANNER_PROMPT)
        self.assertIn("one self-contained tool call", PLANNER_PROMPT)
        self.assertIn("never behavior_tree tasks", PLANNER_PROMPT)

    def test_known_location_catalog_is_supplied_to_planner(self):
        model = CapturingStructuredModel()

        LeadTaskPlanner(model).plan(
            "Navigate to location1",
            robot_state={"pose": None},
            available_capabilities={"navigation"},
            known_locations=["location1", "location2"],
        )

        human_prompt = model.messages[1][1]
        self.assertIn('Known navigation locations: ["location1", "location2"]', human_prompt)

    def test_lead_agent_enforces_search_before_multi_view_inspection(self):
        prompt = build_lead_agent_prompt(
            known_locations=["east_view"],
            tool_names=["search_for_object", "circle_object_for_inspection"],
            max_tool_calls=12,
        )

        self.assertIn("Follow ordered plan dependencies strictly", prompt)
        self.assertIn(
            "call search_for_object before\ncircle_object_for_inspection",
            prompt,
        )
        self.assertIn("previous run", prompt)


if __name__ == "__main__":
    unittest.main()
