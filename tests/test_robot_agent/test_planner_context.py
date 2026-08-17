from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
