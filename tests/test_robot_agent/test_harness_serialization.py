from __future__ import annotations

import json
import unittest

from robot_agent.harness import build_bounded_goal
from robot_agent.state import Pose2D, RobotState


class HarnessSerializationTest(unittest.TestCase):
    def test_bounded_goal_embeds_plan_and_robot_state_as_json(self):
        plan = [{"description": "go to Bob's room", "status": "pending"}]
        robot_state = RobotState(last_planned_pose=Pose2D(1.0, 2.0, 0.5))

        prompt = build_bounded_goal("navigate", plan, robot_state)
        plan_text = prompt.split("<task_plan_json>\n", 1)[1].split("\n</task_plan_json>", 1)[0]
        state_text = prompt.split("<robot_state_json>\n", 1)[1].split("\n</robot_state_json>", 1)[0]

        self.assertEqual(json.loads(plan_text), plan)
        self.assertEqual(json.loads(state_text), robot_state.to_agent_context())


if __name__ == "__main__":
    unittest.main()
