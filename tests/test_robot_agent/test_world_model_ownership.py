from __future__ import annotations

from pathlib import Path
import unittest

from robot_agent.state import Pose2D, RobotState
from robot_agent.world_model import WorldModel


class WorldModelOwnershipTest(unittest.TestCase):
    def test_navigation_state_updates_through_world_model(self):
        state = RobotState()
        world = WorldModel(state)
        planned = Pose2D(1.0, 2.0, 0.5)

        world.update_navigation_status("planned", planned)
        self.assertEqual(state.navigation_status, "planned")
        self.assertIs(state.last_planned_pose, planned)

        world.update_navigation_status("succeeded", None)
        world.update_pose(planned)
        self.assertEqual(state.navigation_status, "succeeded")
        self.assertIsNone(state.last_planned_pose)
        self.assertIs(state.pose, planned)

    def test_registry_does_not_bypass_world_model_robot_state(self):
        repository_root = Path(__file__).resolve().parents[2]
        registry_source = (
            repository_root / "src/robot_agent/tools/registry.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("robot_state.", registry_source)


if __name__ == "__main__":
    unittest.main()
