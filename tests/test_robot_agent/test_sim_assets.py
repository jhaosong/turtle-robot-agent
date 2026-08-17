from __future__ import annotations

import ast
from pathlib import Path
import unittest
import xml.etree.ElementTree as ElementTree

import yaml


class SimulationAssetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_fire_extinguisher_xml_is_well_formed(self):
        model_root = self.root / "sim_assets/models/fire_extinguisher"

        config = ElementTree.parse(model_root / "model.config").getroot()
        sdf = ElementTree.parse(model_root / "model.sdf").getroot()

        self.assertEqual(config.findtext("name"), "fire_extinguisher")
        self.assertEqual(sdf.find("model").attrib["name"], "fire_extinguisher")

    def test_packaged_model_matches_importable_asset(self):
        asset_root = self.root / "sim_assets/models/fire_extinguisher"
        package_root = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/models/fire_extinguisher"
        )

        for filename in ("model.config", "model.sdf"):
            self.assertEqual(
                (asset_root / filename).read_text(encoding="utf-8"),
                (package_root / filename).read_text(encoding="utf-8"),
            )

    def test_demo_launch_uses_deterministic_extinguisher_spawner(self):
        package_root = self.root / "turtlebot3_behavior_demos/tb3_worlds"
        spawner_path = package_root / "scripts/fire_extinguisher_spawner.py"
        launch_source = (
            package_root / "launch/tb3_demo_world.launch.py"
        ).read_text(encoding="utf-8")

        ast.parse(spawner_path.read_text(encoding="utf-8"))
        self.assertIn('executable="fire_extinguisher_spawner.py"', launch_source)
        self.assertNotIn("block_spawner.py", launch_source)

    def test_fire_extinguisher_station_matches_documented_observation_pose(self):
        locations_path = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml"
        )
        locations = yaml.safe_load(locations_path.read_text(encoding="utf-8"))

        self.assertEqual(
            locations["fire_extinguisher_station"],
            [4.0, 1.5, 1.571],
        )


if __name__ == "__main__":
    unittest.main()
