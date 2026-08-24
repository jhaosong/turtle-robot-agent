from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ElementTree

import yaml


class SimulationAssetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_fire_extinguisher_xml_is_well_formed(self):
        model_root = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/models/fire_extinguisher"
        )

        config = ElementTree.parse(model_root / "model.config").getroot()
        sdf = ElementTree.parse(model_root / "model.sdf").getroot()

        self.assertEqual(config.findtext("name"), "fire_extinguisher")
        self.assertEqual(sdf.find("model").attrib["name"], "fire_extinguisher")

    def test_packaged_models_are_well_formed(self):
        for model_name in ("fire_extinguisher", "inspection_platform"):
            package_root = (
                self.root
                / f"turtlebot3_behavior_demos/tb3_worlds/models/{model_name}"
            )
            config = ElementTree.parse(package_root / "model.config").getroot()
            sdf = ElementTree.parse(package_root / "model.sdf").getroot()
            self.assertEqual(config.findtext("name"), model_name)
            self.assertEqual(sdf.find("model").attrib["name"], model_name)

    def test_yoloe_prompt_catalog_has_usable_reference(self):
        prompt_root = self.root / "sim_assets/yoloe_prompts"
        catalog = json.loads(
            (prompt_root / "catalog.json").read_text(encoding="utf-8")
        )
        prompt = catalog["fire extinguisher"]
        self.assertTrue(prompt["descriptions"])
        self.assertTrue((prompt_root / prompt["reference_image"]).is_file())

    def test_demo_launch_uses_deterministic_extinguisher_spawner(self):
        package_root = self.root / "turtlebot3_behavior_demos/tb3_worlds"
        spawner_path = package_root / "scripts/fire_extinguisher_spawner.py"
        launch_source = (
            package_root / "launch/tb3_demo_world.launch.py"
        ).read_text(encoding="utf-8")

        ast.parse(spawner_path.read_text(encoding="utf-8"))
        self.assertIn('executable="fire_extinguisher_spawner.py"', launch_source)
        self.assertIn('"object_z", default=0.30', launch_source)
        self.assertNotIn("block_spawner.py", launch_source)

    def test_extinguisher_room_is_ten_meters_and_does_not_leak_object_pose(self):
        map_root = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/maps"
        )
        metadata = yaml.safe_load(
            (map_root / "extinguisher_room_map.yaml").read_text(encoding="utf-8")
        )
        tokens = (map_root / metadata["image"]).read_text(encoding="ascii").split()
        width, height = int(tokens[1]), int(tokens[2])
        self.assertEqual(width * float(metadata["resolution"]), 10.0)
        self.assertEqual(height * float(metadata["resolution"]), 10.0)

        locations = yaml.safe_load(
            (map_root / "extinguisher_room_locations.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            list(locations),
            ["inspection_start", "east_view", "north_view", "west_view"],
        )
        self.assertNotIn("fire_extinguisher_station", locations)

    def test_extinguisher_room_contains_only_walls_and_central_platform(self):
        world_path = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/worlds/extinguisher_room.world"
        )
        world = ElementTree.parse(world_path).getroot().find("world")
        self.assertEqual(world.attrib["name"], "extinguisher_room")
        self.assertEqual(world.findall("include"), [])
        self.assertEqual(
            [
                model.attrib["name"]
                for model in world.findall("model")
                if model.attrib["name"] != "ground_plane"
            ],
            ["room_walls", "inspection_platform"],
        )
        platform = world.find("model[@name='inspection_platform']")
        size = platform.findtext("./link/collision/geometry/box/size")
        pose = platform.findtext("./link/collision/pose")
        self.assertEqual(size, "1.0 1.0 0.30")
        self.assertEqual(pose, "0 0 0.15 0 0 0")

    def test_static_map_marks_the_platform_footprint(self):
        map_path = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/maps/extinguisher_room_map.pgm"
        )
        tokens = map_path.read_text(encoding="ascii").split()
        pixels = [int(value) for value in tokens[4:]]
        width = int(tokens[1])
        center_cells = [pixels[row * width + column] for row in (9, 10) for column in (9, 10)]
        self.assertEqual(center_cells, [0, 0, 0, 0])

    def test_extinguisher_is_spawned_above_waffle_pi_scan_plane(self):
        spawner = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/scripts/fire_extinguisher_spawner.py"
        ).read_text(encoding="utf-8")
        self.assertIn('declare_parameter("z", 0.30)', spawner)
        # Official Humble Waffle Pi model: 0.010 m base joint + 0.121 m ray pose.
        self.assertGreater(0.30, 0.010 + 0.121)

    def test_ros_launcher_waits_for_robot_topics_and_tf(self):
        launcher = (self.root / "run_robot_agent_ros.sh").read_text(
            encoding="utf-8"
        )
        world_launch = (
            self.root
            / "turtlebot3_behavior_demos/tb3_worlds/launch/tb3_world.launch.py"
        ).read_text(encoding="utf-8")

        self.assertIn("Refreshed tb3_worlds runtime resources from /app", launcher)
        self.assertIn("ros2 pkg prefix tb3_worlds", launcher)
        self.assertIn('"${odom_ready}" == "true"', launcher)
        self.assertIn('"${scan_ready}" == "true"', launcher)
        self.assertIn('"pause": "false"', world_launch)
        self.assertIn('"-timeout", "180.0"', world_launch)
        self.assertIn('"-unpause"', world_launch)
        self.assertNotIn("spawn_turtlebot3.launch.py", world_launch)
        self.assertIn("exec python /app/src/robot_agent/cli.py", launcher)
        self.assertNotIn("--execute-ros2", launcher)
        self.assertNotIn("--ros-backend", launcher)


if __name__ == "__main__":
    unittest.main()
