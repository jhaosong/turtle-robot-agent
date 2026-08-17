from __future__ import annotations

import unittest

from robot_agent.runtime.rviz_config import configure_rviz


class RvizConfigTest(unittest.TestCase):
    def test_focuses_layout_on_navigation_and_two_camera_views(self):
        config = {
            "Panels": [
                {"Class": "rviz_common/Displays", "Name": "Displays"},
                {"Class": "rviz_common/Selection", "Name": "Selection"},
                {"Class": "rviz_common/Tool Properties", "Name": "Tools"},
                {"Class": "rviz_common/Views", "Name": "Views"},
                {
                    "Class": "nav2_rviz_plugins/Navigation 2",
                    "Name": "Navigation 2",
                },
                {"Class": "nav2_rviz_plugins/Selector", "Name": "Selector"},
                {"Class": "nav2_rviz_plugins/Docking", "Name": "Docking"},
            ],
            "Visualization Manager": {
                "Displays": [
                    {"Class": "rviz_default_plugins/Grid", "Name": "Grid"},
                    {"Class": "rviz_common/Group", "Name": "Realsense"},
                ]
            },
            "Window Geometry": {"QMainWindow State": "stale-layout"},
        }

        configured = configure_rviz(config)

        self.assertEqual(
            [panel["Class"] for panel in configured["Panels"]],
            ["rviz_common/Displays", "nav2_rviz_plugins/Navigation 2"],
        )
        displays = configured["Visualization Manager"]["Displays"]
        image_displays = [
            display
            for display in displays
            if display["Class"] == "rviz_default_plugins/Image"
        ]
        self.assertEqual(
            [display["Name"] for display in image_displays],
            ["Raw Image", "YOLOE Annotated"],
        )
        self.assertEqual(
            [display["Topic"]["Value"] for display in image_displays],
            ["/camera/image_raw", "/camera/yoloe_annotated"],
        )
        self.assertNotIn("QMainWindow State", configured["Window Geometry"])
        self.assertFalse(configured["Window Geometry"]["Hide Right Dock"])


if __name__ == "__main__":
    unittest.main()
