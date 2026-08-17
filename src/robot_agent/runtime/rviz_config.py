"""Build the focused RViz layout used by the robot-agent demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


RAW_CAMERA_TOPIC = "/camera/image_raw"
ANNOTATED_CAMERA_TOPIC = "/camera/yoloe_annotated"


def _image_display(name: str, topic: str, reliability: str) -> dict[str, Any]:
    return {
        "Class": "rviz_default_plugins/Image",
        "Enabled": True,
        "Max Value": 1,
        "Median window": 5,
        "Min Value": 0,
        "Name": name,
        "Normalize Range": True,
        "Topic": {
            "Depth": 5,
            "Durability Policy": "Volatile",
            "History Policy": "Keep Last",
            "Reliability Policy": reliability,
            "Value": topic,
        },
        "Value": True,
    }


def configure_rviz(config: dict[str, Any]) -> dict[str, Any]:
    """Keep navigation displays while replacing auxiliary docks with cameras."""
    allowed_panels = {
        "rviz_common/Displays",
        "nav2_rviz_plugins/Navigation 2",
    }
    config["Panels"] = [
        panel
        for panel in config.get("Panels", [])
        if panel.get("Class") in allowed_panels
    ]

    manager = config.setdefault("Visualization Manager", {})
    displays = [
        display
        for display in manager.get("Displays", [])
        if display.get("Class") != "rviz_default_plugins/Image"
        and display.get("Name") != "Realsense"
    ]
    displays.extend(
        [
            _image_display("Raw Image", RAW_CAMERA_TOPIC, "Best Effort"),
            _image_display(
                "YOLOE Annotated",
                ANNOTATED_CAMERA_TOPIC,
                "Reliable",
            ),
        ]
    )
    manager["Displays"] = displays

    # Omitting the stale serialized Qt state lets both Image displays dock in
    # their plugin's default right-side area instead of restoring old panes.
    config["Window Geometry"] = {
        "Displays": {"collapsed": False},
        "Navigation 2": {"collapsed": False},
        "Height": 900,
        "Hide Left Dock": False,
        "Hide Right Dock": False,
        "Width": 1600,
        "X": 0,
        "Y": 0,
    }
    return config


def write_robot_agent_config(source: Path, destination: Path) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    configured = configure_rviz(config)
    destination.write_text(
        yaml.safe_dump(configured, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    write_robot_agent_config(args.source, args.destination)


if __name__ == "__main__":
    main()
