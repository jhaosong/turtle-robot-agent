"""A small semantic world model; raw ROS messages never enter agent context."""

from __future__ import annotations

from datetime import datetime, timezone

from robot_agent.state import Detection, Pose2D, RobotState


class WorldModel:
    def __init__(self, robot_state: RobotState) -> None:
        self.robot_state = robot_state

    def update_pose(self, pose: Pose2D) -> None:
        self.robot_state.pose = pose

    def update_navigation_status(self, status: str) -> None:
        self.robot_state.navigation_status = status

    def update_detections(self, detections: list[Detection]) -> None:
        self.robot_state.visible_objects = detections
        observed_at = datetime.now(timezone.utc).isoformat()
        self.robot_state.last_perception_at = observed_at

    def find(self, color: str | None, label: str | None) -> list[Detection]:
        return [
            item
            for item in self.robot_state.visible_objects
            if (color is None or item.color == color)
            and (label is None or item.label == label)
        ]

    def has_perception_observation(self) -> bool:
        return self.robot_state.last_perception_at is not None

    def context(self) -> dict:
        return self.robot_state.to_agent_context()
