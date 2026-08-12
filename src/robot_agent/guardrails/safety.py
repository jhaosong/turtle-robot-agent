"""Safety checks deliberately kept outside LLM prompts."""

from __future__ import annotations

import math

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.state import Pose2D


class SafetyValidator:
    def __init__(self, settings: RobotAgentSettings) -> None:
        self.settings = settings

    def validate_pose(self, pose: Pose2D) -> None:
        for value, name in ((pose.x, "x"), (pose.y, "y"), (pose.yaw, "yaw")):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Target pose {name} must be a finite number")
        if not self.settings.workspace_min_x <= pose.x <= self.settings.workspace_max_x:
            raise ValueError("Target pose x is outside the configured safety workspace")
        if not self.settings.workspace_min_y <= pose.y <= self.settings.workspace_max_y:
            raise ValueError("Target pose is outside the configured safety workspace")

    def validate_wait(self, seconds: float) -> float:
        if not math.isfinite(seconds) or seconds <= 0 or seconds > self.settings.tool_timeout_sec:
            raise ValueError(f"Wait duration must be in (0, {self.settings.tool_timeout_sec}]")
        return seconds
