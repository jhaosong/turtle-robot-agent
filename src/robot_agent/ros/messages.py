"""ROS-independent helpers for geometry message conversion."""

from __future__ import annotations

import math


def yaw_to_quaternion(yaw: float) -> dict[str, float]:
    """Return the planar quaternion shape expected by ROS2 Pose messages."""
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw / 2.0),
        "w": math.cos(yaw / 2.0),
    }


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert a ROS quaternion into a planar yaw angle."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
