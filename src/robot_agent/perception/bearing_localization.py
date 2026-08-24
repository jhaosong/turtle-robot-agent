"""Pure bearing geometry for monocular two-view object localization."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_agent.state import ImagePosition, Pose2D


@dataclass(frozen=True)
class TriangulatedEstimate:
    """Intersection estimate plus geometry-derived reliability metadata."""

    position: Pose2D | None
    confidence: float
    baseline_m: float
    ray_angle_rad: float
    baseline_to_distance_ratio: float
    valid: bool
    reason: str | None = None
    first_range_m: float | None = None
    second_range_m: float | None = None


def bearing_from_detection(
    image_position: ImagePosition,
    robot_yaw: float,
    horizontal_fov_rad: float,
) -> float:
    """Convert a bbox center into an absolute map-frame horizontal bearing.

    ROS optical images increase x to the right. With the camera looking along
    the robot's forward axis, an object to image-right has a negative yaw
    offset, hence ``0.5 - x_normalized`` below.
    """
    if not 0.0 < horizontal_fov_rad < math.pi:
        raise ValueError("horizontal_fov_rad must be between 0 and pi")
    if not 0.0 <= image_position.x_normalized <= 1.0:
        raise ValueError("image_position.x_normalized must be in [0, 1]")
    image_offset = 2.0 * (0.5 - image_position.x_normalized)
    camera_bearing = math.atan(image_offset * math.tan(horizontal_fov_rad / 2.0))
    return _normalize_angle(robot_yaw + camera_bearing)


def triangulate_from_bearings(
    pose_at_t1: Pose2D,
    bearing_at_t1: float,
    pose_at_t2: Pose2D,
    bearing_at_t2: float,
    *,
    minimum_baseline_m: float = 0.25,
    min_ray_angle_rad: float = math.radians(3.0),
) -> TriangulatedEstimate:
    """Intersect two map-frame rays with closed-form 2D line geometry.

    Degenerate geometry is returned as an explicitly invalid, low-confidence
    estimate instead of raising. This lets the tool report a localization
    failure distinct from navigation or perception failure.
    """
    displacement = (pose_at_t2.x - pose_at_t1.x, pose_at_t2.y - pose_at_t1.y)
    baseline = math.hypot(*displacement)
    first_direction = (math.cos(bearing_at_t1), math.sin(bearing_at_t1))
    second_direction = (math.cos(bearing_at_t2), math.sin(bearing_at_t2))
    determinant = _cross(first_direction, second_direction)
    # Conditioning depends on |sin(theta)|, so fold the ray angle to [0, pi/2].
    ray_angle = math.asin(min(1.0, abs(determinant)))

    if baseline < minimum_baseline_m:
        return _invalid_estimate(
            baseline,
            ray_angle,
            "Triangulation baseline is too short",
        )
    if abs(determinant) < 1e-9:
        return _invalid_estimate(
            baseline,
            ray_angle,
            "Bearing rays are parallel or collinear",
        )

    first_range = _cross(displacement, second_direction) / determinant
    second_range = _cross(displacement, first_direction) / determinant
    if first_range <= 0.0 or second_range <= 0.0:
        return TriangulatedEstimate(
            position=None,
            confidence=0.0,
            baseline_m=baseline,
            ray_angle_rad=ray_angle,
            baseline_to_distance_ratio=0.0,
            valid=False,
            reason="Triangulated object lies behind a camera observation",
            first_range_m=first_range,
            second_range_m=second_range,
        )

    x = pose_at_t1.x + first_range * first_direction[0]
    y = pose_at_t1.y + first_range * first_direction[1]
    mean_range = (first_range + second_range) / 2.0
    baseline_ratio = baseline / max(mean_range, 1e-9)
    angle_score = min(1.0, ray_angle / max(2.0 * min_ray_angle_rad, 1e-9))
    ratio_score = min(1.0, baseline_ratio / 0.25)
    confidence = math.sqrt(angle_score * ratio_score)
    valid = ray_angle >= min_ray_angle_rad
    return TriangulatedEstimate(
        position=Pose2D(x=x, y=y, yaw=0.0, frame_id=pose_at_t1.frame_id),
        confidence=confidence,
        baseline_m=baseline,
        ray_angle_rad=ray_angle,
        baseline_to_distance_ratio=baseline_ratio,
        valid=valid,
        reason=None if valid else "Bearing rays have insufficient angular separation",
        first_range_m=first_range,
        second_range_m=second_range,
    )


def _invalid_estimate(
    baseline_m: float,
    ray_angle_rad: float,
    reason: str,
) -> TriangulatedEstimate:
    return TriangulatedEstimate(
        position=None,
        confidence=0.0,
        baseline_m=baseline_m,
        ray_angle_rad=ray_angle_rad,
        baseline_to_distance_ratio=0.0,
        valid=False,
        reason=reason,
    )


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
