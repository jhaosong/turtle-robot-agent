"""Generate and rank safe motions that create useful bearing parallax."""

from __future__ import annotations

from dataclasses import dataclass
import math

from robot_agent.state import Pose2D


class NoFeasibleBaselineError(ValueError):
    """Raised when every costmap-checked baseline candidate is rejected."""


@dataclass(frozen=True)
class BaselineCandidate:
    name: str
    pose: Pose2D
    tangential_displacement_m: float
    straight_line_distance_m: float


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: BaselineCandidate
    feasible: bool
    path_length_m: float
    obstacle_risk: float
    reason: str | None = None

    def score(self, *, alpha: float, beta: float, gamma: float) -> float:
        if not self.feasible:
            return -math.inf
        return (
            alpha * self.candidate.tangential_displacement_m
            - beta * self.path_length_m
            - gamma * self.obstacle_risk
        )


def generate_baseline_candidates(
    robot_pose: Pose2D,
    target_bearing_rad: float,
    *,
    radius_m: float = 0.75,
    assumed_object_distance_m: float = 1.5,
    offset_angles_deg: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0),
) -> list[BaselineCandidate]:
    """Create a candidate fan whose poses keep facing the assumed object."""
    if radius_m <= 0.0:
        raise ValueError("baseline candidate radius must be positive")
    if assumed_object_distance_m <= 0.0:
        raise ValueError("assumed object distance must be positive")
    assumed_object_x = (
        robot_pose.x + assumed_object_distance_m * math.cos(target_bearing_rad)
    )
    assumed_object_y = (
        robot_pose.y + assumed_object_distance_m * math.sin(target_bearing_rad)
    )
    candidates: list[BaselineCandidate] = []
    for magnitude_deg in offset_angles_deg:
        if not 0.0 < magnitude_deg < 180.0:
            raise ValueError("baseline offset angles must be between 0 and 180")
        for side_name, sign in (("left", 1.0), ("right", -1.0)):
            offset = sign * math.radians(magnitude_deg)
            travel_bearing = target_bearing_rad + offset
            x = robot_pose.x + radius_m * math.cos(travel_bearing)
            y = robot_pose.y + radius_m * math.sin(travel_bearing)
            candidates.append(
                BaselineCandidate(
                    name=f"{side_name}_{magnitude_deg:g}deg",
                    pose=Pose2D(
                        x=x,
                        y=y,
                        yaw=math.atan2(
                            assumed_object_y - y,
                            assumed_object_x - x,
                        ),
                        frame_id=robot_pose.frame_id,
                    ),
                    tangential_displacement_m=abs(radius_m * math.sin(offset)),
                    straight_line_distance_m=radius_m,
                )
            )
    return candidates


def cheap_candidate_score(
    candidate: BaselineCandidate,
    *,
    alpha: float,
    beta: float,
) -> float:
    """First-pass score before an expensive Nav2 path/costmap query."""
    return (
        alpha * candidate.tangential_displacement_m
        - beta * candidate.straight_line_distance_m
    )


def select_baseline_candidate(
    evaluations: list[CandidateEvaluation],
    *,
    alpha: float = 3.0,
    beta: float = 0.35,
    gamma: float = 2.0,
) -> CandidateEvaluation:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        raise NoFeasibleBaselineError(
            "All baseline candidates are infeasible in the Nav2 costmap"
        )
    return max(
        feasible,
        key=lambda item: item.score(alpha=alpha, beta=beta, gamma=gamma),
    )


def generate_object_viewpoints(
    object_pose: Pose2D,
    current_pose: Pose2D,
    *,
    radius_m: float,
    count: int = 4,
) -> list[Pose2D]:
    """Return evenly spaced poses, starting near the robot and facing the object."""
    if count < 2:
        raise ValueError("viewpoint count must be at least two")
    if radius_m <= 0:
        raise ValueError("viewpoint radius must be positive")
    observed_angle = math.atan2(
        current_pose.y - object_pose.y,
        current_pose.x - object_pose.x,
    )
    # Lock the orbit phase to the nearest evenly spaced map-frame direction.
    # This keeps repeated runs deterministic while still starting near the
    # robot instead of selecting an arbitrary side of the object.
    angular_step = 2.0 * math.pi / count
    start_angle = round(observed_angle / angular_step) * angular_step
    viewpoints: list[Pose2D] = []
    for index in range(count):
        angle = start_angle + index * angular_step
        x = object_pose.x + radius_m * math.cos(angle)
        y = object_pose.y + radius_m * math.sin(angle)
        viewpoints.append(
            Pose2D(
                x=x,
                y=y,
                yaw=math.atan2(object_pose.y - y, object_pose.x - x),
                frame_id=object_pose.frame_id,
            )
        )
    return viewpoints
