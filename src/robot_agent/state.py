"""Structured working state exposed to the runtime, never raw ROS streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELED = "canceled"
    PLANNED = "planned"
    NEEDS_INPUT = "needs_input"


class GoalBlocker(str, Enum):
    NONE = "none"
    MISSING_EVIDENCE = "missing_evidence"
    NEEDS_USER_INPUT = "needs_user_input"
    RUN_FAILED = "run_failed"
    EXTERNAL_WAIT = "external_wait"
    GOAL_NOT_MET_YET = "goal_not_met_yet"
    NO_PROGRESS = "no_progress"


@dataclass
class ToolResult:
    """The only result format returned by a robotics tool to the agent."""

    status: ToolStatus
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_sec: float = 0.0
    timestamp: str = field(default_factory=utc_now)
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class GoalEvaluation:
    """Deterministic completion verdict, separate from the LLM final message."""

    satisfied: bool
    blocker: GoalBlocker
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocker"] = self.blocker.value
        return payload


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = "map"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImagePosition:
    """Detection box in one camera image, not a world-frame coordinate."""

    x_px: float
    y_px: float
    x_normalized: float
    y_normalized: float
    width_normalized: float = 0.0
    height_normalized: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Detection:
    label: str
    confidence: float
    color: str | None = None
    position: Pose2D | None = None
    image_position: ImagePosition | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.position is not None:
            result["position"] = self.position.to_dict()
        if self.image_position is not None:
            result["image_position"] = self.image_position.to_dict()
        return result

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "Detection":
        position = payload.get("position")
        image_position = payload.get("image_position")
        return cls(
            label=str(payload["label"]),
            confidence=float(payload["confidence"]),
            color=payload.get("color"),
            position=Pose2D(**position) if isinstance(position, dict) else None,
            image_position=(
                ImagePosition(**image_position)
                if isinstance(image_position, dict)
                else None
            ),
        )


def matching_goal_detections(
    goal_requirements: dict[str, Any],
    visible_objects: list[Detection],
) -> list[Detection]:
    """Return detections satisfying every explicitly requested attribute."""
    requested_colors = set(goal_requirements.get("requested_colors") or [])
    requested_labels = set(goal_requirements.get("requested_labels") or [])
    if not requested_colors and not requested_labels:
        return list(visible_objects)
    return [
        detection
        for detection in visible_objects
        if (not requested_colors or detection.color in requested_colors)
        and (not requested_labels or detection.label in requested_labels)
    ]


def goal_requirements_satisfied(
    goal_requirements: dict[str, Any],
    visible_objects: list[Detection],
) -> bool:
    """Determine whether semantic detections satisfy a perception goal."""
    return bool(matching_goal_detections(goal_requirements, visible_objects))


@dataclass
class RobotState:
    pose: Pose2D | None = None
    last_planned_pose: Pose2D | None = None
    navigation_status: str = "unknown"
    visible_objects: list[Detection] = field(default_factory=list)
    obstacle_ahead: bool | None = None
    last_perception_at: str | None = None

    def to_agent_context(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_dict() if self.pose else None,
            "last_planned_pose": self.last_planned_pose.to_dict() if self.last_planned_pose else None,
            "navigation_status": self.navigation_status,
            "visible_objects": [item.to_dict() for item in self.visible_objects],
            "obstacle_ahead": self.obstacle_ahead,
            "last_perception_at": self.last_perception_at,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "RobotState":
        pose = payload.get("pose")
        last_planned_pose = payload.get("last_planned_pose")
        visible_objects = payload.get("visible_objects") or []
        return cls(
            pose=Pose2D(**pose) if isinstance(pose, dict) else None,
            last_planned_pose=Pose2D(**last_planned_pose)
            if isinstance(last_planned_pose, dict)
            else None,
            navigation_status=str(payload.get("navigation_status", "unknown")),
            visible_objects=[Detection.from_snapshot(item) for item in visible_objects],
            obstacle_ahead=payload.get("obstacle_ahead"),
            last_perception_at=payload.get("last_perception_at"),
        )


@dataclass
class SemanticSessionState:
    """Robot-world facts that survive independent goal runs in one session."""

    robot_state: RobotState = field(default_factory=RobotState)
    visited_locations: list[str] = field(default_factory=list)

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "robot_state": self.robot_state.to_agent_context(),
            "visited_locations": list(self.visited_locations),
            "updated_at": utc_now(),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "SemanticSessionState":
        robot_state = payload.get("robot_state")
        visited = payload.get("visited_locations") or []
        if not isinstance(robot_state, dict) or not isinstance(visited, list):
            raise ValueError("Invalid semantic session checkpoint")
        return cls(
            robot_state=RobotState.from_snapshot(robot_state),
            visited_locations=[str(item) for item in visited],
        )


@dataclass
class RunState:
    run_id: str
    goal: str
    robot_state: RobotState = field(default_factory=RobotState)
    plan: list[dict[str, Any]] = field(default_factory=list)
    goal_requirements: dict[str, Any] = field(default_factory=dict)
    last_tool_result: ToolResult | None = None
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    continuation_count: int = 0
    max_continuations: int = 12
    no_progress_count: int = 0
    max_no_progress_continuations: int = 3
    visited_locations: list[str] = field(default_factory=list)
    current_bt_node_index: int | None = None
    goal_evaluation: GoalEvaluation | None = None
    model_stop_reason: str | None = None
    status: str = "running"

    def to_agent_context(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "plan": self.plan,
            "goal_requirements": self.goal_requirements,
            "robot_state": self.robot_state.to_agent_context(),
            "last_tool_result": self.last_tool_result.to_dict() if self.last_tool_result else None,
            "failures": self.failures[-3:],
            "continuation_count": self.continuation_count,
            "max_continuations": self.max_continuations,
            "no_progress_count": self.no_progress_count,
            "max_no_progress_continuations": self.max_no_progress_continuations,
            "visited_locations": self.visited_locations,
            "current_bt_node_index": self.current_bt_node_index,
            "goal_evaluation": self.goal_evaluation.to_dict() if self.goal_evaluation else None,
            "model_stop_reason": self.model_stop_reason,
            "status": self.status,
        }

    def to_snapshot(self) -> dict[str, Any]:
        """Persist full recovery state; agent context intentionally stays compact."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "robot_state": self.robot_state.to_agent_context(),
            "plan": self.plan,
            "goal_requirements": self.goal_requirements,
            "last_tool_result": self.last_tool_result.to_dict() if self.last_tool_result else None,
            "tool_history": self.tool_history,
            "failures": self.failures,
            "continuation_count": self.continuation_count,
            "max_continuations": self.max_continuations,
            "no_progress_count": self.no_progress_count,
            "max_no_progress_continuations": self.max_no_progress_continuations,
            "visited_locations": self.visited_locations,
            "current_bt_node_index": self.current_bt_node_index,
            "goal_evaluation": self.goal_evaluation.to_dict() if self.goal_evaluation else None,
            "model_stop_reason": self.model_stop_reason,
            "status": self.status,
        }

    def to_semantic_session_state(self) -> SemanticSessionState:
        return SemanticSessionState(
            robot_state=RobotState.from_snapshot(self.robot_state.to_agent_context()),
            visited_locations=list(self.visited_locations),
        )

    def progress_signature(self) -> dict[str, Any]:
        """Return only state whose change is observable task progress."""
        return {
            "robot_state": self.robot_state.to_agent_context(),
            "visited_locations": list(self.visited_locations),
            "current_bt_node_index": self.current_bt_node_index,
        }
