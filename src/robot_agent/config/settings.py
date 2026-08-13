"""Environment-backed settings for a single robotics-agent run.

The shape intentionally mirrors DeerFlow's explicit runtime configuration,
without importing its application-wide configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RobotAgentSettings:
    """Configuration that changes runtime behaviour but not agent code."""

    location_file: Path
    run_directory: Path
    session_id: str = "default"
    execute_ros2: bool = False
    trace: bool = True
    ros_backend: str = "cli"
    nav_action_name: str = "/navigate_to_pose"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    camera_topic: str = "/camera/image_raw"
    map_frame: str = "map"
    base_frame: str = "base_link"
    tool_timeout_sec: float = 30.0
    loop_warn_threshold: int = 3
    repeated_tool_limit: int = 5
    max_tool_calls: int = 12
    max_continuations: int = 12
    max_no_progress_continuations: int = 3
    bt_navigation_retries: int = 1
    active_perception_retry_enabled: bool = False
    active_perception_backoff_speed: float = -0.05
    active_perception_backoff_duration_sec: float = 0.5
    detector_backend: str = "color_blob"
    detection_interval_sec: float = 1.0
    detection_confidence_threshold: float = 0.8
    center_on_detection: bool = True
    image_center_tolerance: float = 0.08
    centering_max_angular_speed: float = 0.2
    centering_gain: float = 0.5
    centering_timeout_sec: float = 10.0
    yolo_model: str = "yolov8n.pt"
    yolo_input_size: int = 640
    workspace_min_x: float = -10.0
    workspace_max_x: float = 10.0
    workspace_min_y: float = -10.0
    workspace_max_y: float = 10.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.session_id) or self.session_id in {".", ".."}:
            raise ValueError("session_id must be a safe 1-64 character identifier")
        if self.ros_backend not in {"cli", "rclpy"}:
            raise ValueError("ros_backend must be 'cli' or 'rclpy'")
        if self.tool_timeout_sec <= 0:
            raise ValueError("tool_timeout_sec must be positive")
        if (
            self.repeated_tool_limit < 1
            or self.max_tool_calls < 1
            or self.max_continuations < 1
            or self.max_no_progress_continuations < 1
        ):
            raise ValueError("tool-call limits must be at least one")
        if self.loop_warn_threshold >= self.repeated_tool_limit:
            raise ValueError("loop_warn_threshold must be below repeated_tool_limit")
        if not 0 <= self.bt_navigation_retries <= 3:
            raise ValueError("bt_navigation_retries must be between 0 and 3")
        if not -0.1 <= self.active_perception_backoff_speed < 0:
            raise ValueError("active_perception_backoff_speed must be in [-0.1, 0)")
        if not 0 < self.active_perception_backoff_duration_sec <= 2:
            raise ValueError("active_perception_backoff_duration_sec must be in (0, 2]")
        if self.detector_backend not in {"color_blob", "yolo", "vlm"}:
            raise ValueError("detector_backend must be 'color_blob', 'yolo', or 'vlm'")
        if not 0.1 <= self.detection_interval_sec <= 60:
            raise ValueError("detection_interval_sec must be in [0.1, 60]")
        if not 0 <= self.detection_confidence_threshold <= 1:
            raise ValueError("detection_confidence_threshold must be in [0, 1]")
        if not 0.01 <= self.image_center_tolerance <= 0.25:
            raise ValueError("image_center_tolerance must be in [0.01, 0.25]")
        if not 0 < self.centering_max_angular_speed <= 0.5:
            raise ValueError("centering_max_angular_speed must be in (0, 0.5]")
        if not 0 < self.centering_gain <= 2:
            raise ValueError("centering_gain must be in (0, 2]")
        if not 0 < self.centering_timeout_sec <= self.tool_timeout_sec:
            raise ValueError("centering_timeout_sec must be in (0, tool_timeout_sec]")
        if not 160 <= self.yolo_input_size <= 1280:
            raise ValueError("yolo_input_size must be in [160, 1280]")
        if self.workspace_min_x >= self.workspace_max_x or self.workspace_min_y >= self.workspace_max_y:
            raise ValueError("workspace minimum bounds must be smaller than maximum bounds")

    @classmethod
    def from_env(cls, project_root: Path) -> "RobotAgentSettings":
        turtlebot_root = project_root / "turtlebot3_behavior_demos"
        location_file = Path(
            os.getenv(
                "ROBOT_AGENT_LOCATION_FILE",
                str(turtlebot_root / "tb3_worlds/maps/sim_house_locations.yaml"),
            )
        )
        run_directory = Path(
            os.getenv("ROBOT_AGENT_RUN_DIRECTORY", str(project_root / "robot_agent_runs"))
        )
        return cls(
            location_file=location_file,
            run_directory=run_directory,
            session_id=os.getenv("ROBOT_AGENT_SESSION_ID", "default"),
            execute_ros2=_as_bool(os.getenv("ROBOT_AGENT_EXECUTE_ROS2")),
            trace=_as_bool(os.getenv("ROBOT_AGENT_TRACE"), default=True),
            ros_backend=os.getenv("ROBOT_AGENT_ROS_BACKEND", "cli").strip().lower(),
            nav_action_name=os.getenv("ROBOT_AGENT_NAV_ACTION", "/navigate_to_pose"),
            cmd_vel_topic=os.getenv("ROBOT_AGENT_CMD_VEL_TOPIC", "/cmd_vel"),
            odom_topic=os.getenv("ROBOT_AGENT_ODOM_TOPIC", "/odom"),
            camera_topic=os.getenv("ROBOT_AGENT_CAMERA_TOPIC", "/camera/image_raw"),
            map_frame=os.getenv("ROBOT_AGENT_MAP_FRAME", "map"),
            base_frame=os.getenv("ROBOT_AGENT_BASE_FRAME", "base_link"),
            tool_timeout_sec=float(os.getenv("ROBOT_AGENT_TOOL_TIMEOUT_SEC", "30")),
            loop_warn_threshold=int(os.getenv("ROBOT_AGENT_LOOP_WARN_THRESHOLD", "3")),
            repeated_tool_limit=int(os.getenv("ROBOT_AGENT_REPEATED_TOOL_LIMIT", "5")),
            max_tool_calls=int(os.getenv("ROBOT_AGENT_MAX_TOOL_CALLS", "12")),
            max_continuations=int(os.getenv("ROBOT_AGENT_MAX_CONTINUATIONS", "12")),
            max_no_progress_continuations=int(
                os.getenv("ROBOT_AGENT_MAX_NO_PROGRESS_CONTINUATIONS", "3")
            ),
            bt_navigation_retries=int(os.getenv("ROBOT_AGENT_BT_NAVIGATION_RETRIES", "1")),
            active_perception_retry_enabled=_as_bool(
                os.getenv("ROBOT_AGENT_ACTIVE_PERCEPTION_RETRY")
            ),
            active_perception_backoff_speed=float(
                os.getenv("ROBOT_AGENT_ACTIVE_PERCEPTION_BACKOFF_SPEED", "-0.05")
            ),
            active_perception_backoff_duration_sec=float(
                os.getenv("ROBOT_AGENT_ACTIVE_PERCEPTION_BACKOFF_DURATION_SEC", "0.5")
            ),
            detector_backend=os.getenv(
                "ROBOT_AGENT_DETECTOR_BACKEND", "color_blob"
            ).strip().lower(),
            detection_interval_sec=float(
                os.getenv("ROBOT_AGENT_DETECTION_INTERVAL_SEC", "1.0")
            ),
            detection_confidence_threshold=float(
                os.getenv("ROBOT_AGENT_DETECTION_CONFIDENCE_THRESHOLD", "0.8")
            ),
            center_on_detection=_as_bool(
                os.getenv("ROBOT_AGENT_CENTER_ON_DETECTION"), default=True
            ),
            image_center_tolerance=float(
                os.getenv("ROBOT_AGENT_IMAGE_CENTER_TOLERANCE", "0.08")
            ),
            centering_max_angular_speed=float(
                os.getenv("ROBOT_AGENT_CENTERING_MAX_ANGULAR_SPEED", "0.2")
            ),
            centering_gain=float(
                os.getenv("ROBOT_AGENT_CENTERING_GAIN", "0.5")
            ),
            centering_timeout_sec=float(
                os.getenv("ROBOT_AGENT_CENTERING_TIMEOUT_SEC", "10.0")
            ),
            yolo_model=os.getenv("ROBOT_AGENT_YOLO_MODEL", "yolov8n.pt"),
            yolo_input_size=int(os.getenv("ROBOT_AGENT_YOLO_INPUT_SIZE", "640")),
            workspace_min_x=float(os.getenv("ROBOT_AGENT_WORKSPACE_MIN_X", "-10")),
            workspace_max_x=float(os.getenv("ROBOT_AGENT_WORKSPACE_MAX_X", "10")),
            workspace_min_y=float(os.getenv("ROBOT_AGENT_WORKSPACE_MIN_Y", "-10")),
            workspace_max_y=float(os.getenv("ROBOT_AGENT_WORKSPACE_MAX_Y", "10")),
        )
