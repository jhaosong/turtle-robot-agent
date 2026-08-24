"""Environment-backed settings for a single robotics-agent run.

The shape intentionally mirrors DeerFlow's explicit runtime configuration,
without importing its application-wide configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class RobotAgentSettings:
    """Configuration that changes runtime behaviour but not agent code."""

    location_file: Path
    run_directory: Path
    session_id: str = "default"
    trace: bool = True
    nav_action_name: str = "/navigate_to_pose"
    compute_path_action_name: str = "/compute_path_to_pose"
    global_costmap_service: str = "/global_costmap/get_costmap"
    cmd_vel_topic: str = "/cmd_vel"
    camera_topic: str = "/camera/image_raw"
    annotated_camera_topic: str = "/camera/yoloe_annotated"
    map_frame: str = "map"
    base_frame: str = "base_link"
    tool_timeout_sec: float = 30.0
    loop_warn_threshold: int = 3
    repeated_tool_limit: int = 5
    max_tool_calls: int = 12
    max_continuations: int = 12
    max_no_progress_continuations: int = 3
    bt_navigation_retries: int = 1
    detection_interval_sec: float = 0.2
    detection_box_threshold: float = 0.05
    detection_confidence_threshold: float = 0.05
    detection_tracking_confidence_threshold: float = 0.01
    detection_tracking_max_center_jump: float = 0.25
    detection_confirmation_frames: int = 1
    center_on_detection: bool = True
    image_center_tolerance: float = 0.10
    centering_max_angular_speed: float = 0.25
    centering_min_angular_speed: float = 0.10
    centering_gain: float = 0.8
    target_box_size_normalized: float = 0.4
    box_size_tolerance: float = 0.05
    centering_max_linear_speed: float = 0.25
    centering_min_linear_speed: float = 0.08
    centering_linear_gain: float = 1.0
    centering_stable_frames: int = 3
    centering_timeout_sec: float = 30.0
    centering_detection_hold_sec: float = 1.0
    post_cancel_settle_sec: float = 0.5
    yoloe_model: str = "yoloe-26s-seg.pt"
    yoloe_prompt_catalog: Path | None = None
    yolo_input_size: int = 640
    camera_horizontal_fov_rad: float = 1.085595
    triangulation_min_baseline_m: float = 0.25
    triangulation_min_ray_angle_deg: float = 3.0
    triangulation_min_confidence: float = 0.25
    baseline_candidate_radius_m: float = 0.75
    baseline_assumed_object_distance_m: float = 1.5
    baseline_score_alpha: float = 3.0
    baseline_score_beta: float = 0.35
    baseline_score_gamma: float = 2.0
    baseline_nav2_candidate_count: int = 4
    inspection_radius_m: float = 2.0
    inspection_min_radius_m: float = 2.0
    inspection_max_radius_m: float = 2.0
    workspace_min_x: float = -10.0
    workspace_max_x: float = 10.0
    workspace_min_y: float = -10.0
    workspace_max_y: float = 10.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.session_id) or self.session_id in {".", ".."}:
            raise ValueError("session_id must be a safe 1-64 character identifier")
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
        if not 0.1 <= self.detection_interval_sec <= 60:
            raise ValueError("detection_interval_sec must be in [0.1, 60]")
        if not 0 <= self.detection_box_threshold <= 1:
            raise ValueError("detection_box_threshold must be in [0, 1]")
        if not 0 <= self.detection_confidence_threshold <= 1:
            raise ValueError("detection_confidence_threshold must be in [0, 1]")
        if not 0 <= self.detection_tracking_confidence_threshold <= 1:
            raise ValueError(
                "detection_tracking_confidence_threshold must be in [0, 1]"
            )
        if (
            self.detection_tracking_confidence_threshold
            > self.detection_confidence_threshold
        ):
            raise ValueError(
                "detection_tracking_confidence_threshold must not exceed "
                "detection_confidence_threshold"
            )
        if not 0 < self.detection_tracking_max_center_jump <= 1:
            raise ValueError(
                "detection_tracking_max_center_jump must be in (0, 1]"
            )
        if self.detection_box_threshold > self.detection_confidence_threshold:
            raise ValueError(
                "detection_box_threshold must not exceed "
                "detection_confidence_threshold"
            )
        if not 1 <= self.detection_confirmation_frames <= 5:
            raise ValueError("detection_confirmation_frames must be in [1, 5]")
        if not 0.01 <= self.image_center_tolerance <= 0.25:
            raise ValueError("image_center_tolerance must be in [0.01, 0.25]")
        if not 0 < self.centering_max_angular_speed <= 0.5:
            raise ValueError("centering_max_angular_speed must be in (0, 0.5]")
        if not 0 < self.centering_min_angular_speed <= self.centering_max_angular_speed:
            raise ValueError(
                "centering_min_angular_speed must be positive and no greater than the maximum"
            )
        if not 0 < self.centering_gain <= 2:
            raise ValueError("centering_gain must be in (0, 2]")
        if not 0.05 <= self.target_box_size_normalized <= 0.9:
            raise ValueError("target_box_size_normalized must be in [0.05, 0.9]")
        if not 0.01 <= self.box_size_tolerance <= 0.25:
            raise ValueError("box_size_tolerance must be in [0.01, 0.25]")
        if not 0 < self.centering_max_linear_speed <= 0.25:
            raise ValueError("centering_max_linear_speed must be in (0, 0.25]")
        if not 0 < self.centering_min_linear_speed <= self.centering_max_linear_speed:
            raise ValueError(
                "centering_min_linear_speed must be positive and no greater than the maximum"
            )
        if not 0 < self.centering_linear_gain <= 2:
            raise ValueError("centering_linear_gain must be in (0, 2]")
        if not 1 <= self.centering_stable_frames <= 10:
            raise ValueError("centering_stable_frames must be in [1, 10]")
        if not 0 < self.centering_timeout_sec <= self.tool_timeout_sec:
            raise ValueError("centering_timeout_sec must be in (0, tool_timeout_sec]")
        if not 0 <= self.centering_detection_hold_sec <= 2:
            raise ValueError("centering_detection_hold_sec must be in [0, 2]")
        if not 0 <= self.post_cancel_settle_sec <= 2:
            raise ValueError("post_cancel_settle_sec must be in [0, 2]")
        if not 160 <= self.yolo_input_size <= 1280:
            raise ValueError("yolo_input_size must be in [160, 1280]")
        if not 0.5 <= self.camera_horizontal_fov_rad <= 2.5:
            raise ValueError("camera_horizontal_fov_rad must be in [0.5, 2.5]")
        if not 0.1 <= self.triangulation_min_baseline_m <= 2.0:
            raise ValueError("triangulation_min_baseline_m must be in [0.1, 2.0]")
        if not 0.1 <= self.triangulation_min_ray_angle_deg <= 45.0:
            raise ValueError("triangulation_min_ray_angle_deg must be in [0.1, 45]")
        if not 0.0 <= self.triangulation_min_confidence <= 1.0:
            raise ValueError("triangulation_min_confidence must be in [0, 1]")
        if not 0.2 <= self.baseline_candidate_radius_m <= 3.0:
            raise ValueError("baseline_candidate_radius_m must be in [0.2, 3.0]")
        if not 0.2 <= self.baseline_assumed_object_distance_m <= 10.0:
            raise ValueError(
                "baseline_assumed_object_distance_m must be in [0.2, 10.0]"
            )
        if min(
            self.baseline_score_alpha,
            self.baseline_score_beta,
            self.baseline_score_gamma,
        ) < 0.0:
            raise ValueError("baseline score weights must be non-negative")
        if not 1 <= self.baseline_nav2_candidate_count <= 8:
            raise ValueError("baseline_nav2_candidate_count must be in [1, 8]")
        if not 0.6 <= self.inspection_radius_m <= 3.0:
            raise ValueError("inspection_radius_m must be in [0.6, 3.0]")
        if not 0.6 <= self.inspection_min_radius_m <= self.inspection_max_radius_m:
            raise ValueError("inspection radius bounds are invalid")
        if self.inspection_radius_m > self.inspection_max_radius_m:
            raise ValueError("inspection_radius_m must not exceed inspection_max_radius_m")
        if self.workspace_min_x >= self.workspace_max_x or self.workspace_min_y >= self.workspace_max_y:
            raise ValueError("workspace minimum bounds must be smaller than maximum bounds")

    @classmethod
    def from_env(cls, project_root: Path) -> "RobotAgentSettings":
        turtlebot_root = project_root / "turtlebot3_behavior_demos"
        location_file = Path(
            os.getenv(
                "ROBOT_AGENT_LOCATION_FILE",
                str(turtlebot_root / "tb3_worlds/maps/extinguisher_room_locations.yaml"),
            )
        )
        run_directory = Path(
            os.getenv("ROBOT_AGENT_RUN_DIRECTORY", str(project_root / "robot_agent_runs"))
        )
        return cls(
            location_file=location_file,
            run_directory=run_directory,
            session_id=os.getenv("ROBOT_AGENT_SESSION_ID", "default"),
            trace=os.getenv("ROBOT_AGENT_TRACE", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            nav_action_name=os.getenv("ROBOT_AGENT_NAV_ACTION", "/navigate_to_pose"),
            compute_path_action_name=os.getenv(
                "ROBOT_AGENT_COMPUTE_PATH_ACTION", "/compute_path_to_pose"
            ),
            global_costmap_service=os.getenv(
                "ROBOT_AGENT_GLOBAL_COSTMAP_SERVICE", "/global_costmap/get_costmap"
            ),
            cmd_vel_topic=os.getenv("ROBOT_AGENT_CMD_VEL_TOPIC", "/cmd_vel"),
            camera_topic=os.getenv("ROBOT_AGENT_CAMERA_TOPIC", "/camera/image_raw"),
            annotated_camera_topic=os.getenv(
                "ROBOT_AGENT_ANNOTATED_CAMERA_TOPIC", "/camera/yoloe_annotated"
            ),
            map_frame=os.getenv("ROBOT_AGENT_MAP_FRAME", "map"),
            base_frame=os.getenv("ROBOT_AGENT_BASE_FRAME", "base_link"),
            tool_timeout_sec=_env_float("ROBOT_AGENT_TOOL_TIMEOUT_SEC", 30.0),
            loop_warn_threshold=_env_int("ROBOT_AGENT_LOOP_WARN_THRESHOLD", 3),
            repeated_tool_limit=_env_int("ROBOT_AGENT_REPEATED_TOOL_LIMIT", 5),
            max_tool_calls=_env_int("ROBOT_AGENT_MAX_TOOL_CALLS", 12),
            max_continuations=_env_int("ROBOT_AGENT_MAX_CONTINUATIONS", 12),
            max_no_progress_continuations=_env_int(
                "ROBOT_AGENT_MAX_NO_PROGRESS_CONTINUATIONS", 3
            ),
            bt_navigation_retries=_env_int("ROBOT_AGENT_BT_NAVIGATION_RETRIES", 1),
            detection_interval_sec=_env_float(
                "ROBOT_AGENT_DETECTION_INTERVAL_SEC", 0.2
            ),
            detection_box_threshold=_env_float(
                "ROBOT_AGENT_DETECTION_BOX_THRESHOLD", 0.05
            ),
            detection_confidence_threshold=_env_float(
                "ROBOT_AGENT_DETECTION_CONFIDENCE_THRESHOLD", 0.05
            ),
            detection_tracking_confidence_threshold=_env_float(
                "ROBOT_AGENT_DETECTION_TRACKING_CONFIDENCE_THRESHOLD", 0.01
            ),
            detection_tracking_max_center_jump=_env_float(
                "ROBOT_AGENT_DETECTION_TRACKING_MAX_CENTER_JUMP", 0.25
            ),
            detection_confirmation_frames=_env_int(
                "ROBOT_AGENT_DETECTION_CONFIRMATION_FRAMES", 1
            ),
            center_on_detection=os.getenv(
                "ROBOT_AGENT_CENTER_ON_DETECTION", "true"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            image_center_tolerance=_env_float(
                "ROBOT_AGENT_IMAGE_CENTER_TOLERANCE", 0.10
            ),
            centering_max_angular_speed=_env_float(
                "ROBOT_AGENT_CENTERING_MAX_ANGULAR_SPEED", 0.25
            ),
            centering_min_angular_speed=_env_float(
                "ROBOT_AGENT_CENTERING_MIN_ANGULAR_SPEED", 0.10
            ),
            centering_gain=_env_float("ROBOT_AGENT_CENTERING_GAIN", 0.8),
            target_box_size_normalized=_env_float(
                "ROBOT_AGENT_TARGET_BOX_SIZE_NORMALIZED", 0.4
            ),
            box_size_tolerance=_env_float(
                "ROBOT_AGENT_BOX_SIZE_TOLERANCE", 0.05
            ),
            centering_max_linear_speed=_env_float(
                "ROBOT_AGENT_CENTERING_MAX_LINEAR_SPEED", 0.25
            ),
            centering_min_linear_speed=_env_float(
                "ROBOT_AGENT_CENTERING_MIN_LINEAR_SPEED", 0.08
            ),
            centering_linear_gain=_env_float(
                "ROBOT_AGENT_CENTERING_LINEAR_GAIN", 1.0
            ),
            centering_stable_frames=_env_int(
                "ROBOT_AGENT_CENTERING_STABLE_FRAMES", 3
            ),
            centering_timeout_sec=_env_float(
                "ROBOT_AGENT_CENTERING_TIMEOUT_SEC", 30.0
            ),
            centering_detection_hold_sec=_env_float(
                "ROBOT_AGENT_CENTERING_DETECTION_HOLD_SEC", 1.0
            ),
            post_cancel_settle_sec=_env_float(
                "ROBOT_AGENT_POST_CANCEL_SETTLE_SEC", 0.5
            ),
            yoloe_model=os.getenv("ROBOT_AGENT_YOLOE_MODEL", "yoloe-26s-seg.pt"),
            yoloe_prompt_catalog=Path(
                os.getenv(
                    "ROBOT_AGENT_YOLOE_PROMPT_CATALOG",
                    str(project_root / "sim_assets/yoloe_prompts/catalog.json"),
                )
            ),
            yolo_input_size=_env_int("ROBOT_AGENT_YOLO_INPUT_SIZE", 640),
            camera_horizontal_fov_rad=_env_float(
                "ROBOT_AGENT_CAMERA_HORIZONTAL_FOV_RAD", 1.085595
            ),
            triangulation_min_baseline_m=_env_float(
                "ROBOT_AGENT_TRIANGULATION_MIN_BASELINE_M", 0.25
            ),
            triangulation_min_ray_angle_deg=_env_float(
                "ROBOT_AGENT_TRIANGULATION_MIN_RAY_ANGLE_DEG", 3.0
            ),
            triangulation_min_confidence=_env_float(
                "ROBOT_AGENT_TRIANGULATION_MIN_CONFIDENCE", 0.25
            ),
            baseline_candidate_radius_m=_env_float(
                "ROBOT_AGENT_BASELINE_CANDIDATE_RADIUS_M", 0.75
            ),
            baseline_assumed_object_distance_m=_env_float(
                "ROBOT_AGENT_BASELINE_ASSUMED_OBJECT_DISTANCE_M", 1.5
            ),
            baseline_score_alpha=_env_float(
                "ROBOT_AGENT_BASELINE_SCORE_ALPHA", 3.0
            ),
            baseline_score_beta=_env_float(
                "ROBOT_AGENT_BASELINE_SCORE_BETA", 0.35
            ),
            baseline_score_gamma=_env_float(
                "ROBOT_AGENT_BASELINE_SCORE_GAMMA", 2.0
            ),
            baseline_nav2_candidate_count=_env_int(
                "ROBOT_AGENT_BASELINE_NAV2_CANDIDATE_COUNT", 4
            ),
            inspection_radius_m=_env_float(
                "ROBOT_AGENT_INSPECTION_RADIUS_M", 2.0
            ),
            inspection_min_radius_m=_env_float(
                "ROBOT_AGENT_INSPECTION_MIN_RADIUS_M", 2.0
            ),
            inspection_max_radius_m=_env_float(
                "ROBOT_AGENT_INSPECTION_MAX_RADIUS_M", 2.0
            ),
            workspace_min_x=_env_float("ROBOT_AGENT_WORKSPACE_MIN_X", -10.0),
            workspace_max_x=_env_float("ROBOT_AGENT_WORKSPACE_MAX_X", 10.0),
            workspace_min_y=_env_float("ROBOT_AGENT_WORKSPACE_MIN_Y", -10.0),
            workspace_max_y=_env_float("ROBOT_AGENT_WORKSPACE_MAX_Y", 10.0),
        )
