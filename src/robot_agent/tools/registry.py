"""High-level tool registry with validation, result normalization, and tracing."""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

import yaml
from langchain_core.tools import StructuredTool

from robot_agent.guardrails import SafetyValidator
from robot_agent.middlewares import ToolLoopDetector
from robot_agent.navigation import (
    BaselineCandidate,
    CandidateEvaluation,
    NoFeasibleBaselineError,
    cheap_candidate_score,
    generate_baseline_candidates,
    generate_object_viewpoints,
    select_baseline_candidate,
)
from robot_agent.perception import Detector, YoloeDetector, crop_detection_image
from robot_agent.perception.bearing_localization import (
    bearing_from_detection,
    triangulate_from_bearings,
)
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime.runtime import RobotAgentRuntime
from robot_agent.skills.behavior_tree import BehaviorTreeSkill
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.contracts import (
    CaptureObjectCropInput,
    CircleObjectForInspectionInput,
    ClarificationInput,
    FindObjectInput,
    BehaviorTreeSkillInput,
    NavigateToLocationInput,
    NavigateToPoseInput,
    MoveRelativeInput,
    SearchForObjectInput,
    WaitInput,
)
from robot_agent.world_model import WorldModel


def load_locations(path: Path) -> dict[str, Pose2D]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    locations: dict[str, Pose2D] = {}
    for name, pose in raw.items():
        if not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"Location {name!r} must be [x, y, yaw]")
        locations[name] = Pose2D(float(pose[0]), float(pose[1]), float(pose[2]))
    return locations


def _angle_is_covered(
    angle_deg: float,
    covered_angles_deg: list[float],
    *,
    epsilon_deg: float = 1e-6,
) -> bool:
    return any(
        abs((angle_deg - covered + 180.0) % 360.0 - 180.0) <= epsilon_deg
        for covered in covered_angles_deg
    )


def _select_detection(
    matches: list[Detection],
    *,
    reference: Detection | None = None,
    max_center_jump: float,
    require_image_position: bool = False,
) -> Detection | None:
    eligible = (
        [item for item in matches if item.image_position is not None]
        if require_image_position
        else matches
    )
    if not eligible:
        return None
    if reference is None or reference.image_position is None:
        return max(eligible, key=lambda item: item.confidence)

    previous = reference.image_position
    positioned = [item for item in eligible if item.image_position is not None]
    if not positioned:
        return None
    def center_distance(item: Detection) -> float:
        position = item.image_position
        assert position is not None
        return math.hypot(
            position.x_normalized - previous.x_normalized,
            position.y_normalized - previous.y_normalized,
        )

    nearest = min(positioned, key=center_distance)
    center_jump = center_distance(nearest)
    return nearest if center_jump <= max_center_jump else None


class RobotToolRegistry:
    """Build the final, model-visible high-level tool set for one run."""

    def __init__(
        self,
        runtime: RobotAgentRuntime,
        ros: Ros2Adapter,
        bt_skill: BehaviorTreeSkill,
        detector: Detector | None = None,
    ) -> None:
        self.runtime = runtime
        self.ros = ros
        self.bt_skill = bt_skill
        self.locations = load_locations(runtime.settings.location_file)
        self.world = WorldModel(runtime.state.robot_state)
        self.safety = SafetyValidator(runtime.settings)
        self.detector = detector
        self._execution_lock = threading.RLock()
        self.loop_detector = ToolLoopDetector(
            warn_threshold=runtime.settings.loop_warn_threshold,
            hard_limit=runtime.settings.repeated_tool_limit,
        )

    def _get_detector(self) -> Detector:
        """Load heavyweight perception models only when a tool needs them."""
        if self.detector is None:
            self.detector = YoloeDetector(
                self.runtime.settings.yoloe_model,
                self.runtime.settings.yolo_input_size,
                min(
                    self.runtime.settings.detection_box_threshold,
                    self.runtime.settings.detection_tracking_confidence_threshold,
                ),
                self.runtime.settings.yoloe_prompt_catalog,
            )
        return self.detector

    def _execute(self, tool_name: str, arguments: dict, operation) -> ToolResult:
        # LangGraph may execute multiple tool calls from one AI message on
        # worker threads. Physical robot actions and shared runtime state must
        # remain strictly serial even if upstream middleware is bypassed.
        with self._execution_lock:
            return self._execute_serialized(tool_name, arguments, operation)

    def _execute_serialized(
        self,
        tool_name: str,
        arguments: dict,
        operation,
    ) -> ToolResult:
        loop_decision = None
        if self.runtime.no_progress_exhausted:
            result = ToolResult(
                status=ToolStatus.FAILED,
                data={"tool": tool_name, "arguments": arguments, "blocker": "no_progress"},
                error="Tool execution blocked because the run made no semantic progress",
                retryable=False,
            )
        else:
            loop_decision = self.loop_detector.check(tool_name, arguments)
        if loop_decision is not None and not loop_decision.allowed:
            result = ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "tool": tool_name,
                    "arguments": arguments,
                    "loop_detection": {
                        "count": loop_decision.count,
                        "hard_limit": self.runtime.settings.repeated_tool_limit,
                    },
                },
                error="Repeated identical tool-call loop blocked at the hard limit",
                retryable=False,
            )
        elif loop_decision is not None:
            started = time.monotonic()
            try:
                result = operation()
            except (ValueError, KeyError) as exc:
                result = ToolResult(
                    status=ToolStatus.FAILED,
                    data={"tool": tool_name, "arguments": arguments},
                    error=str(exc),
                    retryable=False,
                )
            except Exception as exc:  # tool boundary: never leak exceptions to the model
                result = ToolResult(
                    status=ToolStatus.FAILED,
                    data={"tool": tool_name, "arguments": arguments},
                    error=f"Tool execution failed: {type(exc).__name__}",
                    retryable=True,
                )
            result.duration_sec = result.duration_sec or time.monotonic() - started
            if loop_decision.warning:
                warning = {
                    "count": loop_decision.count,
                    "hard_limit": self.runtime.settings.repeated_tool_limit,
                    "message": "Identical tool call is repeating; choose a different action or report the blocker.",
                }
                result.data = {**result.data, "loop_warning": warning}
                self.runtime.emit(
                    "loop_warning",
                    {"tool": tool_name, "arguments": arguments, **warning},
                    category="control",
                )
        self.runtime.record_tool_result(tool_name, arguments, result)
        return result

    def _record(self, tool_name: str, arguments: dict, operation) -> dict[str, Any]:
        return self._execute(tool_name, arguments, operation).to_dict()

    @staticmethod
    def _pose_from_result(result: ToolResult) -> Pose2D | None:
        payload = result.data.get("pose")
        if result.status != ToolStatus.SUCCESS or not isinstance(payload, dict):
            return None
        return Pose2D(**payload)

    @staticmethod
    def _detection_from_payload(payload: Any) -> Detection | None:
        return Detection.from_snapshot(payload) if isinstance(payload, dict) else None

    def _run_visual_alignment(
        self,
        on_tick: Callable[[], Detection | None],
        *,
        target_box_size: float,
        box_size_tolerance: float,
    ) -> ToolResult:
        settings = self.runtime.settings
        return self.ros.align_to_detection(
            on_tick,
            tick_interval_sec=settings.detection_interval_sec,
            horizontal_tolerance=settings.image_center_tolerance,
            target_box_size=target_box_size,
            box_size_tolerance=box_size_tolerance,
            max_angular_speed=settings.centering_max_angular_speed,
            min_angular_speed=settings.centering_min_angular_speed,
            angular_gain=settings.centering_gain,
            max_linear_speed=settings.centering_max_linear_speed,
            min_linear_speed=settings.centering_min_linear_speed,
            linear_gain=settings.centering_linear_gain,
            timeout_sec=settings.centering_timeout_sec,
            stable_frames=settings.centering_stable_frames,
            detection_hold_sec=settings.centering_detection_hold_sec,
        )

    def _navigate_to_pose(self, pose: Pose2D, location: str | None = None) -> ToolResult:
        self.safety.validate_pose(pose)
        result = self.ros.navigate_to_pose(pose)
        if result.status == ToolStatus.SUCCESS:
            observed = self.ros.get_pose()
            actual_pose = self._pose_from_result(observed)
            if actual_pose is not None:
                if actual_pose.frame_id != pose.frame_id:
                    self.world.update_navigation_status("needs_verification")
                    result.status = ToolStatus.FAILED
                    result.error = (
                        "Nav2 reported success, but the observed pose is in "
                        f"{actual_pose.frame_id!r} instead of {pose.frame_id!r}"
                    )
                    result.retryable = True
                    result.data = {
                        **result.data,
                        "navigation_goal_succeeded": True,
                        "pose_observation": observed.to_dict(),
                    }
                    return result
                position_error = math.hypot(
                    actual_pose.x - pose.x,
                    actual_pose.y - pose.y,
                )
                yaw_error = math.atan2(
                    math.sin(actual_pose.yaw - pose.yaw),
                    math.cos(actual_pose.yaw - pose.yaw),
                )
                self.world.update_navigation_status("succeeded")
                self.world.update_pose(actual_pose)
                result.data = {
                    **result.data,
                    "actual_pose": actual_pose.to_dict(),
                    "position_error_m": position_error,
                    "yaw_error_rad": yaw_error,
                    "pose_observation": observed.data,
                }
                if location is not None and location not in self.runtime.state.visited_locations:
                    self.runtime.state.visited_locations.append(location)
            else:
                self.world.update_navigation_status("needs_verification")
                result.status = ToolStatus.FAILED
                result.error = "Nav2 reported success, but the final map-frame pose could not be observed"
                result.retryable = True
                result.data = {
                    **result.data,
                    "navigation_goal_succeeded": True,
                    "pose_observation": observed.to_dict(),
                }
        else:
            self.world.update_navigation_status("failed")
        return result

    def _navigate_to_location(self, location: str) -> ToolResult:
        return self._navigate_to_pose(self.locations[location], location=location)

    def _move_relative(self, distance_m: float) -> ToolResult:
        observed = self.ros.get_pose()
        start_pose = self._pose_from_result(observed)
        if start_pose is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "move_relative", "distance_m": distance_m},
                error="Cannot compute relative motion without a live map-frame pose",
                retryable=True,
            )
        if start_pose.frame_id != self.runtime.settings.map_frame:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "move_relative",
                    "distance_m": distance_m,
                    "start_pose": start_pose.to_dict(),
                },
                error="Relative motion requires a live pose in the configured map frame",
                retryable=True,
            )
        target = Pose2D(
            x=start_pose.x + distance_m * math.cos(start_pose.yaw),
            y=start_pose.y + distance_m * math.sin(start_pose.yaw),
            yaw=start_pose.yaw,
            frame_id=start_pose.frame_id,
        )
        result = self._navigate_to_pose(target)
        result.data = {
            **result.data,
            "operation": "move_relative",
            "distance_m": distance_m,
            "start_pose": start_pose.to_dict(),
            "computed_target_pose": target.to_dict(),
        }
        return result

    def _search_for_object(
        self,
        route: list[str],
        color: str | None,
        label: str | None,
    ) -> ToolResult:
        unknown = [location for location in route if location not in self.locations]
        if unknown:
            raise ValueError(f"Unknown search locations: {unknown}")
        detector = self._get_detector()
        detector.validate_query(color=color, label=label)

        observed_frames = 0
        legs: list[dict[str, Any]] = []
        detection_streak = 0
        target_acquired = False
        tracked_detection: Detection | None = None

        # Clear detections from a previous query. During this search, a brief
        # candidate remains visible until the adapter's overlay TTL expires.
        self.ros.update_detection_overlay([])

        def detect_latest() -> Detection | None:
            nonlocal observed_frames
            nonlocal detection_streak
            nonlocal target_acquired
            nonlocal tracked_detection
            image = self.ros.get_camera_frame()
            if image is None:
                return None
            observed_frames += 1
            matches = detector.detect(image, color=color, label=label)
            # Tracking can use weaker boxes after acquisition, but RViz only
            # displays boxes that meet the user-facing box threshold.
            overlay_matches = [
                item
                for item in matches
                if item.confidence >= self.runtime.settings.detection_box_threshold
            ]
            self.ros.update_detection_overlay(overlay_matches)
            active_threshold = (
                self.runtime.settings.detection_tracking_confidence_threshold
                if target_acquired
                else self.runtime.settings.detection_confidence_threshold
            )
            matches = [
                item
                for item in matches
                if item.confidence >= active_threshold
            ]
            if not matches:
                detection_streak = 0
                return None

            candidate = _select_detection(
                matches,
                reference=tracked_detection if target_acquired else None,
                max_center_jump=(
                    self.runtime.settings.detection_tracking_max_center_jump
                ),
            )
            if candidate is None:
                return None
            if not target_acquired:
                detection_streak += 1
                if detection_streak < self.runtime.settings.detection_confirmation_frames:
                    tracked_detection = candidate
                    return None
                target_acquired = True
            tracked_detection = candidate
            return candidate

        for location in route:
            pose = self.locations[location]
            self.safety.validate_pose(pose)
            navigation = self.ros.navigate_to_pose_with_watch(
                pose,
                detect_latest,
                self.runtime.settings.detection_interval_sec,
            )
            legs.append(
                {
                    "location": location,
                    "status": navigation.status.value,
                    "target_pose": pose.to_dict(),
                }
            )
            found_payload = navigation.data.get("found")
            found = self._detection_from_payload(found_payload)
            if found is not None:
                cancellation_confirmed = (
                    navigation.status == ToolStatus.SUCCESS
                    and navigation.data.get("navigation_canceled") is True
                )
                navigation_stopped = (
                    navigation.status == ToolStatus.SUCCESS
                    and (
                        cancellation_confirmed
                        or navigation.data.get("navigation_stopped") is True
                    )
                )
                centering = None
                prealignment = None
                centered: bool | None = None
                final_status = navigation.status
                final_error = navigation.error
                final_retryable = navigation.retryable
                if not navigation_stopped:
                    final_status = ToolStatus.FAILED
                    final_error = (
                        "Target detection was reported while Nav2 was not confirmed stopped"
                    )
                    final_retryable = True
                if (
                    navigation_stopped
                    and self.runtime.settings.center_on_detection
                ):
                    detection_pose_payload = navigation.data.get("detection_pose")
                    if isinstance(detection_pose_payload, dict):
                        detection_pose = Pose2D(**detection_pose_payload)
                    else:
                        detection_pose = self._pose_from_result(self.ros.get_pose())
                    current_pose = self._pose_from_result(self.ros.get_pose())
                    if (
                        detection_pose is not None
                        and current_pose is not None
                        and found.image_position is not None
                    ):
                        target_bearing = bearing_from_detection(
                            found.image_position,
                            detection_pose.yaw,
                            self.runtime.settings.camera_horizontal_fov_rad,
                        )
                        yaw_error = math.atan2(
                            math.sin(target_bearing - current_pose.yaw),
                            math.cos(target_bearing - current_pose.yaw),
                        )
                        if abs(yaw_error) > self.runtime.settings.image_center_tolerance:
                            prealignment = self._navigate_to_pose(
                                Pose2D(
                                    x=current_pose.x,
                                    y=current_pose.y,
                                    yaw=target_bearing,
                                    frame_id=current_pose.frame_id,
                                )
                            )
                    # Nav2 can still rotate slightly while cancellation
                    # settles, so the moving-search bbox is not a safe visual
                    # tracking reference. Reacquire once from rest instead of
                    # rejecting every fresh box as an excessive center jump.
                    tracked_detection = None

                    def detect_for_alignment() -> Detection | None:
                        # The hand-off box predates Nav2 cancellation. Always
                        # reacquire from the stationary camera before issuing
                        # a visual-servo command.
                        return detect_latest()

                    centering = self._run_visual_alignment(
                        detect_for_alignment,
                        target_box_size=self.runtime.settings.target_box_size_normalized,
                        box_size_tolerance=self.runtime.settings.box_size_tolerance,
                    )
                    centered = centering.status == ToolStatus.SUCCESS
                    centered_detection = self._detection_from_payload(
                        centering.data.get("found")
                    )
                    if centered_detection is not None:
                        found = centered_detection
                    # Confirmed detection plus Nav2 cancellation completes the
                    # search. Alignment is best-effort preparation for the
                    # subsequent inspection, not a reason to discard evidence.

                self.world.update_detections([found])
                self.world.update_navigation_status(
                    (
                        "centered_on_detection"
                        if centered
                        else "interrupted_for_detection"
                        if navigation_stopped and centered is None
                        else "needs_verification"
                    ),
                )
                observed = self.ros.get_pose()
                observation_pose = self._pose_from_result(observed)
                if observation_pose is not None:
                    self.world.update_pose(observation_pose)
                return ToolResult(
                    status=final_status,
                    data={
                        "operation": "search_for_object",
                        "found": found.to_dict(),
                        "object_position": (
                            found.position.to_dict() if found.position else None
                        ),
                        "image_position": (
                            found.image_position.to_dict()
                            if found.image_position
                            else None
                        ),
                        "observation_pose": (
                            observation_pose.to_dict() if observation_pose else None
                        ),
                        "route": route,
                        "completed_legs": legs,
                        "observed_frames": observed_frames,
                        "detection_interval_sec": self.runtime.settings.detection_interval_sec,
                        "confidence_threshold": self.runtime.settings.detection_confidence_threshold,
                        "tracking_confidence_threshold": self.runtime.settings.detection_tracking_confidence_threshold,
                        "tracking_max_center_jump": self.runtime.settings.detection_tracking_max_center_jump,
                        "confirmation_frames": self.runtime.settings.detection_confirmation_frames,
                        "navigation_canceled": cancellation_confirmed,
                        "navigation_stopped": navigation_stopped,
                        "centering_requested": self.runtime.settings.center_on_detection,
                        "centered": centered,
                        "centering_incomplete": centered is False,
                        "centering": centering.to_dict() if centering else None,
                        "bearing_prealignment": (
                            prealignment.to_dict()
                            if self.runtime.settings.center_on_detection
                            and prealignment is not None
                            else None
                        ),
                    },
                    error=final_error,
                    retryable=final_retryable,
                )
            if navigation.status != ToolStatus.SUCCESS:
                self.world.update_navigation_status("failed")
                return ToolResult(
                    status=navigation.status,
                    data={
                        "operation": "search_for_object",
                        "route": route,
                        "completed_legs": legs,
                        "navigation": navigation.to_dict(),
                    },
                    error=navigation.error,
                    retryable=navigation.retryable,
                )
            if location not in self.runtime.state.visited_locations:
                self.runtime.state.visited_locations.append(location)

        if observed_frames == 0:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "search_for_object",
                    "route": route,
                    "completed_legs": legs,
                    "observed_frames": 0,
                },
                error="Search route completed without receiving a camera image",
                retryable=True,
            )
        self.world.update_detections([])
        self.world.update_navigation_status("succeeded")
        observed = self.ros.get_pose()
        observed_pose = self._pose_from_result(observed)
        if observed_pose is not None:
            self.world.update_pose(observed_pose)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "search_for_object",
                "found": None,
                "route": route,
                "completed_legs": legs,
                "observed_frames": observed_frames,
                "detection_interval_sec": self.runtime.settings.detection_interval_sec,
            },
        )

    def _detect_object_once(
        self,
        detector: Detector,
        *,
        label: str,
        color: str | None,
        reference: Detection | None = None,
        minimum_confidence: float | None = None,
    ) -> Detection | None:
        image = self.ros.get_camera_frame()
        if image is None:
            return None
        matches = detector.detect(image, color=color, label=label)
        confidence_threshold = (
            self.runtime.settings.detection_confidence_threshold
            if minimum_confidence is None
            else minimum_confidence
        )
        visible = [
            item
            for item in matches
            if item.confidence >= confidence_threshold
        ]
        self.ros.update_detection_overlay(
            [
                item
                for item in matches
                if item.confidence >= self.runtime.settings.detection_box_threshold
            ]
        )
        return _select_detection(
            visible,
            reference=reference,
            max_center_jump=(
                self.runtime.settings.detection_tracking_max_center_jump
            ),
            require_image_position=True,
        )

    def _capture_object_crop(
        self,
        color: str | None,
        label: str | None,
        padding_ratio: float,
    ) -> ToolResult:
        image = self.ros.get_camera_frame()
        if image is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "capture_object_crop"},
                error="No camera image is available",
                retryable=True,
            )

        matches = self._get_detector().detect(image, color=color, label=label)
        visible = [
            item
            for item in matches
            if item.confidence >= self.runtime.settings.detection_box_threshold
        ]
        self.ros.update_detection_overlay(visible)
        selected = _select_detection(
            visible,
            max_center_jump=self.runtime.settings.detection_tracking_max_center_jump,
            require_image_position=True,
        )
        if selected is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "capture_object_crop",
                    "matches": [item.to_dict() for item in visible],
                },
                error="No matching bbox was detected in the current camera frame",
                retryable=True,
            )

        crop = crop_detection_image(
            image,
            selected,
            padding_ratio=padding_ratio,
        )
        if crop is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "capture_object_crop",
                    "detection": selected.to_dict(),
                },
                error="The selected detection has no usable bbox",
                retryable=True,
            )

        try:
            import cv2

            directory = self.runtime.run_path / "object_crops"
            directory.mkdir(parents=True, exist_ok=True)
            capture_index = len(list(directory.glob("crop_*.jpg"))) + 1
            path = directory / f"crop_{capture_index:02d}.jpg"
            if not cv2.imwrite(str(path), crop.image):
                raise RuntimeError("OpenCV did not write the crop")
        except Exception as error:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "capture_object_crop",
                    "detection": selected.to_dict(),
                },
                error=f"Failed to save detection crop: {error}",
                retryable=True,
            )

        self.world.update_detections([selected])
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "capture_object_crop",
                "detection": selected.to_dict(),
                "image_path": str(path),
                "crop": crop.metadata(),
                "padding_ratio": padding_ratio,
            },
        )

    def _detect_object_bounded(
        self,
        detector: Detector,
        *,
        label: str,
        color: str | None,
        attempts: int = 5,
        reference: Detection | None = None,
        minimum_confidence: float | None = None,
    ) -> Detection | None:
        """Tolerate intermittent inference misses without creating an open loop."""
        for attempt in range(attempts):
            detection = self._detect_object_once(
                detector,
                label=label,
                color=color,
                reference=reference,
                minimum_confidence=minimum_confidence,
            )
            if detection is not None:
                return detection
            if attempt + 1 < attempts:
                time.sleep(self.runtime.settings.detection_interval_sec)
        return None

    def _confirmed_search_reference(
        self,
        *,
        label: str,
        color: str | None,
    ) -> Detection | None:
        """Return evidence that this run already acquired the requested target."""
        for entry in reversed(self.runtime.state.tool_history):
            if entry.get("tool") != "search_for_object":
                continue
            result = entry.get("result") or {}
            if result.get("status") != ToolStatus.SUCCESS.value:
                continue
            detection = self._detection_from_payload(
                (result.get("data") or {}).get("found")
            )
            if detection is None:
                continue
            if detection.label != label:
                continue
            if color is not None and detection.color != color:
                continue
            return detection
        return None

    def _evaluate_candidate(self, candidate) -> CandidateEvaluation:
        try:
            self.safety.validate_pose(candidate.pose)
        except ValueError as exc:
            return CandidateEvaluation(
                candidate=candidate,
                feasible=False,
                path_length_m=math.inf,
                obstacle_risk=1.0,
                reason=str(exc),
            )
        result = self.ros.evaluate_navigation_candidate(candidate.pose)
        data = result.data
        feasible = (
            result.status == ToolStatus.SUCCESS
            and data.get("feasible") is True
        )
        return CandidateEvaluation(
            candidate=candidate,
            feasible=feasible,
            path_length_m=float(data.get("path_length_m", math.inf)),
            obstacle_risk=float(data.get("obstacle_risk", 1.0)),
            reason=result.error,
        )

    def _select_feasible_baseline(
        self,
        candidates: list[BaselineCandidate],
    ) -> tuple[CandidateEvaluation, list[CandidateEvaluation]]:
        settings = self.runtime.settings
        ranked = sorted(
            candidates,
            key=lambda item: cheap_candidate_score(
                item,
                alpha=settings.baseline_score_alpha,
                beta=settings.baseline_score_beta,
            ),
            reverse=True,
        )
        top_count = min(settings.baseline_nav2_candidate_count, len(ranked))
        evaluations = [self._evaluate_candidate(item) for item in ranked[:top_count]]

        # Usually only the strongest cheap-score candidates need expensive
        # Nav2 queries. If all are blocked, continue so "all infeasible" is
        # based on the complete fan rather than an arbitrary shortlist.
        if not any(item.feasible for item in evaluations):
            evaluations.extend(
                self._evaluate_candidate(item) for item in ranked[top_count:]
            )
        selected = select_baseline_candidate(
            evaluations,
            alpha=settings.baseline_score_alpha,
            beta=settings.baseline_score_beta,
            gamma=settings.baseline_score_gamma,
        )
        return selected, evaluations

    def _align_at_viewpoint(
        self,
        detector: Detector,
        *,
        label: str,
        color: str | None,
        initial_detection: Detection | None,
        adjust_size: bool = True,
    ) -> ToolResult:
        tracked: Detection | None = None

        def detect_for_alignment() -> Detection | None:
            nonlocal tracked
            # Never drive from the hand-off box: it may predate Nav2
            # cancellation or the latest viewpoint move. Reacquire once from
            # the stationary camera, then enforce continuity between frames.
            detection = self._detect_object_once(
                detector,
                label=label,
                color=color,
                reference=tracked,
                minimum_confidence=(
                    self.runtime.settings.detection_tracking_confidence_threshold
                ),
            )
            if detection is not None:
                tracked = detection
            return detection

        settings = self.runtime.settings
        seed_height = (
            initial_detection.image_position.height_normalized
            if initial_detection is not None
            and initial_detection.image_position is not None
            else settings.target_box_size_normalized
        )
        return self._run_visual_alignment(
            detect_for_alignment,
            target_box_size=(
                settings.target_box_size_normalized if adjust_size else seed_height
            ),
            box_size_tolerance=(settings.box_size_tolerance if adjust_size else 1.0),
        )

    def _scan_for_object_in_place(
        self,
        detector: Detector,
        *,
        label: str,
        color: str | None,
        pose: Pose2D,
        minimum_confidence: float,
    ) -> tuple[Detection | None, list[dict[str, Any]], ToolResult | None]:
        """Sweep counterclockwise once without changing the robot position."""
        attempts: list[dict[str, Any]] = []
        step_rad = math.pi / 4.0
        for step in range(1, 9):
            yaw = math.atan2(
                math.sin(pose.yaw + step * step_rad),
                math.cos(pose.yaw + step * step_rad),
            )
            scan_pose = Pose2D(
                x=pose.x,
                y=pose.y,
                yaw=yaw,
                frame_id=pose.frame_id,
            )
            navigation = self._navigate_to_pose(scan_pose)
            attempt = {
                "step": step,
                "yaw": yaw,
                "navigation_status": navigation.status.value,
                "detected": False,
            }
            attempts.append(attempt)
            if navigation.status != ToolStatus.SUCCESS:
                return None, attempts, navigation
            detection = self._detect_object_bounded(
                detector,
                label=label,
                color=color,
                minimum_confidence=minimum_confidence,
            )
            if detection is not None:
                attempt["detected"] = True
                return detection, attempts, None
        return None, attempts, None

    @staticmethod
    def _inspection_arrival_pose(
        viewpoint: Pose2D,
        object_pose: Pose2D,
    ) -> Pose2D:
        """Approach each viewpoint along the counterclockwise orbit tangent."""
        orbit_angle = math.atan2(
            viewpoint.y - object_pose.y,
            viewpoint.x - object_pose.x,
        )
        tangent_yaw = math.atan2(
            math.sin(orbit_angle + math.pi / 2.0),
            math.cos(orbit_angle + math.pi / 2.0),
        )
        return Pose2D(
            x=viewpoint.x,
            y=viewpoint.y,
            yaw=tangent_yaw,
            frame_id=viewpoint.frame_id,
        )

    def _save_inspection_frame(
        self,
        index: int,
        detection: Detection,
    ) -> str | None:
        image = self.ros.get_camera_frame()
        if image is None:
            return None
        try:
            import cv2

            crop = crop_detection_image(image, detection)
            if crop is None:
                return None
            directory = self.runtime.run_path / "object_views"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"view_{index:02d}_crop.jpg"
            if not cv2.imwrite(str(path), crop.image):
                return None
            return str(path)
        except Exception:  # Capturing evidence must not interrupt robot control.
            return None

    def _inspection_failure(
        self,
        extra: dict[str, Any],
        error: str,
        *,
        retryable: bool = True,
        status: ToolStatus = ToolStatus.FAILED,
    ) -> ToolResult:
        return ToolResult(
            status=status,
            data={"operation": "circle_object_for_inspection", **extra},
            error=error,
            retryable=retryable,
        )

    def _resolve_inspection_viewpoint(
        self,
        target: Pose2D,
        object_pose: Pose2D,
        *,
        radius_m: float,
        base_angle_deg: float,
        excluded_candidate_names: set[str] | None = None,
    ) -> CandidateEvaluation:
        excluded = excluded_candidate_names or set()
        direct = BaselineCandidate(
            name="planned_viewpoint",
            pose=target,
            tangential_displacement_m=0.0,
            straight_line_distance_m=0.0,
        )
        candidates = [direct]
        for offset_deg in (15.0, 30.0, 45.0):
            angle = math.radians(base_angle_deg + offset_deg)
            x = object_pose.x + radius_m * math.cos(angle)
            y = object_pose.y + radius_m * math.sin(angle)
            offset_rad = math.radians(offset_deg)
            candidates.append(
                BaselineCandidate(
                    name=f"orbit_fallback_{offset_deg:+g}deg",
                    pose=Pose2D(
                        x=x,
                        y=y,
                        yaw=math.atan2(object_pose.y - y, object_pose.x - x),
                        frame_id=target.frame_id,
                    ),
                    tangential_displacement_m=abs(
                        2.0 * radius_m * math.sin(offset_rad / 2.0)
                    ),
                    straight_line_distance_m=math.hypot(x - target.x, y - target.y),
                )
            )

        for candidate in candidates:
            if candidate.name in excluded:
                continue
            evaluation = self._evaluate_candidate(candidate)
            if evaluation.feasible:
                return evaluation

        raise NoFeasibleBaselineError(
            "No untried feasible planned viewpoint or orbit-angle fallback remains"
        )

    def _circle_object_for_inspection(
        self,
        label: str,
        color: str | None,
        viewpoint_count: int,
        radius_m: float | None,
    ) -> ToolResult:
        detector = self._get_detector()
        detector.validate_query(color=color, label=label)
        search_reference = self._confirmed_search_reference(label=label, color=color)
        tracking_confidence = (
            self.runtime.settings.detection_tracking_confidence_threshold
            if search_reference is not None
            else None
        )
        # Bearing geometry requires a camera observation synchronized with the
        # live pose below. Persisted world-model boxes may come from a previous
        # robot pose and must never be reused for triangulation.
        first_detection = self._detect_object_bounded(
            detector,
            label=label,
            color=color,
            reference=search_reference,
            minimum_confidence=tracking_confidence,
        )
        if first_detection is None or first_detection.image_position is None:
            return self._inspection_failure(
                {}, "Object must be visible before active baseline planning"
            )
        first_pose_result = self.ros.get_pose()
        first_pose = self._pose_from_result(first_pose_result)
        if first_pose is None:
            return self._inspection_failure(
                {}, "A live map-frame robot pose is required for object localization"
            )
        first_bearing = bearing_from_detection(
            first_detection.image_position,
            first_pose.yaw,
            self.runtime.settings.camera_horizontal_fov_rad,
        )

        candidates = generate_baseline_candidates(
            first_pose,
            first_bearing,
            radius_m=self.runtime.settings.baseline_candidate_radius_m,
            assumed_object_distance_m=(
                self.runtime.settings.baseline_assumed_object_distance_m
            ),
        )
        try:
            selected, evaluations = self._select_feasible_baseline(candidates)
        except NoFeasibleBaselineError as exc:
            return self._inspection_failure(
                {
                    "baseline_candidates": [
                        {
                            "name": item.candidate.name,
                            "pose": item.candidate.pose.to_dict(),
                            "feasible": item.feasible,
                            "reason": item.reason,
                        }
                        for item in evaluations
                    ],
                },
                str(exc),
            )

        baseline_navigation = self._navigate_to_pose(selected.candidate.pose)
        if baseline_navigation.status != ToolStatus.SUCCESS:
            return self._inspection_failure(
                {
                    "selected_baseline": selected.candidate.pose.to_dict(),
                    "navigation": baseline_navigation.to_dict(),
                },
                baseline_navigation.error or "Baseline navigation failed",
                retryable=baseline_navigation.retryable,
                status=baseline_navigation.status,
            )

        second_detection = self._detect_object_bounded(
            detector,
            label=label,
            color=color,
            minimum_confidence=tracking_confidence,
        )
        second_pose_result = self.ros.get_pose()
        if second_detection is None or second_detection.image_position is None:
            return self._inspection_failure(
                {
                    "selected_baseline": selected.candidate.pose.to_dict(),
                },
                "Object was not visible after baseline navigation",
            )
        second_pose = self._pose_from_result(second_pose_result)
        if second_pose is None:
            return self._inspection_failure(
                {}, "Robot pose was unavailable after baseline motion"
            )
        second_bearing = bearing_from_detection(
            second_detection.image_position,
            second_pose.yaw,
            self.runtime.settings.camera_horizontal_fov_rad,
        )
        localization = triangulate_from_bearings(
            first_pose,
            first_bearing,
            second_pose,
            second_bearing,
            minimum_baseline_m=self.runtime.settings.triangulation_min_baseline_m,
            min_ray_angle_rad=math.radians(
                self.runtime.settings.triangulation_min_ray_angle_deg
            ),
        )
        if (
            not localization.valid
            or localization.position is None
            or localization.confidence
            < self.runtime.settings.triangulation_min_confidence
        ):
            return self._inspection_failure(
                {
                    "first_pose": first_pose.to_dict(),
                    "second_pose": second_pose.to_dict(),
                    "first_bearing_rad": first_bearing,
                    "second_bearing_rad": second_bearing,
                    "baseline_m": localization.baseline_m,
                    "ray_angle_rad": localization.ray_angle_rad,
                    "confidence": localization.confidence,
                },
                localization.reason or "Bearing triangulation confidence is too low",
            )

        localized_detection = Detection(
            label=second_detection.label,
            confidence=second_detection.confidence,
            color=second_detection.color,
            position=localization.position,
            image_position=second_detection.image_position,
        )
        self.world.update_detections([localized_detection])
        settings = self.runtime.settings
        requested_radius = radius_m or settings.inspection_radius_m
        bbox_height = second_detection.image_position.height_normalized
        legibility_radius = (
            (localization.second_range_m or requested_radius)
            * bbox_height
            / settings.target_box_size_normalized
        )
        inspection_radius = max(
            settings.inspection_min_radius_m,
            min(requested_radius, legibility_radius, settings.inspection_max_radius_m),
        )

        viewpoints = generate_object_viewpoints(
            localization.position,
            second_pose,
            radius_m=inspection_radius,
            count=viewpoint_count,
        )
        inspection_state = {
            "label": label,
            "color": color,
            "status": "running",
            "object_position": localization.position.to_dict(),
            "inspection_radius_m": inspection_radius,
            "viewpoint_count": viewpoint_count,
            "captures": [],
        }
        self.runtime.state.inspection_state = inspection_state
        captures: list[dict[str, Any]] = []
        covered_angles: list[float] = []
        for index, planned_viewpoint in enumerate(viewpoints, start=1):
            planned_angle = math.degrees(
                math.atan2(
                    planned_viewpoint.y - localization.position.y,
                    planned_viewpoint.x - localization.position.x,
                )
            ) % 360.0
            if _angle_is_covered(planned_angle, covered_angles):
                continue
            attempted_candidates: set[str] = set()
            last_alignment: ToolResult | None = None
            while True:
                try:
                    evaluation = self._resolve_inspection_viewpoint(
                        planned_viewpoint,
                        localization.position,
                        radius_m=inspection_radius,
                        base_angle_deg=planned_angle,
                        excluded_candidate_names=attempted_candidates,
                    )
                except NoFeasibleBaselineError as exc:
                    failure_data: dict[str, Any] = {
                        "object_position": localization.position.to_dict(),
                        "captures": captures,
                        "failed_viewpoint": planned_viewpoint.to_dict(),
                    }
                    if last_alignment is not None:
                        failure_data["alignment"] = last_alignment.to_dict()
                    error = (
                        "Target was not visible from the planned viewpoint or any "
                        "feasible orbit-angle fallback"
                        if attempted_candidates
                        else str(exc)
                    )
                    return self._inspection_failure(failure_data, error)

                viewpoint = evaluation.candidate.pose
                arrival_pose = self._inspection_arrival_pose(
                    viewpoint,
                    localization.position,
                )
                navigation = self._navigate_to_pose(arrival_pose)
                if navigation.status != ToolStatus.SUCCESS:
                    return self._inspection_failure(
                        {
                            "object_position": localization.position.to_dict(),
                            "captures": captures,
                            "failed_viewpoint": viewpoint.to_dict(),
                        },
                        navigation.error or "Inspection viewpoint navigation failed",
                        retryable=navigation.retryable,
                        status=navigation.status,
                    )

                observed = self._detect_object_once(
                    detector,
                    label=label,
                    color=color,
                    minimum_confidence=tracking_confidence,
                )
                scan_attempts: list[dict[str, Any]] = []
                if observed is None:
                    scan_pose = self._pose_from_result(self.ros.get_pose()) or viewpoint
                    observed, scan_attempts, scan_failure = (
                        self._scan_for_object_in_place(
                            detector,
                            label=label,
                            color=color,
                            pose=scan_pose,
                            minimum_confidence=tracking_confidence,
                        )
                    )
                    if scan_failure is not None:
                        return self._inspection_failure(
                            {
                                "object_position": localization.position.to_dict(),
                                "captures": captures,
                                "failed_viewpoint": viewpoint.to_dict(),
                                "scan_attempts": scan_attempts,
                            },
                            scan_failure.error or "In-place visual scan failed",
                            retryable=scan_failure.retryable,
                            status=scan_failure.status,
                        )
                    if observed is None:
                        self.ros.stop_robot()
                        return self._inspection_failure(
                            {
                                "object_position": localization.position.to_dict(),
                                "captures": captures,
                                "failed_viewpoint": viewpoint.to_dict(),
                                "scan_attempts": scan_attempts,
                            },
                            "Target was not visible during a full in-place scan",
                        )
                alignment = self._align_at_viewpoint(
                    detector,
                    label=label,
                    color=color,
                    initial_detection=observed,
                )
                aligned_detection = self._detection_from_payload(
                    alignment.data.get("found")
                )
                detection_age_sec = alignment.data.get("detection_age_sec")
                stale_timeout_detection = (
                    alignment.status == ToolStatus.TIMEOUT
                    and (
                        not isinstance(detection_age_sec, (int, float))
                        or detection_age_sec
                        > self.runtime.settings.centering_detection_hold_sec
                    )
                )
                if aligned_detection is None or stale_timeout_detection:
                    self.ros.stop_robot()
                    return self._inspection_failure(
                        {
                            "object_position": localization.position.to_dict(),
                            "captures": captures,
                            "failed_viewpoint": viewpoint.to_dict(),
                            "scan_attempts": scan_attempts,
                            "alignment": alignment.to_dict(),
                        },
                        alignment.error
                        or "Target was lost during viewpoint alignment",
                        retryable=alignment.retryable,
                        status=alignment.status,
                    )

                observed = aligned_detection
                alignment_timed_out_with_fresh_detection = (
                    alignment.status == ToolStatus.TIMEOUT
                    and aligned_detection.image_position is not None
                    and isinstance(detection_age_sec, (int, float))
                    and detection_age_sec
                    <= self.runtime.settings.centering_detection_hold_sec
                )
                if (
                    alignment.status != ToolStatus.SUCCESS
                    and not alignment_timed_out_with_fresh_detection
                ):
                    return self._inspection_failure(
                        {
                            "object_position": localization.position.to_dict(),
                            "captures": captures,
                            "failed_viewpoint": viewpoint.to_dict(),
                            "alignment": alignment.to_dict(),
                        },
                        alignment.error or "Object alignment failed at inspection viewpoint",
                        retryable=alignment.retryable,
                        status=alignment.status,
                    )
                if alignment_timed_out_with_fresh_detection:
                    # Alignment already holds zero velocity on timeout. Reinforce
                    # the stop before recording the best available view and moving
                    # on, rather than blocking the complete inspection orbit.
                    self.ros.stop_robot()
                break

            actual_pose_result = self.ros.get_pose()
            actual_pose_observation = self._pose_from_result(actual_pose_result)
            actual_pose = (
                actual_pose_observation.to_dict()
                if actual_pose_observation is not None
                else None
            )
            capture = {
                "index": index,
                "planned_pose": planned_viewpoint.to_dict(),
                "arrival_pose": arrival_pose.to_dict(),
                "pose": actual_pose or viewpoint.to_dict(),
                "planned_angle_deg": planned_angle,
                "used_fallback": evaluation.candidate.name != "planned_viewpoint",
                "detected": True,
                "confidence": observed.confidence,
                "image_position": (
                    observed.image_position.to_dict()
                    if observed.image_position
                    else None
                ),
                "aligned": alignment.status == ToolStatus.SUCCESS,
                "alignment_status": alignment.status.value,
                "alignment_error": alignment.error,
                "scan_attempts": scan_attempts,
                "path_length_m": evaluation.path_length_m,
                "image_path": self._save_inspection_frame(index, observed),
            }
            captures.append(capture)
            if not _angle_is_covered(planned_angle, covered_angles):
                covered_angles.append(planned_angle)
            inspection_state["captures"] = captures

        inspection_state["status"] = "completed"
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "circle_object_for_inspection",
                "object_position": localization.position.to_dict(),
                "baseline_m": localization.baseline_m,
                "parallax_rad": localization.ray_angle_rad,
                "triangulation_confidence": localization.confidence,
                "inspection_radius_m": inspection_radius,
                "viewpoint_count": len(viewpoints),
                "covered_angles_deg": covered_angles,
                "captures": captures,
            },
        )

    def _wait_for(self, seconds: float) -> ToolResult:
        duration = self.safety.validate_wait(seconds)
        time.sleep(duration)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"operation": "wait", "seconds": duration},
        )

    def build(self) -> list[StructuredTool]:
        def get_known_locations() -> dict[str, Any]:
            return self._record(
                "get_known_locations",
                {},
                lambda: ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={"locations": {name: pose.to_dict() for name, pose in self.locations.items()}},
                ),
            )

        def get_robot_state() -> dict[str, Any]:
            def operation() -> ToolResult:
                observed = self.ros.get_pose()
                observed_pose = self._pose_from_result(observed)
                if observed_pose is not None:
                    self.world.update_pose(observed_pose)
                return ToolResult(
                    status=observed.status,
                    data={
                        "robot_state": self.world.context(),
                        "transport": observed.data,
                    },
                    error=observed.error,
                    retryable=observed.retryable,
                )

            return self._record("get_robot_state", {}, operation)

        def navigate_to(location: str) -> dict[str, Any]:
            return self._record(
                "navigate_to",
                {"location": location},
                lambda: self._navigate_to_location(location),
            )

        def navigate_to_pose(
            x: float,
            y: float,
            yaw: float = 0.0,
            frame_id: str = "map",
        ) -> dict[str, Any]:
            arguments = {"x": x, "y": y, "yaw": yaw, "frame_id": frame_id}
            pose = Pose2D(x=x, y=y, yaw=yaw, frame_id=frame_id)
            return self._record(
                "navigate_to_pose",
                arguments,
                lambda: self._navigate_to_pose(pose),
            )

        def move_relative(distance_m: float) -> dict[str, Any]:
            return self._record(
                "move_relative",
                {"distance_m": distance_m},
                lambda: self._move_relative(distance_m),
            )

        def find_object(color: str | None = None, label: str | None = None) -> dict[str, Any]:
            def operation() -> ToolResult:
                if not self.world.has_perception_observation():
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        data={"matches": [], "observation_available": False},
                        error="No perception observation is available",
                        retryable=True,
                    )
                matches = self.world.find(color=color, label=label)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "matches": [item.to_dict() for item in matches],
                        "observation_available": True,
                    },
                )

            return self._record("find_object", {"color": color, "label": label}, operation)

        def capture_object_crop(
            color: str | None = None,
            label: str | None = None,
            padding_ratio: float = 0.05,
        ) -> dict[str, Any]:
            arguments = {
                "color": color,
                "label": label,
                "padding_ratio": padding_ratio,
            }
            return self._record(
                "capture_object_crop",
                arguments,
                lambda: self._capture_object_crop(color, label, padding_ratio),
            )

        def search_for_object(
            route: list[str],
            color: str | None = None,
            label: str | None = None,
        ) -> dict[str, Any]:
            arguments = {"route": route, "color": color, "label": label}
            return self._record(
                "search_for_object",
                arguments,
                lambda: self._search_for_object(route, color, label),
            )

        def circle_object_for_inspection(
            label: str,
            color: str | None = None,
            viewpoint_count: int = 4,
            radius_m: float | None = None,
        ) -> dict[str, Any]:
            arguments = {
                "label": label,
                "color": color,
                "viewpoint_count": viewpoint_count,
                "radius_m": radius_m,
            }
            return self._record(
                "circle_object_for_inspection",
                arguments,
                lambda: self._circle_object_for_inspection(
                    label,
                    color,
                    viewpoint_count,
                    radius_m,
                ),
            )

        def run_behavior_tree(goal: str) -> dict[str, Any]:
            def node_started(index: int, node) -> None:
                self.runtime.state.current_bt_node_index = index
                self.runtime.emit(
                    "behavior_tree_node_started",
                    {"index": index, "node": node.model_dump()},
                    category="skill",
                )
                self.runtime.save_checkpoint()

            def navigate(location: str, index: int) -> ToolResult:
                return self._execute(
                    "run_behavior_tree.GoToPose",
                    {"location": location, "node_index": index},
                    lambda: self._navigate_to_location(location),
                )

            def wait(seconds: float, index: int) -> ToolResult:
                return self._execute(
                    "run_behavior_tree.Wait",
                    {"seconds": seconds, "node_index": index},
                    lambda: self._wait_for(seconds),
                )

            def stop(index: int) -> ToolResult:
                return self._execute(
                    "run_behavior_tree.Stop",
                    {"node_index": index},
                    self.ros.stop_robot,
                )

            def abort() -> ToolResult:
                return self._execute(
                    "run_behavior_tree.AbortStop",
                    {"node_index": self.runtime.state.current_bt_node_index or 0},
                    self.ros.stop_robot,
                )

            return self._record(
                "run_behavior_tree",
                {"goal": goal},
                lambda: self.bt_skill.run(
                    goal,
                    navigate=navigate,
                    stop=stop,
                    wait=wait,
                    abort=abort,
                    on_node_started=node_started,
                ),
            )

        def stop_robot() -> dict[str, Any]:
            return self._record("stop_robot", {}, self.ros.stop_robot)

        def wait_seconds(seconds: float) -> dict[str, Any]:
            return self._record(
                "wait_seconds",
                {"seconds": seconds},
                lambda: self._wait_for(seconds),
            )

        def request_clarification(question: str, reason: str) -> dict[str, Any]:
            return self._record(
                "request_clarification",
                {"question": question, "reason": reason},
                lambda: ToolResult(
                    status=ToolStatus.NEEDS_INPUT,
                    data={"question": question, "reason": reason},
                    retryable=False,
                ),
            )

        return [
            StructuredTool.from_function(get_known_locations, description="Return known named navigation locations."),
            StructuredTool.from_function(get_robot_state, description="Return concise semantic robot state; never raw ROS messages."),
            StructuredTool.from_function(navigate_to, args_schema=NavigateToLocationInput, description="Navigate with Nav2 to one known named location."),
            StructuredTool.from_function(
                navigate_to_pose,
                args_schema=NavigateToPoseInput,
                description="Navigate with Nav2 to explicit x, y map coordinates and an optional yaw heading in radians.",
            ),
            StructuredTool.from_function(
                move_relative,
                args_schema=MoveRelativeInput,
                description=(
                    "Move a signed distance in meters along the robot's live current heading. "
                    "The tool reads TF and computes the map target deterministically."
                ),
            ),
            StructuredTool.from_function(find_object, args_schema=FindObjectInput, description="Query the semantic world model for a visible object."),
            StructuredTool.from_function(
                capture_object_crop,
                args_schema=CaptureObjectCropInput,
                description=(
                    "Run perception on the current camera frame and save only the "
                    "selected object's bbox crop for compact multimodal inspection. "
                    "This tool does not move the robot."
                ),
            ),
            StructuredTool.from_function(
                search_for_object,
                args_schema=SearchForObjectInput,
                description=(
                    "Search for an object while Nav2 continuously traverses an ordered route of known locations; "
                    "navigation is canceled only when detection reaches the configured confidence threshold. "
                    "Pass a concrete open-vocabulary text label for YOLOE."
                ),
            ),
            StructuredTool.from_function(
                circle_object_for_inspection,
                args_schema=CircleObjectForInspectionInput,
                description=(
                    "Inspect, but never search for, a target already confirmed in "
                    "the current run. In one deterministic call, localize it using a "
                    "costmap-aware baseline, plan evenly spaced viewpoints, then "
                    "navigate, align once, and photograph each viewpoint."
                ),
            ),
            StructuredTool.from_function(run_behavior_tree, args_schema=BehaviorTreeSkillInput, description="Generate, validate, persist, and execute a behavior tree skill for a multi-step TurtleBot goal."),
            StructuredTool.from_function(
                stop_robot,
                description="Stop the robot. The rclpy backend also cancels this run's active Nav2 goal; CLI is best-effort zero velocity.",
            ),
            StructuredTool.from_function(wait_seconds, args_schema=WaitInput, description="Wait for a bounded number of seconds."),
            StructuredTool.from_function(
                request_clarification,
                args_schema=ClarificationInput,
                description="Ask one structured question when no safe unambiguous robot action is available.",
            ),
        ]
