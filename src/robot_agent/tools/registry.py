"""High-level tool registry with validation, result normalization, and tracing."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import yaml
from langchain_core.tools import StructuredTool

from robot_agent.guardrails import SafetyValidator
from robot_agent.middlewares import ToolLoopDetector
from robot_agent.perception import Detector, build_detector
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime.runtime import RobotAgentRuntime
from robot_agent.skills.behavior_tree import BehaviorTreeSkill
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.contracts import (
    ClarificationInput,
    FindObjectInput,
    InspectForColorInput,
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
        self.loop_detector = ToolLoopDetector(
            warn_threshold=runtime.settings.loop_warn_threshold,
            hard_limit=runtime.settings.repeated_tool_limit,
        )

    def _get_detector(self) -> Detector:
        """Load heavyweight perception models only when a tool needs them."""
        if self.detector is None:
            self.detector = build_detector(
                self.runtime.settings.detector_backend,
                yolo_model=self.runtime.settings.yolo_model,
                yoloe_model=self.runtime.settings.yoloe_model,
                yolo_input_size=self.runtime.settings.yolo_input_size,
                confidence_threshold=min(
                    self.runtime.settings.detection_box_threshold,
                    self.runtime.settings.detection_tracking_confidence_threshold,
                ),
            )
            self.runtime.emit(
                "detector_ready",
                {
                    "backend": self.runtime.settings.detector_backend,
                    "model": (
                        self.runtime.settings.yoloe_model
                        if self.runtime.settings.detector_backend == "yoloe"
                        else self.runtime.settings.yolo_model
                        if self.runtime.settings.detector_backend == "yolo"
                        else None
                    ),
                    "interval_sec": self.runtime.settings.detection_interval_sec,
                    "target_frequency_hz": round(
                        1.0 / self.runtime.settings.detection_interval_sec, 3
                    ),
                    "box_threshold": self.runtime.settings.detection_box_threshold,
                    "stop_threshold": self.runtime.settings.detection_confidence_threshold,
                    "tracking_threshold": self.runtime.settings.detection_tracking_confidence_threshold,
                    "tracking_max_center_jump": self.runtime.settings.detection_tracking_max_center_jump,
                },
                category="perception",
            )
        return self.detector

    def _execute(self, tool_name: str, arguments: dict, operation) -> ToolResult:
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

    def _navigate_to_pose(self, pose: Pose2D, location: str | None = None) -> ToolResult:
        self.safety.validate_pose(pose)
        result = self.ros.navigate_to_pose(pose)
        if result.status == ToolStatus.SUCCESS:
            observed = self.ros.get_pose()
            if observed.status == ToolStatus.SUCCESS and observed.data.get("pose"):
                actual_pose = Pose2D(**observed.data["pose"])
                if actual_pose.frame_id != pose.frame_id:
                    self.world.update_navigation_status("needs_verification", planned_pose=None)
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
                self.world.update_navigation_status("succeeded", planned_pose=None)
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
                self.world.update_navigation_status("needs_verification", planned_pose=None)
                result.status = ToolStatus.FAILED
                result.error = "Nav2 reported success, but the final map-frame pose could not be observed"
                result.retryable = True
                result.data = {
                    **result.data,
                    "navigation_goal_succeeded": True,
                    "pose_observation": observed.to_dict(),
                }
        elif result.status == ToolStatus.PLANNED:
            self.world.update_navigation_status("planned", planned_pose=pose)
        else:
            self.world.update_navigation_status("failed", planned_pose=None)
        return result

    def _navigate_to_location(self, location: str) -> ToolResult:
        return self._navigate_to_pose(self.locations[location], location=location)

    def _move_relative(self, distance_m: float) -> ToolResult:
        observed = self.ros.get_pose()
        if observed.status != ToolStatus.SUCCESS or not observed.data.get("pose"):
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "move_relative", "distance_m": distance_m},
                error="Cannot compute relative motion without a live map-frame pose",
                retryable=True,
            )
        start_pose = Pose2D(**observed.data["pose"])
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
        diagnostic_last_emitted_at = 0.0
        diagnostic_samples = 0
        diagnostic_candidate_frames = 0
        diagnostic_candidates = 0
        diagnostic_best_confidence: float | None = None
        detection_streak = 0
        target_acquired = False
        tracked_detection: Detection | None = None

        # Clear detections from a previous query. During this search, a brief
        # candidate remains visible until the adapter's overlay TTL expires.
        self.ros.update_detection_overlay([])

        def detect_latest() -> Detection | None:
            nonlocal observed_frames
            nonlocal diagnostic_last_emitted_at
            nonlocal diagnostic_samples
            nonlocal diagnostic_candidate_frames
            nonlocal diagnostic_candidates
            nonlocal diagnostic_best_confidence
            nonlocal detection_streak
            nonlocal target_acquired
            nonlocal tracked_detection
            image = self.ros.get_camera_frame()
            if image is None:
                return None
            observed_frames += 1
            inference_started = time.monotonic()
            matches = detector.detect(image, color=color, label=label)
            inference_ms = (time.monotonic() - inference_started) * 1000.0
            # Tracking can use weaker boxes after acquisition, but RViz only
            # displays boxes that meet the user-facing box threshold.
            overlay_matches = [
                item
                for item in matches
                if item.confidence >= self.runtime.settings.detection_box_threshold
            ]
            self.ros.update_detection_overlay(overlay_matches)
            diagnostic_samples += 1
            diagnostic_candidates += len(matches)
            if matches:
                diagnostic_candidate_frames += 1
                frame_best = max(item.confidence for item in matches)
                diagnostic_best_confidence = max(
                    diagnostic_best_confidence or frame_best,
                    frame_best,
                )
            now = time.monotonic()
            stop_candidate = any(
                item.confidence
                >= self.runtime.settings.detection_confidence_threshold
                for item in matches
            )
            if (
                diagnostic_last_emitted_at == 0.0
                or now - diagnostic_last_emitted_at >= 1.0
                or stop_candidate
            ):
                self.runtime.emit(
                    "detector_sampled",
                    {
                        "backend": self.runtime.settings.detector_backend,
                        "query": {"label": label, "color": color},
                        "samples": diagnostic_samples,
                        "candidate_frames": diagnostic_candidate_frames,
                        "candidates": diagnostic_candidates,
                        "best_confidence": diagnostic_best_confidence,
                        "last_inference_ms": round(inference_ms, 1),
                        "box_threshold": self.runtime.settings.detection_box_threshold,
                        "stop_threshold": self.runtime.settings.detection_confidence_threshold,
                    },
                    category="perception",
                )
                diagnostic_last_emitted_at = now
                diagnostic_samples = 0
                diagnostic_candidate_frames = 0
                diagnostic_candidates = 0
                diagnostic_best_confidence = None
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

            candidate = max(matches, key=lambda item: item.confidence)
            if (
                target_acquired
                and tracked_detection is not None
                and tracked_detection.image_position is not None
            ):
                previous = tracked_detection.image_position
                positioned_matches = [
                    item for item in matches if item.image_position is not None
                ]
                if not positioned_matches:
                    return None
                candidate = min(
                    positioned_matches,
                    key=lambda item: math.hypot(
                        item.image_position.x_normalized - previous.x_normalized,
                        item.image_position.y_normalized - previous.y_normalized,
                    ),
                )
                center_jump = math.hypot(
                    candidate.image_position.x_normalized - previous.x_normalized,
                    candidate.image_position.y_normalized - previous.y_normalized,
                )
                if center_jump > self.runtime.settings.detection_tracking_max_center_jump:
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
            if isinstance(found_payload, dict):
                found = Detection.from_snapshot(found_payload)
                cancellation_confirmed = (
                    navigation.status == ToolStatus.SUCCESS
                    and navigation.data.get("navigation_canceled") is True
                )
                centering = None
                centered: bool | None = None
                final_status = navigation.status
                final_error = navigation.error
                final_retryable = navigation.retryable
                if not cancellation_confirmed:
                    final_status = ToolStatus.FAILED
                    final_error = (
                        "Target detection was reported without confirmed Nav2 cancellation"
                    )
                    final_retryable = True
                if (
                    cancellation_confirmed
                    and self.runtime.settings.center_on_detection
                ):
                    initial_alignment_detection = found

                    def detect_for_alignment() -> Detection | None:
                        nonlocal initial_alignment_detection
                        if initial_alignment_detection is not None:
                            detection = initial_alignment_detection
                            initial_alignment_detection = None
                            return detection
                        return detect_latest()

                    self.runtime.emit(
                        "visual_alignment_started",
                        {
                            "found": found.to_dict(),
                            "target_x_normalized": 0.5,
                            "target_height_normalized": self.runtime.settings.target_box_size_normalized,
                            "phases": ["rotate", "approach"],
                            "stable_frames": self.runtime.settings.centering_stable_frames,
                            "detection_hold_sec": self.runtime.settings.centering_detection_hold_sec,
                        },
                        category="perception",
                    )
                    centering = self.ros.align_to_detection(
                        detect_for_alignment,
                        tick_interval_sec=self.runtime.settings.detection_interval_sec,
                        horizontal_tolerance=self.runtime.settings.image_center_tolerance,
                        target_box_size=self.runtime.settings.target_box_size_normalized,
                        box_size_tolerance=self.runtime.settings.box_size_tolerance,
                        max_angular_speed=self.runtime.settings.centering_max_angular_speed,
                        min_angular_speed=self.runtime.settings.centering_min_angular_speed,
                        angular_gain=self.runtime.settings.centering_gain,
                        max_linear_speed=self.runtime.settings.centering_max_linear_speed,
                        min_linear_speed=self.runtime.settings.centering_min_linear_speed,
                        linear_gain=self.runtime.settings.centering_linear_gain,
                        timeout_sec=self.runtime.settings.centering_timeout_sec,
                        stable_frames=self.runtime.settings.centering_stable_frames,
                        detection_hold_sec=self.runtime.settings.centering_detection_hold_sec,
                    )
                    self.runtime.emit(
                        "visual_alignment_finished",
                        {
                            "status": centering.status.value,
                            "error": centering.error,
                            "duration_sec": centering.duration_sec,
                            "result": centering.data,
                        },
                        category="perception",
                    )
                    centered = centering.status == ToolStatus.SUCCESS
                    centered_payload = centering.data.get("found")
                    if isinstance(centered_payload, dict):
                        found = Detection.from_snapshot(centered_payload)
                    if not centered:
                        final_status = centering.status
                        final_error = centering.error
                        final_retryable = centering.retryable

                self.world.update_detections([found])
                self.world.update_navigation_status(
                    (
                        "centered_on_detection"
                        if centered
                        else "interrupted_for_detection"
                        if cancellation_confirmed and centered is None
                        else "needs_verification"
                    ),
                    planned_pose=None,
                )
                observed = self.ros.get_pose()
                observation_pose = None
                if observed.status == ToolStatus.SUCCESS and observed.data.get("pose"):
                    observation_pose = Pose2D(**observed.data["pose"])
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
                        "centering_requested": self.runtime.settings.center_on_detection,
                        "centered": centered,
                        "centering": centering.to_dict() if centering else None,
                    },
                    error=final_error,
                    retryable=final_retryable,
                )
            if navigation.status == ToolStatus.PLANNED:
                self.world.update_navigation_status("planned", planned_pose=pose)
                return ToolResult(
                    status=ToolStatus.PLANNED,
                    data={
                        "operation": "search_for_object",
                        "route": route,
                        "completed_legs": legs,
                        "watch_executed": False,
                    },
                )
            if navigation.status != ToolStatus.SUCCESS:
                self.world.update_navigation_status("failed", planned_pose=None)
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
        self.world.update_navigation_status("succeeded", planned_pose=None)
        observed = self.ros.get_pose()
        if observed.status == ToolStatus.SUCCESS and observed.data.get("pose"):
            self.world.update_pose(Pose2D(**observed.data["pose"]))
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

    def _wait_for(self, seconds: float) -> ToolResult:
        duration = self.safety.validate_wait(seconds)
        if self.runtime.settings.execute_ros2:
            time.sleep(duration)
            status = ToolStatus.SUCCESS
        else:
            status = ToolStatus.PLANNED
        return ToolResult(status=status, data={"operation": "wait", "seconds": duration})

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
                if observed.status == ToolStatus.SUCCESS and observed.data.get("pose"):
                    self.world.update_pose(Pose2D(**observed.data["pose"]))
                return ToolResult(
                    status=observed.status,
                    data={"robot_state": self.world.context(), "transport": observed.data},
                    error=observed.error,
                    retryable=observed.retryable,
                )

            return self._record("get_robot_state", {}, operation)

        def navigate_to(location: str) -> dict[str, Any]:
            return self._record("navigate_to", {"location": location}, lambda: self._navigate_to_location(location))

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

        def inspect_for_color(color: str) -> dict[str, Any]:
            def operation() -> ToolResult:
                observed = self.ros.detect_color(color)
                attempts = 1
                adjustment = None
                should_retry = (
                    self.runtime.settings.active_perception_retry_enabled
                    and self.runtime.settings.execute_ros2
                    and (
                        observed.status in {ToolStatus.FAILED, ToolStatus.TIMEOUT}
                        or not observed.data.get("detections", [])
                    )
                )
                if should_retry:
                    # Agent tools execute sequentially and this path does not own or
                    # cancel a Nav2 goal. Cancellation handoff settling is therefore
                    # centralized in RclpyRos2Adapter.cancel_navigation(), not here.
                    def detect_for_alignment() -> Detection | None:
                        nonlocal observed, attempts
                        observed = self.ros.detect_color(color)
                        attempts += 1
                        if observed.status != ToolStatus.SUCCESS:
                            return None
                        detections = [
                            Detection.from_snapshot(item)
                            for item in observed.data.get("detections", [])
                        ]
                        if not detections:
                            return None
                        return max(detections, key=lambda item: item.confidence)

                    adjustment = self.ros.align_to_detection(
                        detect_for_alignment,
                        tick_interval_sec=self.runtime.settings.detection_interval_sec,
                        horizontal_tolerance=self.runtime.settings.image_center_tolerance,
                        target_box_size=self.runtime.settings.target_box_size_normalized,
                        box_size_tolerance=self.runtime.settings.box_size_tolerance,
                        max_angular_speed=self.runtime.settings.centering_max_angular_speed,
                        min_angular_speed=self.runtime.settings.centering_min_angular_speed,
                        angular_gain=self.runtime.settings.centering_gain,
                        max_linear_speed=self.runtime.settings.centering_max_linear_speed,
                        min_linear_speed=self.runtime.settings.centering_min_linear_speed,
                        linear_gain=self.runtime.settings.centering_linear_gain,
                        timeout_sec=self.runtime.settings.centering_timeout_sec,
                        stable_frames=self.runtime.settings.centering_stable_frames,
                        detection_hold_sec=self.runtime.settings.centering_detection_hold_sec,
                    )
                    if adjustment.status != ToolStatus.SUCCESS:
                        return ToolResult(
                            status=adjustment.status,
                            data={"color": color, "attempts": attempts, "adjustment": adjustment.to_dict()},
                            error=adjustment.error or "Active perception adjustment failed",
                            retryable=adjustment.retryable,
                        )
                    aligned_payload = adjustment.data.get("found")
                    if isinstance(aligned_payload, dict):
                        observed = ToolResult(
                            status=ToolStatus.SUCCESS,
                            data={"detections": [aligned_payload]},
                        )
                if observed.status == ToolStatus.SUCCESS:
                    detections = [
                        Detection.from_snapshot(item)
                        for item in observed.data.get("detections", [])
                    ]
                    self.world.update_detections(detections)
                return ToolResult(
                    status=observed.status,
                    data={
                        "color": color,
                        "matches": observed.data.get("detections", []),
                        "observation_available": observed.status == ToolStatus.SUCCESS,
                        "attempts": attempts,
                        "adjustment": adjustment.to_dict() if adjustment else None,
                    },
                    error=observed.error,
                    retryable=observed.retryable,
                )

            return self._record("inspect_for_color", {"color": color}, operation)

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
                inspect_for_color,
                args_schema=InspectForColorInput,
                description="Inspect the camera once for a supported color and update the semantic world model.",
            ),
            StructuredTool.from_function(
                search_for_object,
                args_schema=SearchForObjectInput,
                description=(
                    "Search for an object while Nav2 continuously traverses an ordered route of known locations; "
                    "navigation is canceled only when detection reaches the configured confidence threshold. "
                    "With the default YOLOE backend, pass a concrete open-vocabulary text label; "
                    "the color_blob fallback only supports label='colored_object'."
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
