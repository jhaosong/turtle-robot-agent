"""High-level tool registry with validation, result normalization, and tracing."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import yaml
from langchain_core.tools import StructuredTool

from robot_agent.guardrails import SafetyValidator
from robot_agent.middlewares import ToolLoopDetector
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
    ) -> None:
        self.runtime = runtime
        self.ros = ros
        self.bt_skill = bt_skill
        self.locations = load_locations(runtime.settings.location_file)
        self.world = WorldModel(runtime.state.robot_state)
        self.safety = SafetyValidator(runtime.settings)
        self.loop_detector = ToolLoopDetector(
            warn_threshold=runtime.settings.loop_warn_threshold,
            hard_limit=runtime.settings.repeated_tool_limit,
        )

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
            self.world.update_navigation_status("succeeded", planned_pose=None)
            self.world.update_pose(pose)
            if location is not None and location not in self.runtime.state.visited_locations:
                self.runtime.state.visited_locations.append(location)
        elif result.status == ToolStatus.PLANNED:
            self.world.update_navigation_status("planned", planned_pose=pose)
        else:
            self.world.update_navigation_status("failed", planned_pose=None)
        return result

    def _navigate_to_location(self, location: str) -> ToolResult:
        return self._navigate_to_pose(self.locations[location], location=location)

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
                    adjustment = self.ros.adjust_for_perception(
                        self.runtime.settings.active_perception_backoff_speed,
                        self.runtime.settings.active_perception_backoff_duration_sec,
                    )
                    if adjustment.status != ToolStatus.SUCCESS:
                        return ToolResult(
                            status=adjustment.status,
                            data={"color": color, "attempts": attempts, "adjustment": adjustment.to_dict()},
                            error=adjustment.error or "Active perception adjustment failed",
                            retryable=adjustment.retryable,
                        )
                    observed = self.ros.detect_color(color)
                    attempts += 1
                if observed.status == ToolStatus.SUCCESS:
                    detections = [Detection(**item) for item in observed.data.get("detections", [])]
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
            StructuredTool.from_function(find_object, args_schema=FindObjectInput, description="Query the semantic world model for a visible object."),
            StructuredTool.from_function(
                inspect_for_color,
                args_schema=InspectForColorInput,
                description="Inspect the camera once for a supported color and update the semantic world model.",
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
