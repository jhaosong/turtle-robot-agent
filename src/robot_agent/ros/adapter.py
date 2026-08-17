"""ROS2 transport implementations for high-level robot operations.

The CLI backend is useful for transparent dry runs. The rclpy backend is the
actual in-process ROS2 path and follows the TurtleBot demo's ``GoToPose``
ActionClient usage. Both return ``ToolResult`` rather than terminal output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import subprocess
import time
from typing import Any, Callable

import yaml

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.perception import detect_colored_blobs
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus

from .messages import quaternion_to_yaw, yaw_to_quaternion


class _DetectionTicker:
    """Monotonic rate gate; the first observation is allowed immediately."""

    def __init__(
        self,
        interval_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("tick_interval_sec must be positive")
        self.interval_sec = interval_sec
        self.clock = clock
        self.last_tick: float | None = None

    def ready(self) -> bool:
        now = self.clock()
        if self.last_tick is None or now - self.last_tick >= self.interval_sec:
            self.last_tick = now
            return True
        return False


class Ros2Adapter(ABC):
    @abstractmethod
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise NotImplementedError

    def navigate_to_pose_with_watch(
        self,
        pose: Pose2D,
        on_tick: Callable[[], Detection | None],
        tick_interval_sec: float,
    ) -> ToolResult:
        """Fallback for adapters without cooperative ROS event spinning."""
        _ = on_tick, tick_interval_sec
        return self.navigate_to_pose(pose)

    @abstractmethod
    def stop_robot(self) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def cancel_navigation(self) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def detect_color(self, color: str) -> ToolResult:
        raise NotImplementedError

    def get_camera_frame(self) -> Any | None:
        """Return the latest BGR image without exposing it to agent context."""
        return None

    def update_detection_overlay(self, detections: list[Detection]) -> None:
        """Update boxes rendered on the annotated camera stream when supported."""
        _ = detections

    def align_to_detection(
        self,
        on_tick: Callable[[], Detection | None],
        *,
        tick_interval_sec: float,
        horizontal_tolerance: float,
        target_box_size: float,
        box_size_tolerance: float,
        max_angular_speed: float,
        min_angular_speed: float = 0.025,
        angular_gain: float,
        max_linear_speed: float,
        min_linear_speed: float = 0.02,
        linear_gain: float,
        timeout_sec: float,
        stable_frames: int = 3,
        detection_hold_sec: float = 1.0,
    ) -> ToolResult:
        _ = (
            on_tick,
            tick_interval_sec,
            horizontal_tolerance,
            target_box_size,
            box_size_tolerance,
            max_angular_speed,
            min_angular_speed,
            angular_gain,
            max_linear_speed,
            min_linear_speed,
            linear_gain,
            timeout_sec,
            stable_frames,
            detection_hold_sec,
        )
        return ToolResult(
            status=ToolStatus.FAILED,
            data={"operation": "align_to_detection"},
            error="ROS2 backend does not support visual alignment",
            retryable=False,
        )

    def close(self) -> None:
        """Release transport resources when a run ends."""


class Ros2CliAdapter(Ros2Adapter):
    """CLI implementation used for dry-run and simple shell-based ROS2 runs."""

    def __init__(self, settings: RobotAgentSettings) -> None:
        self.settings = settings

    def _run(self, command: str, details: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        if not self.settings.execute_ros2:
            return ToolResult(
                status=ToolStatus.PLANNED,
                data={"backend": "ros2_cli", "command": command, **details},
            )
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.tool_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={"backend": "ros2_cli", "command": command, **details},
                error="ROS2 command timed out",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        if completed.returncode != 0:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"backend": "ros2_cli", "command": command, **details},
                error=(completed.stderr or "ROS2 command failed").strip()[:500],
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"backend": "ros2_cli", "command": command, **details},
            duration_sec=time.monotonic() - started,
        )

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        quaternion = yaw_to_quaternion(pose.yaw)
        goal = (
            "{pose: {header: {frame_id: '"
            + pose.frame_id
            + "'}, pose: {position: {x: "
            + f"{pose.x:.3f}, y: {pose.y:.3f}, z: 0.0}}, orientation: {{x: {quaternion['x']:.6f}, "
            + f"y: {quaternion['y']:.6f}, z: {quaternion['z']:.6f}, w: {quaternion['w']:.6f}}}}}}}"
        )
        command = (
            f"ros2 action send_goal {self.settings.nav_action_name} "
            f"nav2_msgs/action/NavigateToPose \"{goal}\""
        )
        return self._run(command, {"operation": "navigate_to_pose", "target_pose": pose.to_dict()})

    def navigate_to_pose_with_watch(
        self,
        pose: Pose2D,
        on_tick: Callable[[], Detection | None],
        tick_interval_sec: float,
    ) -> ToolResult:
        """Reject watched navigation because CLI cannot interleave callbacks."""
        _ = on_tick, tick_interval_sec
        return ToolResult(
            status=ToolStatus.FAILED,
            data={
                "backend": "ros2_cli",
                "operation": "navigate_to_pose_with_watch",
                "target_pose": pose.to_dict(),
            },
            error=(
                "Watched navigation and visual alignment require "
                "ROBOT_AGENT_ROS_BACKEND=rclpy"
            ),
            retryable=False,
        )

    def stop_robot(self) -> ToolResult:
        message = "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        command = (
            f"ros2 topic pub --times 3 {self.settings.cmd_vel_topic} "
            f"geometry_msgs/msg/Twist \"{message}\" --rate 2"
        )
        return self._run(
            command,
            {
                "operation": "stop_robot",
                "topic": self.settings.cmd_vel_topic,
                "guarantee": "best_effort_zero_velocity_only",
            },
        )

    def get_pose(self) -> ToolResult:
        command = f"ros2 topic echo {self.settings.odom_topic} --once --spin-time {self.settings.tool_timeout_sec}"
        if not self.settings.execute_ros2:
            return ToolResult(
                status=ToolStatus.PLANNED,
                data={"backend": "ros2_cli", "command": command, "operation": "get_pose", "topic": self.settings.odom_topic},
            )
        started = time.monotonic()
        try:
            completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True, timeout=self.settings.tool_timeout_sec)
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or "Unable to read odometry").strip())
            message = yaml.safe_load(completed.stdout)
            pose = message["pose"]["pose"]
            orientation = pose["orientation"]
            result = Pose2D(
                x=float(pose["position"]["x"]),
                y=float(pose["position"]["y"]),
                yaw=quaternion_to_yaw(float(orientation["x"]), float(orientation["y"]), float(orientation["z"]), float(orientation["w"])),
                frame_id=str(message.get("header", {}).get("frame_id") or self.settings.map_frame),
            )
            return ToolResult(status=ToolStatus.SUCCESS, data={"operation": "get_pose", "pose": result.to_dict()}, duration_sec=time.monotonic() - started)
        except subprocess.TimeoutExpired:
            return ToolResult(status=ToolStatus.TIMEOUT, data={"operation": "get_pose", "topic": self.settings.odom_topic}, error="Odometry read timed out", duration_sec=time.monotonic() - started, retryable=True)
        except (KeyError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            return ToolResult(status=ToolStatus.FAILED, data={"operation": "get_pose", "topic": self.settings.odom_topic}, error=f"Unable to parse odometry: {exc}", duration_sec=time.monotonic() - started, retryable=True)

    def cancel_navigation(self) -> ToolResult:
        return ToolResult(status=ToolStatus.FAILED, data={"operation": "cancel_navigation"}, error="CLI backend cannot safely cancel an unknown action goal", retryable=False)

    def detect_color(self, color: str) -> ToolResult:
        if not self.settings.execute_ros2:
            return ToolResult(
                status=ToolStatus.PLANNED,
                data={"operation": "detect_color", "color": color, "topic": self.settings.camera_topic, "backend": "ros2_cli"},
                error="Color detection requires the rclpy backend; ROS2 CLI cannot safely interpret camera images",
            )
        return ToolResult(
            status=ToolStatus.FAILED,
            data={"operation": "detect_color", "color": color, "topic": self.settings.camera_topic, "backend": "ros2_cli"},
            error="Color detection requires ROBOT_AGENT_ROS_BACKEND=rclpy",
            retryable=False,
        )

class RclpyRos2Adapter(Ros2Adapter):
    """In-process ROS2 implementation for Nav2 and emergency stopping."""

    def __init__(self, settings: RobotAgentSettings) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav2_msgs.action import NavigateToPose
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Image
            from rclpy.action import ActionClient
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformException, TransformListener
        except ImportError as exc:  # pragma: no cover - depends on ROS install
            raise RuntimeError("rclpy/Nav2 packages are required for ROBOT_AGENT_ROS_BACKEND=rclpy") from exc

        self.settings = settings
        self._rclpy = rclpy
        self._twist_type = Twist
        self._goal_type = NavigateToPose
        self._action_client_type = ActionClient
        self._time_type = Time
        self._transform_exception = TransformException
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node: Node = rclpy.create_node("robot_agent_ros_adapter")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self._node,
            spin_thread=False,
        )
        self._publisher = self._node.create_publisher(Twist, settings.cmd_vel_topic, 10)
        self._annotated_image_publisher = self._node.create_publisher(
            Image,
            settings.annotated_camera_topic,
            10,
        )
        self._latest_pose: Pose2D | None = None
        self._latest_image: Any | None = None
        self._detection_overlay: list[Detection] = []
        self._detection_overlay_updated_at = 0.0
        self._annotation_warning_emitted = False
        self._node.create_subscription(Odometry, settings.odom_topic, self._on_odom, 10)
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._node.create_subscription(
            Image,
            settings.camera_topic,
            self._on_image,
            camera_qos,
        )
        self._nav_client = ActionClient(self._node, NavigateToPose, settings.nav_action_name)
        self._active_goal_handle: Any | None = None
        self._active_result_future: Any | None = None

    def _on_odom(self, message: Any) -> None:
        orientation = message.pose.pose.orientation
        yaw = quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        self._latest_pose = Pose2D(
            x=message.pose.pose.position.x,
            y=message.pose.pose.position.y,
            yaw=yaw,
            frame_id=message.header.frame_id or self.settings.map_frame,
        )

    def _on_image(self, message: Any) -> None:
        self._latest_image = message
        self._publish_annotated_image(message)

    def update_detection_overlay(self, detections: list[Detection]) -> None:
        self._detection_overlay = list(detections)
        self._detection_overlay_updated_at = time.monotonic()

    def _publish_annotated_image(self, message: Any) -> None:
        """Republish camera frames with the latest short-lived detection boxes."""
        get_count = getattr(
            self._annotated_image_publisher,
            "get_subscription_count",
            None,
        )
        if callable(get_count) and get_count() <= 0:
            return
        try:
            import cv2
            from cv_bridge import CvBridge

            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8").copy()
            frame_height, frame_width = frame.shape[:2]
            overlay_ttl = max(0.25, self.settings.detection_interval_sec * 1.5)
            detections = (
                self._detection_overlay
                if time.monotonic() - self._detection_overlay_updated_at <= overlay_ttl
                else []
            )
            for detection in detections:
                position = detection.image_position
                if position is None:
                    continue
                half_width = position.width_normalized * frame_width / 2.0
                half_height = position.height_normalized * frame_height / 2.0
                center_x = position.x_normalized * frame_width
                center_y = position.y_normalized * frame_height
                left = max(0, min(frame_width - 1, round(center_x - half_width)))
                top = max(0, min(frame_height - 1, round(center_y - half_height)))
                right = max(0, min(frame_width - 1, round(center_x + half_width)))
                bottom = max(0, min(frame_height - 1, round(center_y + half_height)))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                caption = f"{detection.label} {detection.confidence:.2f}"
                cv2.putText(
                    frame,
                    caption,
                    (left, max(18, top - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            annotated = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            annotated.header = message.header
            self._annotated_image_publisher.publish(annotated)
        except Exception as exc:  # Visualization must never interrupt control.
            if not self._annotation_warning_emitted:
                self._node.get_logger().warning(
                    f"Annotated camera stream disabled after rendering error: {exc}"
                )
                self._annotation_warning_emitted = True
            return

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        started = time.monotonic()
        if self._active_goal_handle is not None:
            return ToolResult(status=ToolStatus.FAILED, data={"operation": "navigate_to_pose", "target_pose": pose.to_dict()}, error="A navigation goal is already active", retryable=False)
        if not self._nav_client.wait_for_server(timeout_sec=self.settings.tool_timeout_sec):
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={"operation": "navigate_to_pose", "target_pose": pose.to_dict()},
                error="Nav2 action server unavailable",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        goal = self._goal_type.Goal()
        goal.pose.header.frame_id = pose.frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = pose.x
        goal.pose.pose.position.y = pose.y
        quaternion = yaw_to_quaternion(pose.yaw)
        goal.pose.pose.orientation.x = quaternion["x"]
        goal.pose.pose.orientation.y = quaternion["y"]
        goal.pose.pose.orientation.z = quaternion["z"]
        goal.pose.pose.orientation.w = quaternion["w"]
        send_future = self._nav_client.send_goal_async(goal, feedback_callback=self._on_navigation_feedback)
        self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=self.settings.tool_timeout_sec)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "navigate_to_pose", "target_pose": pose.to_dict()},
                error="Nav2 rejected goal",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        self._active_goal_handle = handle
        result_future = handle.get_result_async()
        self._active_result_future = result_future
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=self.settings.tool_timeout_sec)
        result = result_future.result()
        if result is None:
            self.cancel_navigation()
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={"operation": "navigate_to_pose", "target_pose": pose.to_dict(), "cancel_requested": True},
                error="Nav2 result timed out",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        self._active_goal_handle = None
        self._active_result_future = None
        success = result.status == 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED
        return ToolResult(
            status=ToolStatus.SUCCESS if success else ToolStatus.FAILED,
            data={"operation": "navigate_to_pose", "target_pose": pose.to_dict(), "goal_status": result.status},
            error=None if success else f"Nav2 goal ended with status {result.status}",
            duration_sec=time.monotonic() - started,
            retryable=not success,
        )

    def navigate_to_pose_with_watch(
        self,
        pose: Pose2D,
        on_tick: Callable[[], Detection | None],
        tick_interval_sec: float,
    ) -> ToolResult:
        """Navigate while cooperatively checking the latest camera frame."""
        started = time.monotonic()
        if self._active_goal_handle is not None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "navigate_to_pose_with_watch",
                    "target_pose": pose.to_dict(),
                },
                error="A navigation goal is already active",
                retryable=False,
            )
        if not self._nav_client.wait_for_server(
            timeout_sec=self.settings.tool_timeout_sec
        ):
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={
                    "operation": "navigate_to_pose_with_watch",
                    "target_pose": pose.to_dict(),
                },
                error="Nav2 action server unavailable",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )

        goal = self._goal_type.Goal()
        goal.pose.header.frame_id = pose.frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = pose.x
        goal.pose.pose.position.y = pose.y
        quaternion = yaw_to_quaternion(pose.yaw)
        goal.pose.pose.orientation.x = quaternion["x"]
        goal.pose.pose.orientation.y = quaternion["y"]
        goal.pose.pose.orientation.z = quaternion["z"]
        goal.pose.pose.orientation.w = quaternion["w"]
        send_future = self._nav_client.send_goal_async(
            goal,
            feedback_callback=self._on_navigation_feedback,
        )
        self._rclpy.spin_until_future_complete(
            self._node,
            send_future,
            timeout_sec=self.settings.tool_timeout_sec,
        )
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "navigate_to_pose_with_watch",
                    "target_pose": pose.to_dict(),
                },
                error="Nav2 rejected goal",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )

        self._active_goal_handle = handle
        result_future = handle.get_result_async()
        self._active_result_future = result_future
        ticker = _DetectionTicker(tick_interval_sec)
        while not result_future.done():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if result_future.done():
                break
            if ticker.ready():
                try:
                    found = on_tick()
                except Exception as exc:
                    cancel_result = self.cancel_navigation()
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        data={
                            "operation": "navigate_to_pose_with_watch",
                            "target_pose": pose.to_dict(),
                            "cancel_result": cancel_result.to_dict(),
                        },
                        error=f"Detection callback failed: {type(exc).__name__}",
                        duration_sec=time.monotonic() - started,
                        retryable=False,
                    )
                if found is not None:
                    cancel_result = self.cancel_navigation()
                    canceled = cancel_result.status == ToolStatus.CANCELED
                    return ToolResult(
                        status=ToolStatus.SUCCESS if canceled else ToolStatus.FAILED,
                        data={
                            "operation": "navigate_to_pose_with_watch",
                            "found": found.to_dict(),
                            "target_pose": pose.to_dict(),
                            "navigation_canceled": canceled,
                            "cancel_result": cancel_result.to_dict(),
                        },
                        error=(
                            None
                            if canceled
                            else "Target was detected but Nav2 cancellation was not confirmed"
                        ),
                        duration_sec=time.monotonic() - started,
                        retryable=not canceled,
                    )
            if time.monotonic() - started > self.settings.tool_timeout_sec:
                cancel_result = self.cancel_navigation()
                return ToolResult(
                    status=ToolStatus.TIMEOUT,
                    data={
                        "operation": "navigate_to_pose_with_watch",
                        "target_pose": pose.to_dict(),
                        "cancel_result": cancel_result.to_dict(),
                    },
                    error="Nav2 result timed out during watched navigation",
                    duration_sec=time.monotonic() - started,
                    retryable=True,
                )

        result = result_future.result()
        self._active_goal_handle = None
        self._active_result_future = None
        if result is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={
                    "operation": "navigate_to_pose_with_watch",
                    "target_pose": pose.to_dict(),
                },
                error="Nav2 returned no result",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        success = result.status == 4
        return ToolResult(
            status=ToolStatus.SUCCESS if success else ToolStatus.FAILED,
            data={
                "operation": "navigate_to_pose_with_watch",
                "target_pose": pose.to_dict(),
                "goal_status": result.status,
                "found": None,
            },
            error=None if success else f"Nav2 goal ended with status {result.status}",
            duration_sec=time.monotonic() - started,
            retryable=not success,
        )

    def stop_robot(self) -> ToolResult:
        cancel_result = self.cancel_navigation() if self._active_goal_handle is not None else None
        zero_command_count = (
            self._publish_zero_velocity()
            if cancel_result is not None
            else self._hold_zero_velocity(self.settings.post_cancel_settle_sec)
        )
        cancellation_failed = cancel_result is not None and cancel_result.status != ToolStatus.CANCELED
        return ToolResult(
            status=ToolStatus.FAILED if cancellation_failed else ToolStatus.SUCCESS,
            data={
                "operation": "stop_robot",
                "topic": self.settings.cmd_vel_topic,
                "navigation_canceled": cancel_result is not None and not cancellation_failed,
                "zero_command_count": zero_command_count,
            },
            error="Zero velocity was published but Nav2 cancellation was not confirmed" if cancellation_failed else None,
            retryable=cancellation_failed,
        )

    def get_pose(self) -> ToolResult:
        started = time.monotonic()
        deadline = started + min(self.settings.tool_timeout_sec, 5.0)
        last_error = "Map transform is not available"
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.settings.map_frame,
                    self.settings.base_frame,
                    self._time_type(),
                )
            except self._transform_exception as exc:
                last_error = str(exc)
                continue
            rotation = transform.transform.rotation
            pose = Pose2D(
                x=float(transform.transform.translation.x),
                y=float(transform.transform.translation.y),
                yaw=quaternion_to_yaw(
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ),
                frame_id=self.settings.map_frame,
            )
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "operation": "get_pose",
                    "pose": pose.to_dict(),
                    "source": "tf",
                    "transform": f"{self.settings.map_frame}->{self.settings.base_frame}",
                },
                duration_sec=time.monotonic() - started,
            )
        return ToolResult(
            status=ToolStatus.FAILED,
            data={
                "operation": "get_pose",
                "source": "tf",
                "transform": f"{self.settings.map_frame}->{self.settings.base_frame}",
            },
            error=f"Unable to observe the robot map pose: {last_error}",
            duration_sec=time.monotonic() - started,
            retryable=True,
        )

    def _on_navigation_feedback(self, feedback: Any) -> None:
        # Feedback is intentionally not sent to the LLM; it remains transport telemetry.
        _ = feedback

    def _publish_zero_velocity(self, count: int = 3) -> int:
        message = self._twist_type()
        for _ in range(count):
            self._publisher.publish(message)
        return count

    def _hold_zero_velocity(self, duration_sec: float) -> int:
        """Publish stop commands throughout a bounded settling window."""
        published = self._publish_zero_velocity()
        if duration_sec <= 0:
            return published
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self._publisher.publish(self._twist_type())
            published += 1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._rclpy.spin_once(self._node, timeout_sec=0.0)
                time.sleep(min(0.05, remaining))
        return published

    def _settle_after_navigation_cancel(self) -> int:
        """Hold zero velocity while residual Nav2 controller output drains."""
        return self._hold_zero_velocity(self.settings.post_cancel_settle_sec)

    def cancel_navigation(self) -> ToolResult:
        if self._active_goal_handle is None:
            return ToolResult(status=ToolStatus.SUCCESS, data={"operation": "cancel_navigation", "active_goal": False})
        handle = self._active_goal_handle
        result_future = self._active_result_future
        future = handle.cancel_goal_async()
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=self.settings.tool_timeout_sec)
        response = future.result()
        cancel_accepted = bool(
            response is not None and getattr(response, "goals_canceling", [])
        )
        terminal_canceled = False
        if cancel_accepted and result_future is not None:
            self._rclpy.spin_until_future_complete(
                self._node,
                result_future,
                timeout_sec=self.settings.tool_timeout_sec,
            )
            terminal_result = result_future.result()
            terminal_canceled = (
                terminal_result is not None and terminal_result.status == 5
            )
        self._active_goal_handle = None
        self._active_result_future = None
        zero_command_count = (
            self._settle_after_navigation_cancel() if terminal_canceled else 0
        )
        return ToolResult(
            status=ToolStatus.CANCELED if terminal_canceled else ToolStatus.FAILED,
            data={
                "operation": "cancel_navigation",
                "active_goal": True,
                "cancel_accepted": cancel_accepted,
                "terminal_canceled": terminal_canceled,
                "post_cancel_settle_sec": (
                    self.settings.post_cancel_settle_sec
                    if terminal_canceled
                    else 0.0
                ),
                "zero_command_count": zero_command_count,
            },
            error=(
                None
                if terminal_canceled
                else "Nav2 did not reach the canceled terminal state"
            ),
            retryable=not terminal_canceled,
        )

    def detect_color(self, color: str) -> ToolResult:
        started = time.monotonic()
        while self._latest_image is None and time.monotonic() - started < self.settings.tool_timeout_sec:
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
        if self._latest_image is None:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={"operation": "detect_color", "color": color, "topic": self.settings.camera_topic},
                error="No camera image received",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        try:
            from cv_bridge import CvBridge

            image = CvBridge().imgmsg_to_cv2(self._latest_image, desired_encoding="bgr8")
            detections: list[Detection] = detect_colored_blobs(image, color)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"operation": "detect_color", "color": color, "detections": [item.to_dict() for item in detections]},
                duration_sec=time.monotonic() - started,
            )
        except (RuntimeError, ValueError) as exc:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "detect_color", "color": color, "topic": self.settings.camera_topic},
                error=str(exc),
                duration_sec=time.monotonic() - started,
                retryable=False,
            )

    def get_camera_frame(self) -> Any | None:
        if self._latest_image is None:
            return None
        try:
            from cv_bridge import CvBridge
        except ImportError as exc:  # pragma: no cover - ROS dependency
            raise RuntimeError("cv_bridge is required for camera detection") from exc
        return CvBridge().imgmsg_to_cv2(
            self._latest_image,
            desired_encoding="bgr8",
        )

    def align_to_detection(
        self,
        on_tick: Callable[[], Detection | None],
        *,
        tick_interval_sec: float,
        horizontal_tolerance: float,
        target_box_size: float,
        box_size_tolerance: float,
        max_angular_speed: float,
        min_angular_speed: float = 0.025,
        angular_gain: float,
        max_linear_speed: float,
        min_linear_speed: float = 0.02,
        linear_gain: float,
        timeout_sec: float,
        stable_frames: int = 3,
        detection_hold_sec: float = 1.0,
    ) -> ToolResult:
        """Rotate to center, then approach to the target bbox size."""
        started = time.monotonic()
        ticker = _DetectionTicker(tick_interval_sec)
        phase = "rotate"
        stable_count = 0
        last_detection: Detection | None = None
        last_detection_at: float | None = None
        last_errors: dict[str, Any] = {}
        previous_diagnostic_pose: Pose2D | None = getattr(
            self,
            "_latest_pose",
            None,
        )
        last_log_at = 0.0
        last_log_state: str | None = None

        def publish_stop(*, hold: bool = False) -> None:
            if hold:
                settle_sec = getattr(
                    getattr(self, "settings", None),
                    "post_cancel_settle_sec",
                    0.0,
                )
                self._hold_zero_velocity(settle_sec)
            else:
                self._publish_zero_velocity()

        def bounded_velocity(
            error: float,
            gain: float,
            minimum: float,
            maximum: float,
        ) -> float:
            magnitude = min(maximum, max(minimum, gain * abs(error)))
            return math.copysign(magnitude, error)

        def log_tick(
            *,
            state: str,
            linear_x: float,
            angular_z: float,
            force: bool = False,
        ) -> None:
            nonlocal previous_diagnostic_pose
            nonlocal last_log_at
            nonlocal last_log_state
            now = time.monotonic()
            if not force and state == last_log_state and now - last_log_at < 1.0:
                return
            pose = getattr(self, "_latest_pose", None)
            pose_delta_m = None
            pose_delta_yaw = None
            if pose is not None and previous_diagnostic_pose is not None:
                pose_delta_m = math.hypot(
                    pose.x - previous_diagnostic_pose.x,
                    pose.y - previous_diagnostic_pose.y,
                )
                pose_delta_yaw = math.atan2(
                    math.sin(pose.yaw - previous_diagnostic_pose.yaw),
                    math.cos(pose.yaw - previous_diagnostic_pose.yaw),
                )
            if pose is not None:
                previous_diagnostic_pose = pose
            get_logger = getattr(self._node, "get_logger", None)
            if not callable(get_logger):
                return
            command_summary = (
                "cmd=stop"
                if linear_x == 0.0 and angular_z == 0.0
                else f"v={linear_x:.3f} w={angular_z:.3f}"
            )
            motion_summary = ""
            if pose_delta_m is not None and pose_delta_yaw is not None:
                motion_summary = (
                    f" moved={pose_delta_m:.4f}m turned={pose_delta_yaw:.4f}rad"
                )
            get_logger().info(
                f"visual_alignment state={state} {command_summary}{motion_summary}"
            )
            last_log_at = now
            last_log_state = state

        while time.monotonic() - started <= timeout_sec:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if not ticker.ready():
                continue
            detection = on_tick()
            if detection is None:
                stable_count = 0
                within_hold = (
                    last_detection is not None
                    and last_detection_at is not None
                    and time.monotonic() - last_detection_at <= detection_hold_sec
                )
                publish_stop()
                log_tick(
                    state="waiting" if within_hold else "lost",
                    linear_x=0.0,
                    angular_z=0.0,
                )
                continue
            last_detection = detection
            last_detection_at = time.monotonic()
            image_position = detection.image_position
            if image_position is None:
                publish_stop(hold=True)
                log_tick(
                    state="invalid_bbox",
                    linear_x=0.0,
                    angular_z=0.0,
                    force=True,
                )
                return ToolResult(
                    status=ToolStatus.FAILED,
                    data={
                        "operation": "align_to_detection",
                        "found": detection.to_dict(),
                    },
                    error="Detection has no image position for visual alignment",
                    retryable=False,
                )
            if image_position.height_normalized <= 0:
                publish_stop(hold=True)
                log_tick(
                    state="invalid_bbox",
                    linear_x=0.0,
                    angular_z=0.0,
                    force=True,
                )
                return ToolResult(
                    status=ToolStatus.FAILED,
                    data={
                        "operation": "align_to_detection",
                        "found": detection.to_dict(),
                    },
                    error="Detection bounding box has no usable height",
                    retryable=False,
                )
            horizontal_error = image_position.x_normalized - 0.5
            vertical_error = image_position.y_normalized - 0.5
            size_error = target_box_size - image_position.height_normalized
            last_errors = {
                "phase": phase,
                "horizontal_error": horizontal_error,
                "vertical_error": vertical_error,
                "size_error": size_error,
                "target_box_size": target_box_size,
            }

            if phase == "rotate":
                if abs(horizontal_error) <= horizontal_tolerance:
                    stable_count += 1
                    publish_stop()
                    if stable_count >= stable_frames:
                        phase = "approach"
                        stable_count = 0
                        log_tick(
                            state="rotation_complete",
                            linear_x=0.0,
                            angular_z=0.0,
                            force=True,
                        )
                    else:
                        log_tick(
                            state=f"rotate_stable_{stable_count}/{stable_frames}",
                            linear_x=0.0,
                            angular_z=0.0,
                        )
                    continue
                stable_count = 0
                command = self._twist_type()
                command.angular.z = bounded_velocity(
                    -horizontal_error,
                    angular_gain,
                    min_angular_speed,
                    max_angular_speed,
                )
                self._publisher.publish(command)
                log_tick(
                    state="rotate",
                    linear_x=command.linear.x,
                    angular_z=command.angular.z,
                )
                continue

            # Approach is translation-only. If the target drifts, stop and
            # reacquire heading before allowing any more linear motion.
            if abs(horizontal_error) > horizontal_tolerance * 1.5:
                phase = "rotate"
                stable_count = 0
                publish_stop()
                log_tick(
                    state="recenter",
                    linear_x=0.0,
                    angular_z=0.0,
                    force=True,
                )
                continue
            if abs(size_error) <= box_size_tolerance:
                stable_count += 1
                publish_stop()
                if stable_count < stable_frames:
                    log_tick(
                        state=f"approach_stable_{stable_count}/{stable_frames}",
                        linear_x=0.0,
                        angular_z=0.0,
                    )
                    continue
                publish_stop(hold=True)
                log_tick(
                    state="aligned",
                    linear_x=0.0,
                    angular_z=0.0,
                    force=True,
                )
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "operation": "align_to_detection",
                        "found": detection.to_dict(),
                        "centered": True,
                        "phase": "complete",
                        "stable_frames": stable_frames,
                        "horizontal_error": horizontal_error,
                        "vertical_error": vertical_error,
                        "size_error": size_error,
                        "target_box_size": target_box_size,
                    },
                    duration_sec=time.monotonic() - started,
                )
            stable_count = 0
            command = self._twist_type()
            command.linear.x = bounded_velocity(
                size_error,
                linear_gain,
                min_linear_speed,
                max_linear_speed,
            )
            self._publisher.publish(command)
            log_tick(
                state="approach",
                linear_x=command.linear.x,
                angular_z=command.angular.z,
            )

        publish_stop(hold=True)
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            data={
                "operation": "align_to_detection",
                "found": last_detection.to_dict() if last_detection else None,
                "centered": False,
                **last_errors,
            },
            error="Target was not aligned before the visual-alignment timeout",
            duration_sec=time.monotonic() - started,
            retryable=True,
        )

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def build_ros2_adapter(settings: RobotAgentSettings, backend: str = "cli") -> Ros2Adapter:
    if backend == "rclpy":
        return RclpyRos2Adapter(settings)
    if backend != "cli":
        raise ValueError(f"Unsupported ROS2 backend: {backend}")
    return Ros2CliAdapter(settings)
