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
from typing import Any

import yaml

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.perception import detect_colored_blobs
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus

from .messages import quaternion_to_yaw, yaw_to_quaternion


class Ros2Adapter(ABC):
    @abstractmethod
    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        raise NotImplementedError

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

    def adjust_for_perception(self, linear_x: float, duration_sec: float) -> ToolResult:
        return ToolResult(
            status=ToolStatus.FAILED,
            data={"operation": "adjust_for_perception"},
            error="ROS2 backend does not support active perception adjustment",
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

    def adjust_for_perception(self, linear_x: float, duration_sec: float) -> ToolResult:
        rate_hz = 10
        count = max(1, math.ceil(duration_sec * rate_hz))
        move = (
            f"{{linear: {{x: {linear_x:.3f}, y: 0.0, z: 0.0}}, "
            "angular: {x: 0.0, y: 0.0, z: 0.0}}"
        )
        stop = "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        command = (
            f"ros2 topic pub --times {count} {self.settings.cmd_vel_topic} geometry_msgs/msg/Twist \"{move}\" --rate {rate_hz} "
            f"&& ros2 topic pub --times 3 {self.settings.cmd_vel_topic} geometry_msgs/msg/Twist \"{stop}\" --rate 10"
        )
        return self._run(
            command,
            {
                "operation": "adjust_for_perception",
                "linear_x": linear_x,
                "duration_sec": duration_sec,
            },
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
        except ImportError as exc:  # pragma: no cover - depends on ROS install
            raise RuntimeError("rclpy/Nav2 packages are required for ROBOT_AGENT_ROS_BACKEND=rclpy") from exc

        self.settings = settings
        self._rclpy = rclpy
        self._twist_type = Twist
        self._goal_type = NavigateToPose
        self._action_client_type = ActionClient
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node: Node = rclpy.create_node("robot_agent_ros_adapter")
        self._publisher = self._node.create_publisher(Twist, settings.cmd_vel_topic, 10)
        self._latest_pose: Pose2D | None = None
        self._latest_image: Any | None = None
        self._node.create_subscription(Odometry, settings.odom_topic, self._on_odom, 10)
        self._node.create_subscription(Image, settings.camera_topic, self._on_image, 10)
        self._nav_client = ActionClient(self._node, NavigateToPose, settings.nav_action_name)
        self._active_goal_handle: Any | None = None

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
        success = result.status == 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED
        return ToolResult(
            status=ToolStatus.SUCCESS if success else ToolStatus.FAILED,
            data={"operation": "navigate_to_pose", "target_pose": pose.to_dict(), "goal_status": result.status},
            error=None if success else f"Nav2 goal ended with status {result.status}",
            duration_sec=time.monotonic() - started,
            retryable=not success,
        )

    def stop_robot(self) -> ToolResult:
        cancel_result = self.cancel_navigation() if self._active_goal_handle is not None else None
        message = self._twist_type()
        for _ in range(3):
            self._publisher.publish(message)
        cancellation_failed = cancel_result is not None and cancel_result.status != ToolStatus.CANCELED
        return ToolResult(
            status=ToolStatus.FAILED if cancellation_failed else ToolStatus.SUCCESS,
            data={
                "operation": "stop_robot",
                "topic": self.settings.cmd_vel_topic,
                "navigation_canceled": cancel_result is not None and not cancellation_failed,
            },
            error="Zero velocity was published but Nav2 cancellation was not confirmed" if cancellation_failed else None,
            retryable=cancellation_failed,
        )

    def get_pose(self) -> ToolResult:
        self._rclpy.spin_once(self._node, timeout_sec=0.2)
        if self._latest_pose is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"operation": "get_pose", "topic": self.settings.odom_topic},
                error="No odometry message received yet",
                retryable=True,
            )
        return ToolResult(status=ToolStatus.SUCCESS, data={"pose": self._latest_pose.to_dict()})

    def _on_navigation_feedback(self, feedback: Any) -> None:
        # Feedback is intentionally not sent to the LLM; it remains transport telemetry.
        _ = feedback

    def cancel_navigation(self) -> ToolResult:
        if self._active_goal_handle is None:
            return ToolResult(status=ToolStatus.SUCCESS, data={"operation": "cancel_navigation", "active_goal": False})
        future = self._active_goal_handle.cancel_goal_async()
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=self.settings.tool_timeout_sec)
        canceled = future.result() is not None
        self._active_goal_handle = None
        return ToolResult(
            status=ToolStatus.CANCELED if canceled else ToolStatus.FAILED,
            data={"operation": "cancel_navigation", "active_goal": True},
            error=None if canceled else "Nav2 did not acknowledge cancellation",
            retryable=not canceled,
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

    def adjust_for_perception(self, linear_x: float, duration_sec: float) -> ToolResult:
        started = time.monotonic()
        message = self._twist_type()
        message.linear.x = linear_x
        while time.monotonic() - started < duration_sec:
            self._publisher.publish(message)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(0.1)
        stop = self._twist_type()
        for _ in range(3):
            self._publisher.publish(stop)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "adjust_for_perception",
                "linear_x": linear_x,
                "duration_sec": duration_sec,
            },
            duration_sec=time.monotonic() - started,
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
