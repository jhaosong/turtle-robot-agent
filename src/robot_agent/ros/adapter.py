"""In-process rclpy transport for high-level robot operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import time
from typing import Any, Callable

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.navigation.costmap import CostmapSnapshot, LETHAL_COST, path_risk
from robot_agent.state import Detection, Pose2D, ToolResult, ToolStatus

from .messages import quaternion_to_yaw, yaw_to_quaternion


def _populate_pose_stamped(message: Any, pose: Pose2D, stamp: Any) -> None:
    message.header.frame_id = pose.frame_id
    message.header.stamp = stamp
    message.pose.position.x = pose.x
    message.pose.position.y = pose.y
    quaternion = yaw_to_quaternion(pose.yaw)
    message.pose.orientation.x = quaternion["x"]
    message.pose.orientation.y = quaternion["y"]
    message.pose.orientation.z = quaternion["z"]
    message.pose.orientation.w = quaternion["w"]


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

    def get_camera_frame(self) -> Any | None:
        """Return the latest BGR image without exposing it to agent context."""
        return None

    def update_detection_overlay(self, detections: list[Detection]) -> None:
        """Update boxes rendered on the annotated camera stream when supported."""
        _ = detections

    def evaluate_navigation_candidate(self, pose: Pose2D) -> ToolResult:
        """Compute a path and costmap risk without moving the robot."""
        return ToolResult(
            status=ToolStatus.FAILED,
            data={
                "operation": "evaluate_navigation_candidate",
                "target_pose": pose.to_dict(),
            },
            error="ROS2 backend does not support Nav2 candidate evaluation",
            retryable=False,
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
        min_angular_speed: float = 0.10,
        angular_gain: float,
        max_linear_speed: float,
        min_linear_speed: float = 0.08,
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


class RclpyRos2Adapter(Ros2Adapter):
    """In-process ROS2 implementation for Nav2 and emergency stopping."""

    def __init__(self, settings: RobotAgentSettings) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav2_msgs.action import ComputePathToPose, NavigateToPose
            from nav2_msgs.srv import GetCostmap
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
            raise RuntimeError("rclpy and Nav2 packages are required") from exc

        self.settings = settings
        self._rclpy = rclpy
        self._twist_type = Twist
        self._goal_type = NavigateToPose
        self._path_goal_type = ComputePathToPose
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
        self._latest_image: Any | None = None
        self._detection_overlay: list[Detection] = []
        self._detection_overlay_updated_at = 0.0
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
        self._path_client = ActionClient(
            self._node,
            ComputePathToPose,
            settings.compute_path_action_name,
        )
        self._costmap_client = self._node.create_client(
            GetCostmap,
            settings.global_costmap_service,
        )
        self._costmap_request_type = GetCostmap.Request
        self._active_goal_handle: Any | None = None
        self._active_result_future: Any | None = None

    def _on_image(self, message: Any) -> None:
        self._latest_image = message
        self._publish_annotated_image(message)

    def update_detection_overlay(self, detections: list[Detection]) -> None:
        self._detection_overlay = list(detections)
        self._detection_overlay_updated_at = time.monotonic()

    def _publish_annotated_image(self, message: Any) -> None:
        """Republish camera frames with the latest short-lived detection boxes."""
        if self._annotated_image_publisher.get_subscription_count() <= 0:
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
        except Exception:  # Visualization must never interrupt control.
            return

    def _start_navigation_goal(
        self,
        pose: Pose2D,
        *,
        operation: str,
        started: float,
    ) -> tuple[Any | None, ToolResult | None]:
        """Send one Nav2 goal and return its result future or an early failure."""
        details = {"operation": operation, "target_pose": pose.to_dict()}
        if self._active_goal_handle is not None:
            return None, ToolResult(
                status=ToolStatus.FAILED,
                data=details,
                error="A navigation goal is already active",
                retryable=False,
            )
        if not self._nav_client.wait_for_server(
            timeout_sec=self.settings.tool_timeout_sec
        ):
            return None, ToolResult(
                status=ToolStatus.TIMEOUT,
                data=details,
                error="Nav2 action server unavailable",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        goal = self._goal_type.Goal()
        _populate_pose_stamped(
            goal.pose,
            pose,
            self._node.get_clock().now().to_msg(),
        )
        send_future = self._nav_client.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(
            self._node,
            send_future,
            timeout_sec=self.settings.tool_timeout_sec,
        )
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return None, ToolResult(
                status=ToolStatus.FAILED,
                data=details,
                error="Nav2 rejected goal",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        self._active_goal_handle = handle
        result_future = handle.get_result_async()
        self._active_result_future = result_future
        return result_future, None

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        started = time.monotonic()
        result_future, failure = self._start_navigation_goal(
            pose,
            operation="navigate_to_pose",
            started=started,
        )
        if failure is not None:
            return failure
        assert result_future is not None
        self._rclpy.spin_until_future_complete(
            self._node,
            result_future,
            timeout_sec=self.settings.tool_timeout_sec,
        )
        result = result_future.result()
        if result is None:
            self.cancel_navigation()
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={
                    "operation": "navigate_to_pose",
                    "target_pose": pose.to_dict(),
                    "cancel_requested": True,
                },
                error="Nav2 result timed out",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        self._active_goal_handle = None
        self._active_result_future = None
        success = result.status == 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED
        return ToolResult(
            status=ToolStatus.SUCCESS if success else ToolStatus.FAILED,
            data={
                "operation": "navigate_to_pose",
                "target_pose": pose.to_dict(),
                "goal_status": result.status,
            },
            error=None if success else f"Nav2 goal ended with status {result.status}",
            duration_sec=time.monotonic() - started,
            retryable=not success,
        )

    def evaluate_navigation_candidate(self, pose: Pose2D) -> ToolResult:
        """Use Nav2's planner and global costmap as a read-only feasibility query."""
        started = time.monotonic()
        details = {
            "operation": "evaluate_navigation_candidate",
            "target_pose": pose.to_dict(),
        }
        timeout = min(self.settings.tool_timeout_sec, 10.0)
        if not self._path_client.wait_for_server(timeout_sec=timeout):
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data=details,
                error="Nav2 ComputePathToPose action server unavailable",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )

        goal = self._path_goal_type.Goal()
        _populate_pose_stamped(
            goal.goal,
            pose,
            self._node.get_clock().now().to_msg(),
        )
        goal.use_start = False

        send_future = self._path_client.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=timeout)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={**details, "feasible": False},
                error="Nav2 planner rejected candidate",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        result_future = handle.get_result_async()
        self._rclpy.spin_until_future_complete(
            self._node,
            result_future,
            timeout_sec=timeout,
        )
        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.status != 4:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={**details, "feasible": False},
                error="Nav2 could not compute a path to candidate",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )

        path = wrapped_result.result.path
        points = [
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in path.poses
        ]
        if not points:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={**details, "feasible": False, "path_pose_count": 0},
                error="Nav2 returned an empty candidate path",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        path_length = sum(
            math.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(points, points[1:])
        )

        if not self._costmap_client.wait_for_service(timeout_sec=timeout):
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={**details, "path_length_m": path_length},
                error="Nav2 global costmap service unavailable",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        costmap_future = self._costmap_client.call_async(self._costmap_request_type())
        self._rclpy.spin_until_future_complete(
            self._node,
            costmap_future,
            timeout_sec=timeout,
        )
        response = costmap_future.result()
        if response is None:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={**details, "path_length_m": path_length},
                error="Nav2 global costmap query timed out",
                duration_sec=time.monotonic() - started,
                retryable=True,
            )
        metadata = response.map.metadata
        origin = metadata.origin
        origin_yaw = quaternion_to_yaw(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )
        if abs(origin_yaw) > 1e-4:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={**details, "path_length_m": path_length},
                error="Rotated Nav2 costmap origins are not supported",
                retryable=False,
            )
        snapshot = CostmapSnapshot(
            width=int(metadata.size_x),
            height=int(metadata.size_y),
            resolution=float(metadata.resolution),
            origin_x=float(origin.position.x),
            origin_y=float(origin.position.y),
            data=tuple(int(value) for value in response.map.data),
        )
        feasible, risk, max_cost, clearance = path_risk(snapshot, points)
        target_cost = snapshot.cost_at(pose.x, pose.y)
        feasible = feasible and target_cost < LETHAL_COST
        max_cost = max(max_cost, target_cost)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                **details,
                "feasible": feasible,
                "path_length_m": path_length,
                "path_pose_count": len(points),
                "target_cost": target_cost,
                "max_path_cost": max_cost,
                "obstacle_risk": risk,
                "min_clearance_m": clearance,
            },
            error=None if feasible else "Candidate path intersects lethal or unknown costmap cells",
            duration_sec=time.monotonic() - started,
            retryable=False,
        )

    def navigate_to_pose_with_watch(
        self,
        pose: Pose2D,
        on_tick: Callable[[], Detection | None],
        tick_interval_sec: float,
    ) -> ToolResult:
        """Navigate while cooperatively checking the latest camera frame."""
        started = time.monotonic()
        result_future, failure = self._start_navigation_goal(
            pose,
            operation="navigate_to_pose_with_watch",
            started=started,
        )
        if failure is not None:
            return failure
        assert result_future is not None
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
                    try:
                        detection_pose = self._map_pose_from_tf()
                    except self._transform_exception:
                        detection_pose = None
                    cancel_result = self.cancel_navigation()
                    canceled = cancel_result.status == ToolStatus.CANCELED
                    navigation_stopped = bool(
                        cancel_result.data.get("navigation_stopped")
                    )
                    return ToolResult(
                        status=(
                            ToolStatus.SUCCESS
                            if navigation_stopped
                            else ToolStatus.FAILED
                        ),
                        data={
                            "operation": "navigate_to_pose_with_watch",
                            "found": found.to_dict(),
                            "detection_pose": (
                                detection_pose.to_dict()
                                if detection_pose is not None
                                else None
                            ),
                            "target_pose": pose.to_dict(),
                            "navigation_canceled": canceled,
                            "navigation_stopped": navigation_stopped,
                            "cancel_result": cancel_result.to_dict(),
                        },
                        error=(
                            None
                            if navigation_stopped
                            else "Target was detected but Nav2 did not reach a confirmed terminal state"
                        ),
                        duration_sec=time.monotonic() - started,
                        retryable=not navigation_stopped,
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
        navigation_stopped = (
            cancel_result is None
            or cancel_result.data.get("navigation_stopped") is True
        )
        cancellation_failed = not navigation_stopped
        return ToolResult(
            status=ToolStatus.FAILED if cancellation_failed else ToolStatus.SUCCESS,
            data={
                "operation": "stop_robot",
                "topic": self.settings.cmd_vel_topic,
                "navigation_canceled": (
                    cancel_result is not None
                    and cancel_result.status == ToolStatus.CANCELED
                ),
                "navigation_stopped": navigation_stopped,
                "zero_command_count": zero_command_count,
            },
            error="Zero velocity was published but Nav2 did not reach a confirmed terminal state" if cancellation_failed else None,
            retryable=cancellation_failed,
        )

    def _map_pose_from_tf(self) -> Pose2D:
        transform = self._tf_buffer.lookup_transform(
            self.settings.map_frame,
            self.settings.base_frame,
            self._time_type(),
        )
        rotation = transform.transform.rotation
        return Pose2D(
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

    def get_pose(self) -> ToolResult:
        started = time.monotonic()
        deadline = started + min(self.settings.tool_timeout_sec, 5.0)
        last_error = "Map transform is not available"
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
            try:
                pose = self._map_pose_from_tf()
            except self._transform_exception as exc:
                last_error = str(exc)
                continue
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

    def cancel_navigation(self) -> ToolResult:
        if self._active_goal_handle is None:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "operation": "cancel_navigation",
                    "active_goal": False,
                    "navigation_stopped": True,
                },
            )
        handle = self._active_goal_handle
        result_future = self._active_result_future
        future = handle.cancel_goal_async()
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=self.settings.tool_timeout_sec)
        response = future.result()
        cancel_accepted = bool(
            response is not None and getattr(response, "goals_canceling", [])
        )
        terminal_result = None
        if result_future is not None:
            self._rclpy.spin_until_future_complete(
                self._node,
                result_future,
                timeout_sec=(
                    self.settings.tool_timeout_sec
                    if cancel_accepted
                    else self.settings.post_cancel_settle_sec
                ),
            )
            terminal_result = result_future.result()
        terminal_status = (
            getattr(terminal_result, "status", None)
            if terminal_result is not None
            else None
        )
        terminal_inactive = terminal_status in {4, 5, 6}
        terminal_canceled = terminal_status == 5
        if terminal_inactive:
            self._active_goal_handle = None
            self._active_result_future = None
        zero_command_count = (
            self._hold_zero_velocity(self.settings.post_cancel_settle_sec)
            if terminal_inactive
            else 0
        )
        return ToolResult(
            status=(
                ToolStatus.CANCELED
                if terminal_canceled
                else ToolStatus.SUCCESS
                if terminal_inactive
                else ToolStatus.FAILED
            ),
            data={
                "operation": "cancel_navigation",
                "active_goal": True,
                "cancel_accepted": cancel_accepted,
                "terminal_canceled": terminal_canceled,
                "terminal_status": terminal_status,
                "navigation_stopped": terminal_inactive,
                "post_cancel_settle_sec": (
                    self.settings.post_cancel_settle_sec
                    if terminal_inactive
                    else 0.0
                ),
                "zero_command_count": zero_command_count,
            },
            error=(
                None
                if terminal_inactive
                else "Nav2 did not reach a confirmed terminal state"
            ),
            retryable=not terminal_inactive,
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
        min_angular_speed: float = 0.10,
        angular_gain: float,
        max_linear_speed: float,
        min_linear_speed: float = 0.08,
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

        def publish_stop(*, hold: bool = False) -> None:
            if hold:
                self._hold_zero_velocity(self.settings.post_cancel_settle_sec)
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

        while time.monotonic() - started <= timeout_sec:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if not ticker.ready():
                continue
            # Never keep executing the previous command while detector
            # inference blocks. Each control update is a bounded motion pulse.
            publish_stop()
            detection = on_tick()
            if detection is None:
                stable_count = 0
                publish_stop()
                if (
                    last_detection is None
                    and time.monotonic() - started >= detection_hold_sec
                ):
                    publish_stop(hold=True)
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        data={
                            "operation": "align_to_detection",
                            "found": None,
                            "centered": False,
                        },
                        error="Target was not visible at visual-alignment start",
                        duration_sec=time.monotonic() - started,
                        retryable=True,
                    )
                continue
            last_detection = detection
            last_detection_at = time.monotonic()
            image_position = detection.image_position
            if image_position is None:
                publish_stop(hold=True)
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
                continue

            # Approach is translation-only. If the target drifts, stop and
            # reacquire heading before allowing any more linear motion.
            if abs(horizontal_error) > horizontal_tolerance * 1.5:
                phase = "rotate"
                stable_count = 0
                publish_stop()
                continue
            if abs(size_error) <= box_size_tolerance:
                stable_count += 1
                publish_stop()
                if stable_count < stable_frames:
                    continue
                publish_stop(hold=True)
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

        publish_stop(hold=True)
        detection_age_sec = (
            time.monotonic() - last_detection_at
            if last_detection_at is not None
            else None
        )
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            data={
                "operation": "align_to_detection",
                "found": last_detection.to_dict() if last_detection else None,
                "detection_age_sec": detection_age_sec,
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
