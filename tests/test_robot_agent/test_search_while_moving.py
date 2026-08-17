from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robot_agent.config import RobotAgentSettings
from robot_agent.harness import build_verified_final_response
from robot_agent.perception import Detector
from robot_agent.ros import Ros2Adapter
from robot_agent.ros import adapter as adapter_module
from robot_agent.ros.adapter import RclpyRos2Adapter, Ros2CliAdapter, _DetectionTicker
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.skills.behavior_tree import BehaviorTreeSkill
from robot_agent.state import (
    Detection,
    GoalBlocker,
    GoalEvaluation,
    ImagePosition,
    Pose2D,
    ToolResult,
    ToolStatus,
)
from robot_agent.tools.registry import RobotToolRegistry


class ThirdTickDetector(Detector):
    def __init__(self) -> None:
        self.calls = 0

    def validate_query(self, *, color: str | None, label: str | None) -> None:
        if color != "blue" or label is not None:
            raise ValueError("This test detector only accepts blue")

    def detect(
        self,
        image,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        self.calls += 1
        if self.calls >= 3:
            x_px = 480.0 if self.calls == 3 else 320.0
            x_normalized = 0.75 if self.calls == 3 else 0.5
            return [
                Detection(
                    label="colored_object",
                    color="blue",
                    confidence=0.91,
                    image_position=ImagePosition(
                        x_px,
                        180.0,
                        x_normalized,
                        0.5,
                        0.2,
                        0.35,
                    ),
                )
            ]
        return []


class LowThenHighConfidenceDetector(ThirdTickDetector):
    def detect(self, image, *, color=None, label=None):
        self.calls += 1
        confidence = 0.04 if self.calls == 1 else 0.05
        x_normalized = 0.25 if self.calls == 2 else 0.5
        return [
            Detection(
                label="colored_object",
                color="blue",
                confidence=confidence,
                image_position=ImagePosition(
                    100.0 if self.calls == 2 else 200.0,
                    50.0,
                    x_normalized,
                    0.25,
                    0.2,
                    0.35,
                ),
            )
        ]


class AcquiredThenLowConfidenceDetector(ThirdTickDetector):
    def detect(self, image, *, color=None, label=None):
        self.calls += 1
        confidence = 0.06 if self.calls <= 2 else 0.02
        x_normalized = 0.3 if self.calls <= 2 else 0.5
        return [
            Detection(
                label="colored_object",
                color="blue",
                confidence=confidence,
                image_position=ImagePosition(
                    200.0,
                    180.0,
                    x_normalized,
                    0.5,
                    0.2,
                    0.8,
                ),
            )
        ]


class WatchedNavigationAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.watched_targets: list[Pose2D] = []
        self.plain_navigation_calls = 0
        self.cancel_calls = 0
        self.center_calls = 0
        self.center_kwargs = {}
        self.current_pose = Pose2D(0.0, 0.0, 0.0)
        self.overlay_updates: list[list[Detection]] = []

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self.plain_navigation_calls += 1
        self.current_pose = pose
        return ToolResult(status=ToolStatus.SUCCESS)

    def navigate_to_pose_with_watch(self, pose, on_tick, tick_interval_sec):
        self.watched_targets.append(pose)
        for _ in range(3):
            found = on_tick()
            if found is not None:
                self.cancel_calls += 1
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "operation": "navigate_to_pose_with_watch",
                        "found": found.to_dict(),
                        "target_pose": pose.to_dict(),
                        "navigation_canceled": True,
                    },
                )
        self.current_pose = pose
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"operation": "navigate_to_pose_with_watch", "found": None},
        )

    def get_camera_frame(self):
        return object()

    def update_detection_overlay(self, detections: list[Detection]) -> None:
        self.overlay_updates.append(list(detections))

    def align_to_detection(self, on_tick, **kwargs) -> ToolResult:
        self.center_calls += 1
        self.center_kwargs = kwargs
        for _ in range(3):
            found = on_tick()
            if found is None or found.image_position is None:
                return ToolResult(status=ToolStatus.FAILED, error="target lost")
            horizontal_error = found.image_position.x_normalized - 0.5
            if abs(horizontal_error) <= kwargs["horizontal_tolerance"]:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "operation": "align_to_detection",
                        "found": found.to_dict(),
                        "centered": True,
                        "horizontal_error": horizontal_error,
                    },
                )
        return ToolResult(status=ToolStatus.TIMEOUT, error="target not centered")

    def stop_robot(self) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"pose": self.current_pose.to_dict()},
        )

    def cancel_navigation(self) -> ToolResult:
        self.cancel_calls += 1
        return ToolResult(status=ToolStatus.CANCELED)

    def detect_color(self, color: str) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data={"detections": []})


class SearchWhileMovingTest(unittest.TestCase):
    def _registry(self, root: Path, adapter: Ros2Adapter, detector: Detector):
        locations = root / "locations.yaml"
        locations.write_text(
            "location1: [1.0, 0.0, 0.0]\n"
            "location2: [2.0, 0.0, 0.0]\n"
            "location3: [3.0, 0.0, 0.0]\n",
            encoding="utf-8",
        )
        settings = RobotAgentSettings(
            location_file=locations,
            run_directory=root / "runs",
            execute_ros2=True,
            trace=False,
            detection_interval_sec=0.75,
            max_no_progress_continuations=5,
        )
        runtime = RobotAgentRuntime(settings, "find a blue object while moving")
        registry = RobotToolRegistry(
            runtime,
            adapter,
            bt_skill=object(),
            detector=detector,
        )
        return runtime, registry

    def test_match_on_third_tick_cancels_once_and_stops_route(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = ThirdTickDetector()
            runtime, registry = self._registry(
                Path(temporary_directory), adapter, detector
            )
            tool = {tool.name: tool for tool in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke(
                {
                    "route": ["location1", "location2", "location3"],
                    "color": "blue",
                }
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(result["data"]["found"]["color"], "blue")
            self.assertEqual(
                result["data"]["image_position"],
                {
                    "x_px": 320.0,
                    "y_px": 180.0,
                    "x_normalized": 0.5,
                    "y_normalized": 0.5,
                    "width_normalized": 0.2,
                    "height_normalized": 0.35,
                },
            )
            self.assertEqual(detector.calls, 4)
            self.assertEqual(len(adapter.overlay_updates), 5)
            self.assertEqual(adapter.overlay_updates[0], [])
            self.assertEqual(adapter.overlay_updates[1], [])
            self.assertEqual(adapter.overlay_updates[2], [])
            self.assertEqual(
                adapter.overlay_updates[-1][0].label,
                "colored_object",
            )
            self.assertEqual(adapter.cancel_calls, 1)
            self.assertEqual(adapter.center_calls, 1)
            self.assertEqual(adapter.center_kwargs["target_box_size"], 0.6)
            self.assertEqual(len(adapter.watched_targets), 2)
            self.assertEqual(adapter.watched_targets[-1].x, 2.0)
            self.assertEqual(runtime.state.robot_state.visible_objects[0].color, "blue")
            self.assertIsNotNone(runtime.state.robot_state.last_perception_at)
            self.assertTrue(result["data"]["centered"])
            runtime.state.goal_evaluation = GoalEvaluation(
                satisfied=True,
                blocker=GoalBlocker.NONE,
                reason="Perception verified the requested semantic object evidence",
            )
            final = build_verified_final_response(runtime, "succeeded")
            self.assertIn("confidence=0.910", final)
            self.assertIn("x=320.0px, y=180.0px", final)
            self.assertIn("Robot pose at detection", final)

    def test_detection_stops_at_five_percent_threshold(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = LowThenHighConfidenceDetector()
            _, registry = self._registry(Path(temporary_directory), adapter, detector)
            tool = {tool.name: tool for tool in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke({"route": ["location1"], "color": "blue"})

            self.assertEqual(detector.calls, 3)
            self.assertEqual(adapter.overlay_updates[0], [])
            self.assertEqual(adapter.overlay_updates[1], [])
            self.assertEqual(result["data"]["found"]["confidence"], 0.05)
            self.assertEqual(result["data"]["confidence_threshold"], 0.05)
            self.assertEqual(adapter.cancel_calls, 1)
            self.assertEqual(adapter.center_calls, 1)

    def test_confirmed_target_uses_lower_tracking_threshold_during_alignment(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = AcquiredThenLowConfidenceDetector()
            _, registry = self._registry(Path(temporary_directory), adapter, detector)
            tool = {item.name: item for item in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke({"route": ["location1"], "color": "blue"})

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(result["data"]["found"]["confidence"], 0.02)
            self.assertEqual(result["data"]["tracking_confidence_threshold"], 0.01)
            self.assertTrue(result["data"]["centered"])
            self.assertEqual(adapter.cancel_calls, 1)
            self.assertEqual(adapter.center_calls, 1)

    def test_single_frame_candidate_does_not_cancel_navigation(self):
        class SingleFrameDetector(Detector):
            def __init__(self):
                self.calls = 0

            def validate_query(self, *, color=None, label=None):
                return None

            def detect(self, image, *, color=None, label=None):
                self.calls += 1
                if self.calls == 1:
                    return [
                        Detection(
                            "fire extinguisher",
                            0.9,
                            image_position=ImagePosition(
                                320.0, 180.0, 0.5, 0.5, 0.2, 0.4
                            ),
                        )
                    ]
                return []

        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = SingleFrameDetector()
            _, registry = self._registry(Path(temporary_directory), adapter, detector)
            tool = {item.name: item for item in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke(
                {"route": ["location1"], "label": "fire extinguisher"}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertIsNone(result["data"]["found"])
            self.assertEqual(adapter.cancel_calls, 0)
            self.assertEqual(adapter.center_calls, 0)
            self.assertEqual(adapter.overlay_updates[-1], [])

    def test_unconfirmed_navigation_cancellation_blocks_alignment(self):
        class UnconfirmedCancellationAdapter(WatchedNavigationAdapter):
            def navigate_to_pose_with_watch(self, pose, on_tick, tick_interval_sec):
                self.watched_targets.append(pose)
                found = None
                for _ in range(3):
                    found = on_tick()
                    if found is not None:
                        break
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "operation": "navigate_to_pose_with_watch",
                        "found": found.to_dict() if found else None,
                        "navigation_canceled": False,
                    },
                )

        with TemporaryDirectory() as temporary_directory:
            adapter = UnconfirmedCancellationAdapter()
            _, registry = self._registry(
                Path(temporary_directory),
                adapter,
                LowThenHighConfidenceDetector(),
            )
            tool = {item.name: item for item in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke({"route": ["location1"], "color": "blue"})

            self.assertEqual(result["status"], ToolStatus.FAILED.value)
            self.assertIn("without confirmed Nav2 cancellation", result["error"])
            self.assertFalse(result["data"]["navigation_canceled"])
            self.assertEqual(adapter.center_calls, 0)

    def test_detector_health_is_reported_during_search(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = LowThenHighConfidenceDetector()
            runtime, registry = self._registry(
                Path(temporary_directory), adapter, detector
            )
            tool = {tool.name: tool for tool in registry.build()}[
                "search_for_object"
            ]

            with patch.object(runtime, "emit") as emit:
                tool.invoke({"route": ["location1"], "color": "blue"})

            samples = [
                call.args[1]
                for call in emit.call_args_list
                if call.args[0] == "detector_sampled"
            ]
            self.assertGreaterEqual(len(samples), 2)
            self.assertEqual(samples[0]["backend"], "yoloe")
            self.assertEqual(samples[0]["best_confidence"], 0.04)
            self.assertEqual(samples[0]["box_threshold"], 0.05)
            self.assertEqual(samples[-1]["best_confidence"], 0.05)
            self.assertEqual(samples[-1]["stop_threshold"], 0.05)

    def test_plain_navigation_and_bt_do_not_use_watched_navigation(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            _, registry = self._registry(
                Path(temporary_directory), adapter, ThirdTickDetector()
            )
            navigate = {tool.name: tool for tool in registry.build()}["navigate_to"]

            result = navigate.invoke({"location": "location1"})

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value)
            self.assertEqual(adapter.plain_navigation_calls, 1)
            self.assertEqual(adapter.watched_targets, [])
            self.assertNotIn(
                "navigate_to_pose_with_watch",
                inspect.getsource(BehaviorTreeSkill),
            )


class DetectionTickerTest(unittest.TestCase):
    def test_never_ticks_more_often_than_interval(self):
        clock = Mock(side_effect=[0.0, 0.2, 0.74, 0.75, 1.49, 1.50])
        ticker = _DetectionTicker(0.75, clock=clock)

        ready = [ticker.ready() for _ in range(6)]

        self.assertEqual(ready, [True, False, False, True, False, True])


class RclpyCancellationHandoffTest(unittest.TestCase):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    class FakeTwist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0)
            self.angular = SimpleNamespace(z=0.0)

    def _adapter(self, clock, *, active=True):
        class Future:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class Handle:
            def cancel_goal_async(self):
                return Future(SimpleNamespace(goals_canceling=[object()]))

        class Publisher:
            def __init__(self):
                self.timestamps = []

            def publish(self, message):
                self.timestamps.append(clock.now)

        adapter = object.__new__(RclpyRos2Adapter)
        adapter.settings = SimpleNamespace(
            tool_timeout_sec=1.0,
            post_cancel_settle_sec=0.25,
        )
        adapter._node = object()
        adapter._twist_type = self.FakeTwist
        adapter._publisher = Publisher()
        adapter._rclpy = SimpleNamespace(
            spin_until_future_complete=lambda node, future, timeout_sec: None,
            spin_once=lambda node, timeout_sec: clock.advance(timeout_sec),
        )
        adapter._active_goal_handle = Handle() if active else None
        adapter._active_result_future = (
            Future(SimpleNamespace(status=5)) if active else None
        )
        return adapter

    def test_confirmed_cancel_outlasts_simulated_residual_velocity(self):
        clock = self.FakeClock()
        adapter = self._adapter(clock)
        residual_command_until = 0.15

        with patch.object(adapter_module.time, "monotonic", clock.monotonic):
            with patch.object(adapter_module.time, "sleep", clock.advance):
                result = adapter.cancel_navigation()

        self.assertEqual(result.status, ToolStatus.CANCELED)
        self.assertGreaterEqual(clock.now, 0.25)
        self.assertGreater(clock.now, residual_command_until)
        self.assertTrue(
            any(
                timestamp >= residual_command_until
                for timestamp in adapter._publisher.timestamps
            )
        )
        self.assertEqual(result.data["post_cancel_settle_sec"], 0.25)
        self.assertGreaterEqual(result.data["zero_command_count"], 4)

    def test_no_active_goal_has_no_settle_delay(self):
        clock = self.FakeClock()
        adapter = self._adapter(clock, active=False)

        with patch.object(adapter_module.time, "monotonic", clock.monotonic):
            with patch.object(adapter_module.time, "sleep", clock.advance):
                result = adapter.cancel_navigation()

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(clock.now, 0.0)
        self.assertEqual(adapter._publisher.timestamps, [])

    def test_settle_is_centralized_outside_tool_call_sites(self):
        source = inspect.getsource(RobotToolRegistry)
        self.assertNotIn("post_cancel_settle_sec", source)


class RclpyCenteringControlTest(unittest.TestCase):
    def test_alignment_logs_compact_state_command_and_pose_delta(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        logger = Logger()
        adapter = object.__new__(RclpyRos2Adapter)
        adapter._node = SimpleNamespace(get_logger=lambda: logger)
        adapter._twist_type = FakeTwist
        adapter._publisher = SimpleNamespace(publish=lambda message: None)
        adapter._latest_pose = Pose2D(0.0, 0.0, 0.0)

        def spin_once(node, timeout_sec):
            pose = adapter._latest_pose
            adapter._latest_pose = Pose2D(
                pose.x + 0.01,
                pose.y,
                pose.yaw + 0.01,
            )

        adapter._rclpy = SimpleNamespace(spin_once=spin_once)
        detections = iter(
            [
                Detection(
                    "fire extinguisher",
                    0.11,
                    image_position=ImagePosition(
                        480.0, 180.0, 0.75, 0.5, 0.2, 0.2
                    ),
                ),
                Detection(
                    "fire extinguisher",
                    0.12,
                    image_position=ImagePosition(
                        320.0, 180.0, 0.5, 0.5, 0.2, 0.8
                    ),
                ),
                Detection(
                    "fire extinguisher",
                    0.12,
                    image_position=ImagePosition(
                        320.0, 180.0, 0.5, 0.5, 0.2, 0.8
                    ),
                ),
            ]
        )

        result = adapter.align_to_detection(
            lambda: next(detections),
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            target_box_size=0.8,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=1.0,
            stable_frames=1,
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(len(logger.messages), 3)
        self.assertIn("state=rotate", logger.messages[0])
        self.assertIn("v=0.000", logger.messages[0])
        self.assertIn("w=-0.125", logger.messages[0])
        self.assertIn("moved=0.0100m", logger.messages[0])
        self.assertIn("state=rotation_complete", logger.messages[1])
        self.assertIn("state=aligned", logger.messages[2])
        self.assertIn("cmd=stop", logger.messages[2])

    def test_repeated_lost_detection_logs_are_throttled(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        logger = Logger()
        adapter = object.__new__(RclpyRos2Adapter)
        adapter._node = SimpleNamespace(get_logger=lambda: logger)
        adapter._twist_type = FakeTwist
        adapter._publisher = SimpleNamespace(publish=lambda message: None)
        adapter._latest_pose = Pose2D(0.0, 0.0, 0.0)
        adapter._rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)

        result = adapter.align_to_detection(
            lambda: None,
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            target_box_size=0.8,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=0.01,
            detection_hold_sec=0.0,
        )

        self.assertEqual(result.status, ToolStatus.TIMEOUT)
        self.assertEqual(
            logger.messages,
            ["visual_alignment state=lost cmd=stop moved=0.0000m turned=0.0000rad"],
        )

    def test_brief_detector_dropout_stops_instead_of_reusing_motion(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        class FakePublisher:
            def __init__(self):
                self.commands = []

            def publish(self, message):
                self.commands.append((message.linear.x, message.angular.z))

        adapter = object.__new__(RclpyRos2Adapter)
        adapter._rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)
        adapter._node = object()
        adapter._twist_type = FakeTwist
        adapter._publisher = FakePublisher()
        initial = Detection(
            "fire extinguisher",
            0.11,
            image_position=ImagePosition(480.0, 180.0, 0.75, 0.5, 0.2, 0.2),
        )
        centered = Detection(
            "fire extinguisher",
            0.12,
            image_position=ImagePosition(320.0, 180.0, 0.5, 0.5, 0.2, 0.8),
        )
        detections = iter([initial, None, centered, centered])

        result = adapter.align_to_detection(
            lambda: next(detections),
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            target_box_size=0.8,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=1.0,
            stable_frames=1,
            detection_hold_sec=1.0,
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertNotEqual(adapter._publisher.commands[0], (0.0, 0.0))
        self.assertEqual(adapter._publisher.commands[1:4], [(0.0, 0.0)] * 3)
        self.assertEqual(adapter._publisher.commands[-3:], [(0.0, 0.0)] * 3)

    def test_rotation_and_approach_never_drive_both_axes_together(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        class FakePublisher:
            def __init__(self):
                self.commands = []

            def publish(self, message):
                self.commands.append((message.linear.x, message.angular.z))

        adapter = object.__new__(RclpyRos2Adapter)
        adapter._rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)
        adapter._node = object()
        adapter._twist_type = FakeTwist
        adapter._publisher = FakePublisher()
        detections = iter(
            [
                Detection(
                    "colored_object",
                    0.9,
                    color="blue",
                    image_position=ImagePosition(480.0, 180.0, 0.75, 0.5, 0.2, 0.10),
                ),
                Detection(
                    "colored_object",
                    0.9,
                    color="blue",
                    image_position=ImagePosition(320.0, 180.0, 0.5, 0.5, 0.2, 0.10),
                ),
                Detection(
                    "colored_object",
                    0.9,
                    color="blue",
                    image_position=ImagePosition(320.0, 180.0, 0.5, 0.5, 0.2, 0.25),
                ),
                Detection(
                    "colored_object",
                    0.9,
                    color="blue",
                    image_position=ImagePosition(320.0, 180.0, 0.5, 0.5, 0.2, 0.35),
                ),
            ]
        )

        result = adapter.align_to_detection(
            lambda: next(detections),
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            target_box_size=0.35,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=1.0,
            stable_frames=1,
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        moving_commands = [
            command
            for command in adapter._publisher.commands
            if command != (0.0, 0.0)
        ]
        self.assertEqual(len(moving_commands), 2)
        first_linear, first_angular = moving_commands[0]
        second_linear, second_angular = moving_commands[1]
        self.assertEqual(first_linear, 0.0)
        self.assertLess(first_angular, 0.0)
        self.assertGreater(second_linear, 0.0)
        self.assertEqual(second_angular, 0.0)
        self.assertEqual(adapter._publisher.commands[-3:], [(0.0, 0.0)] * 3)
        self.assertAlmostEqual(result.data["vertical_error"], 0.0)

    def test_alignment_requires_consecutive_stable_frames_per_phase(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        adapter = object.__new__(RclpyRos2Adapter)
        adapter._rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)
        adapter._node = object()
        adapter._twist_type = FakeTwist
        adapter._publisher = SimpleNamespace(publish=lambda message: None)
        centered = lambda x=0.5: Detection(
            "fire extinguisher",
            0.9,
            image_position=ImagePosition(320.0, 180.0, x, 0.5, 0.2, 0.8),
        )
        detections = iter(
            [
                centered(),
                centered(0.7),
                centered(),
                centered(),
                centered(),
                centered(),
            ]
        )
        calls = 0

        def detect():
            nonlocal calls
            calls += 1
            return next(detections)

        result = adapter.align_to_detection(
            detect,
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.03,
            target_box_size=0.8,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=1.0,
            stable_frames=2,
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(calls, 6)
        self.assertEqual(result.data["stable_frames"], 2)

    def test_missing_image_position_fails_cleanly(self):
        class FakeTwist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0.0)
                self.angular = SimpleNamespace(z=0.0)

        adapter = object.__new__(RclpyRos2Adapter)
        adapter._rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)
        adapter._node = object()
        adapter._twist_type = FakeTwist
        adapter._publisher = SimpleNamespace(publish=lambda message: None)

        result = adapter.align_to_detection(
            lambda: Detection("fire extinguisher", 0.9),
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            target_box_size=0.35,
            box_size_tolerance=0.05,
            max_angular_speed=0.2,
            angular_gain=0.5,
            max_linear_speed=0.12,
            linear_gain=0.5,
            timeout_sec=1.0,
        )

        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("no image position", result.error)


class RclpyDetectionOverlayTest(unittest.TestCase):
    def test_normalized_detection_box_is_published_on_annotated_topic(self):
        rectangles = []
        captions = []

        class FakeFrame:
            shape = (400, 800, 3)

            def copy(self):
                return self

        class FakeBridge:
            def imgmsg_to_cv2(self, message, desired_encoding):
                self.source_message = message
                self.desired_encoding = desired_encoding
                return FakeFrame()

            def cv2_to_imgmsg(self, frame, encoding):
                return SimpleNamespace(header=None, frame=frame, encoding=encoding)

        class FakeImagePublisher:
            def __init__(self):
                self.messages = []

            def get_subscription_count(self):
                return 1

            def publish(self, message):
                self.messages.append(message)

        fake_cv2 = SimpleNamespace(
            FONT_HERSHEY_SIMPLEX=0,
            LINE_AA=1,
            rectangle=lambda frame, top_left, bottom_right, color, thickness: rectangles.append(
                (top_left, bottom_right, color, thickness)
            ),
            putText=lambda frame, text, origin, font, scale, color, thickness, line: captions.append(
                (text, origin)
            ),
        )
        adapter = object.__new__(RclpyRos2Adapter)
        adapter.settings = SimpleNamespace(detection_interval_sec=1.0)
        adapter._annotated_image_publisher = FakeImagePublisher()
        adapter._detection_overlay = []
        adapter._detection_overlay_updated_at = 0.0
        adapter.update_detection_overlay(
            [
                Detection(
                    "fire extinguisher",
                    0.91,
                    image_position=ImagePosition(
                        400.0,
                        200.0,
                        0.5,
                        0.5,
                        0.25,
                        0.5,
                    ),
                )
            ]
        )
        source = SimpleNamespace(header=object())

        with patch.dict(
            "sys.modules",
            {
                "cv2": fake_cv2,
                "cv_bridge": SimpleNamespace(CvBridge=FakeBridge),
            },
        ):
            adapter._publish_annotated_image(source)

        self.assertEqual(rectangles, [((300, 100), (500, 300), (0, 255, 0), 2)])
        self.assertEqual(captions[0][0], "fire extinguisher 0.91")
        self.assertIs(
            adapter._annotated_image_publisher.messages[0].header,
            source.header,
        )


class CliWatchedNavigationTest(unittest.TestCase):
    def test_cli_fails_loudly_without_blind_navigation(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = RobotAgentSettings(
                location_file=root / "locations.yaml",
                run_directory=root / "runs",
                execute_ros2=False,
                trace=False,
            )
            adapter = Ros2CliAdapter(settings)
            pose = Pose2D(1.0, 2.0, 0.3)
            on_tick = Mock()
            with patch.object(adapter, "navigate_to_pose") as navigate:
                result = adapter.navigate_to_pose_with_watch(pose, on_tick, 1.0)

            self.assertEqual(result.status, ToolStatus.FAILED)
            self.assertIn("require ROBOT_AGENT_ROS_BACKEND=rclpy", result.error)
            navigate.assert_not_called()
            on_tick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
