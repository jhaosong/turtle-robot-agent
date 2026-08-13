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
                    ),
                )
            ]
        return []


class LowThenHighConfidenceDetector(ThirdTickDetector):
    def detect(self, image, *, color=None, label=None):
        self.calls += 1
        confidence = 0.79 if self.calls == 1 else 0.8
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
                ),
            )
        ]


class WatchedNavigationAdapter(Ros2Adapter):
    def __init__(self) -> None:
        self.watched_targets: list[Pose2D] = []
        self.plain_navigation_calls = 0
        self.cancel_calls = 0
        self.center_calls = 0
        self.current_pose = Pose2D(0.0, 0.0, 0.0)

    def navigate_to_pose(self, pose: Pose2D) -> ToolResult:
        self.plain_navigation_calls += 1
        self.current_pose = pose
        return ToolResult(status=ToolStatus.SUCCESS)

    def navigate_to_pose_with_watch(self, pose, on_tick, tick_interval_sec):
        self.watched_targets.append(pose)
        for _ in range(2):
            found = on_tick()
            if found is not None:
                self.cancel_calls += 1
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "operation": "navigate_to_pose_with_watch",
                        "found": found.to_dict(),
                        "target_pose": pose.to_dict(),
                    },
                )
        self.current_pose = pose
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"operation": "navigate_to_pose_with_watch", "found": None},
        )

    def get_camera_frame(self):
        return object()

    def center_target_in_view(self, on_tick, **kwargs) -> ToolResult:
        self.center_calls += 1
        found = on_tick()
        if found is None:
            return ToolResult(status=ToolStatus.FAILED, error="target lost")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "center_target_in_view",
                "found": found.to_dict(),
                "centered": True,
                "horizontal_error": 0.0,
            },
        )

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
                },
            )
            self.assertEqual(detector.calls, 4)
            self.assertEqual(adapter.cancel_calls, 1)
            self.assertEqual(adapter.center_calls, 1)
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

    def test_detection_below_eighty_percent_does_not_stop_navigation(self):
        with TemporaryDirectory() as temporary_directory:
            adapter = WatchedNavigationAdapter()
            detector = LowThenHighConfidenceDetector()
            _, registry = self._registry(Path(temporary_directory), adapter, detector)
            tool = {tool.name: tool for tool in registry.build()}[
                "search_for_object"
            ]

            result = tool.invoke({"route": ["location1"], "color": "blue"})

            self.assertEqual(detector.calls, 3)
            self.assertEqual(result["data"]["found"]["confidence"], 0.8)
            self.assertEqual(result["data"]["confidence_threshold"], 0.8)
            self.assertEqual(adapter.cancel_calls, 1)
            self.assertEqual(adapter.center_calls, 1)

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


class RclpyCenteringControlTest(unittest.TestCase):
    def test_rotates_toward_right_side_target_then_publishes_stop(self):
        class FakeTwist:
            def __init__(self):
                self.angular = SimpleNamespace(z=0.0)

        class FakePublisher:
            def __init__(self):
                self.angular_commands = []

            def publish(self, message):
                self.angular_commands.append(message.angular.z)

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
                    image_position=ImagePosition(480.0, 180.0, 0.75, 0.5),
                ),
                Detection(
                    "colored_object",
                    0.9,
                    color="blue",
                    image_position=ImagePosition(320.0, 180.0, 0.5, 0.5),
                ),
            ]
        )

        result = adapter.center_target_in_view(
            lambda: next(detections),
            tick_interval_sec=0.000001,
            horizontal_tolerance=0.08,
            max_angular_speed=0.2,
            gain=0.5,
            timeout_sec=1.0,
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertLess(adapter._publisher.angular_commands[0], 0.0)
        self.assertEqual(adapter._publisher.angular_commands[-3:], [0.0, 0.0, 0.0])


class CliWatchedNavigationTest(unittest.TestCase):
    def test_cli_delegates_without_calling_tick(self):
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
            expected = ToolResult(status=ToolStatus.PLANNED, data={"sentinel": True})

            with patch.object(adapter, "navigate_to_pose", return_value=expected) as navigate:
                result = adapter.navigate_to_pose_with_watch(pose, on_tick, 1.0)

            self.assertIs(result, expected)
            navigate.assert_called_once_with(pose)
            on_tick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
