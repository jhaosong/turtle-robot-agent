from __future__ import annotations

import inspect
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robot_agent.config.settings import RobotAgentSettings
from robot_agent.navigation import (
    CandidateEvaluation,
    CostmapSnapshot,
    NoFeasibleBaselineError,
    generate_baseline_candidates,
    generate_object_viewpoints,
    path_risk,
    select_baseline_candidate,
)
from robot_agent.perception.bearing_localization import (
    bearing_from_detection,
    triangulate_from_bearings,
)
from robot_agent.ros import Ros2Adapter
from robot_agent.runtime.runtime import RobotAgentRuntime
from robot_agent.skills.behavior_tree import BehaviorTreeSkill
from robot_agent.state import Detection, ImagePosition, Pose2D, ToolResult, ToolStatus
from robot_agent.tools.registry import RobotToolRegistry, _angle_is_covered


class GeometryDetector:
    def __init__(self, adapter: "FeasibleAdapter", object_x: float, object_y: float):
        self.adapter = adapter
        self.object_x = object_x
        self.object_y = object_y
        self.horizontal_fov_rad = 1.085595

    def validate_query(self, *, color, label):
        if label != "fire extinguisher":
            raise ValueError("unexpected label")

    def detect(self, image, *, color=None, label=None):
        del image, color
        self.validate_query(color=None, label=label)
        pose = self.adapter.pose
        relative_bearing = math.atan2(
            self.object_y - pose.y,
            self.object_x - pose.x,
        ) - pose.yaw
        relative_bearing = math.atan2(
            math.sin(relative_bearing), math.cos(relative_bearing)
        )
        x_normalized = 0.5 - math.tan(relative_bearing) / (
            2.0 * math.tan(self.horizontal_fov_rad / 2.0)
        )
        if not 0.0 <= x_normalized <= 1.0:
            return []
        return [
            Detection(
                label="fire extinguisher",
                confidence=0.9,
                image_position=ImagePosition(
                    x_px=640.0 * x_normalized,
                    y_px=240.0,
                    x_normalized=x_normalized,
                    y_normalized=0.5,
                    width_normalized=0.2,
                    height_normalized=0.4,
                ),
            )
        ]


class FirstViewBlindDetector(GeometryDetector):
    """Simulate a target hidden only at the first planned east viewpoint."""

    def detect(self, image, *, color=None, label=None):
        if math.hypot(self.adapter.pose.x - 1.0, self.adapter.pose.y) < 0.05:
            return []
        return super().detect(image, color=color, label=label)


class InitialFramesMissDetector(GeometryDetector):
    def __init__(self, *args, misses: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining_misses = misses

    def detect(self, image, *, color=None, label=None):
        if self.remaining_misses > 0:
            self.remaining_misses -= 1
            return []
        return super().detect(image, color=color, label=label)


class LowConfidenceGeometryDetector(GeometryDetector):
    def detect(self, image, *, color=None, label=None):
        detections = super().detect(image, color=color, label=label)
        return [
            Detection(
                label=item.label,
                confidence=0.02,
                color=item.color,
                position=item.position,
                image_position=item.image_position,
            )
            for item in detections
        ]


class FeasibleAdapter(Ros2Adapter):
    def __init__(self):
        self.pose = Pose2D(0.0, -2.0, math.pi / 2.0)
        self.navigation_targets: list[Pose2D] = []
        self.alignment_calls = 0
        self.evaluation_calls = 0
        self.block_evaluation_number: int | None = None

    def navigate_to_pose(self, pose):
        self.navigation_targets.append(pose)
        self.pose = pose
        return ToolResult(status=ToolStatus.SUCCESS)

    def evaluate_navigation_candidate(self, pose):
        self.evaluation_calls += 1
        blocked = self.evaluation_calls == self.block_evaluation_number
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "feasible": not blocked,
                "path_length_m": math.hypot(
                    pose.x - self.pose.x, pose.y - self.pose.y
                ),
                "obstacle_risk": 0.0,
            },
        )

    def align_to_detection(self, on_tick, **kwargs):
        del kwargs
        self.alignment_calls += 1
        on_tick()  # Consume a possibly stale seed, as the real controller does.
        detection = on_tick()
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "align_to_detection",
                "found": detection.to_dict() if detection else None,
            },
        )

    def stop_robot(self):
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_pose(self):
        return ToolResult(status=ToolStatus.SUCCESS, data={"pose": self.pose.to_dict()})

    def cancel_navigation(self):
        return ToolResult(status=ToolStatus.SUCCESS)

    def get_camera_frame(self):
        return object()


class FirstAlignmentTimeoutAdapter(FeasibleAdapter):
    timeout_detection_age_sec = 0.0

    def align_to_detection(self, on_tick, **kwargs):
        self.alignment_calls += 1
        detection = on_tick()
        if self.alignment_calls == 1:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                data={
                    "operation": "align_to_detection",
                    "found": detection.to_dict() if detection else None,
                    "detection_age_sec": self.timeout_detection_age_sec,
                    "centered": False,
                },
                error="Target was not aligned before timeout",
                retryable=True,
            )
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "operation": "align_to_detection",
                "found": detection.to_dict() if detection else None,
            },
        )


class ActiveViewPlanningTest(unittest.TestCase):
    def test_bbox_bearing_sign_center_and_edges(self):
        fov = math.radians(90.0)
        for x, expected_offset in ((0.5, 0.0), (0.0, fov / 2), (1.0, -fov / 2)):
            bearing = bearing_from_detection(
                ImagePosition(640 * x, 240, x, 0.5),
                robot_yaw=0.3,
                horizontal_fov_rad=fov,
            )
            self.assertAlmostEqual(bearing, 0.3 + expected_offset)

    def test_closed_form_triangulation_recovers_exact_position(self):
        target = (2.0, 1.0)
        first = Pose2D(0.0, 0.0, 0.0)
        second = Pose2D(0.0, 1.0, 0.0)
        result = triangulate_from_bearings(
            first,
            math.atan2(target[1] - first.y, target[0] - first.x),
            second,
            math.atan2(target[1] - second.y, target[0] - second.x),
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.position.x, target[0], places=12)
        self.assertAlmostEqual(result.position.y, target[1], places=12)
        self.assertGreater(result.confidence, 0.0)

    def test_near_parallel_geometry_is_flagged_low_confidence(self):
        result = triangulate_from_bearings(
            Pose2D(0.0, 0.0, 0.0),
            math.atan2(0.01, 10.0),
            Pose2D(0.0, 0.5, 0.0),
            math.atan2(-0.49, 10.0),
            min_ray_angle_rad=math.radians(5.0),
        )
        self.assertFalse(result.valid)
        self.assertLess(result.confidence, 0.5)
        self.assertIn("angular separation", result.reason)

    def test_candidate_fan_is_relative_to_target_bearing(self):
        candidates = generate_baseline_candidates(
            Pose2D(1.0, 2.0, -2.0),
            target_bearing_rad=math.pi / 2,
            radius_m=1.0,
            assumed_object_distance_m=2.0,
        )
        self.assertEqual(len(candidates), 8)
        left_90 = next(item for item in candidates if item.name == "left_90deg")
        self.assertAlmostEqual(left_90.pose.x, 0.0)
        self.assertAlmostEqual(left_90.pose.y, 2.0)
        self.assertAlmostEqual(left_90.tangential_displacement_m, 1.0)

        assumed_object = Pose2D(1.0, 4.0, 0.0)
        for candidate in candidates:
            expected_yaw = math.atan2(
                assumed_object.y - candidate.pose.y,
                assumed_object.x - candidate.pose.x,
            )
            self.assertAlmostEqual(candidate.pose.yaw, expected_yaw)
        self.assertGreater(
            abs(left_90.pose.yaw - math.pi / 2),
            math.radians(10.0),
        )

    def test_costmap_hard_rejects_lethal_path(self):
        data = [0] * 25
        data[2 * 5 + 2] = 254
        snapshot = CostmapSnapshot(5, 5, 1.0, 0.0, 0.0, tuple(data))
        feasible, risk, max_cost, clearance = path_risk(
            snapshot,
            [(0.5, 0.5), (2.5, 2.5)],
        )
        self.assertFalse(feasible)
        self.assertEqual(max_cost, 254)
        self.assertEqual(clearance, 0.0)
        self.assertGreater(risk, 0.0)

    def test_best_safe_tangential_candidate_wins(self):
        candidates = generate_baseline_candidates(
            Pose2D(0.0, 0.0, 0.0), 0.0, radius_m=1.0
        )
        left_90 = next(item for item in candidates if item.name == "left_90deg")
        left_30 = next(item for item in candidates if item.name == "left_30deg")
        blocked = next(item for item in candidates if item.name == "right_90deg")
        selected = select_baseline_candidate(
            [
                CandidateEvaluation(blocked, False, 1.0, 0.0, reason="lethal"),
                CandidateEvaluation(left_30, True, 0.8, 0.0),
                CandidateEvaluation(left_90, True, 1.1, 0.0),
            ],
            alpha=3.0,
            beta=0.35,
            gamma=2.0,
        )
        self.assertEqual(selected.candidate.name, "left_90deg")

    def test_all_candidates_rejected_has_distinct_error(self):
        candidate = generate_baseline_candidates(Pose2D(0, 0, 0), 0.0)[0]
        with self.assertRaisesRegex(NoFeasibleBaselineError, "All baseline candidates"):
            select_baseline_candidate(
                [CandidateEvaluation(candidate, False, math.inf, 1.0)]
            )

    def test_four_viewpoints_are_ninety_degrees_apart(self):
        object_pose = Pose2D(2.0, 1.0, 0.0)
        viewpoints = generate_object_viewpoints(
            object_pose,
            Pose2D(2.0, 0.0, 0.0),
            radius_m=1.0,
            count=4,
        )
        angles = [math.atan2(item.y - 1.0, item.x - 2.0) for item in viewpoints]
        for first, second in zip(angles, angles[1:]):
            self.assertAlmostEqual(second - first, math.pi / 2)
        self.assertTrue(all(math.isclose(math.hypot(p.x - 2, p.y - 1), 1.0) for p in viewpoints))

    def test_viewpoint_phase_is_locked_to_map_frame(self):
        viewpoints = generate_object_viewpoints(
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.7, 0.7, 0.0),
            radius_m=1.0,
            count=4,
        )

        self.assertAlmostEqual(viewpoints[0].x, 1.0)
        self.assertAlmostEqual(viewpoints[0].y, 0.0)
        self.assertAlmostEqual(viewpoints[1].x, 0.0)
        self.assertAlmostEqual(viewpoints[1].y, 1.0)

    def _build_circle_tool(self, root, adapter):
        locations = root / "locations.yaml"
        locations.write_text("inspection_start: [0, -2, 1.571]\n", encoding="utf-8")
        settings = RobotAgentSettings(
            location_file=locations,
            run_directory=root / "runs",
            trace=False,
            max_no_progress_continuations=5,
        )
        runtime = RobotAgentRuntime(settings, "inspect the extinguisher")
        detector = GeometryDetector(adapter, object_x=0.0, object_y=0.0)
        tools = {
            item.name: item
            for item in RobotToolRegistry(
                runtime, adapter, bt_skill=object(), detector=detector
            ).build()
        }
        return runtime, tools

    def test_one_circle_tool_plans_once_and_captures_all_viewpoints(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            runtime, tools = self._build_circle_tool(root, adapter)
            self.assertNotIn("capture_next_inspection_viewpoint", tools)
            self.assertNotIn("initialize_object_inspection", tools)
            schema = tools["circle_object_for_inspection"].args_schema.model_json_schema()
            self.assertNotIn("view_angle_deg", schema["properties"])
            with patch(
                "robot_agent.tools.registry.generate_object_viewpoints",
                wraps=generate_object_viewpoints,
            ) as planner:
                result = tools["circle_object_for_inspection"].invoke(
                    {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
                )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(len(result["data"]["captures"]), 4)
            planner.assert_called_once()
            self.assertEqual(planner.call_args.kwargs["count"], 4)
            self.assertEqual(len(adapter.navigation_targets), 13)
            self.assertEqual(adapter.alignment_calls, 4)
            self.assertEqual(adapter.evaluation_calls, 8)
            self.assertTrue(
                all(
                    len(capture["scan_attempts"]) == 2
                    for capture in result["data"]["captures"]
                )
            )
            self.assertEqual(runtime.state.inspection_state["status"], "completed")

    def test_circle_tool_ignores_persisted_bbox_for_first_bearing(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            locations = root / "locations.yaml"
            locations.write_text(
                "inspection_start: [0, -2, 1.571]\n",
                encoding="utf-8",
            )
            settings = RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
                max_no_progress_continuations=5,
            )
            runtime = RobotAgentRuntime(settings, "inspect the extinguisher")
            runtime.state.robot_state.visible_objects = [
                Detection(
                    label="fire extinguisher",
                    confidence=0.99,
                    image_position=ImagePosition(
                        x_px=0.0,
                        y_px=240.0,
                        x_normalized=0.0,
                        y_normalized=0.5,
                        width_normalized=0.2,
                        height_normalized=0.4,
                    ),
                )
            ]
            detector = GeometryDetector(adapter, object_x=0.0, object_y=0.0)
            tools = {
                item.name: item
                for item in RobotToolRegistry(
                    runtime,
                    adapter,
                    bt_skill=object(),
                    detector=detector,
                ).build()
            }

            result = tools["circle_object_for_inspection"].invoke(
                {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(len(result["data"]["captures"]), 4)
            self.assertAlmostEqual(result["data"]["object_position"]["x"], 0.0)
            self.assertAlmostEqual(result["data"]["object_position"]["y"], 0.0)

    def test_circle_tool_tolerates_intermittent_initial_frame_misses(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            locations = root / "locations.yaml"
            locations.write_text("inspection_start: [0, -2, 1.571]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
                detection_interval_sec=0.1,
                max_no_progress_continuations=5,
            )
            runtime = RobotAgentRuntime(settings, "inspect the extinguisher")
            detector = InitialFramesMissDetector(
                adapter,
                object_x=0.0,
                object_y=0.0,
                misses=2,
            )
            tools = {
                item.name: item
                for item in RobotToolRegistry(
                    runtime,
                    adapter,
                    bt_skill=object(),
                    detector=detector,
                ).build()
            }

            result = tools["circle_object_for_inspection"].invoke(
                {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(len(result["data"]["captures"]), 4)

    def test_confirmed_search_hands_off_to_tracking_threshold(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            locations = root / "locations.yaml"
            locations.write_text("inspection_start: [0, -2, 1.571]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
                max_no_progress_continuations=5,
            )
            runtime = RobotAgentRuntime(settings, "inspect the extinguisher")
            confirmed = Detection(
                label="fire extinguisher",
                confidence=0.06,
                image_position=ImagePosition(
                    x_px=320.0,
                    y_px=240.0,
                    x_normalized=0.5,
                    y_normalized=0.5,
                    width_normalized=0.2,
                    height_normalized=0.4,
                ),
            )
            runtime.record_tool_result(
                "search_for_object",
                {"route": ["inspection_start"], "label": "fire extinguisher"},
                ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={"found": confirmed.to_dict()},
                ),
            )
            detector = LowConfidenceGeometryDetector(
                adapter,
                object_x=0.0,
                object_y=0.0,
            )
            tools = {
                item.name: item
                for item in RobotToolRegistry(
                    runtime,
                    adapter,
                    bt_skill=object(),
                    detector=detector,
                ).build()
            }

            result = tools["circle_object_for_inspection"].invoke(
                {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(len(result["data"]["captures"]), 4)
            self.assertTrue(
                all(capture["confidence"] == 0.02 for capture in result["data"]["captures"])
            )

    def test_blocked_viewpoint_uses_orbit_relative_fallback(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            adapter.block_evaluation_number = 5
            _, tools = self._build_circle_tool(root, adapter)

            with patch(
                "robot_agent.tools.registry.generate_baseline_candidates",
                wraps=generate_baseline_candidates,
            ) as candidate_generator:
                result = tools["circle_object_for_inspection"].invoke(
                    {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
                )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertTrue(result["data"]["captures"][0]["used_fallback"])
            self.assertTrue(
                {
                    "planned_pose",
                    "pose",
                    "planned_angle_deg",
                    "used_fallback",
                    "alignment_status",
                }.issubset(result["data"]["captures"][0])
            )
            self.assertLessEqual(adapter.evaluation_calls, 12)
            self.assertEqual(candidate_generator.call_count, 1)

    def test_invisible_viewpoint_fails_after_one_scan_without_moving_fallback(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            locations = root / "locations.yaml"
            locations.write_text("inspection_start: [0, -2, 1.571]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=locations,
                run_directory=root / "runs",
                trace=False,
                max_no_progress_continuations=5,
                inspection_min_radius_m=0.75,
            )
            runtime = RobotAgentRuntime(settings, "inspect the extinguisher")
            detector = FirstViewBlindDetector(adapter, object_x=0.0, object_y=0.0)
            tools = {
                item.name: item
                for item in RobotToolRegistry(
                    runtime,
                    adapter,
                    bt_skill=object(),
                    detector=detector,
                ).build()
            }

            with patch("robot_agent.tools.registry.time.sleep"):
                result = tools["circle_object_for_inspection"].invoke(
                    {
                        "label": "fire extinguisher",
                        "viewpoint_count": 4,
                        "radius_m": 1.0,
                    }
                )

            self.assertEqual(result["status"], ToolStatus.FAILED.value, result)
            self.assertEqual(len(result["data"]["captures"]), 1)
            self.assertIn("full in-place scan", result["error"])
            self.assertFalse(
                any(capture["used_fallback"] for capture in result["data"]["captures"])
            )
            scan_sequence = None
            for start in range(len(adapter.navigation_targets) - 8):
                sequence = adapter.navigation_targets[start : start + 9]
                same_position = all(
                    math.isclose(target.x, sequence[0].x)
                    and math.isclose(target.y, sequence[0].y)
                    for target in sequence
                )
                one_direction = all(
                    math.isclose(
                        (current.yaw - previous.yaw) % (2.0 * math.pi),
                        math.pi / 4.0,
                    )
                    for previous, current in zip(sequence, sequence[1:])
                )
                if same_position and one_direction:
                    scan_sequence = sequence
                    break
            self.assertIsNotNone(scan_sequence)

    def test_in_place_scan_turns_one_direction_until_detection(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            adapter.pose = Pose2D(1.0, 0.0, math.pi / 2.0)
            runtime, _ = self._build_circle_tool(root, adapter)
            detector = GeometryDetector(adapter, object_x=0.0, object_y=0.0)
            registry = RobotToolRegistry(
                runtime,
                adapter,
                bt_skill=object(),
                detector=detector,
            )

            detection, attempts, failure = registry._scan_for_object_in_place(
                detector,
                label="fire extinguisher",
                color=None,
                pose=adapter.pose,
                minimum_confidence=0.05,
            )

            self.assertIsNone(failure)
            self.assertIsNotNone(detection)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts[-1]["detected"])
            self.assertTrue(
                all(
                    math.isclose(target.x, 1.0)
                    and math.isclose(target.y, 0.0)
                    for target in adapter.navigation_targets
                )
            )
            self.assertAlmostEqual(
                (adapter.navigation_targets[1].yaw - adapter.navigation_targets[0].yaw)
                % (2.0 * math.pi),
                math.pi / 4.0,
            )

    def test_inspection_arrival_uses_counterclockwise_orbit_tangent(self):
        object_pose = Pose2D(0.0, 0.0, 0.0)
        east = RobotToolRegistry._inspection_arrival_pose(
            Pose2D(1.0, 0.0, math.pi),
            object_pose,
        )
        north = RobotToolRegistry._inspection_arrival_pose(
            Pose2D(0.0, 1.0, -math.pi / 2.0),
            object_pose,
        )

        self.assertAlmostEqual(east.yaw, math.pi / 2.0)
        self.assertAlmostEqual(abs(north.yaw), math.pi)
        self.assertEqual((east.x, east.y), (1.0, 0.0))
        self.assertEqual((north.x, north.y), (0.0, 1.0))

    def test_orbit_fallbacks_remain_distinct_for_distinct_viewpoints(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            runtime, _ = self._build_circle_tool(root, adapter)
            registry = RobotToolRegistry(
                runtime,
                adapter,
                bt_skill=object(),
                detector=GeometryDetector(adapter, 0.0, 0.0),
            )

            def evaluate(candidate):
                return CandidateEvaluation(
                    candidate=candidate,
                    feasible=candidate.name == "orbit_fallback_+30deg",
                    path_length_m=1.0,
                    obstacle_risk=0.0,
                )

            registry._evaluate_candidate = evaluate
            object_pose = Pose2D(0.0, 0.0, 0.0)
            resolved_east = registry._resolve_inspection_viewpoint(
                Pose2D(1.0, 0.0, math.pi),
                object_pose,
                radius_m=1.0,
                base_angle_deg=0.0,
            )
            resolved_north = registry._resolve_inspection_viewpoint(
                Pose2D(0.0, 1.0, -math.pi / 2),
                object_pose,
                radius_m=1.0,
                base_angle_deg=90.0,
            )

            self.assertAlmostEqual(
                resolved_east.candidate.pose.x,
                math.cos(math.radians(30)),
            )
            self.assertAlmostEqual(
                resolved_east.candidate.pose.y,
                math.sin(math.radians(30)),
            )
            self.assertAlmostEqual(
                resolved_north.candidate.pose.x,
                math.cos(math.radians(120)),
            )
            self.assertAlmostEqual(
                resolved_north.candidate.pose.y,
                math.sin(math.radians(120)),
            )
            self.assertGreater(
                math.hypot(
                    resolved_east.candidate.pose.x - resolved_north.candidate.pose.x,
                    resolved_east.candidate.pose.y - resolved_north.candidate.pose.y,
                ),
                1.0,
            )

    def test_all_orbit_fallbacks_blocked_fails_clearly(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            runtime, _ = self._build_circle_tool(root, adapter)
            registry = RobotToolRegistry(
                runtime,
                adapter,
                bt_skill=object(),
                detector=GeometryDetector(adapter, 0.0, 0.0),
            )

            def reject(candidate):
                return CandidateEvaluation(
                    candidate=candidate,
                    feasible=False,
                    path_length_m=math.inf,
                    obstacle_risk=1.0,
                    reason="blocked",
                )

            registry._evaluate_candidate = reject
            with self.assertRaisesRegex(
                NoFeasibleBaselineError,
                "orbit-angle fallback",
            ):
                registry._resolve_inspection_viewpoint(
                    Pose2D(1.0, 0.0, math.pi),
                    Pose2D(0.0, 0.0, 0.0),
                    radius_m=1.0,
                    base_angle_deg=0.0,
                )

    def test_fresh_alignment_timeout_captures_and_continues_orbit(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FirstAlignmentTimeoutAdapter()
            _, tools = self._build_circle_tool(root, adapter)

            result = tools["circle_object_for_inspection"].invoke(
                {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
            )

            self.assertEqual(result["status"], ToolStatus.SUCCESS.value, result)
            self.assertEqual(len(result["data"]["captures"]), 4)
            self.assertEqual(adapter.alignment_calls, 4)
            self.assertEqual(len(adapter.navigation_targets), 13)
            self.assertFalse(result["data"]["captures"][0]["aligned"])
            self.assertEqual(
                result["data"]["captures"][0]["alignment_status"],
                ToolStatus.TIMEOUT.value,
            )
            self.assertTrue(result["data"]["captures"][1]["aligned"])

    def test_stale_alignment_timeout_fails_without_moving_fallback(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FirstAlignmentTimeoutAdapter()
            adapter.timeout_detection_age_sec = 5.0
            _, tools = self._build_circle_tool(root, adapter)

            result = tools["circle_object_for_inspection"].invoke(
                {"label": "fire extinguisher", "viewpoint_count": 4, "radius_m": 1.0}
            )

            self.assertEqual(result["status"], ToolStatus.TIMEOUT.value, result)
            self.assertEqual(adapter.alignment_calls, 1)
            self.assertEqual(len(adapter.navigation_targets), 4)
            self.assertEqual(result["data"]["captures"], [])
            self.assertEqual(len(result["data"]["scan_attempts"]), 2)

    def test_angle_coverage_uses_tolerance(self):
        self.assertTrue(_angle_is_covered(90.0 + 1e-10, [90.0]))

    def test_registry_uses_direct_triangulation_without_compatibility_objects(self):
        source = inspect.getsource(RobotToolRegistry._circle_object_for_inspection)
        self.assertIn("triangulate_from_bearings", source)
        self.assertNotIn("BearingObservation", source)
        self.assertNotIn("triangulate_bearings", source)

    def test_inspection_failure_helper_preserves_result_shape(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = FeasibleAdapter()
            runtime, _ = self._build_circle_tool(root, adapter)
            registry = RobotToolRegistry(
                runtime,
                adapter,
                bt_skill=object(),
                detector=GeometryDetector(adapter, 0.0, 0.0),
            )

            result = registry._inspection_failure(
                {"failed_viewpoint": {"x": 1.0}},
                "blocked",
                retryable=False,
            )

            self.assertEqual(result.status, ToolStatus.FAILED)
            self.assertEqual(result.data["operation"], "circle_object_for_inspection")
            self.assertEqual(result.data["failed_viewpoint"], {"x": 1.0})
            self.assertEqual(result.error, "blocked")
            self.assertFalse(result.retryable)

    def test_behavior_tree_skill_has_no_active_view_dependencies(self):
        source = inspect.getsource(BehaviorTreeSkill)
        self.assertNotIn("triangulate", source)
        self.assertNotIn("baseline", source)
        self.assertNotIn("align_to_detection", source)


if __name__ == "__main__":
    unittest.main()
