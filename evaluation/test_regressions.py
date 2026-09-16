"""Regression tests for V2 input, state, and resource boundaries."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from gaze_tracking_v2.app import (
    _estimate_processed_frame_pose,
    assess_candidate_model,
    build_calibration_plan,
    build_calibration_points,
    build_evaluation_points,
    calibration_coverage_bin,
    calibration_progress,
    motion_coverage_complete,
    calibration_step_status,
    parse_args,
)
from gaze_tracking_v2.blink import BlinkDetector
from gaze_tracking_v2.evaluation_io import EvaluationLogger
from gaze_tracking_v2.pose_gate import HeadPose, HeadPoseGate

try:
    import numpy as np

    from gaze_tracking_v2.calibration import (
        CalibrationPoint,
        CumulativeCalibrationDataset,
    )
    from gaze_tracking_v2.correction import CumulativeCorrection
    from gaze_tracking_v2.drift import CalibrationDriftMonitor
    from gaze_tracking_v2.mapping import DualEyeMapper, HeadCompensatedMapper, RidgePolynomial2D
    from gaze_tracking_v2.persistence import (
        calibration_state_path,
        clear_calibration_state,
        load_calibration_state,
        save_calibration_state,
    )
except ModuleNotFoundError as exc:
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from .metrics import evaluate_records, load_csv
except ImportError:
    from metrics import evaluate_records, load_csv


class DependencyFreeRegressionTest(unittest.TestCase):
    def test_calibration_progress_tracks_all_completion_requirements(self):
        def progress(elapsed, samples, motion=False, **coverage):
            return calibration_progress(elapsed_ms=elapsed, sample_count=samples,
                min_samples=15, motion=motion, coverage=coverage)
        self.assertEqual(progress(1500, 180), 1.0)
        self.assertEqual(progress(1500, 0), 0.0)
        self.assertLess(progress(600, 180), 1.0)
        self.assertAlmostEqual(progress(6000, 180, True,
            negative=100, neutral=100), 2 / 3)
        self.assertEqual(progress(0, 15, True,
            negative=5, neutral=5, positive=5), 1.0)
        self.assertEqual(progress(6000, 180, True,
            negative=5, neutral=5, positive=5), 1.0)

    def test_motion_requires_every_direction_even_with_excess_samples(self):
        self.assertFalse(motion_coverage_complete({"negative": 180, "neutral": 180}))
        self.assertFalse(motion_coverage_complete({"negative": 5, "neutral": 5, "positive": 4}))
        self.assertTrue(motion_coverage_complete({"negative": 5, "neutral": 5, "positive": 5}))

    def test_translation_bins_require_the_mapper_training_span(self):
        pose = HeadPose(0, 0, 0)
        plan = build_calibration_plan(640, 480)
        for scale in (0.30, 0.4203, 0.60):
            threshold = max(0.02, 0.065 * scale)
            self.assertGreater(2 * threshold / scale, 0.12)
            for step, axis in ((plan[3], 0), (plan[4], 1)):
                for sign, expected in ((-1, "negative"), (1, "positive")):
                    center = [0.5, 0.5]
                    center[axis] += sign * (threshold + 0.001)
                    self.assertEqual(calibration_coverage_bin(step, pose,
                        face_scale=scale, reference_face_scale=scale,
                        face_center=tuple(center), reference_face_center=(0.5, 0.5)), expected)
        # Previously counted as positive although opposite samples at this
        # displacement would not satisfy the model's 0.12 normalized span.
        self.assertEqual(calibration_coverage_bin(plan[3], pose,
            face_scale=0.4203, reference_face_scale=0.4203,
            face_center=(0.521, 0.5), reference_face_center=(0.5, 0.5)), "neutral")

    def test_motion_sampling_rejects_rotation_left_over_from_previous_step(self):
        plan = build_calibration_plan(640, 480)
        args = dict(face_scale=0.4, reference_face_scale=0.4,
                    face_center=(0.5, 0.5), reference_face_center=(0.5, 0.5))
        self.assertFalse(calibration_step_status(plan[1], HeadPose(7, 8, 0), **args)[0])
        self.assertTrue(calibration_step_status(plan[1], HeadPose(7, 0, 0), **args)[0])
        for step in plan[3:6]:
            self.assertFalse(calibration_step_status(step, HeadPose(0, 8, 0), **args)[0])
            self.assertTrue(calibration_step_status(step, HeadPose(0, 0, 0), **args)[0])

    def test_app_helpers_are_importable_without_camera_dependencies(self):
        self.assertEqual(len(build_calibration_points(640, 480)), 25)
        self.assertEqual(parse_args(["--no-console"]).no_console, True)

    def test_v26_plan_starts_with_fixed_target_pose_anchors(self):
        plan = build_calibration_plan(640, 480)
        self.assertEqual(len(plan), 31)
        self.assertEqual({step.target for step in plan[:6]}, {(320, 240)})
        self.assertEqual(
            [step.pose_condition for step in plan[:6]],
            [
                "neutral",
                "yaw_sweep",
                "pitch_sweep",
                "x_sweep",
                "y_sweep",
                "scale_sweep",
            ],
        )

    def test_v26_pose_conditions_gate_the_controlled_samples(self):
        neutral = type("Pose", (), {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})()
        yawed = type("Pose", (), {"yaw": 7.0, "pitch": 0.0, "roll": 0.0})()
        plan = build_calibration_plan(640, 480)
        common = {
            "face_scale": 0.30,
            "reference_face_scale": 0.30,
            "face_center": (0.50, 0.50),
            "reference_face_center": (0.50, 0.50),
        }
        self.assertTrue(calibration_step_status(plan[0], neutral, **common)[0])
        self.assertTrue(calibration_step_status(plan[1], yawed, **common)[0])
        self.assertFalse(calibration_step_status(plan[6], yawed, **common)[0])
        self.assertEqual(
            calibration_coverage_bin(plan[1], yawed, **common),
            "positive",
        )


    def test_missing_blink_frame_breaks_a_partial_close_run(self):
        detector = BlinkDetector(close_frames=2)
        detector.update(0.25)
        detector.update(0.10)
        detector.update(None)
        self.assertFalse(detector.update(0.10).started)
        self.assertTrue(detector.update(0.10).started)

    def test_processed_frame_pose_does_not_reflect_landmarks_twice(self):
        class RecordingEstimator:
            def __init__(self):
                self.mirrored = None

            def estimate(self, landmarks, width, height, *, mirrored=False):
                self.mirrored = mirrored
                return (landmarks, width, height)

        estimator = RecordingEstimator()
        landmarks = [object()]
        result = _estimate_processed_frame_pose(estimator, landmarks, 640, 480)
        self.assertFalse(estimator.mirrored)
        self.assertEqual(result, (landmarks, 640, 480))

    def test_logger_uses_unique_paths_and_rejects_nonfinite_points(self):
        with tempfile.TemporaryDirectory() as directory:
            first = EvaluationLogger(directory, condition="normal", camera_width=640, camera_height=480)
            second = EvaluationLogger(directory, condition="normal", camera_width=640, camera_height=480)
            self.assertNotEqual(first.path, second.path)
            with self.assertRaises(ValueError):
                first.write(
                    trial_id="t1",
                    timestamp_ms=0,
                    target=(10, 10),
                    raw_gaze=(math.nan, 10),
                    filtered_gaze=None,
                    face_detected=False,
                    feature_valid=False,
                    blink=False,
                    blink_started=False,
                    blink_ended=False,
                    blink_openness=None,
                    blink_threshold=None,
                    pose_valid=False,
                    pose_gate_open=False,
                    pose_yaw=None,
                    pose_pitch=None,
                    pose_roll=None,
                    pose_reference_ready=False,
                    pose_candidate_allowed=False,
                    drift_score=None,
                    drift_detected=False,
                )
            first.write(
                trial_id="t2",
                timestamp_ms=1,
                target=(10, 10),
                raw_gaze=(10, 10),
                filtered_gaze=(10, 10),
                face_detected=True,
                feature_valid=True,
                blink=False,
                blink_started=False,
                blink_ended=False,
                blink_openness=0.25,
                blink_threshold=0.10,
                pose_valid=True,
                pose_gate_open=True,
                pose_yaw=0.0,
                pose_pitch=0.0,
                pose_roll=0.0,
                pose_reference_ready=True,
                pose_candidate_allowed=True,
                drift_score=0.0,
                drift_detected=False,
                left_mapper_features=range(10),
                right_mapper_features=range(10, 20),
            )
            first.close()
            second.close()
            with first.path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["model_version"], "v2.6")
            self.assertEqual(row["left_mapper_feature_9"], "9.0")
            self.assertEqual(row["right_mapper_feature_9"], "19.0")

    def test_metrics_rejects_nonfinite_csv_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaze.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["target_x", "target_y", "gaze_x", "gaze_y"],
                )
                writer.writeheader()
                writer.writerow({"target_x": "nan", "target_y": 10, "gaze_x": 10, "gaze_y": 10})
            with self.assertRaises(ValueError):
                load_csv(path)

    def test_metrics_rejects_invalid_configuration(self):
        record = {
            "trial_id": "t1",
            "timestamp_ms": 0,
            "target_x": 10,
            "target_y": 10,
            "gaze_x": 10,
            "gaze_y": 10,
            "valid": True,
            "condition": "normal",
        }
        with self.assertRaises(ValueError):
            evaluate_records([record], screen_width=0)

    def test_force_finish_does_not_bypass_calibration_sample_limit(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"V2 dependencies unavailable: {IMPORT_ERROR}")
        from gaze_tracking_v2.calibration import CalibrationCollector

        collector = CalibrationCollector([(x, y) for y in (0, 1, 2) for x in (0, 1, 2)], min_samples=3)
        collector.add_sample([0.1, 0.2], [0.1, 0.2])
        self.assertFalse(collector.finish_current(force=True))
        self.assertEqual(collector.current_index, 0)

    def test_motion_step_can_be_skipped_without_relaxing_normal_finish(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"V2 dependencies unavailable: {IMPORT_ERROR}")
        from gaze_tracking_v2.calibration import CalibrationCollector

        collector = CalibrationCollector(
            [(x, y) for y in (0, 1, 2) for x in (0, 1, 2)],
            min_samples=3,
        )
        self.assertTrue(collector.skip_current())
        self.assertTrue(collector.points[0].skipped)
        self.assertEqual(collector.current_index, 1)
        self.assertFalse(collector.finish_current(force=True))

    def test_candidate_acceptance_requires_global_and_regional_improvement(self):
        common = dict(screen_diagonal=800, expected_regions=[0, 1],
                      active_regions={0: [100.] * 20, 1: [100.] * 20})
        self.assertTrue(assess_candidate_model(
            [80.] * 40, [100.] * 40,
            candidate_regions={0: [80.] * 20, 1: [80.] * 20}, **common)[0])
        self.assertFalse(assess_candidate_model(
            [10.] * 20 + [120.] * 20, [100.] * 40,
            candidate_regions={0: [10.] * 20, 1: [120.] * 20}, **common)[0])

    def test_acceptance_rejects_missing_sparse_and_unpaired_regions(self):
        for active in (None, [20.] * 20):
            self.assertFalse(assess_candidate_model(
                [10.] * 20, active, screen_diagonal=800,
                expected_regions=range(12), candidate_regions={0: [10.] * 20},
                active_regions={0: [20.] * 20}, baseline_regions={0: 100.})[0])
        for candidate_regions, active_regions in (
            ({0: [10.] * 39, 1: [10.]}, {0: [20.] * 39, 1: [20.]}),
            ({0: [10.] * 20, 1: [10.] * 20}, {0: [20.] * 21, 1: [20.] * 19}),
            ({0: [10.] * 20, 1: [float('nan')] * 20},
             {0: [20.] * 20, 1: [20.] * 20}),
        ):
            self.assertFalse(assess_candidate_model(
                [10.] * 40, [20.] * 40, screen_diagonal=800,
                expected_regions=[0, 1], candidate_regions=candidate_regions,
                active_regions=active_regions)[0])

    def test_initial_acceptance_rejects_center_and_bad_region(self):
        points = build_evaluation_points(640, 480)
        baseline = {i: math.hypot(x - 320, y - 240)
                    for i, (x, y) in enumerate(points)}
        def assess(regions):
            return assess_candidate_model(
                [e for values in regions.values() for e in values], None,
                screen_diagonal=800, expected_regions=range(len(points)),
                candidate_regions=regions, baseline_regions=baseline)[0]
        self.assertFalse(assess({i: [e] * 20 for i, e in baseline.items()}))
        self.assertTrue(assess({i: [20.] * 20 for i in baseline}))
        bad_region = {i: [20.] * 20 for i in baseline}
        bad_region[5] = [150.] * 20
        self.assertFalse(assess(bad_region))
        # A small tail can pass the average and regional means but fail P95.
        self.assertFalse(assess({i: [20.] * 18 + [400.] * 2 for i in baseline}))


@unittest.skipIf(IMPORT_ERROR is not None, f"V2 dependencies unavailable: {IMPORT_ERROR}")
class NumpyRegressionTest(unittest.TestCase):
    def test_calibration_profiles_are_isolated_and_reset_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(camera_id=0, screen_size=(640, 480), mirror=True,
                          condition="normal")
            paths = [calibration_state_path(directory, **config)]
            for change in (dict(camera_id=1), dict(screen_size=(1280, 720)),
                           dict(mirror=False), dict(condition="glasses"),
                           dict(condition="../glasses"), dict(condition="..\\glasses")):
                paths.append(calibration_state_path(directory, **(config | change)))
            self.assertEqual(len(set(paths)), len(paths))
            self.assertEqual(paths[0], calibration_state_path(directory, **config))
            mapper = DualEyeMapper(screen_size=(640, 480)).fit(
                np.array([[x, y] for x in range(3) for y in range(3)]),
                np.array([[x, y] for x in range(3) for y in range(3)]),
                np.array([[x * 100, y * 100] for x in range(3) for y in range(3)]))
            correction = CumulativeCorrection(screen_size=(640, 480))
            for path, condition in ((paths[0], "normal"), (paths[4], "glasses")):
                save_calibration_state(path, mapper=mapper, correction=correction,
                    drift_monitor=None, condition=condition, camera_id=0, mirror=True)
            clear_calibration_state(paths[4])
            self.assertFalse(paths[4].exists())
            self.assertIsNotNone(load_calibration_state(
                paths[0], expected_screen_size=(640, 480), expected_condition="normal",
                expected_camera_id=0, expected_mirror=True))

    def test_display_canvas_preserves_camera_frame_and_overlay_coordinates(self):
        from gaze_tracking_v2.rendering import DisplayCanvas
        from gaze_tracking_v2.app import _draw_calibration_feedback, _draw_gaze, _draw_target
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas = DisplayCanvas(frame, (1280, 960))
        self.assertEqual(canvas.image.shape, (960, 1280, 3))
        self.assertEqual(canvas.shape, frame.shape)
        canvas.drawing.circle(canvas, (100, 120), 4, (0, 255, 0), -1)
        self.assertEqual(tuple(canvas.image[240, 200]), (0, 255, 0))
        self.assertFalse(frame.any())
        _draw_calibration_feedback(canvas, (320, 240), condition="screen",
            status="Collecting", blocked=False, progress=0.5, step_index=6, total_steps=31)
        _draw_gaze(canvas, (100, 100), (110, 110), (105, 105))
        _draw_target(canvas, (50, 50), "Evaluation")
        wide = DisplayCanvas(frame, (1600, 900))
        self.assertEqual(wide.image.shape, (900, 1200, 3))

    def test_head_pose_estimator_rebases_after_persistent_jump(self):
        from gaze_tracking_v2.head_pose import HeadPoseEstimator

        estimator = HeadPoseEstimator(rebase_frames=3)
        self.assertIsNotNone(
            estimator._smooth(HeadPose(0.0, 0.0, 0.0, 0.0, 0.0, 500.0))
        )
        jumped = HeadPose(60.0, 0.0, 0.0, 0.0, 0.0, 500.0)
        self.assertIsNone(estimator._smooth(jumped))
        self.assertIsNone(estimator._smooth(jumped))
        self.assertIsNotNone(estimator._smooth(jumped))

    def test_cumulative_history_keeps_head_anchors_across_sessions(self):
        grid = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
        )
        targets = np.column_stack((grid[:, 0] * 10 + 50, grid[:, 1] * 10 + 50))
        relative_heads = np.vstack((np.zeros((1, 5)), np.eye(5), -np.eye(5)))
        heads = np.column_stack((relative_heads, np.zeros((len(relative_heads), 3))))
        history = CumulativeCalibrationDataset()
        for _ in range(2):
            history.add(
                grid,
                grid,
                targets,
                np.column_stack((np.full(len(grid), 0.5), np.full(len(grid), 0.5), np.full(len(grid), 0.3))),
                anchor_left=np.zeros((len(heads), 2)),
                anchor_right=np.zeros((len(heads), 2)),
                anchor_heads=heads,
                anchor_targets=np.full((len(heads), 2), 50.0),
            )
        restored = CumulativeCalibrationDataset.from_dict(history.to_dict())
        self.assertEqual(restored.session_count, 2)
        self.assertEqual(restored.sample_count, 18)
        self.assertEqual(restored.anchor_count, 22)
        self.assertEqual(restored.anchor_arrays()[2].shape[1], 8)

    def test_v26_two_stage_mapper_separates_eye_and_head_learning(self):
        grid = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)],
            dtype=float,
        )
        grid_targets = np.column_stack(
            [50.0 + 20.0 * grid[:, 0], 50.0 + 20.0 * grid[:, 1]]
        )
        relative_heads = np.vstack((np.zeros((1, 5)), np.eye(5), -np.eye(5)))
        heads = np.column_stack((relative_heads, np.zeros((len(relative_heads), 3))))
        residual = np.column_stack(
            [10.0 * heads[:, 0] + 5.0 * heads[:, 2],
             8.0 * heads[:, 1] + 4.0 * heads[:, 3] + 3.0 * heads[:, 4]]
        )
        anchor_eyes = -residual / 20.0
        anchor_targets = np.full((len(heads), 2), 50.0)
        mapper = HeadCompensatedMapper(
            ridge=1e-9, screen_size=(100, 100)
        ).fit(
            grid, grid, grid_targets,
            anchor_eyes, anchor_eyes, heads, anchor_targets,
        )
        query_head = np.array([0.5, -0.5, 0.25, 0.5, -0.25, 0.0, 0.0, 0.0])
        query_residual = np.array([6.25, -2.75])
        query_eye = -query_residual / 20.0
        self.assertEqual(mapper.input_dimension, 10)
        np.testing.assert_allclose(
            mapper.predict(query_eye, query_eye, query_head).gaze,
            (50.0, 50.0),
            atol=1e-4,
        )

    def test_v26_rejects_missing_head_pose_coverage(self):
        grid = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)],
            dtype=float,
        )
        targets = np.column_stack(
            [50.0 + 20.0 * grid[:, 0], 50.0 + 20.0 * grid[:, 1]]
        )
        anchor_eyes = np.zeros((11, 2), dtype=float)
        anchor_heads = np.zeros((11, 8), dtype=float)
        anchor_targets = np.full((11, 2), 50.0)

        with self.assertRaisesRegex(ValueError, "coverage"):
            HeadCompensatedMapper(ridge=1e-6, screen_size=(100, 100)).fit(
                grid, grid, targets,
                anchor_eyes, anchor_eyes, anchor_heads, anchor_targets,
            )

    def test_v26_two_stage_state_round_trip(self):
        grid = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)],
            dtype=float,
        )
        targets = np.column_stack(
            [50.0 + 20.0 * grid[:, 0], 50.0 + 20.0 * grid[:, 1]]
        )
        relative_heads = np.vstack((np.zeros((1, 5)), np.eye(5)))
        heads = np.column_stack((relative_heads, np.zeros((len(relative_heads), 3))))
        anchor_eyes = np.zeros((6, 2), dtype=float)
        anchor_targets = np.full((6, 2), 50.0)
        mapper = HeadCompensatedMapper(
            ridge=1e-6, screen_size=(100, 100)
        ).fit(
            grid, grid, targets,
            anchor_eyes, anchor_eyes, heads, anchor_targets,
        )
        history = CumulativeCalibrationDataset()
        history.add(
            grid,
            grid,
            targets,
            anchor_left=anchor_eyes,
            anchor_right=anchor_eyes,
            anchor_heads=heads,
            anchor_targets=anchor_targets,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v26.json"
            save_calibration_state(
                path,
                mapper=mapper,
                correction=CumulativeCorrection(screen_size=(100, 100)),
                drift_monitor=None,
                condition="test",
                calibration_data=history,
                camera_id=1,
                mirror=True,
            )
            restored = load_calibration_state(
                path,
                expected_screen_size=(100, 100),
                expected_condition="test",
                expected_mapper_input_dimension=10,
                expected_camera_id=1,
                expected_mirror=True,
            )
            with self.assertRaisesRegex(ValueError, "another camera"):
                load_calibration_state(
                    path,
                    expected_screen_size=(100, 100),
                    expected_condition="test",
                    expected_camera_id=2,
                    expected_mirror=True,
                )
        self.assertIsInstance(restored.mapper, HeadCompensatedMapper)
        self.assertEqual(restored.camera_id, 1)
        self.assertTrue(restored.mirror)
        self.assertEqual(restored.calibration_data.anchor_count, len(heads))
        np.testing.assert_allclose(
            restored.mapper.predict([0.2, -0.1], [0.2, -0.1], np.zeros(8)).gaze,
            mapper.predict([0.2, -0.1], [0.2, -0.1], np.zeros(8)).gaze,
            atol=1e-6,
        )

    def test_invalid_context_does_not_partially_append_a_sample(self):
        point = CalibrationPoint((0, 0))
        with self.assertRaises(ValueError):
            point.add([0.1, 0.2], [0.1, 0.2], [0.5, math.nan, 0.3])
        self.assertEqual(point.sample_count, 0)
        self.assertEqual(len(point.context_samples), 0)

    def test_missing_drift_frame_breaks_persistence(self):
        monitor = CalibrationDriftMonitor(
            np.array([[0.5, 0.5, 0.3], [0.51, 0.5, 0.3], [0.5, 0.49, 0.31]]),
            persistence_frames=3,
        )
        outlier = [0.7, 0.5, 0.3]
        monitor.update(outlier)
        monitor.update(outlier)
        monitor.update(None)
        self.assertFalse(monitor.update(outlier).warning)
        self.assertFalse(monitor.update(outlier).warning)
        self.assertTrue(monitor.update(outlier).warning)

    def test_nonfinite_mapping_parameters_and_predictions_fail(self):
        with self.assertRaises(ValueError):
            RidgePolynomial2D(ridge=math.nan)
        model = RidgePolynomial2D().fit(
            np.array([[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]),
            np.zeros((9, 2)),
        )
        with self.assertRaises(ValueError):
            model.predict([math.nan, 0.0])

    def test_nonfinite_pose_is_treated_as_missing(self):
        gate = HeadPoseGate()
        state = gate.update(HeadPose(math.nan, 0.0, 0.0))
        self.assertIsNone(state.pose)
        self.assertFalse(state.reference_ready)

    def test_pose_reference_preserves_relative_translation(self):
        gate = HeadPoseGate(reference_frames=2)
        gate.update(HeadPose(1.0, 2.0, 3.0, tx=10.0, ty=20.0, tz=300.0))
        state = gate.update(
            HeadPose(1.5, 2.5, 3.5, tx=14.0, ty=25.0, tz=308.0)
        )
        self.assertTrue(state.reference_ready)
        self.assertIsNotNone(state.relative_pose)
        assert state.relative_pose is not None
        self.assertAlmostEqual(state.relative_pose.tx, 2.0)
        self.assertAlmostEqual(state.relative_pose.ty, 2.5)
        self.assertAlmostEqual(state.relative_pose.tz, 4.0)

    def test_calibration_state_round_trips_mapper_and_correction(self):
        points = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
        )
        targets = np.column_stack(
            [50.0 + 20.0 * points[:, 0], 50.0 + 20.0 * points[:, 1]]
        )
        mapper = DualEyeMapper(ridge=1e-9, screen_size=(100, 100)).fit(
            points, points, targets
        )
        correction = CumulativeCorrection(screen_size=(100, 100), ridge=1e-9)
        correction.add(
            targets - np.array([4.0, -6.0]),
            targets,
        )
        mapper.set_output_correction(correction)
        calibration_data = CumulativeCalibrationDataset()
        calibration_data.add(
            points,
            points,
            targets,
            np.column_stack(
                [
                    np.full(len(points), 0.50),
                    np.full(len(points), 0.50),
                    np.full(len(points), 0.30),
                ]
            ),
        )
        drift = CalibrationDriftMonitor(
            np.array(
                [
                    [0.50, 0.50, 0.30],
                    [0.51, 0.50, 0.30],
                    [0.50, 0.49, 0.31],
                ]
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration_state(
                path,
                mapper=mapper,
                correction=correction,
                drift_monitor=drift,
                condition="glasses",
                calibration_data=calibration_data,
            )
            restored = load_calibration_state(
                path,
                expected_screen_size=(100, 100),
                expected_condition="glasses",
            )

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.correction.sample_count, 9)
        self.assertIsNotNone(restored.calibration_data)
        assert restored.calibration_data is not None
        self.assertEqual(restored.calibration_data.sample_count, 9)
        np.testing.assert_allclose(restored.calibration_data.arrays()[2], targets)
        self.assertIsNotNone(restored.drift_monitor)
        assert restored.drift_monitor is not None
        np.testing.assert_allclose(restored.drift_monitor.reference, drift.reference)
        np.testing.assert_allclose(
            restored.mapper.predict([0.25, -0.25], [0.25, -0.25]).gaze,
            mapper.predict([0.25, -0.25], [0.25, -0.25]).gaze,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
