"""Offline tests for the dependency-light parts of the V2 pipeline."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


try:
    import numpy as np

    from gaze_tracking_v2.calibration import (
        CalibrationCollector,
        CumulativeCalibrationDataset,
    )
    from gaze_tracking_v2.correction import CumulativeCorrection
    from gaze_tracking_v2.drift import CalibrationDriftMonitor
    from gaze_tracking_v2.features import extract_face_features
    from gaze_tracking_v2.filters import ExponentialSmoother
    from gaze_tracking_v2.mapping import DualEyeMapper, gaze_features, polynomial_features
except ModuleNotFoundError as exc:  # The repository's minimal test interpreter may lack NumPy.
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"V2 dependencies unavailable: {IMPORT_ERROR}")
class V2ComponentsTest(unittest.TestCase):
    def test_polynomial_features_shape_and_terms(self):
        values = polynomial_features(np.array([[2.0, 3.0]]))
        self.assertEqual(values.shape, (1, 6))
        np.testing.assert_allclose(values[0], [1.0, 2.0, 3.0, 4.0, 6.0, 9.0])

    def test_dual_eye_mapper_fits_quadratic_relationship(self):
        points = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, -0.5],
                [-0.5, 0.5],
                [0.25, 0.75],
            ]
        )
        targets = np.column_stack(
            [
                100.0 + 200.0 * points[:, 0] + 50.0 * points[:, 1] + 20.0 * points[:, 0] * points[:, 1],
                240.0 + 80.0 * points[:, 0] - 100.0 * points[:, 1] + 30.0 * points[:, 1] ** 2,
            ]
        )
        mapper = DualEyeMapper(ridge=1e-9).fit(points, points, targets)
        prediction = mapper.predict([0.25, -0.25], [0.25, -0.25])
        expected = np.array(
            [
                100.0 + 200.0 * 0.25 + 50.0 * -0.25 + 20.0 * 0.25 * -0.25,
                240.0 + 80.0 * 0.25 - 100.0 * -0.25 + 30.0 * (-0.25) ** 2,
            ]
        )
        np.testing.assert_allclose(prediction.gaze, expected, atol=1e-3)
        self.assertLess(mapper.left_model.rmse, 1e-3)

    def test_mapper_can_fit_normalized_targets_and_return_pixels(self):
        points = np.array(
            [
                [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
            ]
        )
        targets = np.column_stack(
            [320.0 + 160.0 * points[:, 0], 240.0 + 120.0 * points[:, 1]]
        )
        mapper = DualEyeMapper(
            ridge=1e-9, screen_size=(640, 480)
        ).fit(points, points, targets)
        prediction = mapper.predict([0.25, -0.5], [0.25, -0.5])
        np.testing.assert_allclose(prediction.gaze, [360.0, 180.0], atol=1e-3)

    def test_mapper_can_generate_leave_one_out_predictions(self):
        points = np.array(
            [[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
        )
        targets = np.column_stack(
            [100.0 + 20.0 * points[:, 0], 200.0 + 30.0 * points[:, 1]]
        )
        mapper = DualEyeMapper(ridge=1e-9).fit(points, points, targets)
        predictions = mapper.leave_one_out_predictions(points, points, targets)
        self.assertEqual(predictions.shape, (9, 2))
        np.testing.assert_allclose(predictions, targets, atol=1e-3)

    def test_five_value_features_keep_context_linear(self):
        values = gaze_features(np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]))
        self.assertEqual(values.shape, (1, 9))
        np.testing.assert_allclose(
            values[0], [1.0, 0.2, 0.3, 0.04, 0.06, 0.09, 0.4, 0.5, 0.6]
        )

    def test_extended_face_context_stays_linear(self):
        values = gaze_features(np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]))
        self.assertEqual(values.shape, (1, 11))
        np.testing.assert_allclose(
            values[0],
            [1.0, 0.2, 0.3, 0.04, 0.06, 0.09, 0.4, 0.5, 0.6, 0.7, 0.8],
        )

    def test_full_face_landmarks_produce_compact_context(self):
        landmarks = [
            SimpleNamespace(
                x=0.1 + (index % 20) * 0.04,
                y=0.1 + (index // 20) * 0.035,
                z=0.0,
            )
            for index in range(478)
        ]
        for index in (33, 133, 362, 263):
            landmarks[index].y = 0.4
        landmarks[33].x, landmarks[133].x = 0.30, 0.50
        landmarks[362].x, landmarks[263].x = 0.55, 0.75
        for index in (159, 160, 386, 387):
            landmarks[index].y = 0.39
        for index in (145, 144, 374, 373):
            landmarks[index].y = 0.43
        for index in (468, 469, 470, 471, 472):
            landmarks[index].x, landmarks[index].y = 0.40, 0.41
        for index in (473, 474, 475, 476, 477):
            landmarks[index].x, landmarks[index].y = 0.65, 0.41

        features = extract_face_features(landmarks, 640, 480)
        self.assertTrue(features.valid)
        self.assertEqual(features.context_vector.shape, (7,))
        self.assertTrue(np.all(np.isfinite(features.context_vector)))
        self.assertEqual(features.left_mapper_vector.shape, (2,))
        self.assertEqual(features.right_mapper_vector.shape, (2,))
        left_mapper, right_mapper = features.head_aware_mapper_vectors(
            SimpleNamespace(
                yaw=2.0,
                pitch=-1.0,
                roll=0.5,
                tz=12.0,
            )
        )
        self.assertEqual(left_mapper.shape, (9,))
        self.assertEqual(right_mapper.shape, (9,))
        pose = SimpleNamespace(yaw=2.0, pitch=-1.0)
        head_vector = features.head_compensation_vector(
            pose,
            reference_pose=pose,
            reference_face_center=features.face_center,
            reference_face_scale=features.face_scale,
        )
        self.assertEqual(head_vector.shape, (8,))
        np.testing.assert_allclose(head_vector[:5], np.zeros(5), atol=1e-12)
        self.assertTrue(np.all(np.isfinite(head_vector[5:])))

    def test_calibration_uses_median_and_requires_multiple_frames(self):
        targets = [(x, y) for y in (0, 100, 200) for x in (0, 100, 200)]
        collector = CalibrationCollector(targets, min_samples=3)
        for index, _target in enumerate(targets):
            for offset in (-0.01, 0.0, 0.01, 8.0):
                collector.add_sample(
                    [index * 0.1 + offset, index * 0.05],
                    [index * 0.1 + offset, index * 0.05],
                    [0.5, 0.5, 0.3, 0.8, 0.9, 0.0, 0.0],
                )
            self.assertTrue(collector.finish_current())
        left, right, fitted_targets, spreads = collector.dataset()
        self.assertEqual(left.shape, (9, 2))
        self.assertEqual(right.shape, (9, 2))
        np.testing.assert_allclose(left[0], [0.0, 0.0], atol=0.01)
        np.testing.assert_allclose(fitted_targets, np.asarray(targets, dtype=float))
        self.assertTrue(np.all(spreads >= 0.0))
        self.assertEqual(collector.context_dataset().shape, (9, 7))

    def test_cumulative_correction_accumulates_affine_bias(self):
        correction = CumulativeCorrection(screen_size=(100, 100), ridge=1e-9)
        predicted = np.array(
            [[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]]
        )
        targets = predicted + np.array([5.0, -8.0])
        correction.add(predicted, targets)
        self.assertEqual(correction.sample_count, 4)
        self.assertTrue(correction.ready)
        np.testing.assert_allclose(correction.apply((50.0, 50.0)), [55.0, 42.0], atol=1e-5)

    def test_cumulative_correction_applies_smooth_vertical_residual(self):
        correction = CumulativeCorrection(screen_size=(100, 100), ridge=1e-9)
        predicted = np.array(
            [[20.0, 10.0], [50.0, 30.0], [80.0, 50.0], [20.0, 70.0], [50.0, 90.0], [80.0, 95.0]]
        )
        targets = np.array(
            [[20.0, 15.0], [50.0, 38.0], [80.0, 55.0], [20.0, 78.0], [50.0, 98.0], [80.0, 100.0]]
        )
        correction.add(predicted, targets)
        for source, target in zip(predicted, targets):
            self.assertAlmostEqual(correction.apply(source)[1], target[1], places=5)

    def test_cumulative_correction_replace_discards_old_output_coordinates(self):
        correction = CumulativeCorrection(screen_size=(100, 100), ridge=1e-9)
        first_predicted = np.array(
            [[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]]
        )
        correction.add(first_predicted, first_predicted + [20.0, -20.0])
        second_predicted = np.array(
            [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]]
        )
        correction.replace(second_predicted, second_predicted + [3.0, -4.0])
        self.assertEqual(correction.sample_count, 4)
        np.testing.assert_allclose(correction.apply((50.0, 50.0)), [53.0, 46.0], atol=1e-5)

    def test_cumulative_calibration_dataset_round_trips_features_and_context(self):
        dataset = CumulativeCalibrationDataset()
        left = np.arange(18.0).reshape(3, 6) / 10.0
        right = left + 0.1
        targets = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        contexts = np.array(
            [[0.5, 0.5, 0.3], [0.51, 0.5, 0.3], [0.5, 0.49, 0.31]]
        )
        dataset.add(left, right, targets, contexts)
        restored = CumulativeCalibrationDataset.from_dict(dataset.to_dict())
        self.assertEqual(restored.sample_count, 3)
        self.assertEqual(restored.feature_dimension, 6)
        np.testing.assert_allclose(restored.arrays()[0], left)
        np.testing.assert_allclose(restored.context_dataset(), contexts)

    def test_drift_monitor_requires_persistent_face_shift(self):
        reference = np.array(
            [[0.50, 0.50, 0.30], [0.51, 0.50, 0.30], [0.50, 0.49, 0.31]]
        )
        monitor = CalibrationDriftMonitor(reference, persistence_frames=3)
        self.assertFalse(monitor.update([0.50, 0.50, 0.30]).warning)
        self.assertFalse(monitor.update([0.70, 0.50, 0.30]).warning)
        self.assertFalse(monitor.update([0.70, 0.50, 0.30]).warning)
        self.assertTrue(monitor.update([0.70, 0.50, 0.30]).warning)

    def test_exponential_smoother_is_explicit_about_alpha(self):
        smoother = ExponentialSmoother(alpha=0.5)
        self.assertEqual(smoother.update((0, 0)), (0.0, 0.0))
        self.assertEqual(smoother.update((10, 20)), (5.0, 10.0))
        smoother.reset()
        self.assertIsNone(smoother.value)


if __name__ == "__main__":
    unittest.main()
