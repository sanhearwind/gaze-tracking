"""Quality gating must reject contamination and recover without locking a point."""
import unittest
from types import SimpleNamespace
import numpy as np
from gaze_tracking_v2.quality import CalibrationQuality
from gaze_tracking_v2.calibration import CalibrationPoint


def features(value=0.0, width=0.08):
    def eye():
        return SimpleNamespace(vector=np.array([value, 0.0]), eye_width=width,
                               eye_height=width * 0.3, openness=0.3, valid=True)
    return SimpleNamespace(left=eye(), right=eye())


class CalibrationQualityTest(unittest.TestCase):
    def test_isolated_jump_rejected_and_recovers(self):
        gate = CalibrationQuality()
        for _ in range(3):
            self.assertEqual(gate.frame_reason(features()), "accepted")
        self.assertEqual(gate.frame_reason(features(0.5)), "geometry_jump")
        self.assertEqual(gate.frame_reason(features()), "accepted")
        self.assertEqual(gate.frame_reason(features(width=0.2)), "geometry_jump")

    def test_smooth_motion_allowed_and_persistent_change_recovers(self):
        gate = CalibrationQuality()
        for x in np.arange(0, 0.4, 0.02):
            self.assertEqual(gate.frame_reason(features(x)), "accepted")
        self.assertEqual(gate.frame_reason(features(-0.3)), "geometry_jump")
        gate.frame_reason(features(-0.3))
        gate.frame_reason(features(-0.3))
        self.assertEqual(gate.frame_reason(features(-0.3)), "accepted")

    def test_invalid_geometry_does_not_poison_reference(self):
        gate = CalibrationQuality()
        self.assertEqual(gate.frame_reason(features(width=0)), "geometry_implausible")
        self.assertEqual(gate.frame_reason(features(float("nan"))), "geometry_invalid")
        self.assertEqual(gate.frame_reason(features()), "accepted")

    def test_multimodal_point_rejected_then_stable_resampling_recovers(self):
        gate = CalibrationQuality()
        point = CalibrationPoint((320, 240))
        for i in range(24):
            v = 0.15 if i % 2 else -0.15
            point.add([v, 0], [0, 0], [i, 0, 0])
            gate.screen_sample(point, i * 33)
        self.assertEqual(gate.screen_reason(point, 15), "point_scattered")
        for i in range(24, 50):
            point.add([0.01, 0], [0, 0], [i, 0, 0])
            gate.screen_sample(point, i * 33)
        self.assertEqual(gate.screen_reason(point, 15), "accepted")
        self.assertEqual(len(point.context_samples), point.sample_count)
        gate.break_screen_run(point)
        self.assertEqual(point.sample_count, 0)
        self.assertEqual(gate.screen_reason(point, 15), "need_samples")

    def test_slow_drift_rejected_even_when_spread_is_small(self):
        gate = CalibrationQuality()
        point = CalibrationPoint((0, 0))
        for i in range(20):
            point.add([i * 0.003, 0], [0, 0])
            gate.screen_sample(point, i * 33)
        self.assertEqual(gate.screen_reason(point, 15), "point_moving")

    def test_large_sample_requirement_not_trapped_by_window(self):
        gate = CalibrationQuality()
        point = CalibrationPoint((0, 0))
        for i in range(50):
            point.add([0, 0], [0, 0])
            gate.screen_sample(point, i * 33, min_samples=50)
        self.assertEqual(gate.screen_reason(point, 50), "accepted")
