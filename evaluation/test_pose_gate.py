"""Tests for head-pose gating without camera dependencies."""

import unittest

from gaze_tracking_v2.pose_gate import HeadPose, HeadPoseGate


class PoseGateTest(unittest.TestCase):
    def test_gate_requires_stable_near_frontal_pose(self):
        gate = HeadPoseGate(open_frames=2, close_frames=2)
        self.assertFalse(gate.update(HeadPose(0, 0, 0)).allowed)
        self.assertTrue(gate.update(HeadPose(4, -3, 2)).allowed)
        self.assertTrue(gate.update(HeadPose(30, 0, 0)).allowed)
        self.assertFalse(gate.update(HeadPose(31, 0, 0)).allowed)

    def test_missing_pose_is_not_accepted(self):
        gate = HeadPoseGate()
        self.assertFalse(gate.update(None).allowed)

    def test_gate_uses_relative_pose_instead_of_absolute_zero(self):
        gate = HeadPoseGate(open_frames=2, close_frames=2, reference_frames=3)
        gate.update(HeadPose(-170, 179, -178))
        gate.update(HeadPose(-171, -179, -177))
        reference_state = gate.update(HeadPose(-170, 180, -178))
        self.assertTrue(reference_state.reference_ready)
        self.assertFalse(reference_state.allowed)
        self.assertTrue(gate.update(HeadPose(-168, -179, -177)).candidate_allowed)
        self.assertTrue(gate.update(HeadPose(-168, -179, -177)).allowed)
        gate.update(HeadPose(-140, -179, -177))
        self.assertFalse(gate.update(HeadPose(-140, -179, -177)).allowed)


if __name__ == "__main__":
    unittest.main()
