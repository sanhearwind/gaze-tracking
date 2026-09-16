"""Tests for the dependency-free temporal blink detector."""

import unittest

from gaze_tracking_v2.blink import BlinkDetector


class BlinkDetectorTest(unittest.TestCase):
    def test_requires_consecutive_closed_frames_and_open_frames(self):
        detector = BlinkDetector(close_frames=2, open_frames=3)
        detector.update(0.25)
        detector.update(0.24)

        first_closed = detector.update(0.10)
        self.assertFalse(first_closed.is_blink)
        started = detector.update(0.09)
        self.assertTrue(started.is_blink)
        self.assertTrue(started.started)

        detector.update(0.20)
        detector.update(0.21)
        ended = detector.update(0.22)
        self.assertFalse(ended.is_blink)
        self.assertTrue(ended.ended)

    def test_missing_openness_does_not_create_a_blink_event(self):
        detector = BlinkDetector()
        state = detector.update(None)
        self.assertFalse(state.is_blink)
        self.assertFalse(state.started)
        self.assertFalse(state.ended)


if __name__ == "__main__":
    unittest.main()

