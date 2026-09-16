"""Conservative calibration quality checks in normalized eye coordinates.

These are geometric heuristics, not MediaPipe confidence probabilities.
"""
from collections import deque
import numpy as np


class CalibrationQuality:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque(maxlen=3)
        self.times = []

    def frame_reason(self, features):
        if features is None:
            self.history.clear()
            return "eyes_missing"
        eyes = (features.left, features.right)
        values = np.array([list(e.vector) + [e.eye_width, e.eye_height, e.openness]
                           for e in eyes], dtype=float)
        if not np.all(np.isfinite(values)) or any(not e.valid for e in eyes):
            self.history.clear()
            return "geometry_invalid"
        if (np.any(values[:, 2:4] <= 0) or np.any(values[:, 4] < 0.10)
                or np.any(np.abs(values[:, 0]) > 0.8)
                or np.any(np.abs(values[:, 1]) > 1.0)):
            self.history.clear()
            return "geometry_implausible"
        previous = np.median(np.asarray(self.history), axis=0) if self.history else None
        # Keep raw observations so a sustained legitimate shift can recover;
        # rejected single-frame spikes do not become the sole reference.
        self.history.append(values)
        if previous is not None:
            eye_jump = np.max(np.linalg.norm(values[:, :2] - previous[:, :2], axis=1))
            size_jump = np.max(np.abs(np.log(values[:, 2:4] / previous[:, 2:4])))
            if eye_jump > 0.12 or size_jump > 0.35:
                return "geometry_jump"
        return "accepted"

    def break_screen_run(self, point):
        point.left_samples.clear()
        point.right_samples.clear()
        point.context_samples.clear()
        self.times.clear()

    def screen_sample(self, point, now_ms, min_samples=15):
        self.times.append(now_ms)
        # Discard earlier unstable clusters rather than contaminating the
        # eventual median or trapping a point permanently in a failed state.
        while len(self.times) > min_samples and self.times[0] < now_ms - 750:
            self.times.pop(0)
            point.left_samples.pop(0)
            point.right_samples.pop(0)
            if point.context_samples:
                point.context_samples.pop(0)

    def screen_reason(self, point, min_samples):
        if point.sample_count < min_samples:
            return "need_samples"
        if len(self.times) != point.sample_count or self.times[-1] - self.times[0] < 500:
            return "need_stable_interval"
        for rows in (point.left_samples, point.right_samples):
            eye = np.asarray(rows, dtype=float)[:, :2]
            radius = np.linalg.norm(eye - np.median(eye, axis=0), axis=1)
            if np.percentile(radius, 90) > 0.045:
                return "point_scattered"
            part = max(1, len(eye) // 3)
            if np.linalg.norm(np.median(eye[:part], axis=0) -
                              np.median(eye[-part:], axis=0)) > 0.03:
                return "point_moving"
        return "accepted"
