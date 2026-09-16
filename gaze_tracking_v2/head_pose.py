"""Approximate head-pose estimation for quality, compensation, and logging.

The camera matrix is an approximation because the current project has no
per-camera intrinsic calibration.  The result is therefore used as a smoothed
relative context signal rather than as a strict prerequisite for gaze output.
The estimator normally consumes landmarks in the same coordinate system as
the image that MediaPipe processed.  The optional ``mirrored`` conversion is
reserved for callers whose landmark X coordinates explicitly need reflection
before solvePnP; it must not be inferred from display mirroring alone.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .pose_gate import HeadPose


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype="double",
)
LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


class HeadPoseEstimator:
    """Estimate and stabilize coarse head rotation and translation."""

    def __init__(
        self,
        *,
        smoothing_alpha: float = 0.35,
        max_jump_degrees: float = 45.0,
        max_jump_translation: float = 250.0,
        rebase_frames: int = 4,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if max_jump_degrees <= 0 or max_jump_translation <= 0:
            raise ValueError("pose jump limits must be positive")
        if int(rebase_frames) != rebase_frames or rebase_frames < 2:
            raise ValueError("rebase_frames must be an integer of at least 2")
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_jump_degrees = float(max_jump_degrees)
        self.max_jump_translation = float(max_jump_translation)
        self.rebase_frames = int(rebase_frames)
        self.reset()

    def reset(self) -> None:
        self._previous: Optional[HeadPose] = None
        self._rebase_candidate: Optional[HeadPose] = None
        self._rebase_count = 0

    @staticmethod
    def _angle_delta(current: float, previous: float) -> float:
        return (float(current) - float(previous) + 180.0) % 360.0 - 180.0

    def _smooth(self, current: HeadPose) -> Optional[HeadPose]:
        previous = self._previous
        if previous is None:
            self._previous = current
            return current

        angle_deltas = tuple(
            self._angle_delta(current_value, previous_value)
            for current_value, previous_value in (
                (current.yaw, previous.yaw),
                (current.pitch, previous.pitch),
                (current.roll, previous.roll),
            )
        )
        translations = tuple(
            value
            for value in (current.tx, current.ty, current.tz)
            if value is not None
        )
        previous_translations = tuple(
            value
            for value in (previous.tx, previous.ty, previous.tz)
            if value is not None
        )
        if len(translations) != 3 or len(previous_translations) != 3:
            return None
        jump_rejected = any(
            abs(delta) > self.max_jump_degrees for delta in angle_deltas
        ) or max(
            abs(value - previous_value)
            for value, previous_value in zip(translations, previous_translations)
        ) > self.max_jump_translation
        if jump_rejected:
            candidate = self._rebase_candidate
            if candidate is None:
                self._rebase_candidate = current
                self._rebase_count = 1
                return None
            candidate_angle_deltas = (
                self._angle_delta(current.yaw, candidate.yaw),
                self._angle_delta(current.pitch, candidate.pitch),
                self._angle_delta(current.roll, candidate.roll),
            )
            candidate_translations = (candidate.tx, candidate.ty, candidate.tz)
            stable_candidate = (
                all(abs(delta) <= 8.0 for delta in candidate_angle_deltas)
                and all(value is not None for value in candidate_translations)
                and max(
                    abs(value - previous_value)
                    for value, previous_value in zip(
                        translations, candidate_translations
                    )
                ) <= 40.0
            )
            if stable_candidate:
                self._rebase_count += 1
            else:
                self._rebase_candidate = current
                self._rebase_count = 1
            if self._rebase_count >= self.rebase_frames:
                self._previous = current
                self._rebase_candidate = None
                self._rebase_count = 0
                return current
            return None

        self._rebase_candidate = None
        self._rebase_count = 0

        alpha = self.smoothing_alpha
        smoothed = HeadPose(
            yaw=previous.yaw + alpha * angle_deltas[0],
            pitch=previous.pitch + alpha * angle_deltas[1],
            roll=previous.roll + alpha * angle_deltas[2],
            tx=previous.tx + alpha * (current.tx - previous.tx),
            ty=previous.ty + alpha * (current.ty - previous.ty),
            tz=previous.tz + alpha * (current.tz - previous.tz),
            reprojection_error_px=current.reprojection_error_px,
        )
        self._previous = smoothed
        return smoothed

    def estimate(
        self,
        landmarks: Sequence[object],
        image_width: int,
        image_height: int,
        *,
        mirrored: bool = False,
    ) -> HeadPose | None:
        try:
            if image_width <= 0 or image_height <= 0:
                return None
            image_points = np.array(
                [
                    (
                        (
                            1.0 - float(landmarks[index].x)
                            if mirrored
                            else float(landmarks[index].x)
                        )
                        * image_width,
                        float(landmarks[index].y) * image_height,
                    )
                    for index in LANDMARK_INDICES
                ],
                dtype="double",
            )
            if not np.all(np.isfinite(image_points)):
                return None
            focal_length = max(image_width, image_height)
            camera_matrix = np.array(
                [
                    [focal_length, 0, image_width / 2],
                    [0, focal_length, image_height / 2],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            dist_coeffs = np.zeros((4, 1), dtype="double")
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None
            translation = np.asarray(translation_vector, dtype=float).reshape(-1)
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                return None
            if translation[2] <= 0.0:
                return None

            projected, _ = cv2.projectPoints(
                MODEL_POINTS,
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs,
            )
            projected = np.asarray(projected, dtype=float).reshape(-1, 2)
            reprojection_error = float(
                np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
            )
            if not np.isfinite(reprojection_error):
                return None

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            if not np.all(np.isfinite(rotation_matrix)):
                return None
            sy = np.sqrt(
                rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
            )
            singular = sy < 1e-6
            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0.0
            measurement = HeadPose(
                yaw=float(np.degrees(yaw)),
                pitch=float(np.degrees(pitch)),
                roll=float(np.degrees(roll)),
                tx=float(translation[0]),
                ty=float(translation[1]),
                tz=float(translation[2]),
                reprojection_error_px=reprojection_error,
            )
            return self._smooth(measurement)
        except (IndexError, TypeError, ValueError, cv2.error, np.linalg.LinAlgError):
            return None
