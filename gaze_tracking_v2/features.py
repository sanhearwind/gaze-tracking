"""Feature extraction for the V2 gaze mapper.

The legacy implementation used the iris position relative to one eye corner.
V2 expresses the iris centre in each eye's local coordinate system: horizontal
and vertical distances are divided by the eye width/height.  This reduces the
effect of camera distance, face scale, and small translations in the frame.

The mapper also receives a compact descriptor computed from *all* MediaPipe
face landmarks. Passing the 468 points directly to a small personal calibration
would overfit, so the descriptor keeps global face position, scale, box shape,
and landmark-cloud offset while remaining small enough to regularize.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


# Face Mesh landmark indices.  The two corners are deliberately treated as an
# unordered pair; the local horizontal axis is oriented toward image +X so the
# feature convention is the same for both eyes.
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_UPPER = (159, 160)
LEFT_EYE_LOWER = (145, 144)
RIGHT_EYE_UPPER = (386, 387)
RIGHT_EYE_LOWER = (374, 373)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)


@dataclass(frozen=True)
class EyeFeatures:
    """Features and drawing metadata for one eye."""

    vector: np.ndarray
    iris_center_px: tuple[int, int]
    eye_width: float
    eye_height: float
    openness: float
    valid: bool


@dataclass(frozen=True)
class FaceFeatures:
    """Features extracted from one detected face."""

    left: EyeFeatures
    right: EyeFeatures
    face_center: tuple[float, float]
    face_scale: float
    face_context: np.ndarray

    @property
    def context_vector(self) -> np.ndarray:
        """Return compact whole-face coordinates for mapping and drift checks."""

        return self.face_context

    @property
    def left_mapper_vector(self) -> np.ndarray:
        """Return the low-dimensional left-eye mapper input.

        The full-face context remains available separately for drift and pose
        monitoring.  It is deliberately not a direct regression input: one
        16-point calibration does not provide enough independent samples for
        the extra context terms to be reliable.
        """

        return self.left.vector

    @property
    def right_mapper_vector(self) -> np.ndarray:
        """Return the low-dimensional right-eye mapper input."""

        return self.right.vector

    def head_aware_mapper_vectors(
        self, relative_pose: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return eye features plus compact position/pose/distance context.

        The context uses image-space face position and scale together with
        relative yaw/pitch/roll and relative PnP depth.  It is intentionally
        small and linear in the mapper; the raw full-face landmark descriptor
        remains dedicated to drift monitoring.
        """

        if relative_pose is None:
            raise ValueError("relative head pose is required")
        try:
            yaw = float(getattr(relative_pose, "yaw")) / 30.0
            pitch = float(getattr(relative_pose, "pitch")) / 20.0
            roll = float(getattr(relative_pose, "roll")) / 20.0
            depth_delta = float(getattr(relative_pose, "tz")) / 100.0
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("relative head pose is incomplete") from exc
        context = np.asarray(
            [
                self.face_context[0] - 0.5,
                self.face_context[1] - 0.5,
                math.log(max(float(self.face_context[2]), 1e-8)),
                yaw,
                pitch,
                roll,
                depth_delta,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(context)):
            raise ValueError("head-aware context must be finite")
        return (
            np.concatenate((self.left.vector, context)),
            np.concatenate((self.right.vector, context)),
        )

    def head_compensation_vector(
        self,
        head_pose: object,
        *,
        reference_pose: object | None = None,
        reference_face_center: tuple[float, float] | None = None,
        reference_face_scale: float | None = None,
    ) -> np.ndarray:
        """Return session-normalized movement plus the session's base context."""

        if head_pose is None:
            raise ValueError("absolute head pose is required")
        try:
            yaw = float(getattr(head_pose, "yaw"))
            pitch = float(getattr(head_pose, "pitch"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("absolute head pose is incomplete") from exc
        base_pose = head_pose if reference_pose is None else reference_pose
        try:
            base_yaw = float(getattr(base_pose, "yaw"))
            base_pitch = float(getattr(base_pose, "pitch"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("reference head pose is incomplete") from exc
        base_center = (
            self.face_center
            if reference_face_center is None
            else tuple(map(float, reference_face_center))
        )
        base_scale = (
            float(self.face_scale)
            if reference_face_scale is None
            else float(reference_face_scale)
        )
        if base_scale <= 0.0:
            raise ValueError("reference face scale must be positive")
        yaw_delta = (yaw - base_yaw + 180.0) % 360.0 - 180.0
        pitch_delta = (pitch - base_pitch + 180.0) % 360.0 - 180.0
        vector = np.asarray(
            [
                yaw_delta / 30.0,
                pitch_delta / 20.0,
                (float(self.face_center[0]) - base_center[0]) / base_scale,
                (float(self.face_center[1]) - base_center[1]) / base_scale,
                math.log(max(float(self.face_scale) / base_scale, 1e-8)),
                (base_center[0] - 0.5) / 0.10,
                (base_center[1] - 0.5) / 0.10,
                math.log(base_scale),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(vector)):
            raise ValueError("head compensation features must be finite")
        return vector

    def separated_mapper_vectors(
        self,
        head_pose: object,
        *,
        reference_pose: object | None = None,
        reference_face_center: tuple[float, float] | None = None,
        reference_face_scale: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return two eye vectors and their shared session-aware head vector."""

        head = self.head_compensation_vector(
            head_pose,
            reference_pose=reference_pose,
            reference_face_center=reference_face_center,
            reference_face_scale=reference_face_scale,
        )
        return self.left.vector.copy(), self.right.vector.copy(), head

    @property
    def valid(self) -> bool:
        return bool(
            self.left.valid
            and self.right.valid
            and np.all(np.isfinite(self.left.vector))
            and np.all(np.isfinite(self.right.vector))
            and np.all(np.isfinite(self.context_vector))
        )

    @property
    def blink_openness(self) -> float:
        return min(self.left.openness, self.right.openness)

    @property
    def blink_candidate(self) -> bool:
        return not self.left.valid or not self.right.valid


def _point(landmarks: Sequence[object], index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([float(landmark.x), float(landmark.y), float(landmark.z)], dtype=float)


def _mean_points(landmarks: Sequence[object], indices: Sequence[int]) -> np.ndarray:
    return np.mean([_point(landmarks, index) for index in indices], axis=0)


def _eye_features(
    landmarks: Sequence[object],
    corners: tuple[int, int],
    upper: Sequence[int],
    lower: Sequence[int],
    iris: Sequence[int],
    image_width: int,
    image_height: int,
) -> EyeFeatures:
    corner_a = _point(landmarks, corners[0])
    corner_b = _point(landmarks, corners[1])
    axis = corner_b[:2] - corner_a[:2]
    width = float(np.linalg.norm(axis))
    if width < 1e-8:
        raise ValueError("eye corner distance is too small")

    # Make the horizontal axis point toward image +X for both eyes.
    if axis[0] < 0:
        axis = -axis
    horizontal = axis / np.linalg.norm(axis)
    vertical = np.array([-horizontal[1], horizontal[0]], dtype=float)

    eye_center = (corner_a + corner_b) / 2.0
    iris_center = _mean_points(landmarks, iris)
    upper_center = _mean_points(landmarks, upper)
    lower_center = _mean_points(landmarks, lower)
    eye_height = abs(float(np.dot(lower_center[:2] - upper_center[:2], vertical)))
    openness = float(eye_height / width)
    eye_is_open = openness >= 0.08
    # Keep the vector finite for blink detection, but mark it invalid so the
    # calibration and mapper never consume a closed-eye feature.
    feature_height = max(eye_height, width * 0.35)

    delta = iris_center[:2] - eye_center[:2]
    normalized_x = float(np.dot(delta, horizontal) / width)
    normalized_y = float(np.dot(delta, vertical) / feature_height)
    iris_center_px = (
        int(round(iris_center[0] * image_width)),
        int(round(iris_center[1] * image_height)),
    )
    vector = np.array([normalized_x, normalized_y], dtype=float)
    return EyeFeatures(vector, iris_center_px, width, eye_height, openness, eye_is_open)


def extract_face_features(
    landmarks: Sequence[object], image_width: int, image_height: int
) -> FaceFeatures:
    """Extract normalized local coordinates from a MediaPipe face landmark list."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(landmarks) <= max(*LEFT_IRIS, *RIGHT_IRIS):
        raise ValueError("face landmarks do not contain the iris points")

    points = np.asarray(
        [[float(landmark.x), float(landmark.y)] for landmark in landmarks],
        dtype=float,
    )
    if points.ndim != 2 or points.shape[0] < 10 or not np.all(np.isfinite(points)):
        raise ValueError("face landmarks must contain finite x/y points")
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    face_size = max_xy - min_xy
    face_scale = float(max(face_size[0], face_size[1]))
    if not np.isfinite(face_scale) or face_scale < 1e-8:
        raise ValueError("face scale is too small")

    # Use the complete landmark cloud, but compress it to a small and stable
    # context vector. The first three values preserve the old context API;
    # the remaining values describe full-face geometry rather than iris motion.
    face_center = (min_xy + max_xy) / 2.0
    landmark_centroid = np.mean(points, axis=0)
    face_context = np.asarray(
        [
            face_center[0],
            face_center[1],
            face_scale,
            face_size[0] / face_scale,
            face_size[1] / face_scale,
            (landmark_centroid[0] - face_center[0]) / face_scale,
            (landmark_centroid[1] - face_center[1]) / face_scale,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(face_context)):
        raise ValueError("face context must contain finite values")

    return FaceFeatures(
        left=_eye_features(
            landmarks,
            LEFT_EYE_CORNERS,
            LEFT_EYE_UPPER,
            LEFT_EYE_LOWER,
            LEFT_IRIS,
            image_width,
            image_height,
        ),
        right=_eye_features(
            landmarks,
            RIGHT_EYE_CORNERS,
            RIGHT_EYE_UPPER,
            RIGHT_EYE_LOWER,
            RIGHT_IRIS,
            image_width,
            image_height,
        ),
        face_center=(float(face_center[0]), float(face_center[1])),
        face_scale=face_scale,
        face_context=face_context,
    )
