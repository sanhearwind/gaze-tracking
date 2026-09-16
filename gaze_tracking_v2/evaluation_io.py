"""CSV logging for reproducible V2 evaluation runs."""

from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
from typing import Optional, Sequence


FIELDNAMES = [
    "trial_id",
    "timestamp_ms",
    "target_x",
    "target_y",
    "raw_gaze_x",
    "raw_gaze_y",
    "gaze_x",
    "gaze_y",
    "face_detected",
    "feature_valid",
    "blink",
    "blink_started",
    "blink_ended",
    "blink_openness",
    "blink_threshold",
    "pose_valid",
    "pose_gate_open",
    "pose_yaw",
    "pose_pitch",
    "pose_roll",
    "pose_tx",
    "pose_ty",
    "pose_tz",
    "pose_reprojection_error_px",
    "pose_reference_ready",
    "pose_candidate_allowed",
    "drift_score",
    "drift_detected",
    "valid",
    "condition",
    "camera_width",
    "camera_height",
    "model_version",
    *[f"left_mapper_feature_{index}" for index in range(10)],
    *[f"right_mapper_feature_{index}" for index in range(10)],
]


class EvaluationLogger:
    """Write frame-level raw and filtered gaze values."""

    def __init__(
        self,
        directory: str | Path,
        *,
        condition: str,
        camera_width: int,
        camera_height: int,
        model_version: str = "v2.6",
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"gaze_log_v2_{stamp}.csv"
        suffix = 1
        while True:
            try:
                self._handle = self.path.open(
                    "x", newline="", encoding="utf-8"
                )
                break
            except FileExistsError:
                self.path = directory / f"gaze_log_v2_{stamp}_{suffix}.csv"
                suffix += 1
        self._writer = csv.DictWriter(self._handle, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self.condition = condition
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.model_version = model_version

    def write(
        self,
        *,
        trial_id: str,
        timestamp_ms: int,
        target: tuple[int, int],
        raw_gaze: Optional[tuple[float, float]],
        filtered_gaze: Optional[tuple[float, float]],
        face_detected: bool,
        feature_valid: bool,
        blink: bool,
        blink_started: bool,
        blink_ended: bool,
        blink_openness: Optional[float],
        blink_threshold: Optional[float],
        pose_valid: bool,
        pose_gate_open: bool,
        pose_yaw: Optional[float],
        pose_pitch: Optional[float],
        pose_roll: Optional[float],
        pose_reference_ready: bool,
        pose_candidate_allowed: bool,
        drift_score: Optional[float],
        drift_detected: bool,
        pose_tx: Optional[float] = None,
        pose_ty: Optional[float] = None,
        pose_tz: Optional[float] = None,
        pose_reprojection_error_px: Optional[float] = None,
        left_mapper_features: Optional[Sequence[float]] = None,
        right_mapper_features: Optional[Sequence[float]] = None,
    ) -> None:
        try:
            target_values = tuple(float(value) for value in target)
        except (TypeError, ValueError) as exc:
            raise ValueError("target must contain two finite coordinates") from exc
        if len(target_values) != 2 or not all(
            math.isfinite(value) for value in target_values
        ):
            raise ValueError("target must contain two coordinates")
        for name, point in (("raw_gaze", raw_gaze), ("filtered_gaze", filtered_gaze)):
            if point is not None and (
                len(point) != 2
                or not all(math.isfinite(float(value)) for value in point)
            ):
                raise ValueError(f"{name} must contain two finite coordinates")
        for name, value in (
            ("timestamp_ms", timestamp_ms),
            ("blink_openness", blink_openness),
            ("blink_threshold", blink_threshold),
            ("pose_yaw", pose_yaw),
            ("pose_pitch", pose_pitch),
            ("pose_roll", pose_roll),
            ("pose_tx", pose_tx),
            ("pose_ty", pose_ty),
            ("pose_tz", pose_tz),
            ("pose_reprojection_error_px", pose_reprojection_error_px),
            ("drift_score", drift_score),
        ):
            if value is not None:
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError(f"{name} must be finite when provided")
        mapper_feature_values: dict[str, Optional[tuple[float, ...]]] = {}
        for name, values in (
            ("left", left_mapper_features),
            ("right", right_mapper_features),
        ):
            if values is None:
                mapper_feature_values[name] = None
                continue
            try:
                vector = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name}_mapper_features must contain finite values"
                ) from exc
            if len(vector) != 10 or not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    f"{name}_mapper_features must contain 10 finite values"
                )
            mapper_feature_values[name] = vector

        row = {
                "trial_id": trial_id,
                "timestamp_ms": timestamp_ms,
                "target_x": target_values[0],
                "target_y": target_values[1],
                "raw_gaze_x": "" if raw_gaze is None else raw_gaze[0],
                "raw_gaze_y": "" if raw_gaze is None else raw_gaze[1],
                "gaze_x": "" if filtered_gaze is None else filtered_gaze[0],
                "gaze_y": "" if filtered_gaze is None else filtered_gaze[1],
                "face_detected": int(face_detected),
                "feature_valid": int(feature_valid),
                "blink": int(blink),
                "blink_started": int(blink_started),
                "blink_ended": int(blink_ended),
                "blink_openness": "" if blink_openness is None else blink_openness,
                "blink_threshold": "" if blink_threshold is None else blink_threshold,
                "pose_valid": int(pose_valid),
                "pose_gate_open": int(pose_gate_open),
                "pose_yaw": "" if pose_yaw is None else pose_yaw,
                "pose_pitch": "" if pose_pitch is None else pose_pitch,
                "pose_roll": "" if pose_roll is None else pose_roll,
                "pose_tx": "" if pose_tx is None else pose_tx,
                "pose_ty": "" if pose_ty is None else pose_ty,
                "pose_tz": "" if pose_tz is None else pose_tz,
                "pose_reprojection_error_px": ""
                if pose_reprojection_error_px is None
                else pose_reprojection_error_px,
                "pose_reference_ready": int(pose_reference_ready),
                "pose_candidate_allowed": int(pose_candidate_allowed),
                "drift_score": "" if drift_score is None else drift_score,
                "drift_detected": int(drift_detected),
                "valid": int(filtered_gaze is not None),
                "condition": self.condition,
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
                "model_version": self.model_version,
            }
        for name in ("left", "right"):
            values = mapper_feature_values[name]
            for index in range(10):
                row[f"{name}_mapper_feature_{index}"] = (
                    "" if values is None else values[index]
                )
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "EvaluationLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
