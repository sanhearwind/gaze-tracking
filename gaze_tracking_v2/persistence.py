"""Atomic persistence for the V2 mapper and cumulative correction state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional
import uuid

import numpy as np

from .calibration import CumulativeCalibrationDataset
from .correction import CumulativeCorrection
from .drift import CalibrationDriftMonitor
from .mapping import DualEyeMapper, HeadCompensatedMapper, gaze_features, linear_gaze_features


CALIBRATION_STATE_SCHEMA_VERSION = 7
CALIBRATION_MODEL_VERSION = "v2.6"


def calibration_state_path(
    directory: str | Path, *, camera_id: int, screen_size: tuple[int, int],
    mirror: bool, condition: str,
) -> Path:
    """Keep each camera/session configuration in its own stable namespace."""
    width, height = _screen_size(screen_size, "screen_size")
    # Hash the exact condition, avoiding path traversal and lossy slug collisions.
    key = json.dumps(
        [CALIBRATION_MODEL_VERSION, camera_id, width, height, mirror, str(condition)],
        ensure_ascii=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(directory) / "calibration_states" / f"v2_6_{digest}.json"


@dataclass(frozen=True)
class CalibrationState:
    """Restored runtime state for one compatible camera/condition."""

    mapper: DualEyeMapper | HeadCompensatedMapper
    correction: CumulativeCorrection
    drift_monitor: Optional[CalibrationDriftMonitor]
    condition: str
    calibration_data: Optional[CumulativeCalibrationDataset] = None
    camera_id: Optional[int] = None
    mirror: Optional[bool] = None


def _screen_size(value: object, field: str) -> tuple[int, int]:
    dimensions = np.asarray(value, dtype=float)
    if (
        dimensions.shape != (2,)
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0)
        or not np.all(dimensions == np.floor(dimensions))
    ):
        raise ValueError(f"{field} must contain two positive integer dimensions")
    return int(dimensions[0]), int(dimensions[1])


def _finite_rmse(value: object, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _mapper_to_dict(mapper: DualEyeMapper | HeadCompensatedMapper) -> dict[str, object]:
    if not mapper.ready or mapper.screen_size is None:
        raise ValueError("cannot save an unfitted mapper")
    if mapper.left_model.coefficients is None or mapper.right_model.coefficients is None:
        raise ValueError("mapper coefficients are missing")
    payload = {
        "ridge": mapper.ridge,
        "model_kind": mapper.model_kind,
        "screen_size": list(mapper.screen_size),
        "left_coefficients": mapper.left_model.coefficients.tolist(),
        "right_coefficients": mapper.right_model.coefficients.tolist(),
        "left_input_dimension": mapper.left_model.input_dimension,
        "right_input_dimension": mapper.right_model.input_dimension,
        "left_feature_mean": mapper.left_model.feature_mean.tolist(),
        "left_feature_scale": mapper.left_model.feature_scale.tolist(),
        "right_feature_mean": mapper.right_model.feature_mean.tolist(),
        "right_feature_scale": mapper.right_model.feature_scale.tolist(),
        "left_rmse": mapper.left_model.rmse,
        "right_rmse": mapper.right_model.rmse,
    }
    if isinstance(mapper, HeadCompensatedMapper):
        model = mapper.compensation_model
        if model.coefficients is None:
            raise ValueError("head compensation coefficients are missing")
        payload.update(
            {
                "compensation_coefficients": model.coefficients.tolist(),
                "compensation_input_dimension": model.input_dimension,
                "compensation_feature_mean": model.feature_mean.tolist(),
                "compensation_feature_scale": model.feature_scale.tolist(),
                "compensation_rmse": model.rmse,
                "head_min": mapper.head_min.tolist(),
                "head_max": mapper.head_max.tolist(),
            }
        )
    return payload


def _mapper_from_dict(
    payload: object, expected_screen_size: tuple[int, int]
) -> DualEyeMapper | HeadCompensatedMapper:
    if not isinstance(payload, dict):
        raise ValueError("mapper state must be an object")
    saved_screen_size = _screen_size(payload.get("screen_size"), "mapper screen_size")
    if saved_screen_size != expected_screen_size:
        raise ValueError("saved mapper screen_size does not match camera")
    try:
        ridge = float(payload.get("ridge"))
    except (TypeError, ValueError) as exc:
        raise ValueError("saved mapper ridge is invalid") from exc
    model_kind = str(payload.get("model_kind") or "")
    if model_kind == HeadCompensatedMapper.model_kind:
        mapper = HeadCompensatedMapper(ridge=ridge, screen_size=saved_screen_size)
    else:
        mapper = DualEyeMapper(
            ridge=ridge,
            screen_size=saved_screen_size,
            model_kind=model_kind,
        )
    left_coefficients = np.asarray(payload.get("left_coefficients"), dtype=float)
    right_coefficients = np.asarray(payload.get("right_coefficients"), dtype=float)
    if (
        left_coefficients.ndim != 2
        or right_coefficients.ndim != 2
        or left_coefficients.shape != right_coefficients.shape
        or left_coefficients.shape[1] != 2
        or left_coefficients.shape[0] < 3
        or not np.all(np.isfinite(left_coefficients))
        or not np.all(np.isfinite(right_coefficients))
    ):
        raise ValueError("saved mapper coefficients are invalid")

    def restore_feature_state(
        name: str,
        coefficients: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        raw_dimension = payload.get(f"{name}_input_dimension")
        if raw_dimension is None:
            raise ValueError(f"saved {name} input dimension is missing")
        try:
            input_dimension = int(raw_dimension)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"saved {name} input dimension is invalid") from exc
        if input_dimension != raw_dimension or input_dimension < 2:
            raise ValueError(f"saved {name} input dimension is invalid")
        feature_builder = (
            linear_gaze_features if model_kind == "linear" else gaze_features
        )
        expected_feature_count = feature_builder(
            np.zeros((1, input_dimension), dtype=float)
        ).shape[1]
        if expected_feature_count != coefficients.shape[0]:
            raise ValueError(f"saved {name} feature dimension is invalid")

        mean_payload = payload.get(f"{name}_feature_mean")
        scale_payload = payload.get(f"{name}_feature_scale")
        mean = np.zeros(coefficients.shape[0], dtype=float) if mean_payload is None else np.asarray(mean_payload, dtype=float)
        scale = np.ones(coefficients.shape[0], dtype=float) if scale_payload is None else np.asarray(scale_payload, dtype=float)
        if (
            mean.shape != (coefficients.shape[0],)
            or scale.shape != (coefficients.shape[0],)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError(f"saved {name} feature normalization is invalid")
        return input_dimension, mean, scale

    left_dimension, left_mean, left_scale = restore_feature_state(
        "left", left_coefficients
    )
    right_dimension, right_mean, right_scale = restore_feature_state(
        "right", right_coefficients
    )
    if left_dimension != right_dimension:
        raise ValueError("saved mapper input dimensions do not match")
    mapper.left_model.coefficients = left_coefficients
    mapper.right_model.coefficients = right_coefficients
    mapper.left_model.input_dimension = left_dimension
    mapper.right_model.input_dimension = right_dimension
    mapper.left_model.feature_mean = left_mean
    mapper.left_model.feature_scale = left_scale
    mapper.right_model.feature_mean = right_mean
    mapper.right_model.feature_scale = right_scale
    mapper.left_model.rmse = _finite_rmse(payload.get("left_rmse"), "left_rmse")
    mapper.right_model.rmse = _finite_rmse(payload.get("right_rmse"), "right_rmse")
    if isinstance(mapper, HeadCompensatedMapper):
        coefficients = np.asarray(
            payload.get("compensation_coefficients"), dtype=float
        )
        dimension = payload.get("compensation_input_dimension")
        mean = np.asarray(payload.get("compensation_feature_mean"), dtype=float)
        scale = np.asarray(payload.get("compensation_feature_scale"), dtype=float)
        head_min = np.asarray(payload.get("head_min"), dtype=float)
        head_max = np.asarray(payload.get("head_max"), dtype=float)
        if (
            dimension != mapper.head_dimension
            or coefficients.shape != (mapper.head_dimension + 1, 2)
            or mean.shape != (mapper.head_dimension + 1,)
            or scale.shape != (mapper.head_dimension + 1,)
            or head_min.shape != (mapper.head_dimension,)
            or head_max.shape != (mapper.head_dimension,)
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(head_min))
            or not np.all(np.isfinite(head_max))
            or np.any(scale <= 0)
            or np.any(head_min > head_max)
        ):
            raise ValueError("saved head compensation model is invalid")
        model = mapper.compensation_model
        model.coefficients = coefficients
        model.input_dimension = int(dimension)
        model.feature_mean = mean
        model.feature_scale = scale
        model.rmse = _finite_rmse(
            payload.get("compensation_rmse"), "compensation_rmse"
        )
        mapper.head_min = head_min
        mapper.head_max = head_max
    return mapper


def _drift_to_dict(
    drift_monitor: Optional[CalibrationDriftMonitor],
) -> Optional[dict[str, object]]:
    if drift_monitor is None:
        return None
    return {
        "reference": drift_monitor.reference.tolist(),
        "scale": drift_monitor.scale.tolist(),
        "threshold": drift_monitor.threshold,
        "persistence_frames": drift_monitor.persistence_frames,
    }


def _drift_from_dict(payload: object) -> Optional[CalibrationDriftMonitor]:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("drift state must be an object or null")
    return CalibrationDriftMonitor.from_state(
        payload.get("reference"),
        payload.get("scale"),
        threshold=float(payload.get("threshold", 3.5)),
        persistence_frames=payload.get("persistence_frames", 15),
    )


def save_calibration_state(
    path: str | Path,
    *,
    mapper: DualEyeMapper | HeadCompensatedMapper,
    correction: CumulativeCorrection,
    drift_monitor: Optional[CalibrationDriftMonitor],
    condition: str,
    calibration_data: Optional[CumulativeCalibrationDataset] = None,
    camera_id: Optional[int] = None,
    mirror: Optional[bool] = None,
    model_version: str = CALIBRATION_MODEL_VERSION,
) -> None:
    """Write a complete calibration state with an atomic replace."""

    if mapper.screen_size != correction.screen_size:
        raise ValueError("mapper and correction screen sizes must match")
    payload = {
        "schema_version": CALIBRATION_STATE_SCHEMA_VERSION,
        "model_version": str(model_version),
        "condition": str(condition),
        "camera_id": camera_id,
        "mirror": mirror,
        "mapper": _mapper_to_dict(mapper),
        "correction": correction.to_dict(),
        "drift": _drift_to_dict(drift_monitor),
        "calibration_data": (
            None if calibration_data is None else calibration_data.to_dict()
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_calibration_state(
    path: str | Path,
    *,
    expected_screen_size: tuple[int, int],
    expected_condition: str,
    expected_mapper_input_dimension: Optional[int] = None,
    expected_camera_id: Optional[int] = None,
    expected_mirror: Optional[bool] = None,
    model_version: str = CALIBRATION_MODEL_VERSION,
) -> Optional[CalibrationState]:
    """Load compatible state or return ``None`` when no state exists."""

    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration state: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration state must be an object")
    if payload.get("schema_version") != CALIBRATION_STATE_SCHEMA_VERSION:
        raise ValueError("calibration state schema version is unsupported")
    if payload.get("model_version") != model_version:
        raise ValueError("calibration state was created by another model version")
    saved_condition = str(payload.get("condition") or "")
    if saved_condition != str(expected_condition):
        raise ValueError(
            f"calibration state condition is {saved_condition!r}, "
            f"but current condition is {expected_condition!r}"
        )
    saved_camera_id = payload.get("camera_id")
    if expected_camera_id is not None and saved_camera_id != expected_camera_id:
        raise ValueError("calibration state was created for another camera")
    saved_mirror = payload.get("mirror")
    if expected_mirror is not None and saved_mirror is not expected_mirror:
        raise ValueError("calibration state mirror mode does not match runtime")

    mapper = _mapper_from_dict(payload.get("mapper"), expected_screen_size)
    if (
        expected_mapper_input_dimension is not None
        and mapper.input_dimension != expected_mapper_input_dimension
    ):
        raise ValueError("saved mapper input dimension does not match runtime")
    correction = CumulativeCorrection.from_dict(
        payload.get("correction"), expected_screen_size=expected_screen_size
    )
    mapper.set_output_correction(correction)
    drift_monitor = _drift_from_dict(payload.get("drift"))
    calibration_data_payload = payload.get("calibration_data")
    calibration_data = (
        None
        if calibration_data_payload is None
        else CumulativeCalibrationDataset.from_dict(calibration_data_payload)
    )
    if calibration_data is not None and calibration_data.sample_count > 0:
        expected_data_dimension = (
            2 if isinstance(mapper, HeadCompensatedMapper) else mapper.input_dimension
        )
        if calibration_data.feature_dimension != expected_data_dimension:
            raise ValueError("saved calibration data does not match mapper input dimension")
        if isinstance(mapper, HeadCompensatedMapper):
            anchors = calibration_data.anchor_arrays()
            if anchors[2].shape[1] != mapper.head_dimension:
                raise ValueError("saved head anchors do not match mapper input dimension")
    return CalibrationState(
        mapper=mapper,
        correction=correction,
        drift_monitor=drift_monitor,
        condition=saved_condition,
        calibration_data=calibration_data,
        camera_id=saved_camera_id,
        mirror=saved_mirror,
    )


def clear_calibration_state(path: str | Path) -> None:
    """Delete the exact persisted state file, if it exists."""

    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
