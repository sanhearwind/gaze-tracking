"""Runnable V2 gaze tracking application.

V2 keeps the same MediaPipe Face Mesh backend as the frozen baseline while
changing the calibration and mapping pipeline.  This makes the first
comparison useful: any accuracy difference is primarily attributable to the
new features/model rather than a simultaneous detector migration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import dataclass
import math
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence


CALIBRATION_SETTLE_MS = 600
CALIBRATION_CAPTURE_MS = 900
CALIBRATION_MIN_SAMPLES = 15
EVALUATION_POINT_DURATION_MS = 2000
EVALUATION_WARMUP_MS = 500
EVALUATION_PREPARE_MS = 1000
MAPPER_INPUT_DIMENSION = 10
HEAD_POSE_ANCHOR_COUNT = 6
HEAD_COVERAGE_MIN_SAMPLES = 5


@dataclass(frozen=True)
class CalibrationStep:
    """One target plus the independently controlled head-pose condition."""

    target: tuple[int, int]
    instruction: str
    pose_condition: str


def assess_candidate_model(
    candidate_errors: Sequence[float],
    active_errors: Optional[Sequence[float]],
    *,
    screen_diagonal: float,
    expected_regions: Sequence[int],
    baseline_regions: Optional[Mapping[int, float]] = None,
    min_region_samples: int = 15,
    candidate_regions: Optional[Mapping[int, Sequence[float]]] = None,
    active_regions: Optional[Mapping[int, Sequence[float]]] = None,
    min_samples: int = 20,
) -> tuple[bool, dict[str, float]]:
    """Apply the V2.6 paired-evaluation acceptance policy."""

    candidate = [float(value) for value in candidate_errors]
    active = None if active_errors is None else [float(value) for value in active_errors]
    if (
        not math.isfinite(screen_diagonal)
        or screen_diagonal <= 0.0
        or min_samples < 1
        or len(candidate) < min_samples
        or not all(math.isfinite(value) and value >= 0.0 for value in candidate)
    ):
        return False, {}

    def summarize(values: Sequence[float]) -> tuple[float, float]:
        ordered = sorted(values)
        position = (len(ordered) - 1) * 0.95
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
        return sum(ordered) / len(ordered), p95

    candidate_mean, candidate_p95 = summarize(candidate)
    metrics = {
        "candidate_mean": candidate_mean,
        "candidate_p95": candidate_p95,
    }
    expected = set(expected_regions)

    def valid_regions(regions, errors):
        if not expected or min_region_samples < 1 or regions is None:
            return False
        if set(regions) != expected:
            return False
        flattened = []
        for values in regions.values():
            if len(values) < min_region_samples:
                return False
            if not all(math.isfinite(value) and value >= 0 for value in values):
                return False
            flattened.extend(values)
        # Regional and global metrics must describe exactly the same frames.
        return sorted(flattened) == sorted(errors)

    if not valid_regions(candidate_regions, candidate):
        return False, metrics
    if active is None:
        if baseline_regions is None or set(baseline_regions) != expected:
            return False, metrics
        if not all(math.isfinite(v) and v >= 0 for v in baseline_regions.values()):
            return False, metrics
        baseline_mean = sum(
            baseline_regions[r] * len(candidate_regions[r]) for r in expected
        ) / len(candidate)
        metrics["baseline_mean"] = baseline_mean
        regional_ok = all(
            sum(candidate_regions[r]) / len(candidate_regions[r])
            <= max(baseline_regions[r], 0.10 * screen_diagonal)
            for r in expected
        )
        return (
            candidate_mean <= 0.35 * screen_diagonal
            and candidate_mean < baseline_mean * 0.90
            and candidate_p95 <= 0.35 * screen_diagonal
            and regional_ok
        ), metrics
    if not valid_regions(active_regions, active) or any(
        len(candidate_regions[r]) != len(active_regions[r]) for r in expected
    ):
        return False, metrics
    if (
        len(active) != len(candidate)
        or not all(math.isfinite(value) and value >= 0.0 for value in active)
    ):
        return False, metrics
    active_mean, active_p95 = summarize(active)
    metrics.update({"active_mean": active_mean, "active_p95": active_p95})
    regional_ok = True
    if candidate_regions is not None and active_regions is not None:
        for region, candidate_values in candidate_regions.items():
            active_values = active_regions.get(region)
            if not candidate_values or not active_values:
                regional_ok = False
                break
            candidate_region_mean = sum(candidate_values) / len(candidate_values)
            active_region_mean = sum(active_values) / len(active_values)
            if candidate_region_mean > max(
                active_region_mean * 1.15, active_region_mean + 5.0
            ):
                regional_ok = False
                break
    accepted = (
        candidate_mean <= active_mean * 0.95
        and candidate_p95 <= active_p95 * 1.05
        and regional_ok
    )
    return accepted, metrics


def build_calibration_plan(
    width: int,
    height: int,
    screen_points: Optional[Sequence[tuple[int, int]]] = None,
) -> list[CalibrationStep]:
    """Build V2.6 fixed-target pose anchors followed by the screen grid."""

    target = (int(width * 0.5), int(height * 0.5))
    anchors = [
        CalibrationStep(target, "A: keep eyes on target; head neutral", "neutral"),
        CalibrationStep(target, "A: slowly sweep head left, center, right", "yaw_sweep"),
        CalibrationStep(target, "A: slowly sweep head up, center, down", "pitch_sweep"),
        CalibrationStep(target, "A: slowly move head left, center, right", "x_sweep"),
        CalibrationStep(target, "A: slowly move head up, center, down", "y_sweep"),
        CalibrationStep(target, "A: slowly move closer, center, farther", "scale_sweep"),
    ]
    points = (
        list(screen_points)
        if screen_points is not None
        else build_calibration_points(width, height)
    )
    anchors.extend(
        CalibrationStep(point, "Keep head neutral; look at the green target", "screen")
        for point in points
    )
    return anchors


def calibration_step_status(
    step: CalibrationStep,
    relative_pose: object,
    *,
    face_scale: float,
    reference_face_scale: Optional[float],
    face_center: tuple[float, float],
    reference_face_center: Optional[tuple[float, float]],
) -> tuple[bool, str]:
    """Return whether the current frame satisfies a V2.6 pose instruction."""

    if relative_pose is None:
        return False, "Blocked: collecting neutral head-pose reference"
    try:
        yaw = float(getattr(relative_pose, "yaw"))
        pitch = float(getattr(relative_pose, "pitch"))
        roll = float(getattr(relative_pose, "roll"))
        scale = float(face_scale)
        center_x, center_y = map(float, face_center)
        dx, dy = (0.0, 0.0) if reference_face_center is None else (center_x - reference_face_center[0], center_y - reference_face_center[1])
    except (AttributeError, TypeError, ValueError):
        return False, "Blocked: head-pose values unavailable"
    values = (yaw, pitch, roll, scale, center_x, center_y, dx, dy)
    if not all(value == value and abs(value) < float("inf") for value in values):
        return False, "Blocked: head-pose values unavailable"

    neutral_rotation = abs(yaw) <= 4.0 and abs(pitch) <= 4.0 and abs(roll) <= 6.0
    condition = step.pose_condition
    accepted = False
    if condition in {"neutral", "screen"}:
        accepted = neutral_rotation
        if accepted and reference_face_scale:
            accepted = 0.90 <= scale / reference_face_scale <= 1.10
        if accepted and reference_face_center is not None:
            accepted = abs(dx) <= 0.035 and abs(dy) <= 0.035
    elif condition == "yaw_sweep":
        accepted = abs(yaw) <= 22.0 and abs(pitch) <= 4.0 and abs(roll) <= 6.0
    elif condition == "pitch_sweep":
        accepted = abs(pitch) <= 18.0 and abs(yaw) <= 4.0 and abs(roll) <= 6.0
    elif condition == "x_sweep" and reference_face_center is not None:
        accepted = abs(dx) <= 0.14 and abs(dy) <= 0.035 and neutral_rotation
    elif condition == "y_sweep" and reference_face_center is not None:
        accepted = abs(dy) <= 0.14 and abs(dx) <= 0.035 and neutral_rotation
    elif condition == "scale_sweep" and reference_face_scale:
        ratio = scale / reference_face_scale
        accepted = 0.82 <= ratio <= 1.22 and neutral_rotation
    if accepted:
        return True, "Status: pose matched; collecting valid samples"
    return False, (
        f"Adjust pose: {step.instruction} | "
        f"yaw {yaw:+.1f}, pitch {pitch:+.1f}, x {dx:+.3f}, y {dy:+.3f}"
    )


def calibration_coverage_bin(
    step: CalibrationStep,
    relative_pose: object,
    *,
    face_scale: float,
    reference_face_scale: Optional[float],
    face_center: tuple[float, float],
    reference_face_center: Optional[tuple[float, float]],
) -> Optional[str]:
    """Classify one accepted motion frame into negative/neutral/positive coverage."""

    if relative_pose is None:
        return None
    condition = step.pose_condition
    if condition == "neutral":
        return "neutral"
    try:
        if condition == "yaw_sweep":
            value, threshold = float(getattr(relative_pose, "yaw")), 5.0
        elif condition == "pitch_sweep":
            value, threshold = float(getattr(relative_pose, "pitch")), 4.0
        elif condition == "x_sweep" and reference_face_center is not None:
            value = float(face_center[0]) - reference_face_center[0]
            threshold = max(0.02, 0.065 * (reference_face_scale or 1.0))
        elif condition == "y_sweep" and reference_face_center is not None:
            value = float(face_center[1]) - reference_face_center[1]
            threshold = max(0.02, 0.065 * (reference_face_scale or 1.0))
        elif condition == "scale_sweep" and reference_face_scale:
            value = float(face_scale) / reference_face_scale - 1.0
            threshold = 0.05
        else:
            return None
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (value == value and abs(value) < float("inf")):
        return None
    if value < -threshold:
        return "negative"
    if value > threshold:
        return "positive"
    return "neutral"



def build_calibration_points(width: int, height: int) -> list[tuple[int, int]]:
    """Return a regular 5x5 calibration grid.

    Calibration points are intentionally independent from the Legacy test
    points.  Only ``build_evaluation_points`` is kept unchanged for fair
    historical comparison.
    """

    x_positions = (0.08, 0.29, 0.50, 0.71, 0.92)
    y_positions = (0.10, 0.30, 0.50, 0.70, 0.90)
    return [
        (int(width * x), int(height * y))
        for y in y_positions
        for x in x_positions
    ]


def build_evaluation_points(width: int, height: int) -> list[tuple[int, int]]:
    """Use the held-out evaluation grid used by the Legacy baseline."""

    x_positions = (0.12, 0.38, 0.62, 0.88)
    y_positions = (0.15, 0.50, 0.85)
    return [
        (int(width * x), int(height * y))
        for y in y_positions
        for x in x_positions
    ]


def build_quick_calibration_points(width: int, height: int) -> list[tuple[int, int]]:
    """Return a 3x3 calibration grid for drift recovery."""

    positions = (0.18, 0.50, 0.82)
    return [
        (int(width * x), int(height * y))
        for y in positions
        for x in positions
    ]


def choose_camera(camera_index: int) -> int:
    """Return the requested camera or the first available camera."""

    import cv2

    if camera_index >= 0:
        return camera_index
    for index in range(10):
        capture = cv2.VideoCapture(index)
        available = capture.isOpened()
        capture.release()
        if available:
            return index
    return -1


def discover_cameras(max_index: int = 10) -> list[tuple[int, int, int]]:
    """Return available camera indices and their currently reported sizes."""

    import cv2

    cameras: list[tuple[int, int, int]] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((index, width, height))
        capture.release()
    return cameras


def _clamp_point(point: Sequence[float], width: int, height: int) -> tuple[float, float]:
    return (
        max(0.0, min(float(width - 1), float(point[0]))),
        max(0.0, min(float(height - 1), float(point[1]))),
    )


def _draw_crosshair(image: np.ndarray, point: tuple[int, int]) -> None:
    import cv2

    cv2 = getattr(image, "drawing", cv2)

    size = 20
    color = (0, 255, 0)
    cv2.line(image, (point[0] - size, point[1]), (point[0] + size, point[1]), color, 2)
    cv2.line(image, (point[0], point[1] - size), (point[0], point[1] + size), color, 2)
    cv2.circle(image, point, 4, color, -1)


def _draw_target(image: np.ndarray, target: tuple[int, int], text: str) -> None:
    import cv2

    cv2 = getattr(image, "drawing", cv2)

    cv2.circle(image, target, 20, (0, 255, 0), -1)
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def motion_coverage_complete(coverage: Mapping[str, int]) -> bool:
    return all(coverage.get(name, 0) >= HEAD_COVERAGE_MIN_SAMPLES
               for name in ("negative", "neutral", "positive"))


def calibration_progress(
    *, elapsed_ms: float, sample_count: int, min_samples: int,
    motion: bool, coverage: Mapping[str, int],
) -> float:
    """Show the least complete requirement, rather than unbounded frame counts."""
    progress = sample_count / max(1, min_samples)
    if not motion:
        duration = CALIBRATION_SETTLE_MS + CALIBRATION_CAPTURE_MS
        progress = min(progress, elapsed_ms / duration)
    if motion:
        covered = sum(min(1.0, max(0, coverage.get(name, 0)) /
                          HEAD_COVERAGE_MIN_SAMPLES)
                      for name in ("negative", "neutral", "positive")) / 3
        progress = min(progress, covered)
    return max(0.0, min(1.0, progress))


def _draw_calibration_feedback(
    image, target: tuple[int, int], *, condition: str, status: str,
    blocked: bool, progress: float, step_index: int, total_steps: int,
) -> None:
    """Keep instructions and two distinct progress rings beside the fixation."""
    import cv2

    cv2 = getattr(image, "drawing", cv2)

    instructions = {
        "neutral": "Hold still",
        "screen": "Head neutral",
        "yaw_sweep": "Turn left / right",
        "pitch_sweep": "Nod up / down",
        "x_sweep": "Move left / right",
        "y_sweep": "Move up / down",
        "scale_sweep": "Closer / farther",
    }
    height, width = image.shape[:2]
    x, y = target
    radius = max(8, min(18, x - 2, y - 2, width - x - 3, height - y - 3))
    inner_radius = max(5, radius - 6)
    step_color = (0, 185, 255) if blocked else (100, 235, 100)
    total_color = (255, 200, 80)
    total_progress = (step_index + progress) / max(1, total_steps)
    for ring_radius, fraction, color in (
        (radius, total_progress, total_color),
        (inner_radius, progress, step_color),
    ):
        cv2.circle(image, target, ring_radius, (25, 25, 25), 6, cv2.LINE_AA)
        cv2.circle(image, target, ring_radius, (95, 95, 95), 3, cv2.LINE_AA)
        if fraction > 0:
            cv2.ellipse(image, target, (ring_radius, ring_radius), -90, 0,
                        360 * min(1.0, fraction), color, 3, cv2.LINE_AA)
    # The fixation itself stays stable even while the status changes.
    cv2.circle(image, target, 4, (60, 255, 60), -1, cv2.LINE_AA)
    cv2.circle(image, target, 2, (255, 255, 255), -1, cv2.LINE_AA)
    # One short action sits immediately outside the rings, rather than below
    # a multi-line panel. Errors replace the action in that same location.
    text = instructions[condition]
    if blocked:
        text = {
            "Face lost - face the camera": "Face camera",
            "Eyes closed - open your eyes": "Open eyes",
            "Eyes unclear - keep eyes visible": "Eyes visible",
            "Keep eyes steady": "Keep eyes steady",
            "Hold still - finding head reference": "Hold still",
            "Return head to neutral": "Return to center",
            "Move gently; avoid tilting": "Move gently",
            "Head outside limits - return to neutral": "Return to center",
            "Hold still - preparing tracking": "Hold still",
        }.get(status, "Hold still")
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(text, baseline, font_scale, color):
        (tw, th), descent = cv2.getTextSize(text, font, font_scale, 1)
        left = max(3, min(x - tw // 2, width - tw - 3))
        baseline = max(th + 3, min(baseline, height - descent - 3))
        cv2.rectangle(image, (left - 2, baseline - th - 2),
                      (left + tw + 2, baseline + descent + 2), (25, 25, 25), -1)
        cv2.putText(image, text, (left, baseline), font, font_scale,
                    color, 1, cv2.LINE_AA)

    (_, text_height), descent = cv2.getTextSize(text, font, 0.50, 1)
    below = y + radius + 4 + text_height
    if below + descent + 3 < height:
        action_baseline = below
        counter_baseline = y - radius - 6
    else:
        action_baseline = y - radius - 6
        counter_baseline = action_baseline - text_height - 8
    label(text, action_baseline, 0.50, step_color)
    label(f"{step_index + 1}/{total_steps}", counter_baseline, 0.32, total_color)


def _draw_gaze(
    image: np.ndarray,
    prediction_left: tuple[float, float],
    prediction_right: tuple[float, float],
    gaze: tuple[float, float],
) -> None:
    import cv2

    cv2 = getattr(image, "drawing", cv2)

    left = (int(round(prediction_left[0])), int(round(prediction_left[1])))
    right = (int(round(prediction_right[0])), int(round(prediction_right[1])))
    filtered = (int(round(gaze[0])), int(round(gaze[1])))
    cv2.circle(image, left, 7, (255, 0, 0), -1)
    cv2.circle(image, right, 7, (0, 0, 255), -1)
    _draw_crosshair(image, filtered)
    cv2.putText(image, f"raw: ({(prediction_left[0] + prediction_right[0]) / 2:.0f}, {(prediction_left[1] + prediction_right[1]) / 2:.0f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"filtered: ({filtered[0]}, {filtered[1]})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)


def _extract_features(
    results: object, width: int, height: int
) -> tuple[bool, Optional[FaceFeatures]]:
    from .features import FaceFeatures, extract_face_features

    faces = getattr(results, "multi_face_landmarks", None)
    if not faces:
        return False, None
    try:
        return True, extract_face_features(faces[0].landmark, width, height)
    except (IndexError, TypeError, ValueError, FloatingPointError):
        return True, None


def _estimate_processed_frame_pose(
    pose_estimator: object,
    landmarks: Sequence[object],
    width: int,
    height: int,
) -> object:
    """Estimate pose in the same coordinates passed to MediaPipe.

    The application mirrors the video frame before Face Mesh runs.  Its
    landmarks therefore already belong to that mirrored frame.  Reflecting
    their X coordinates a second time can move the solvePnP solution behind
    the camera and must not depend on the display-mirroring option.
    """

    return pose_estimator.estimate(
        landmarks,
        width,
        height,
        mirrored=False,
    )


def run_gaze_tracking(
    *,
    camera_index: int = 0,
    requested_width: Optional[int] = None,
    requested_height: Optional[int] = None,
    mirror: bool = True,
    condition: str = "normal",
    min_calibration_samples: int = CALIBRATION_MIN_SAMPLES,
    filter_alpha: float = 0.45,
    ridge: float = 0.05,
    randomize_evaluation: bool = False,
    max_abs_yaw: float = 25.0,
    max_abs_pitch: float = 20.0,
    max_abs_roll: float = 20.0,
    strict_pose_gate: bool = False,
) -> int:
    """Run the V2 camera application and return a process-style status code."""

    try:
        import cv2
        import mediapipe as mp
        import numpy as np

        from .calibration import (
            CalibrationCollector,
            CumulativeCalibrationDataset,
        )
        from .blink import BlinkDetector
        from .correction import CumulativeCorrection
        from .drift import CalibrationDriftMonitor
        from .evaluation_io import EvaluationLogger
        from .quality import CalibrationQuality
        from .rendering import DisplayCanvas
        from .filters import ExponentialSmoother
        from .head_pose import HeadPoseEstimator
        from .mapping import HeadCompensatedMapper
        from .persistence import (
            calibration_state_path,
            clear_calibration_state,
            load_calibration_state,
            save_calibration_state,
        )
        from .pose_gate import HeadPoseGate
    except (ImportError, OSError) as exc:
        print(
            "运行 V2 需要安装 NumPy、OpenCV 和 MediaPipe；"
            f"当前依赖不可用：{exc}"
        )
        return 1

    try:
        smoother = ExponentialSmoother(alpha=filter_alpha)
        blink_detector = BlinkDetector()
        pose_estimator = HeadPoseEstimator()
        pose_gate = HeadPoseGate(
            max_abs_yaw=max_abs_yaw,
            max_abs_pitch=max_abs_pitch,
            max_abs_roll=max_abs_roll,
            reference_frames=8,
        )
        HeadCompensatedMapper(ridge=ridge)
        if min_calibration_samples < 3:
            raise ValueError("每点最少校准帧数不能小于 3")
        if any(
            dimension is not None and dimension <= 0
            for dimension in (requested_width, requested_height)
        ):
            raise ValueError("请求的摄像头宽度和高度必须为正数")
    except ValueError as exc:
        print(f"V2 参数无效：{exc}")
        return 2

    selected_camera = choose_camera(camera_index)
    if selected_camera < 0:
        print("未检测到可用摄像头。")
        return 1

    capture = cv2.VideoCapture(selected_camera)
    if not capture.isOpened():
        print(f"无法打开摄像头 {selected_camera}。")
        return 1
    if requested_width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    if requested_height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    success, initial_frame = capture.read()
    if not success or initial_frame is None or getattr(initial_frame, "ndim", 0) < 2:
        capture.release()
        print("无法读取摄像头的第一帧。")
        return 1
    height, width = initial_frame.shape[:2]
    if width <= 0 or height <= 0:
        capture.release()
        print("摄像头没有返回有效分辨率。")
        return 1
    pending_frame = initial_frame

    points = build_calibration_points(width, height)
    quick_points = build_quick_calibration_points(width, height)
    full_calibration_plan = build_calibration_plan(width, height, points)
    quick_calibration_plan = build_calibration_plan(width, height, quick_points)
    evaluation_points = build_evaluation_points(width, height)
    if randomize_evaluation:
        random.shuffle(evaluation_points)

    root_dir = Path(__file__).resolve().parents[1]
    window_name = "Gaze Tracking V2.6"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, round(1280 * height / width))
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        capture.release()
        cv2.destroyAllWindows()
        raise
    collector: Optional[CalibrationCollector] = None
    active_calibration_plan: list[CalibrationStep] = []
    reference_face_scale: Optional[float] = None
    reference_face_scale_samples: list[float] = []
    reference_face_center: Optional[tuple[float, float]] = None
    reference_face_center_samples: list[tuple[float, float]] = []
    anchor_left_frames: list[object] = []
    anchor_right_frames: list[object] = []
    anchor_head_frames: list[object] = []
    anchor_target_frames: list[tuple[int, int]] = []
    mapper: Optional[HeadCompensatedMapper] = None
    candidate_mapper: Optional[HeadCompensatedMapper] = None
    drift_monitor: Optional[CalibrationDriftMonitor] = None
    candidate_drift_monitor: Optional[CalibrationDriftMonitor] = None
    state_path = calibration_state_path(
        root_dir / "evaluation", camera_id=selected_camera,
        screen_size=(width, height), mirror=mirror, condition=condition,
    )
    cumulative_correction = CumulativeCorrection(screen_size=(width, height))
    calibration_data = CumulativeCalibrationDataset()
    candidate_calibration_data: Optional[CumulativeCalibrationDataset] = None
    try:
        loaded_state = load_calibration_state(
            state_path,
            expected_screen_size=(width, height),
            expected_condition=condition,
            expected_mapper_input_dimension=MAPPER_INPUT_DIMENSION,
            expected_camera_id=selected_camera,
            expected_mirror=mirror,
        )
    except (OSError, ValueError) as exc:
        loaded_state = None
        print(f"警告：已有校准状态未加载：{exc}")
    if loaded_state is not None:
        mapper = loaded_state.mapper
        cumulative_correction = loaded_state.correction
        drift_monitor = loaded_state.drift_monitor
        if loaded_state.calibration_data is not None:
            calibration_data = loaded_state.calibration_data
    calibrating = False
    calibration_started_ms: Optional[float] = None
    evaluating = False
    evaluation_index = 0
    evaluation_started_ms: Optional[float] = None
    logger: Optional[EvaluationLogger] = None
    evaluating_candidate = False
    active_evaluation_errors: list[float] = []
    candidate_evaluation_errors: list[float] = []
    active_evaluation_regions: dict[int, list[float]] = {}
    candidate_evaluation_regions: dict[int, list[float]] = {}
    motion_coverage: dict[str, int] = {}
    calibration_quality = CalibrationQuality()
    quality_trace = None

    print(f"摄像头 {selected_camera} 已打开，实际分辨率 {width}x{height}。")
    if loaded_state is not None:
        print(
            "已加载上次 V2.6 两阶段校准模型："
            f"{calibration_data.session_count} 次会话、"
            f"{calibration_data.sample_count} 个屏幕样本。"
        )
    print(
        "V2.6 操作：c 完整校准，k 快速重新校准，t 独立评估，"
        "n 完成当前步骤（需覆盖达标），Esc 取消候选流程，r 重置，q 退出。"
    )
    print(f"每个校准点自动采集约 {CALIBRATION_CAPTURE_MS}ms，至少 {min_calibration_samples} 帧。")
    if strict_pose_gate:
        print("严格姿态门控已启用：先记录中性姿态，再按相对姿态变化暂停采样/输出。")
    else:
        print("默认只记录头部姿态，不阻断校准；需要门控时使用 --strict-pose-gate。")

    def begin_calibration(*, quick: bool = False) -> None:
        nonlocal collector, calibrating, quality_trace
        nonlocal calibration_started_ms, evaluating, evaluation_started_ms
        nonlocal evaluation_index, logger
        nonlocal active_calibration_plan, reference_face_scale, reference_face_center
        nonlocal reference_face_scale_samples, reference_face_center_samples
        nonlocal anchor_left_frames, anchor_right_frames
        nonlocal anchor_head_frames, anchor_target_frames
        nonlocal candidate_mapper, candidate_calibration_data
        nonlocal candidate_drift_monitor, evaluating_candidate, motion_coverage
        if quick and (mapper is None or not mapper.ready):
            print("快速校准需要先有一个可用的 V2.6 校准模型。")
            return
        if evaluating:
            if logger is not None:
                logger.close()
                logger = None
            evaluating = False
            evaluation_started_ms = None
            evaluation_index = 0
            print("当前评估已停止，将进入重新校准。")
        candidate_mapper = None
        candidate_calibration_data = None
        candidate_drift_monitor = None
        evaluating_candidate = False
        active_calibration_plan = (
            quick_calibration_plan if quick else full_calibration_plan
        )
        collector = CalibrationCollector(
            [step.target for step in active_calibration_plan], min_samples=min_calibration_samples
        )
        calibration_quality.reset()
        if quality_trace is not None:
            quality_trace.close()
        diagnostic_dir = root_dir / "evaluation" / "diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        quality_trace = (diagnostic_dir / (
            "calibration_quality_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"
        )).open("x", encoding="utf-8", buffering=1)
        calibrating = True
        evaluating = False
        calibration_started_ms = time.monotonic() * 1000
        reference_face_scale = None
        reference_face_scale_samples = []
        reference_face_center = None
        reference_face_center_samples = []
        anchor_left_frames = []
        anchor_right_frames = []
        anchor_head_frames = []
        anchor_target_frames = []
        motion_coverage = {}
        smoother.reset()
        blink_detector.reset()
        pose_gate.reset()
        pose_estimator.reset()
        mode = "快速" if quick else "完整"
        print(
            f"开始 V2.6 {mode}校准：先始终注视中心 A 点并按提示移动头部。"
        )

    def finish_calibration_point(now_ms: float, *, force: bool = False) -> None:
        nonlocal calibrating, calibration_started_ms
        nonlocal reference_face_scale, reference_face_center
        nonlocal candidate_mapper, candidate_calibration_data
        nonlocal candidate_drift_monitor, evaluating, evaluating_candidate
        nonlocal evaluation_index, evaluation_started_ms, logger
        nonlocal active_evaluation_errors, candidate_evaluation_errors
        nonlocal active_evaluation_regions, candidate_evaluation_regions
        nonlocal motion_coverage
        if collector is None or collector.current is None:
            return
        point = collector.current
        finishing_index = collector.current_index
        if 0 < finishing_index < HEAD_POSE_ANCHOR_COUNT and not motion_coverage_complete(motion_coverage):
            print("当前头部动作覆盖不足，请继续补采；Esc 可取消。")
            return
        if finishing_index >= HEAD_POSE_ANCHOR_COUNT:
            reason = calibration_quality.screen_reason(point, collector.min_samples)
            if reason != "accepted":
                if force:
                    print(f"当前点需继续采样：{reason}")
                return
        if not collector.finish_current(force=force):
            print(
                f"当前点 {point.target} 只有 {point.sample_count} 帧，"
                f"至少需要 {collector.min_samples} 帧。"
            )
            return
        if finishing_index == 0:
            if not reference_face_scale_samples or not reference_face_center_samples:
                raise RuntimeError("中性 A 点没有完整的脸部参考样本")
            reference_face_scale = float(
                np.median(np.asarray(reference_face_scale_samples, dtype=float))
            )
            center = np.median(
                np.asarray(reference_face_center_samples, dtype=float), axis=0
            )
            reference_face_center = (float(center[0]), float(center[1]))
            print(
                f"中性参考已建立：脸部中心 {reference_face_center}，"
                f"尺度 {reference_face_scale:.4f}。"
            )
        if collector.complete:
            try:
                grid_points = collector.points[HEAD_POSE_ANCHOR_COUNT:]
                if any(
                    point.skipped or point.sample_count < collector.min_samples
                    for point in grid_points
                ):
                    raise ValueError("屏幕校准点不完整")
                aggregated = [point.aggregate() for point in grid_points]
                grid_left = np.asarray([row[0] for row in aggregated], dtype=float)[:, :2]
                grid_right = np.asarray([row[1] for row in aggregated], dtype=float)[:, :2]
                grid_targets = np.asarray(
                    [point.target for point in grid_points], dtype=float
                )
                spreads = np.asarray([row[2] for row in aggregated], dtype=float)
                try:
                    grid_context_rows = np.asarray(
                        [
                            np.median(
                                np.asarray(point.context_samples, dtype=float),
                                axis=0,
                            )
                            for point in grid_points
                        ],
                        dtype=float,
                    )
                except (ValueError, TypeError):
                    grid_context_rows = None
                if not anchor_left_frames:
                    raise ValueError("固定 A 点没有逐帧头姿样本")
                next_history = (
                    calibration_data.clone()
                    if calibration_data.sample_count
                    else CumulativeCalibrationDataset()
                )
                next_history.add(
                    grid_left,
                    grid_right,
                    grid_targets,
                    grid_context_rows,
                    anchor_left=np.asarray(anchor_left_frames, dtype=float),
                    anchor_right=np.asarray(anchor_right_frames, dtype=float),
                    anchor_heads=np.asarray(anchor_head_frames, dtype=float),
                    anchor_targets=np.asarray(anchor_target_frames, dtype=float),
                )
                all_left, all_right, all_targets = next_history.arrays()
                (
                    all_anchor_left,
                    all_anchor_right,
                    all_anchor_heads,
                    all_anchor_targets,
                ) = next_history.anchor_arrays()
                next_mapper = HeadCompensatedMapper(
                    ridge=ridge, screen_size=(width, height)
                ).fit(
                    all_left,
                    all_right,
                    all_targets,
                    all_anchor_left,
                    all_anchor_right,
                    all_anchor_heads,
                    all_anchor_targets,
                )
                next_mapper.set_output_correction(None)
                try:
                    if not next_history.has_context:
                        raise ValueError("没有脸部上下文数据")
                    next_drift_monitor = CalibrationDriftMonitor(
                        next_history.context_dataset()
                    )
                except ValueError:
                    next_drift_monitor = None
                    print("警告：本次校准没有足够的脸部上下文数据，漂移监测未启用。")
            except ValueError as exc:
                calibrating = False
                calibration_started_ms = None
                print(f"候选模型训练失败：{exc}。" + ("继续使用原模型。" if mapper is not None else "尚未建立模型，请重新校准。"))
                return
            candidate_mapper = next_mapper
            candidate_calibration_data = next_history
            candidate_drift_monitor = next_drift_monitor
            calibrating = False
            calibration_started_ms = None
            print(
                "V2.6 候选模型训练完成。基础眼动模型 RMSE: "
                f"{candidate_mapper.left_model.rmse:.2f}px / {candidate_mapper.right_model.rmse:.2f}px；"
                f"头姿残差 RMSE: {candidate_mapper.compensation_model.rmse:.2f}px；"
                f"屏幕点平均帧内离散度: {float(np.mean(spreads)):.4f}。"
            )
            print(
                f"候选历史包含 {candidate_calibration_data.session_count} 次会话、"
                f"{candidate_calibration_data.sample_count} 个屏幕样本；"
                "现在自动进入独立评估。"
            )
            evaluation_index = 0
            evaluation_started_ms = now_ms
            smoother.reset()
            blink_detector.reset()
            active_evaluation_errors = []
            candidate_evaluation_errors = []
            active_evaluation_regions = {}
            candidate_evaluation_regions = {}
            logger = EvaluationLogger(
                root_dir / "evaluation",
                condition=condition,
                camera_width=width,
                camera_height=height,
            )
            # Preserve rejected candidates for offline decomposition and replay.
            # This directory is never used by the active-model loader.
            try:
                save_calibration_state(
                    root_dir / "evaluation" / "diagnostics" / (logger.path.stem + "_candidate.json"),
                    mapper=candidate_mapper,
                    correction=CumulativeCorrection(screen_size=(width, height)),
                    drift_monitor=candidate_drift_monitor, condition=condition,
                    calibration_data=candidate_calibration_data,
                    camera_id=selected_camera, mirror=mirror,
                )
            except (OSError, ValueError) as exc:
                print(f"警告：候选诊断快照保存失败：{exc}")
            # Training and disk writes must not consume the fixation warmup.
            evaluation_started_ms = time.monotonic() * 1000 + EVALUATION_PREPARE_MS
            evaluating_candidate = True
            evaluating = True
        else:
            calibration_started_ms = now_ms
            motion_coverage = {}
            calibration_quality.reset()
            print(
                f"完成校准点 {collector.current_index}/{len(collector.points)}，"
                f"下一个目标：{collector.current.target}。"
            )

    status = 0
    try:
        while capture.isOpened():
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                success = True
            else:
                success, frame = capture.read()
            if not success or frame is None:
                print("无法读取摄像头帧。")
                status = 1
                break
            if (
                getattr(frame, "ndim", 0) < 2
                or frame.shape[1] != width
                or frame.shape[0] != height
            ):
                print(
                    "摄像头帧分辨率发生变化："
                    f"收到 {frame.shape[1]}x{frame.shape[0]}，期望 {width}x{height}。"
                )
                status = 1
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face_detected, features = _extract_features(results, width, height)
            feature_valid = bool(features is not None and features.valid)
            faces = getattr(results, "multi_face_landmarks", None)
            if faces:
                head_pose = _estimate_processed_frame_pose(
                    pose_estimator,
                    faces[0].landmark,
                    width,
                    height,
                )
            else:
                pose_estimator.reset()
                head_pose = None
            pose_state = pose_gate.update(head_pose)
            pose_valid = head_pose is not None
            pose_gate_open = pose_state.allowed
            pose_allowed_for_data = pose_gate_open or not strict_pose_gate
            blink_state = blink_detector.update(
                features.blink_openness if features is not None else None
            )
            blink_detected = bool(
                blink_state.is_blink
                or (features is not None and features.blink_candidate)
            )
            if (
                not calibrating
                and mapper is not None
                and reference_face_center is None
                and feature_valid
                and features is not None
                and pose_state.reference_ready
                and not blink_detected
            ):
                reference_face_center_samples.append(features.face_center)
                reference_face_scale_samples.append(features.face_scale)
                if len(reference_face_center_samples) >= min_calibration_samples:
                    center = np.median(
                        np.asarray(reference_face_center_samples, dtype=float),
                        axis=0,
                    )
                    reference_face_center = (float(center[0]), float(center[1]))
                    reference_face_scale = float(
                        np.median(
                            np.asarray(reference_face_scale_samples, dtype=float)
                        )
                    )
                    print(
                        "本次运行自然姿态参考已建立："
                        f"脸部中心 {reference_face_center}，"
                        f"尺度 {reference_face_scale:.4f}。"
                    )
            separated_vectors = None
            mapper_vectors = None
            if (
                feature_valid
                and features is not None
                and head_pose is not None
                and (
                    calibrating
                    or (
                        reference_face_center is not None
                        and reference_face_scale is not None
                    )
                )
            ):
                try:
                    separated_vectors = features.separated_mapper_vectors(
                        head_pose,
                        reference_pose=pose_gate.reference,
                        reference_face_center=reference_face_center,
                        reference_face_scale=reference_face_scale,
                    )
                    mapper_vectors = (
                        np.concatenate((separated_vectors[0], separated_vectors[2])),
                        np.concatenate((separated_vectors[1], separated_vectors[2])),
                    )
                except ValueError:
                    mapper_vectors = None
                    separated_vectors = None
            now_ms = time.monotonic() * 1000

            drift_state = (
                drift_monitor.update(
                    features.context_vector
                    if features is not None and not blink_detected
                    else None
                )
                if drift_monitor is not None
                else None
            )

            raw_gaze: Optional[tuple[float, float]] = None
            filtered_gaze: Optional[tuple[float, float]] = None

            try:
                _, _, display_width, display_height = cv2.getWindowImageRect(window_name)
            except (cv2.error, AttributeError):
                display_width, display_height = 1280, round(1280 * height / width)
            if display_width <= 0 or display_height <= 0:
                display_width, display_height = width, height
            display = DisplayCanvas(frame, (display_width, display_height))

            if calibrating and collector is not None and collector.current is not None:
                step = active_calibration_plan[collector.current_index]
                target = step.target
                pose_condition_met = False
                pose_condition_status = "Blocked: eye features invalid"
                if features is not None:
                    pose_condition_met, pose_condition_status = calibration_step_status(
                        step,
                        pose_state.relative_pose,
                        face_scale=features.face_scale,
                        reference_face_scale=reference_face_scale,
                        face_center=features.face_center,
                        reference_face_center=reference_face_center,
                    )
                motion_step = 0 < collector.current_index < HEAD_POSE_ANCHOR_COUNT
                screen_step = collector.current_index >= HEAD_POSE_ANCHOR_COUNT
                frame_quality_reason = calibration_quality.frame_reason(features)
                frame_added = False
                elapsed = now_ms - (calibration_started_ms or now_ms)
                if (
                    (motion_step or elapsed >= CALIBRATION_SETTLE_MS)
                    and feature_valid
                    and not blink_detected
                    and (
                        pose_allowed_for_data
                        or collector.current_index < HEAD_POSE_ANCHOR_COUNT
                    )
                    and mapper_vectors is not None
                    and pose_condition_met
                    and frame_quality_reason == "accepted"
                ):
                    added = collector.add_sample(
                        mapper_vectors[0],
                        mapper_vectors[1],
                        features.context_vector,
                    )
                    frame_added = added
                    if added and screen_step:
                        calibration_quality.screen_sample(collector.current, now_ms, collector.min_samples)
                    if (
                        added
                        and 0 < collector.current_index < HEAD_POSE_ANCHOR_COUNT
                    ):
                        anchor_left_frames.append(separated_vectors[0].copy())
                        anchor_right_frames.append(separated_vectors[1].copy())
                        anchor_head_frames.append(separated_vectors[2].copy())
                        anchor_target_frames.append(target)
                    if added and collector.current_index == 0:
                        reference_face_center_samples.append(features.face_center)
                    if added and collector.current_index == 0:
                        reference_face_scale_samples.append(features.face_scale)
                    if (
                        added
                        and 0 < collector.current_index < HEAD_POSE_ANCHOR_COUNT
                    ):
                        coverage_bin = calibration_coverage_bin(
                            step,
                            pose_state.relative_pose,
                            face_scale=features.face_scale,
                            reference_face_scale=reference_face_scale,
                            face_center=features.face_center,
                            reference_face_center=reference_face_center,
                        )
                        if coverage_bin is not None:
                            motion_coverage[coverage_bin] = (
                                motion_coverage.get(coverage_bin, 0) + 1
                            )
                motion_step = (
                    0 < collector.current_index < HEAD_POSE_ANCHOR_COUNT
                )
                if screen_step and not frame_added:
                    calibration_quality.break_screen_run(collector.current)
                point_quality_reason = (
                    calibration_quality.screen_reason(collector.current, collector.min_samples)
                    if screen_step else "not_screen"
                )
                if quality_trace is not None:
                    reason = frame_quality_reason
                    if reason == "accepted" and not frame_added:
                        reason = ("blink" if blink_detected else "pose_or_settle_blocked")
                    quality_trace.write(json.dumps({
                        "timestamp_ms": now_ms, "step": collector.current_index,
                        "condition": step.pose_condition, "target": target,
                        "accepted": frame_added, "reason": reason,
                        "point_quality": point_quality_reason,
                        "eye_geometry": ([
                            [float(v) if math.isfinite(float(v)) else None
                             for v in (eye.eye_width, eye.eye_height, eye.openness)]
                            for eye in (features.left, features.right)
                        ] if features is not None else None),
                        "eye_vectors": ([features.left.vector.tolist(), features.right.vector.tolist()]
                            if features is not None and feature_valid else None),
                        "mapper_vectors": ([v.tolist() for v in mapper_vectors]
                            if mapper_vectors is not None else None),
                    }, allow_nan=False) + "\n")
                coverage_ready = motion_coverage_complete(motion_coverage)
                progress = calibration_progress(
                    elapsed_ms=elapsed, sample_count=collector.current.sample_count,
                    min_samples=collector.min_samples, motion=motion_step,
                    coverage=motion_coverage,
                )
                if screen_step and point_quality_reason != "accepted":
                    progress = min(progress, 0.95)
                blocked = True
                if not face_detected:
                    nearby_status = "Face lost - face the camera"
                elif blink_detected:
                    nearby_status = "Eyes closed - open your eyes"
                elif not feature_valid:
                    nearby_status = "Eyes unclear - keep eyes visible"
                elif frame_quality_reason != "accepted":
                    nearby_status = "Eyes unclear - keep eyes visible"
                elif screen_step and point_quality_reason in {"point_scattered", "point_moving"}:
                    nearby_status = "Keep eyes steady"
                elif head_pose is None or pose_state.relative_pose is None:
                    nearby_status = "Hold still - finding head reference"
                elif not motion_step and elapsed < CALIBRATION_SETTLE_MS:
                    nearby_status = "Keep looking - getting ready"
                    blocked = False
                elif not pose_condition_met:
                    nearby_status = ("Return head to neutral" if not motion_step
                                     else "Move gently; avoid tilting")
                elif strict_pose_gate and not pose_allowed_for_data and not motion_step:
                    nearby_status = "Head outside limits - return to neutral"
                elif mapper_vectors is None:
                    nearby_status = "Hold still - preparing tracking"
                else:
                    blocked = False
                    if motion_step and not coverage_ready:
                        labels = {
                            "yaw_sweep": ("Side A", "Center", "Side B"),
                            "pitch_sweep": ("Side A", "Center", "Side B"),
                            "x_sweep": ("Side A", "Center", "Side B"),
                            "y_sweep": ("Side A", "Center", "Side B"),
                            "scale_sweep": ("Far", "Center", "Near"),
                        }[step.pose_condition]
                        nearby_status = " | ".join(
                            f"{label} {min(HEAD_COVERAGE_MIN_SAMPLES, motion_coverage.get(name, 0))}/5"
                            for label, name in zip(labels, ("negative", "neutral", "positive"))
                        )
                    elif motion_step:
                        nearby_status = "Coverage ready - keep looking"
                    else:
                        nearby_status = "Collecting - keep looking at dot"
                _draw_calibration_feedback(
                    display, target, condition=step.pose_condition, status=nearby_status,
                    blocked=blocked, progress=progress,
                    step_index=collector.current_index, total_steps=len(collector.points),
                )
                normal_point_ready = (
                    not motion_step
                    and elapsed >= CALIBRATION_SETTLE_MS + CALIBRATION_CAPTURE_MS
                    and (not screen_step or point_quality_reason == "accepted")
                )
                motion_point_ready = (
                    motion_step
                    and coverage_ready
                )
                if (
                    (normal_point_ready or motion_point_ready)
                    and collector.current.sample_count >= collector.min_samples
                ):
                    finish_calibration_point(now_ms)
                    if evaluating and not calibrating:
                        display = DisplayCanvas(frame, (display_width, display_height))
            display_mapper = (
                candidate_mapper
                if evaluating and evaluating_candidate and candidate_mapper is not None
                else mapper
            )
            if (
                not calibrating
                and display_mapper is not None
                and display_mapper.ready
                and feature_valid
                and not blink_detected
                and pose_allowed_for_data
                and mapper_vectors is not None
            ):
                prediction = display_mapper.predict(
                    separated_vectors[0],
                    separated_vectors[1],
                    separated_vectors[2],
                )
                left = _clamp_point(prediction.left, width, height)
                right = _clamp_point(prediction.right, width, height)
                raw_gaze = _clamp_point(prediction.gaze, width, height)
                filtered_gaze = smoother.update(raw_gaze)
                filtered_gaze = _clamp_point(filtered_gaze, width, height)
                _draw_gaze(display, left, right, filtered_gaze)
            elif not calibrating and mapper is None and candidate_mapper is None:
                display.drawing.putText(display, "Press C to start calibration", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            if blink_detected and not calibrating:
                display.drawing.putText(
                    display,
                    "Blink/closed eyes: sampling and output paused",
                    (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            if not calibrating and head_pose is not None and pose_state.relative_pose is not None:
                relative_pose = pose_state.relative_pose
                display.drawing.putText(
                    display,
                    f"Pose delta yaw:{relative_pose.yaw:.1f} pitch:{relative_pose.pitch:.1f} roll:{relative_pose.roll:.1f} deg",
                    (10, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            elif not calibrating and head_pose is not None:
                display.drawing.putText(
                    display,
                    "Pose baseline: collecting neutral reference",
                    (10, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            if (
                not calibrating
                and not pose_gate_open
                and strict_pose_gate
                and not (
                    calibrating
                    and collector is not None
                    and collector.current_index < HEAD_POSE_ANCHOR_COUNT
                )
            ):
                display.drawing.putText(
                    display,
                    (
                        "Strict gate: capturing neutral pose"
                        if not pose_state.reference_ready
                        else "Head pose outside limits: strict gate paused sampling/output"
                    ),
                    (10, height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            if not calibrating and drift_state is not None and drift_state.warning:
                display.drawing.putText(
                    display,
                    "Calibration drift detected: press K for quick recalibration",
                    (10, height - 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            if evaluating and evaluation_started_ms is not None:
                target = evaluation_points[evaluation_index]
                _draw_target(
                    display,
                    target,
                    (
                        "Evaluation ready - look at this target"
                        if now_ms < evaluation_started_ms
                        else f"Evaluation {evaluation_index + 1}/{len(evaluation_points)} | Keep looking; Q exits"
                    ),
                )
                if (
                    evaluating_candidate
                    and raw_gaze is not None
                    and now_ms - evaluation_started_ms >= EVALUATION_WARMUP_MS
                ):
                    candidate_error = math.hypot(
                        raw_gaze[0] - target[0], raw_gaze[1] - target[1]
                    )
                    candidate_evaluation_errors.append(candidate_error)
                    candidate_evaluation_regions.setdefault(
                        evaluation_index, []
                    ).append(candidate_error)
                    if (
                        mapper is not None
                        and mapper.ready
                        and separated_vectors is not None
                    ):
                        active_prediction = mapper.predict(
                            separated_vectors[0],
                            separated_vectors[1],
                            separated_vectors[2],
                        )
                        active_point = _clamp_point(
                            active_prediction.gaze, width, height
                        )
                        active_error = math.hypot(
                            active_point[0] - target[0],
                            active_point[1] - target[1],
                        )
                        active_evaluation_errors.append(active_error)
                        active_evaluation_regions.setdefault(
                            evaluation_index, []
                        ).append(active_error)
                if logger is not None and now_ms >= evaluation_started_ms:
                    logger.write(
                        trial_id=f"eval_{evaluation_index + 1:02d}",
                        timestamp_ms=int(now_ms),
                        target=target,
                        raw_gaze=raw_gaze,
                        filtered_gaze=filtered_gaze,
                        face_detected=face_detected,
                        feature_valid=feature_valid,
                        blink=blink_detected,
                        blink_started=blink_state.started,
                        blink_ended=blink_state.ended,
                        blink_openness=(
                            features.blink_openness if features is not None else None
                        ),
                        blink_threshold=blink_state.threshold,
                        pose_valid=pose_valid,
                        pose_gate_open=pose_gate_open,
                        pose_yaw=(
                            pose_state.relative_pose.yaw
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_pitch=(
                            pose_state.relative_pose.pitch
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_roll=(
                            pose_state.relative_pose.roll
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_tx=(
                            pose_state.relative_pose.tx
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_ty=(
                            pose_state.relative_pose.ty
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_tz=(
                            pose_state.relative_pose.tz
                            if pose_state.relative_pose is not None
                            else None
                        ),
                        pose_reprojection_error_px=(
                            head_pose.reprojection_error_px
                            if head_pose is not None
                            else None
                        ),
                        pose_reference_ready=pose_state.reference_ready,
                        pose_candidate_allowed=pose_state.candidate_allowed,
                        drift_score=(
                            drift_state.score if drift_state is not None else None
                        ),
                        drift_detected=(
                            drift_state.warning if drift_state is not None else False
                        ),
                        left_mapper_features=(
                            mapper_vectors[0] if mapper_vectors is not None else None
                        ),
                        right_mapper_features=(
                            mapper_vectors[1] if mapper_vectors is not None else None
                        ),
                    )
                if now_ms - evaluation_started_ms >= EVALUATION_POINT_DURATION_MS:
                    if evaluation_index + 1 >= len(evaluation_points):
                        evaluating = False
                        evaluation_started_ms = None
                        if logger is not None:
                            logger.close()
                            print(f"V2.6 评估完成，日志已保存：{logger.path}")
                        logger = None
                        evaluation_index = 0
                        if evaluating_candidate and candidate_mapper is not None:
                            accepted, comparison = assess_candidate_model(
                                candidate_evaluation_errors,
                                (
                                    active_evaluation_errors
                                    if mapper is not None
                                    else None
                                ),
                                screen_diagonal=math.hypot(width, height),
                                expected_regions=range(len(evaluation_points)),
                                baseline_regions={
                                    i: math.hypot(x - width / 2, y - height / 2)
                                    for i, (x, y) in enumerate(evaluation_points)
                                },
                                candidate_regions=candidate_evaluation_regions,
                                active_regions=(
                                    active_evaluation_regions
                                    if mapper is not None
                                    else None
                                ),
                            )
                            if comparison:
                                if mapper is None:
                                    print(
                                        "首次候选模型评估："
                                        f"mean={comparison['candidate_mean']:.1f}px, "
                                        f"P95={comparison['candidate_p95']:.1f}px。"
                                    )
                                elif "active_mean" in comparison:
                                    print(
                                        "新旧模型同场评估："
                                        f"旧 mean/P95={comparison['active_mean']:.1f}/"
                                        f"{comparison['active_p95']:.1f}px，"
                                        f"新 mean/P95={comparison['candidate_mean']:.1f}/"
                                        f"{comparison['candidate_p95']:.1f}px。"
                                    )
                            if accepted and candidate_calibration_data is not None:
                                mapper = candidate_mapper
                                calibration_data = candidate_calibration_data
                                drift_monitor = candidate_drift_monitor
                                cumulative_correction.reset()
                                try:
                                    save_calibration_state(
                                        state_path,
                                        mapper=mapper,
                                        correction=cumulative_correction,
                                        drift_monitor=drift_monitor,
                                        condition=condition,
                                        calibration_data=calibration_data,
                                        camera_id=selected_camera,
                                        mirror=mirror,
                                    )
                                except (OSError, ValueError) as exc:
                                    print(
                                        "候选模型通过评估，但保存失败；"
                                        f"当前会话继续使用新模型：{exc}"
                                    )
                                else:
                                    print(
                                        "候选模型通过评估并已成为正式模型："
                                        f"{state_path}"
                                    )
                            else:
                                print("候选模型未通过评估（每个测试点需至少 15 帧且满足误差门槛），继续使用原正式模型。")
                            candidate_mapper = None
                            candidate_calibration_data = None
                            candidate_drift_monitor = None
                            evaluating_candidate = False
                    else:
                        evaluation_index += 1
                        evaluation_started_ms = now_ms
                        smoother.reset()

            cv2.imshow(window_name, display.image)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            if key == 27:
                if calibrating or (evaluating and evaluating_candidate):
                    if logger is not None:
                        logger.close()
                    logger = None
                    collector = None
                    calibrating = False
                    evaluating = False
                    evaluating_candidate = False
                    evaluation_started_ms = None
                    candidate_mapper = None
                    candidate_calibration_data = None
                    candidate_drift_monitor = None
                    reference_face_center = None
                    reference_face_scale = None
                    reference_face_center_samples = []
                    reference_face_scale_samples = []
                    pose_gate.reset()
                    pose_estimator.reset()
                    print("已取消本次校准/候选评估，正式模型保持不变。")
                continue
            if key == ord("c"):
                begin_calibration()
            elif key == ord("k"):
                begin_calibration(quick=True)
            elif key == ord("n") and calibrating:
                finish_calibration_point(now_ms, force=True)
            elif key == ord("t"):
                if calibrating:
                    print("请先完成校准。")
                elif mapper is None or not mapper.ready:
                    print("尚未完成 V2.6 校准。")
                elif evaluating:
                    print("评估已经在进行中。")
                else:
                    evaluation_index = 0
                    evaluation_started_ms = time.monotonic() * 1000 + EVALUATION_PREPARE_MS
                    smoother.reset()
                    blink_detector.reset()
                    active_evaluation_errors = []
                    candidate_evaluation_errors = []
                    active_evaluation_regions = {}
                    candidate_evaluation_regions = {}
                    evaluating_candidate = False
                    logger = EvaluationLogger(
                        root_dir / "evaluation",
                        condition=condition,
                        camera_width=width,
                        camera_height=height,
                    )
                    evaluating = True
                    print(f"V2.6 独立评估开始，共 {len(evaluation_points)} 个测试点。")
            elif key == ord("r"):
                if logger is not None:
                    logger.close()
                logger = None
                collector = None
                mapper = None
                candidate_mapper = None
                drift_monitor = None
                candidate_drift_monitor = None
                cumulative_correction.reset()
                calibration_data = CumulativeCalibrationDataset()
                candidate_calibration_data = None
                evaluating_candidate = False
                try:
                    clear_calibration_state(state_path)
                except OSError as exc:
                    print(f"警告：校准状态文件删除失败：{exc}")
                calibrating = False
                evaluating = False
                evaluation_started_ms = None
                smoother.reset()
                blink_detector.reset()
                pose_gate.reset()
                pose_estimator.reset()
                reference_face_center = None
                reference_face_scale = None
                reference_face_center_samples = []
                reference_face_scale_samples = []
                print("V2.6 校准数据已重置。")
    finally:
        if quality_trace is not None:
            quality_trace.close()
        if logger is not None:
            logger.close()
        face_mesh.close()
        capture.release()
        cv2.destroyAllWindows()
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用受控头姿校准的 V2.6 MediaPipe 注视追踪")
    parser.add_argument("--camera", type=int, default=None, help="摄像头索引；交互模式下可省略")
    parser.add_argument("--width", type=int, help="请求的摄像头宽度")
    parser.add_argument("--height", type=int, help="请求的摄像头高度")
    parser.add_argument("--no-mirror", action="store_true", help="不镜像摄像头画面")
    parser.add_argument("--condition", default="normal", help="写入评估 CSV 的测试条件标签")
    parser.add_argument("--min-calibration-samples", type=int, default=CALIBRATION_MIN_SAMPLES, help="每个校准点所需的最少有效帧数")
    parser.add_argument("--filter-alpha", type=float, default=0.45, help="EMA 响应系数，范围为 (0,1]")
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.05,
        help="基础二次映射与头姿线性补偿的岭正则化强度",
    )
    parser.add_argument("--randomize-evaluation", action="store_true", help="随机排列独立评估点顺序")
    parser.add_argument("--max-abs-yaw", type=float, default=25.0, help="允许的最大相对偏航角变化（度）")
    parser.add_argument("--max-abs-pitch", type=float, default=20.0, help="允许的最大相对俯仰角变化（度）")
    parser.add_argument("--max-abs-roll", type=float, default=20.0, help="允许的最大相对翻滚角变化（度）")
    parser.add_argument("--strict-pose-gate", action="store_true", help="相对头姿超出限制时暂停校准和输出")
    console_group = parser.add_mutually_exclusive_group()
    console_group.add_argument("--interactive", action="store_true", help="启动前显示配置窗口")
    console_group.add_argument("--no-console", action="store_true", help="不显示配置窗口")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_setup_ui = bool(
        not args.no_console
        and (
            args.interactive
            or not raw_argv
        )
    )
    if show_setup_ui:
        try:
            from .setup_ui import configure_from_ui

            args = configure_from_ui(args, discover_cameras())
        except RuntimeError as exc:
            print(f"启动配置窗口失败：{exc}")
            return 2
    camera_index = args.camera if args.camera is not None else 0
    return run_gaze_tracking(
        camera_index=camera_index,
        requested_width=args.width,
        requested_height=args.height,
        mirror=not args.no_mirror,
        condition=args.condition,
        min_calibration_samples=args.min_calibration_samples,
        filter_alpha=args.filter_alpha,
        ridge=args.ridge,
        randomize_evaluation=args.randomize_evaluation,
        max_abs_yaw=args.max_abs_yaw,
        max_abs_pitch=args.max_abs_pitch,
        max_abs_roll=args.max_abs_roll,
        strict_pose_gate=args.strict_pose_gate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
