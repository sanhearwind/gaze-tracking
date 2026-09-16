#!/usr/bin/env python3
"""Compute gaze accuracy metrics from a frame-level CSV log.

The evaluator intentionally uses only the Python standard library so it can be
used even when the camera/model dependencies are not installed.  The expected
CSV columns are:

    trial_id,timestamp_ms,target_x,target_y,gaze_x,gaze_y,valid,condition

    ``valid`` may be 0/1, true/false, or omitted.  Invalid frames still count in
    the denominator of ``valid_rate`` but do not contribute to positional error.
    V2 logs may additionally contain ``raw_gaze_x`` and ``raw_gaze_y``; when
    present, the report includes a separate ``raw`` summary. V2 blink fields
    produce blink-frame and blink-invalid-frame statistics; pose fields produce
    head-pose validity and gate statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Record = Dict[str, object]


def _float(value: object, field: str, row_number: int) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: {field!r} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"row {row_number}: {field!r} must be finite, got {value!r}"
        )
    return parsed


def _bool(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def _bool_false_when_empty(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return False
    return _bool(value)


def load_csv(path: str | Path) -> List[Record]:
    """Load and validate a frame-level gaze CSV."""

    records: List[Record] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        required = {"target_x", "target_y"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        has_blink_fields = "blink" in reader.fieldnames
        has_pose_fields = "pose_gate_open" in reader.fieldnames
        has_pose_reference_fields = "pose_reference_ready" in reader.fieldnames
        has_drift_fields = "drift_detected" in reader.fieldnames

        for row_number, row in enumerate(reader, start=2):
            target_x = _float(row.get("target_x"), "target_x", row_number)
            target_y = _float(row.get("target_y"), "target_y", row_number)
            if target_x is None or target_y is None:
                raise ValueError(
                    f"row {row_number}: target_x and target_y are required"
                )
            gaze_x = _float(row.get("gaze_x"), "gaze_x", row_number)
            gaze_y = _float(row.get("gaze_y"), "gaze_y", row_number)
            raw_gaze_x = _float(row.get("raw_gaze_x"), "raw_gaze_x", row_number)
            raw_gaze_y = _float(row.get("raw_gaze_y"), "raw_gaze_y", row_number)
            timestamp_ms = _float(row.get("timestamp_ms"), "timestamp_ms", row_number)

            records.append(
                {
                    "trial_id": str(row.get("trial_id") or "trial-1"),
                    "timestamp_ms": timestamp_ms,
                    "target_x": target_x,
                    "target_y": target_y,
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "valid": _bool(row.get("valid")) and gaze_x is not None and gaze_y is not None,
                    "raw_gaze_x": raw_gaze_x,
                    "raw_gaze_y": raw_gaze_y,
                    "raw_valid": raw_gaze_x is not None and raw_gaze_y is not None,
                    "blink": _bool_false_when_empty(row.get("blink")),
                    "blink_started": _bool_false_when_empty(row.get("blink_started")),
                    "blink_ended": _bool_false_when_empty(row.get("blink_ended")),
                    "blink_openness": _float(row.get("blink_openness"), "blink_openness", row_number),
                    "blink_threshold": _float(row.get("blink_threshold"), "blink_threshold", row_number),
                    "blink_available": has_blink_fields,
                    "pose_valid": _bool_false_when_empty(row.get("pose_valid")),
                    "pose_gate_open": _bool_false_when_empty(row.get("pose_gate_open")),
                    "pose_yaw": _float(row.get("pose_yaw"), "pose_yaw", row_number),
                    "pose_pitch": _float(row.get("pose_pitch"), "pose_pitch", row_number),
                    "pose_roll": _float(row.get("pose_roll"), "pose_roll", row_number),
                    "pose_tx": _float(row.get("pose_tx"), "pose_tx", row_number),
                    "pose_ty": _float(row.get("pose_ty"), "pose_ty", row_number),
                    "pose_tz": _float(row.get("pose_tz"), "pose_tz", row_number),
                    "pose_reprojection_error_px": _float(
                        row.get("pose_reprojection_error_px"),
                        "pose_reprojection_error_px",
                        row_number,
                    ),
                    "pose_available": has_pose_fields,
                    "pose_reference_ready": _bool_false_when_empty(
                        row.get("pose_reference_ready")
                    ),
                    "pose_candidate_allowed": _bool_false_when_empty(
                        row.get("pose_candidate_allowed")
                    ),
                    "pose_reference_available": has_pose_reference_fields,
                    "drift_score": _float(row.get("drift_score"), "drift_score", row_number),
                    "drift_detected": _bool_false_when_empty(row.get("drift_detected")),
                    "drift_available": has_drift_fields,
                    "condition": str(row.get("condition") or "default"),
                }
            )
    return records


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _trial_key(record: Record) -> str:
    return str(record.get("trial_id") or "trial-1")


_NUMERIC_RECORD_FIELDS = (
    "timestamp_ms",
    "target_x",
    "target_y",
    "gaze_x",
    "gaze_y",
    "raw_gaze_x",
    "raw_gaze_y",
    "blink_openness",
    "blink_threshold",
    "pose_yaw",
    "pose_pitch",
    "pose_roll",
    "pose_tx",
    "pose_ty",
    "pose_tz",
    "pose_reprojection_error_px",
    "drift_score",
)


def _validate_records(records: Sequence[Record]) -> None:
    """Reject non-finite programmatic records before statistics are computed."""

    for row_number, record in enumerate(records, start=1):
        if record.get("target_x") is None or record.get("target_y") is None:
            raise ValueError(f"record {row_number}: target coordinates are required")
        for field in _NUMERIC_RECORD_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"record {row_number}: {field!r} must be numeric"
                ) from exc
            if not math.isfinite(parsed):
                raise ValueError(
                    f"record {row_number}: {field!r} must be finite"
                )


def _after_warmup(records: Iterable[Record], warmup_ms: float) -> List[Record]:
    """Drop the first warmup interval independently for every trial."""

    grouped: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        grouped[_trial_key(record)].append(record)

    result: List[Record] = []
    for trial_records in grouped.values():
        timestamps = [
            float(record.get("timestamp_ms"))
            for record in trial_records
            if record.get("timestamp_ms") is not None
        ]
        if not timestamps or warmup_ms <= 0:
            result.extend(trial_records)
            continue

        start = min(timestamps)
        result.extend(
            record
            for record in trial_records
            if record.get("timestamp_ms") is None
            or float(record["timestamp_ms"]) - start >= warmup_ms
        )
    return result


def _summary(
    records: Sequence[Record],
    screen_width: Optional[float],
    screen_height: Optional[float],
    hit_radii_px: Sequence[float],
    *,
    x_field: str = "gaze_x",
    y_field: str = "gaze_y",
    valid_field: str = "valid",
) -> Dict[str, object]:
    valid_records = [
        record
        for record in records
        if bool(record.get(valid_field, True))
        and record.get(x_field) is not None
        and record.get(y_field) is not None
    ]
    errors: List[float] = []
    normalized_errors: List[float] = []
    for record in valid_records:
        dx = float(record[x_field]) - float(record["target_x"])
        dy = float(record[y_field]) - float(record["target_y"])
        error = math.hypot(dx, dy)
        errors.append(error)
        if screen_width is not None and screen_height is not None:
            diagonal = math.hypot(screen_width, screen_height)
            normalized_errors.append(error / diagonal)

    result: Dict[str, object] = {
        "frame_count": len(records),
        "valid_frame_count": len(valid_records),
        "valid_rate": len(valid_records) / len(records) if records else None,
        "mean_error_px": _mean(errors),
        "median_error_px": statistics.median(errors) if errors else None,
        "p95_error_px": _percentile(errors, 0.95),
        "mean_normalized_error": _mean(normalized_errors),
    }

    for radius in hit_radii_px:
        result[f"hit_rate_{radius:g}px"] = (
            sum(error <= radius for error in errors) / len(errors) if errors else None
        )
    result.update(_blink_stats(records))
    result.update(_pose_stats(records))
    result.update(_drift_stats(records))
    return result


def _blink_stats(records: Sequence[Record]) -> Dict[str, object]:
    """Return blink metrics only when the input log declares blink fields."""

    if not any(bool(record.get("blink_available")) for record in records):
        return {}
    blink_frames = sum(bool(record.get("blink")) for record in records)
    blink_events = sum(bool(record.get("blink_started")) for record in records)
    invalid_blink_frames = sum(
        bool(record.get("blink")) and not bool(record.get("valid"))
        for record in records
    )
    return {
        "blink_frame_count": blink_frames,
        "blink_rate": blink_frames / len(records) if records else None,
        "blink_event_count": blink_events,
        "invalid_blink_frame_count": invalid_blink_frames,
        "invalid_blink_rate": (
            invalid_blink_frames / len(records) if records else None
        ),
    }


def _pose_stats(records: Sequence[Record]) -> Dict[str, object]:
    """Return head-pose quality metrics when the input log declares them."""

    if not any(bool(record.get("pose_available")) for record in records):
        return {}
    pose_valid_frames = sum(bool(record.get("pose_valid")) for record in records)
    pose_gate_open_frames = sum(bool(record.get("pose_gate_open")) for record in records)
    result = {
        "pose_valid_frame_count": pose_valid_frames,
        "pose_valid_rate": pose_valid_frames / len(records) if records else None,
        "pose_gate_open_frame_count": pose_gate_open_frames,
        "pose_gate_open_rate": (
            pose_gate_open_frames / len(records) if records else None
        ),
        "pose_gate_closed_frame_count": len(records) - pose_gate_open_frames,
    }
    if any(bool(record.get("pose_reference_available")) for record in records):
        result.update(
            {
                "pose_reference_ready_frame_count": sum(
                    bool(record.get("pose_reference_ready")) for record in records
                ),
                "pose_candidate_allowed_frame_count": sum(
                    bool(record.get("pose_candidate_allowed")) for record in records
                ),
            }
        )
    for field, name in (
        ("pose_tz", "pose_tz_delta"),
        ("pose_reprojection_error_px", "pose_reprojection_error_px"),
    ):
        values = [
            float(record[field])
            for record in records
            if record.get(field) is not None
        ]
        if values:
            result[f"mean_{name}"] = _mean(values)
            result[f"p95_{name}"] = _percentile(values, 0.95)
    return result


def _drift_stats(records: Sequence[Record]) -> Dict[str, object]:
    """Return calibration-drift metrics when V2 logs declare them."""

    if not any(bool(record.get("drift_available")) for record in records):
        return {}
    scores = [
        float(record["drift_score"])
        for record in records
        if record.get("drift_score") is not None
    ]
    detected_frames = sum(bool(record.get("drift_detected")) for record in records)
    return {
        "drift_detected_frame_count": detected_frames,
        "drift_detected_rate": detected_frames / len(records) if records else None,
        "mean_drift_score": _mean(scores),
        "p95_drift_score": _percentile(scores, 0.95),
    }


def _jitter_px(
    records: Sequence[Record],
    *,
    x_field: str = "gaze_x",
    y_field: str = "gaze_y",
    valid_field: str = "valid",
) -> Optional[float]:
    valid = [
        record
        for record in records
        if bool(record.get(valid_field, True))
        and record.get(x_field) is not None
        and record.get(y_field) is not None
    ]
    if not valid:
        return None
    mean_x = statistics.fmean(float(record[x_field]) for record in valid)
    mean_y = statistics.fmean(float(record[y_field]) for record in valid)
    squared = [
        math.hypot(float(record[x_field]) - mean_x, float(record[y_field]) - mean_y) ** 2
        for record in valid
    ]
    return math.sqrt(statistics.fmean(squared))


def evaluate_records(
    records: Sequence[Record],
    *,
    warmup_ms: float = 500.0,
    screen_width: Optional[float] = None,
    screen_height: Optional[float] = None,
    hit_radii_px: Sequence[float] = (30.0, 60.0, 100.0),
) -> Dict[str, object]:
    """Evaluate frame errors, conditions, and trial-level stability."""

    if not math.isfinite(float(warmup_ms)) or warmup_ms < 0:
        raise ValueError("warmup_ms must be a finite non-negative number")
    if (screen_width is None) != (screen_height is None):
        raise ValueError("screen_width and screen_height must be provided together")
    if screen_width is not None and (
        not math.isfinite(float(screen_width)) or screen_width <= 0
    ):
        raise ValueError("screen_width must be positive and finite")
    if screen_height is not None and (
        not math.isfinite(float(screen_height)) or screen_height <= 0
    ):
        raise ValueError("screen_height must be positive and finite")
    hit_radii_px = tuple(float(radius) for radius in hit_radii_px)
    if any(not math.isfinite(radius) or radius < 0 for radius in hit_radii_px):
        raise ValueError("hit radii must be finite and non-negative")

    records = list(records)
    _validate_records(records)
    records = _after_warmup(records, warmup_ms)
    by_condition: Dict[str, List[Record]] = defaultdict(list)
    by_trial: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        by_condition[str(record.get("condition") or "default")].append(record)
        by_trial[_trial_key(record)].append(record)

    trials: Dict[str, Dict[str, object]] = {}
    for trial_id, trial_records in sorted(by_trial.items()):
        summary = _summary(trial_records, screen_width, screen_height, hit_radii_px)
        valid = [record for record in trial_records if bool(record.get("valid", True))]
        summary["jitter_px_rms"] = _jitter_px(trial_records)
        summary["target_x"] = trial_records[0]["target_x"] if trial_records else None
        summary["target_y"] = trial_records[0]["target_y"] if trial_records else None
        summary["valid_target_count"] = len(valid)
        trials[trial_id] = summary

    report: Dict[str, object] = {
        "config": {
            "warmup_ms": warmup_ms,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "hit_radii_px": list(hit_radii_px),
        },
        "overall": _summary(records, screen_width, screen_height, hit_radii_px),
        "conditions": {
            condition: _summary(condition_records, screen_width, screen_height, hit_radii_px)
            for condition, condition_records in sorted(by_condition.items())
        },
        "trials": trials,
    }

    raw_available = any(
        record.get("raw_gaze_x") is not None and record.get("raw_gaze_y") is not None
        for record in records
    )
    if raw_available:
        raw_trials: Dict[str, Dict[str, object]] = {}
        for trial_id, trial_records in sorted(by_trial.items()):
            summary = _summary(
                trial_records,
                screen_width,
                screen_height,
                hit_radii_px,
                x_field="raw_gaze_x",
                y_field="raw_gaze_y",
                valid_field="raw_valid",
            )
            summary["jitter_px_rms"] = _jitter_px(
                trial_records,
                x_field="raw_gaze_x",
                y_field="raw_gaze_y",
                valid_field="raw_valid",
            )
            summary["target_x"] = trial_records[0]["target_x"] if trial_records else None
            summary["target_y"] = trial_records[0]["target_y"] if trial_records else None
            raw_trials[trial_id] = summary
        report["raw"] = _summary(
            records,
            screen_width,
            screen_height,
            hit_radii_px,
            x_field="raw_gaze_x",
            y_field="raw_gaze_y",
            valid_field="raw_valid",
        )
        report["raw_conditions"] = {
            condition: _summary(
                condition_records,
                screen_width,
                screen_height,
                hit_radii_px,
                x_field="raw_gaze_x",
                y_field="raw_gaze_y",
                valid_field="raw_valid",
            )
            for condition, condition_records in sorted(by_condition.items())
        }
        report["raw_trials"] = raw_trials
    return report


def _print_report(report: Dict[str, object]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="frame-level gaze CSV")
    parser.add_argument("--width", type=float, help="screen/image width in pixels")
    parser.add_argument("--height", type=float, help="screen/image height in pixels")
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=500.0,
        help="ignore this time at the start of each trial (default: 500)",
    )
    parser.add_argument(
        "--hit-radii-px",
        default="30,60,100",
        help="comma-separated hit radii in pixels (default: 30,60,100)",
    )
    args = parser.parse_args(argv)
    radii = tuple(float(value) for value in args.hit_radii_px.split(",") if value.strip())
    report = evaluate_records(
        load_csv(args.csv_path),
        warmup_ms=args.warmup_ms,
        screen_width=args.width,
        screen_height=args.height,
        hit_radii_px=radii,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
