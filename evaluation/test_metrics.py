import csv
import tempfile
import unittest
from pathlib import Path

try:
    from .metrics import evaluate_records, load_csv
except ImportError:
    from metrics import evaluate_records, load_csv


class MetricsTest(unittest.TestCase):
    def test_csv_loader_accepts_invalid_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaze.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "trial_id",
                        "timestamp_ms",
                        "target_x",
                        "target_y",
                        "gaze_x",
                        "gaze_y",
                        "valid",
                        "condition",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trial_id": "t1",
                        "timestamp_ms": 0,
                        "target_x": 10,
                        "target_y": 20,
                        "gaze_x": "",
                        "gaze_y": "",
                        "valid": 0,
                        "condition": "normal",
                    }
                )

            records = load_csv(path)
            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["valid"])

    def test_warmup_error_and_hit_rate(self):
        records = [
            {
                "trial_id": "t1",
                "timestamp_ms": 0,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": 0,
                "gaze_y": 0,
                "valid": True,
                "condition": "normal",
            },
            {
                "trial_id": "t1",
                "timestamp_ms": 600,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": 103,
                "gaze_y": 104,
                "valid": True,
                "condition": "normal",
            },
            {
                "trial_id": "t1",
                "timestamp_ms": 700,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": None,
                "gaze_y": None,
                "valid": False,
                "condition": "normal",
            },
        ]

        report = evaluate_records(
            records,
            warmup_ms=500,
            screen_width=1280,
            screen_height=720,
            hit_radii_px=(6, 10),
        )
        overall = report["overall"]
        self.assertEqual(overall["frame_count"], 2)
        self.assertEqual(overall["valid_frame_count"], 1)
        self.assertEqual(overall["valid_rate"], 0.5)
        self.assertAlmostEqual(overall["mean_error_px"], 5.0)
        self.assertEqual(overall["hit_rate_6px"], 1.0)
        self.assertEqual(overall["hit_rate_10px"], 1.0)

    def test_condition_and_trial_breakdown(self):
        records = []
        for condition, offset in (("normal", 0), ("low_light", 20)):
            for index in range(2):
                records.append(
                    {
                        "trial_id": f"{condition}-{index}",
                        "timestamp_ms": 1000,
                        "target_x": 50,
                        "target_y": 50,
                        "gaze_x": 50 + offset,
                        "gaze_y": 50,
                        "valid": True,
                        "condition": condition,
                    }
                )

        report = evaluate_records(records, warmup_ms=0, hit_radii_px=(10,))
        self.assertAlmostEqual(report["conditions"]["normal"]["mean_error_px"], 0.0)
        self.assertAlmostEqual(report["conditions"]["low_light"]["mean_error_px"], 20.0)
        self.assertEqual(report["trials"]["low_light-0"]["jitter_px_rms"], 0.0)

    def test_v2_raw_and_filtered_summaries_are_separate(self):
        records = [
            {
                "trial_id": "t1",
                "timestamp_ms": 1000,
                "target_x": 100,
                "target_y": 100,
                "raw_gaze_x": 80,
                "raw_gaze_y": 100,
                "raw_valid": True,
                "gaze_x": 95,
                "gaze_y": 100,
                "valid": True,
                "condition": "normal",
            }
        ]
        report = evaluate_records(records, warmup_ms=0)
        self.assertAlmostEqual(report["overall"]["mean_error_px"], 5.0)
        self.assertAlmostEqual(report["raw"]["mean_error_px"], 20.0)

    def test_blink_frames_are_reported_separately(self):
        records = [
            {
                "trial_id": "t1",
                "timestamp_ms": 1000,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": 100,
                "gaze_y": 100,
                "valid": True,
                "blink": False,
                "blink_started": False,
                "blink_ended": False,
                "blink_available": True,
                "condition": "normal",
            },
            {
                "trial_id": "t1",
                "timestamp_ms": 1030,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": None,
                "gaze_y": None,
                "valid": False,
                "blink": True,
                "blink_started": True,
                "blink_ended": False,
                "blink_available": True,
                "condition": "normal",
            },
        ]
        report = evaluate_records(records, warmup_ms=0)
        self.assertEqual(report["overall"]["blink_frame_count"], 1)
        self.assertEqual(report["overall"]["blink_event_count"], 1)
        self.assertEqual(report["overall"]["invalid_blink_frame_count"], 1)

    def test_pose_gate_frames_are_reported_separately(self):
        records = [
            {
                "trial_id": "t1",
                "timestamp_ms": 1000,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": 100,
                "gaze_y": 100,
                "valid": True,
                "pose_valid": True,
                "pose_gate_open": True,
                "pose_available": True,
                "condition": "normal",
            },
            {
                "trial_id": "t1",
                "timestamp_ms": 1030,
                "target_x": 100,
                "target_y": 100,
                "gaze_x": None,
                "gaze_y": None,
                "valid": False,
                "pose_valid": True,
                "pose_gate_open": False,
                "pose_available": True,
                "condition": "normal",
            },
        ]
        report = evaluate_records(records, warmup_ms=0)
        self.assertEqual(report["overall"]["pose_gate_closed_frame_count"], 1)
        self.assertEqual(report["overall"]["pose_gate_open_rate"], 0.5)

    def test_drift_fields_are_reported_separately(self):
        records = [
            {
                "trial_id": "t1",
                "timestamp_ms": 0,
                "target_x": 10,
                "target_y": 10,
                "gaze_x": 10,
                "gaze_y": 10,
                "valid": True,
                "condition": "normal",
                "drift_score": 1.0,
                "drift_detected": False,
                "drift_available": True,
            },
            {
                "trial_id": "t1",
                "timestamp_ms": 1000,
                "target_x": 10,
                "target_y": 10,
                "gaze_x": 20,
                "gaze_y": 10,
                "valid": True,
                "condition": "normal",
                "drift_score": 4.0,
                "drift_detected": True,
                "drift_available": True,
            },
        ]
        report = evaluate_records(records, warmup_ms=0)
        self.assertEqual(report["overall"]["drift_detected_frame_count"], 1)
        self.assertAlmostEqual(report["overall"]["drift_detected_rate"], 0.5)
        self.assertAlmostEqual(report["overall"]["p95_drift_score"], 3.85)


if __name__ == "__main__":
    unittest.main()
