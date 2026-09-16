# Evaluation

Run `t` in the application to record an independent 12-point evaluation. Candidate calibration models are evaluated automatically before acceptance.

```sh
python evaluation/metrics.py path/to/gaze_log.csv --width 640 --height 480 --warmup-ms 500
```

Use the capture dimensions, not the window dimensions. Reports include raw and filtered mean, median, P95, hit rates, valid-frame rates, and per-target results. Compare runs with the same camera setup and condition label.

Acceptance requires at least 15 valid samples at every target. Initial candidates must outperform a fixed-center baseline and satisfy global and regional error limits. Updates must improve mean error without excessive P95 or regional regression.

Local artifacts:

- `gaze_log_v2_*.csv`: evaluation frames and mapper inputs.
- `calibration_states/`: accepted models and cumulative calibration data.
- `diagnostics/*_candidate.json`: candidate snapshots for replay, including rejected candidates.
- `diagnostics/calibration_quality_*.jsonl`: calibration features and sample-rejection reasons.

Diagnostic snapshots are never loaded as active profiles. Keep these local artifacts out of source distributions.

Calibration profiles, diagnostic snapshots, and frame-level CSV logs may contain derived iris, eye, face-position, and head-pose measurements. Treat them as personal data: do not commit, publish, or attach them to bug reports unless they have been deliberately reviewed and anonymized.

Run regression tests from the repository root:

```sh
python -m unittest discover -s evaluation -p "test_*.py" -v
```
