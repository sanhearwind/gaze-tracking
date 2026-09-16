# Development Roadmap

V2.6 uses equal-weight binocular predictions with an eye-only base mapper and a separate head-pose residual model. Calibration includes geometric frame checks, stable screen-point sampling, and independent candidate acceptance.

Next priorities:

- Validate quality thresholds across users, lighting, glasses, and camera setups.
- Improve head-motion guidance when an unrelated pose axis blocks sampling.
- Use saved candidates to measure base-model and compensation errors separately.
- Reduce calibration effort without weakening coverage or evaluation quality.

Keep evaluation targets separate from training data. Validate changes with regression tests and comparable live runs; offline tests alone do not establish tracking accuracy.
