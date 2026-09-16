# Gaze Tracking V2.6

A webcam gaze-tracking prototype built with MediaPipe, OpenCV, and NumPy. It combines normalized iris features, multi-frame calibration, a two-stage gaze mapper, and independent evaluation.

## Origin and attribution

This project originally started from [Antoine Lamé's GazeTracking](https://github.com/antoinelame/GazeTracking), which is licensed under the MIT License. The current public source tree contains the rewritten MediaPipe implementation; the original dlib implementation is not required or included.

The current V2.6 implementation under `gaze_tracking_v2/` has been substantially rewritten around MediaPipe Face Mesh/Iris. It adds personalized screen calibration, a two-stage gaze mapper, relative head-pose compensation, drift monitoring, filtering, persistence, and independent evaluation. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution details.

## Quick start

Use Python 3.10 or 3.11 and a virtual environment:

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python gaze_tracking_mediapipe_app.py
```

The setup window selects the camera, resolution, and evaluation condition. To skip it:

```sh
python gaze_tracking_mediapipe_app.py --no-console --camera 0 --width 640 --height 480 --condition glasses
```

## Calibration

Press `c` for full calibration: follow five head-motion instructions while looking at the center target, then follow a 25-point grid. Each motion requires directional coverage; screen points require stable eye samples. The inner ring shows step progress and the outer ring shows calibration progress.

Calibration automatically starts a separate 12-point evaluation. Only accepted candidates become the active model. Press `k` for quick recalibration after a model exists, `t` for evaluation, `Esc` to cancel calibration, `r` to reset the current profile, or `q` to exit.

Profiles are isolated by camera, resolution, mirror mode, and condition. Camera processing stays at the capture resolution; overlays are rendered at the window resolution. Logged coordinates remain in camera pixels.

## Privacy

Webcam frames are processed locally. The application does not contain code that uploads camera frames, calibration data, or evaluation results. Raw images and video are not saved by default.

Personalized calibration features and evaluation records may be stored under `evaluation/`, including `calibration_states/`, `diagnostics/`, and `gaze_log_*.csv`. These files can contain derived iris, eye, face-position, and head-pose measurements. They are excluded from Git and Docker build contexts and should not be shared. Press `r` to reset the active profile, or delete the corresponding local calibration state to remove it.

## Limitations

This project is intended for research, experimentation, and prototyping. It is not designed or validated for medical diagnosis, biometric identification, accessibility safety systems, or other safety-critical applications. Accuracy varies with lighting, camera placement, glasses, occlusion, screen size, and head movement.

## Development

- `gaze_tracking_v2/`: current application, features, mapping, quality checks, rendering, and persistence.
- `evaluation/`: offline metrics and regression tests.
- `tools/`: optional development and visualization utilities.

```sh
python -m unittest discover -s evaluation -p "test_*.py" -v
```

See [evaluation notes](evaluation/README.md) and the [development roadmap](IMPROVEMENT_PLAN.md). Local logs, calibration profiles, and diagnostic snapshots are excluded by Git and Docker ignore rules. This is an experimental system; calibration quality and accuracy vary with camera placement and capture conditions.
