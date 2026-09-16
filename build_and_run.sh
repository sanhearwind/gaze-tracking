#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="gaze-tracking"

if [[ "${DISPLAY:-}" == "" ]]; then
  echo "DISPLAY is not set; a local X11 display is required for the OpenCV window." >&2
  exit 1
fi

docker build -t "${IMAGE_NAME}" .
xhost +local:docker >/dev/null
trap 'xhost -local:docker >/dev/null' EXIT

docker run --rm \
  --device /dev/video0 \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --env QT_X11_NO_MITSHM=1 \
  "${IMAGE_NAME}"
