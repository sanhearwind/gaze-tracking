"""Optional MediaPipe landmark drawing helpers.

This file is a visualization utility, not a test module.  It is kept outside
the test discovery path so the offline CI does not require plotting libraries.
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw face mesh contours and iris landmarks on an RGB image."""

    annotated_image = np.copy(rgb_image)
    for face_landmarks in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
            )
            for landmark in face_landmarks
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """Plot blendshape scores when matplotlib is installed."""

    import matplotlib.pyplot as plt

    names = [item.category_name for item in face_blendshapes]
    scores = [item.score for item in face_blendshapes]
    ranks = range(len(names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bars = ax.barh(ranks, scores)
    ax.set_yticks(list(ranks), names)
    ax.invert_yaxis()
    for score, patch in zip(scores, bars.patches):
        ax.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}")
    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

