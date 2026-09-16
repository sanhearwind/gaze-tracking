"""Runtime detection of face-position and distance drift after calibration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DriftState:
    """Current drift score and whether a persistent warning is active."""

    score: Optional[float]
    candidate: bool
    warning: bool


class CalibrationDriftMonitor:
    """Detect persistent changes in face position or scale.

    Iris coordinates are deliberately excluded: they are expected to change
    whenever the user looks at another target.  The reference is built from
    calibration-time whole-face context instead of iris coordinates.
    Robust median/MAD statistics and minimum scale floors make the check less
    sensitive to ordinary landmark noise.
    """

    def __init__(
        self,
        reference_contexts: np.ndarray,
        *,
        threshold: float = 3.5,
        persistence_frames: int = 15,
        scale_floor: Sequence[float] | float = 0.025,
    ) -> None:
        values = np.asarray(reference_contexts, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3 or len(values) < 3:
            raise ValueError("reference_contexts must have shape (n>=3, d>=3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("reference_contexts must be finite")
        try:
            persistence_limit = int(persistence_frames)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("threshold and persistence_frames must be positive") from exc
        if (
            not isfinite(threshold)
            or threshold <= 0
            or persistence_limit != persistence_frames
            or persistence_limit < 1
        ):
            raise ValueError("threshold and persistence_frames must be positive")
        floor = np.asarray(scale_floor, dtype=float)
        if floor.ndim == 0:
            floor = np.full(values.shape[1], float(floor), dtype=float)
        if (
            floor.shape != (values.shape[1],)
            or not np.all(np.isfinite(floor))
            or np.any(floor <= 0)
        ):
            raise ValueError("scale_floor must match context dimensions")

        self.reference = np.median(values, axis=0)
        mad = np.median(np.abs(values - self.reference), axis=0)
        self.scale = np.maximum(1.4826 * mad, floor)
        self.threshold = float(threshold)
        self.persistence_frames = persistence_limit
        self._candidate_frames = 0

    @classmethod
    def from_state(
        cls,
        reference: Sequence[float],
        scale: Sequence[float],
        *,
        threshold: float = 3.5,
        persistence_frames: int = 15,
    ) -> "CalibrationDriftMonitor":
        """Restore persisted robust statistics without calibration samples."""

        reference_values = np.asarray(reference, dtype=float)
        scale_values = np.asarray(scale, dtype=float)
        if (
            reference_values.ndim != 1
            or reference_values.shape[0] < 3
            or scale_values.shape != reference_values.shape
            or not np.all(np.isfinite(reference_values))
            or not np.all(np.isfinite(scale_values))
            or np.any(scale_values <= 0)
            or not isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("persisted drift statistics are invalid")
        try:
            persistence_limit = int(persistence_frames)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("persistence_frames must be a positive integer") from exc
        if persistence_limit != persistence_frames or persistence_limit < 1:
            raise ValueError("persistence_frames must be a positive integer")

        monitor = cls.__new__(cls)
        monitor.reference = reference_values
        monitor.scale = scale_values
        monitor.threshold = float(threshold)
        monitor.persistence_frames = persistence_limit
        monitor._candidate_frames = 0
        return monitor

    def reset(self) -> None:
        self._candidate_frames = 0

    def update(self, context: Optional[Sequence[float]]) -> DriftState:
        if context is None:
            # Persistence means consecutive observed drift frames. A missing
            # face must break the run instead of preserving it.
            self._candidate_frames = 0
            return DriftState(score=None, candidate=False, warning=False)
        value = np.asarray(context, dtype=float)
        if value.shape != self.reference.shape or not np.all(np.isfinite(value)):
            self._candidate_frames = 0
            return DriftState(score=None, candidate=False, warning=False)

        standardized = (value - self.reference) / self.scale
        score = float(np.sqrt(np.mean(standardized * standardized)))
        candidate = score > self.threshold
        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = max(0, self._candidate_frames - 1)
        warning = self._candidate_frames >= self.persistence_frames
        return DriftState(score=score, candidate=candidate, warning=warning)
