"""Temporal blink detection from normalized eye opening."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class BlinkState:
    """Blink state for one frame."""

    is_blink: bool
    started: bool
    ended: bool
    openness: Optional[float]
    threshold: Optional[float]


class BlinkDetector:
    """Use an adaptive baseline and hysteresis to reject short blink events.

    ``close_frames`` and ``open_frames`` prevent one noisy landmark frame from
    toggling the state.  The baseline adapts only while the eyes are open, so
    a closed-eye interval does not redefine the normal opening level.
    """

    def __init__(
        self,
        *,
        threshold_ratio: float = 0.55,
        minimum_openness: float = 0.08,
        close_frames: int = 2,
        open_frames: int = 3,
        baseline_alpha: float = 0.05,
    ) -> None:
        if not isfinite(threshold_ratio) or not 0.0 < threshold_ratio < 1.0:
            raise ValueError("threshold_ratio must be in (0, 1)")
        if not isfinite(minimum_openness) or minimum_openness <= 0.0:
            raise ValueError("minimum_openness must be positive")
        try:
            close_limit = int(close_frames)
            open_limit = int(open_frames)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("close_frames and open_frames must be positive integers") from exc
        if (
            close_limit != close_frames
            or open_limit != open_frames
            or close_limit < 1
            or open_limit < 1
        ):
            raise ValueError("close_frames and open_frames must be positive")
        if not isfinite(baseline_alpha) or not 0.0 < baseline_alpha <= 1.0:
            raise ValueError("baseline_alpha must be in (0, 1]")
        self.threshold_ratio = float(threshold_ratio)
        self.minimum_openness = float(minimum_openness)
        self.close_frames = close_limit
        self.open_frames = open_limit
        self.baseline_alpha = float(baseline_alpha)
        self.reset()

    def reset(self) -> None:
        self.baseline: Optional[float] = None
        self.is_blink = False
        self._closed_count = 0
        self._open_count = 0

    def update(self, openness: Optional[float]) -> BlinkState:
        """Update the detector and return the state for the current frame."""

        started = False
        ended = False
        value = None if openness is None else float(openness)
        if value is None or not isfinite(value):
            # Missing landmarks break a consecutive close/open run. Keep an
            # active blink open until observed open frames arrive, but do not
            # let a partial run resume after a tracking gap.
            self._closed_count = 0
            self._open_count = 0
            threshold = (
                max(self.minimum_openness, self.baseline * self.threshold_ratio)
                if self.baseline is not None
                else None
            )
            return BlinkState(self.is_blink, False, False, None, threshold)

        if self.baseline is None and value >= self.minimum_openness:
            self.baseline = value
        threshold = (
            max(self.minimum_openness, self.baseline * self.threshold_ratio)
            if self.baseline is not None
            else self.minimum_openness
        )
        candidate_closed = value < threshold

        if self.is_blink:
            if candidate_closed:
                self._open_count = 0
            else:
                self._open_count += 1
                if self._open_count >= self.open_frames:
                    self.is_blink = False
                    ended = True
                    self._closed_count = 0
                    self._open_count = 0
        else:
            if candidate_closed:
                self._closed_count += 1
                self._open_count = 0
                if self._closed_count >= self.close_frames:
                    self.is_blink = True
                    started = True
                    self._closed_count = 0
            else:
                self._closed_count = 0
                self._open_count = 0
                self.baseline = (
                    (1.0 - self.baseline_alpha) * self.baseline
                    + self.baseline_alpha * value
                )
                threshold = max(self.minimum_openness, self.baseline * self.threshold_ratio)

        return BlinkState(self.is_blink, started, ended, value, threshold)
