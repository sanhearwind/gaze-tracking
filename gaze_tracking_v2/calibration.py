"""Multi-frame calibration collection and robust point aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .mapping import DualEyeMapper


@dataclass
class CalibrationPoint:
    target: tuple[int, int]
    left_samples: list[np.ndarray] = field(default_factory=list)
    right_samples: list[np.ndarray] = field(default_factory=list)
    context_samples: list[np.ndarray] = field(default_factory=list)
    skipped: bool = False

    @property
    def sample_count(self) -> int:
        return min(len(self.left_samples), len(self.right_samples))

    def add(
        self,
        left: Sequence[float],
        right: Sequence[float],
        context: Optional[Sequence[float]] = None,
    ) -> None:
        left_value = np.asarray(left, dtype=float)
        right_value = np.asarray(right, dtype=float)
        if left_value.ndim != 1 or right_value.ndim != 1:
            raise ValueError("eye features must each be one-dimensional")
        if left_value.shape != right_value.shape:
            raise ValueError("left and right eye features must have the same shape")
        if not np.all(np.isfinite(left_value)) or not np.all(np.isfinite(right_value)):
            return
        context_value = None
        if context is not None:
            context_value = np.asarray(context, dtype=float)
            if context_value.ndim != 1 or not np.all(np.isfinite(context_value)):
                raise ValueError("context features must be a finite vector")

        # Validate every part before mutating the point.  Otherwise an invalid
        # context could leave the two eye sample lists one frame ahead of the
        # context list and make a later model fit use misaligned rows.
        self.left_samples.append(left_value)
        self.right_samples.append(right_value)
        if context_value is not None:
            self.context_samples.append(context_value)

    def aggregate(self) -> tuple[np.ndarray, np.ndarray, float]:
        if not self.left_samples or not self.right_samples:
            raise ValueError("calibration point has no valid samples")
        left = np.asarray(self.left_samples, dtype=float)
        right = np.asarray(self.right_samples, dtype=float)
        left_median = np.median(left, axis=0)
        right_median = np.median(right, axis=0)
        spread = float(
            np.mean(
                np.concatenate(
                    [np.linalg.norm(left - left_median, axis=1),
                     np.linalg.norm(right - right_median, axis=1)]
                )
            )
        )
        return left_median, right_median, spread


class CalibrationCollector:
    """Collect several valid frames for every calibration target."""

    def __init__(self, points: Sequence[tuple[int, int]], min_samples: int = 15) -> None:
        if len(points) < 6:
            raise ValueError("at least 6 calibration targets are required")
        try:
            sample_limit = int(min_samples)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("min_samples must be an integer of at least 3") from exc
        if sample_limit != min_samples or sample_limit < 3:
            raise ValueError("min_samples must be at least 3")
        self.points = [CalibrationPoint((int(x), int(y))) for x, y in points]
        self.min_samples = sample_limit
        self.current_index = 0

    @property
    def complete(self) -> bool:
        return self.current_index >= len(self.points)

    @property
    def current(self) -> Optional[CalibrationPoint]:
        return None if self.complete else self.points[self.current_index]

    def add_sample(
        self,
        left: Sequence[float],
        right: Sequence[float],
        context: Optional[Sequence[float]] = None,
    ) -> bool:
        if self.current is None:
            return False
        before = self.current.sample_count
        self.current.add(left, right, context)
        return self.current.sample_count > before

    def context_dataset(self) -> np.ndarray:
        """Return one robust face-context vector for each calibration target."""

        if not self.complete:
            raise ValueError("calibration is not complete")
        rows = []
        for point in self.points:
            if point.skipped:
                continue
            if not point.context_samples:
                raise ValueError("calibration has no face-context samples")
            rows.append(np.median(np.asarray(point.context_samples), axis=0))
        return np.asarray(rows, dtype=float)

    def finish_current(self, *, force: bool = False) -> bool:
        point = self.current
        if point is None:
            return False
        # ``force`` only means that the caller wants to finish before the
        # normal timer. It must not bypass the sample requirement, otherwise a
        # later dataset() call can advance to an unfit calibration set.
        if point.sample_count < self.min_samples:
            return False
        if point.sample_count == 0:
            return False
        self.current_index += 1
        return True

    def skip_current(self) -> bool:
        """Advance past a motion step that could not reach its sample target."""

        point = self.current
        if point is None:
            return False
        point.skipped = True
        self.current_index += 1
        return True

    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.complete:
            raise ValueError("calibration is not complete")
        left_rows = []
        right_rows = []
        targets = []
        spreads = []
        for point in self.points:
            if point.skipped:
                continue
            if point.sample_count < self.min_samples:
                raise ValueError(
                    f"target {point.target} has only {point.sample_count} samples"
                )
            left, right, spread = point.aggregate()
            left_rows.append(left)
            right_rows.append(right)
            targets.append(point.target)
            spreads.append(spread)
        return (
            np.asarray(left_rows, dtype=float),
            np.asarray(right_rows, dtype=float),
            np.asarray(targets, dtype=float),
            np.asarray(spreads, dtype=float),
        )

    def fit_mapper(
        self,
        ridge: float = 1e-3,
        screen_size: Optional[tuple[int, int]] = None,
    ) -> tuple[DualEyeMapper, np.ndarray]:
        left, right, targets, spreads = self.dataset()
        mapper = DualEyeMapper(ridge=ridge, screen_size=screen_size).fit(
            left, right, targets
        )
        return mapper, spreads


class CumulativeCalibrationDataset:
    """Keep calibration features so repeated sessions share one model.

    Storing only output-space residuals is unsafe after a recalibration because
    the mapper can move those outputs to a different coordinate system.  This
    container preserves the aggregated eye features and their known targets so
    the application can refit one mapper over every compatible session.
    """

    def __init__(self) -> None:
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._contexts: Optional[np.ndarray] = None
        self._anchor_left: Optional[np.ndarray] = None
        self._anchor_right: Optional[np.ndarray] = None
        self._anchor_heads: Optional[np.ndarray] = None
        self._anchor_targets: Optional[np.ndarray] = None
        self._session_count = 0

    @property
    def sample_count(self) -> int:
        return 0 if self._targets is None else len(self._targets)

    @property
    def feature_dimension(self) -> Optional[int]:
        return None if self._left is None else int(self._left.shape[1])

    @property
    def has_context(self) -> bool:
        return self._contexts is not None

    @property
    def anchor_count(self) -> int:
        return 0 if self._anchor_targets is None else len(self._anchor_targets)

    @property
    def session_count(self) -> int:
        return self._session_count

    def add(
        self,
        left: np.ndarray,
        right: np.ndarray,
        targets: np.ndarray,
        contexts: Optional[np.ndarray] = None,
        *,
        anchor_left: Optional[np.ndarray] = None,
        anchor_right: Optional[np.ndarray] = None,
        anchor_heads: Optional[np.ndarray] = None,
        anchor_targets: Optional[np.ndarray] = None,
    ) -> "CumulativeCalibrationDataset":
        """Append one completed calibration session after validation."""

        left_values = np.asarray(left, dtype=float)
        right_values = np.asarray(right, dtype=float)
        target_values = np.asarray(targets, dtype=float)
        if (
            left_values.ndim != 2
            or right_values.shape != left_values.shape
            or target_values.shape != (len(left_values), 2)
            or len(left_values) == 0
        ):
            raise ValueError(
                "left, right, and targets must contain matched non-empty rows"
            )
        if not np.all(np.isfinite(left_values)) or not np.all(
            np.isfinite(right_values)
        ) or not np.all(np.isfinite(target_values)):
            raise ValueError("calibration dataset must be finite")

        context_values: Optional[np.ndarray] = None
        if contexts is not None:
            context_values = np.asarray(contexts, dtype=float)
            if (
                context_values.ndim != 2
                or context_values.shape[0] != len(left_values)
                or context_values.shape[1] < 3
                or not np.all(np.isfinite(context_values))
            ):
                raise ValueError("contexts must have matched finite rows")

        anchor_payload = (anchor_left, anchor_right, anchor_heads, anchor_targets)
        anchor_values = None
        if any(value is not None for value in anchor_payload):
            if not all(value is not None for value in anchor_payload):
                raise ValueError("all anchor arrays must be provided together")
            a_left = np.asarray(anchor_left, dtype=float)
            a_right = np.asarray(anchor_right, dtype=float)
            a_heads = np.asarray(anchor_heads, dtype=float)
            a_targets = np.asarray(anchor_targets, dtype=float)
            if (
                a_left.ndim != 2
                or a_left.shape[1] != 2
                or a_right.shape != a_left.shape
                or a_heads.ndim != 2
                or a_heads.shape[0] != len(a_left)
                or a_targets.shape != (len(a_left), 2)
                or len(a_left) == 0
                or not all(
                    np.all(np.isfinite(value))
                    for value in (a_left, a_right, a_heads, a_targets)
                )
            ):
                raise ValueError("anchor arrays must contain matched finite rows")
            if self._anchor_heads is not None and (
                a_heads.shape[1] != self._anchor_heads.shape[1]
            ):
                raise ValueError("anchor head feature dimensions do not match")
            anchor_values = (a_left, a_right, a_heads, a_targets)

        if self._left is not None:
            if left_values.shape[1] != self._left.shape[1]:
                raise ValueError("calibration feature dimensions do not match")
            if (self._contexts is None) != (context_values is None):
                raise ValueError("calibration context presence does not match")
            if (
                context_values is not None
                and context_values.shape[1] != self._contexts.shape[1]
            ):
                raise ValueError("calibration context dimensions do not match")

            self._left = np.vstack((self._left, left_values))
            self._right = np.vstack((self._right, right_values))
            self._targets = np.vstack((self._targets, target_values))
            if context_values is not None:
                self._contexts = np.vstack((self._contexts, context_values))
        else:
            self._left = left_values.copy()
            self._right = right_values.copy()
            self._targets = target_values.copy()
            self._contexts = (
                None if context_values is None else context_values.copy()
            )
        if anchor_values is not None:
            a_left, a_right, a_heads, a_targets = anchor_values
            if self._anchor_left is None:
                self._anchor_left = a_left.copy()
                self._anchor_right = a_right.copy()
                self._anchor_heads = a_heads.copy()
                self._anchor_targets = a_targets.copy()
            else:
                self._anchor_left = np.vstack((self._anchor_left, a_left))
                self._anchor_right = np.vstack((self._anchor_right, a_right))
                self._anchor_heads = np.vstack((self._anchor_heads, a_heads))
                self._anchor_targets = np.vstack((self._anchor_targets, a_targets))
        self._session_count += 1
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of the feature-level training arrays."""

        if self._left is None or self._right is None or self._targets is None:
            raise ValueError("calibration dataset is empty")
        return self._left.copy(), self._right.copy(), self._targets.copy()

    def context_dataset(self) -> np.ndarray:
        """Return all stored face-context rows for drift/reference analysis."""

        if self._contexts is None:
            raise ValueError("calibration dataset has no face-context rows")
        return self._contexts.copy()

    def anchor_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of all fixed-target head-pose training frames."""

        if (
            self._anchor_left is None
            or self._anchor_right is None
            or self._anchor_heads is None
            or self._anchor_targets is None
        ):
            raise ValueError("calibration dataset has no head anchor rows")
        return (
            self._anchor_left.copy(),
            self._anchor_right.copy(),
            self._anchor_heads.copy(),
            self._anchor_targets.copy(),
        )

    def clone(self) -> "CumulativeCalibrationDataset":
        """Return a deep validated copy suitable for candidate training."""

        return self.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible feature-level calibration data."""

        if self._left is None or self._right is None or self._targets is None:
            return {
                "left": [],
                "right": [],
                "targets": [],
                "contexts": None,
                "anchor_left": [],
                "anchor_right": [],
                "anchor_heads": [],
                "anchor_targets": [],
                "session_count": 0,
            }
        return {
            "left": self._left.tolist(),
            "right": self._right.tolist(),
            "targets": self._targets.tolist(),
            "contexts": None if self._contexts is None else self._contexts.tolist(),
            "anchor_left": (
                [] if self._anchor_left is None else self._anchor_left.tolist()
            ),
            "anchor_right": (
                [] if self._anchor_right is None else self._anchor_right.tolist()
            ),
            "anchor_heads": (
                [] if self._anchor_heads is None else self._anchor_heads.tolist()
            ),
            "anchor_targets": (
                [] if self._anchor_targets is None else self._anchor_targets.tolist()
            ),
            "session_count": self._session_count,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "CumulativeCalibrationDataset":
        """Restore and validate persisted feature-level calibration data."""

        dataset = cls()
        if payload is None:
            return dataset
        if not isinstance(payload, dict):
            raise ValueError("calibration dataset state must be an object")
        left = np.asarray(payload.get("left", []), dtype=float)
        right = np.asarray(payload.get("right", []), dtype=float)
        targets = np.asarray(payload.get("targets", []), dtype=float)
        contexts_payload = payload.get("contexts")
        contexts = (
            None
            if contexts_payload is None
            else np.asarray(contexts_payload, dtype=float)
        )
        if left.size == 0 and right.size == 0 and targets.size == 0:
            if contexts is not None and contexts.size != 0:
                raise ValueError("empty calibration dataset has unexpected contexts")
            return dataset
        anchor_left = np.asarray(payload.get("anchor_left", []), dtype=float)
        anchor_right = np.asarray(payload.get("anchor_right", []), dtype=float)
        anchor_heads = np.asarray(payload.get("anchor_heads", []), dtype=float)
        anchor_targets = np.asarray(payload.get("anchor_targets", []), dtype=float)
        has_anchors = any(
            value.size > 0
            for value in (anchor_left, anchor_right, anchor_heads, anchor_targets)
        )
        dataset.add(
            left,
            right,
            targets,
            contexts,
            anchor_left=anchor_left if has_anchors else None,
            anchor_right=anchor_right if has_anchors else None,
            anchor_heads=anchor_heads if has_anchors else None,
            anchor_targets=anchor_targets if has_anchors else None,
        )
        try:
            session_count = int(payload.get("session_count", 1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("session_count must be a positive integer") from exc
        if session_count < 1 or session_count != payload.get("session_count", 1):
            raise ValueError("session_count must be a positive integer")
        dataset._session_count = session_count
        return dataset
