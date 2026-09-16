"""Cumulative output correction learned only from known calibration targets."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


class CumulativeCorrection:
    """Accumulate global and smooth vertical spatial bias corrections.

    The correction is trained from mapper predictions and their known
    calibration targets. It stores observations across completed calibration
    sessions in the current process, so repeated calibrations refine the same
    affine residual model and a one-dimensional PCHIP residual curve instead
    of throwing the previous bias away. The evaluation target is never passed
    to this class.
    """

    def __init__(self, screen_size: tuple[int, int], ridge: float = 1e-2) -> None:
        dimensions = np.asarray(screen_size, dtype=float)
        if (
            dimensions.shape != (2,)
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions <= 0)
        ):
            raise ValueError("screen_size must contain positive dimensions")
        width, height = int(dimensions[0]), int(dimensions[1])
        if width <= 0 or height <= 0:
            raise ValueError("screen_size must contain positive dimensions")
        if not math.isfinite(float(ridge)) or ridge < 0:
            raise ValueError("ridge must be non-negative and finite")

        self.screen_size = (width, height)
        self._target_scale = np.asarray([float(width), float(height)], dtype=float)
        self.ridge = float(ridge)
        self._predicted = np.empty((0, 2), dtype=float)
        self._targets = np.empty((0, 2), dtype=float)
        self._coefficients: Optional[np.ndarray] = None
        self._vertical_x: Optional[np.ndarray] = None
        self._vertical_values: Optional[np.ndarray] = None
        self._vertical_slopes: Optional[np.ndarray] = None

    @property
    def sample_count(self) -> int:
        return len(self._targets)

    @property
    def ready(self) -> bool:
        return self._coefficients is not None

    def reset(self) -> None:
        """Clear accumulated sessions, normally when the user presses reset."""

        self._predicted = np.empty((0, 2), dtype=float)
        self._targets = np.empty((0, 2), dtype=float)
        self._coefficients = None
        self._vertical_x = None
        self._vertical_values = None
        self._vertical_slopes = None

    @staticmethod
    def _affine_features(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        return np.column_stack((np.ones(len(values)), values[:, 0], values[:, 1]))

    @staticmethod
    def _fit_pchip(
        x_values: np.ndarray, y_values: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit monotone cubic slopes for a one-dimensional residual curve."""

        order = np.argsort(x_values)
        sorted_x = np.asarray(x_values, dtype=float)[order]
        sorted_y = np.asarray(y_values, dtype=float)[order]
        unique_x: list[float] = []
        grouped_y: list[list[float]] = []
        for x_value, y_value in zip(sorted_x, sorted_y):
            if unique_x and abs(float(x_value) - unique_x[-1]) <= 1e-6:
                grouped_y[-1].append(float(y_value))
            else:
                unique_x.append(float(x_value))
                grouped_y.append([float(y_value)])
        knots = np.asarray(unique_x, dtype=float)
        values = np.asarray([np.mean(group) for group in grouped_y], dtype=float)
        if len(knots) < 2:
            return None, None, None

        intervals = np.diff(knots)
        secants = np.diff(values) / intervals
        slopes = np.zeros(len(knots), dtype=float)

        def endpoint_slope(
            first_interval: float,
            second_interval: float,
            first_secant: float,
            second_secant: float,
        ) -> float:
            slope = (
                (2.0 * first_interval + second_interval) * first_secant
                - first_interval * second_secant
            ) / (first_interval + second_interval)
            if slope * first_secant <= 0.0:
                return 0.0
            if first_secant * second_secant < 0.0 and abs(slope) > abs(3.0 * first_secant):
                return 3.0 * first_secant
            return slope

        if len(knots) == 2:
            slopes[:] = secants[0]
        else:
            slopes[0] = endpoint_slope(
                intervals[0], intervals[1], secants[0], secants[1]
            )
            slopes[-1] = endpoint_slope(
                intervals[-1], intervals[-2], secants[-1], secants[-2]
            )
            for index in range(1, len(knots) - 1):
                previous_secant = secants[index - 1]
                next_secant = secants[index]
                if previous_secant * next_secant <= 0.0:
                    slopes[index] = 0.0
                else:
                    weight_left = 2.0 * intervals[index] + intervals[index - 1]
                    weight_right = intervals[index] + 2.0 * intervals[index - 1]
                    slopes[index] = (weight_left + weight_right) / (
                        weight_left / previous_secant + weight_right / next_secant
                    )
        return knots, values, slopes

    @staticmethod
    def _pchip_eval(
        value: float,
        knots: np.ndarray,
        values: np.ndarray,
        slopes: np.ndarray,
    ) -> float:
        clamped = float(np.clip(value, knots[0], knots[-1]))
        index = int(np.searchsorted(knots, clamped, side="right") - 1)
        index = max(0, min(index, len(knots) - 2))
        interval = knots[index + 1] - knots[index]
        fraction = (clamped - knots[index]) / interval
        fraction_squared = fraction * fraction
        fraction_cubed = fraction_squared * fraction
        h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
        h10 = fraction_cubed - 2.0 * fraction_squared + fraction
        h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
        h11 = fraction_cubed - fraction_squared
        return float(
            h00 * values[index]
            + h10 * interval * slopes[index]
            + h01 * values[index + 1]
            + h11 * interval * slopes[index + 1]
        )

    def _fit(self) -> None:
        if len(self._targets) < 3:
            self._coefficients = None
            self._vertical_x = None
            self._vertical_values = None
            self._vertical_slopes = None
            return
        normalized_predictions = self._predicted / self._target_scale
        residuals = (self._targets - self._predicted) / self._target_scale
        features = self._affine_features(normalized_predictions)
        regularizer = np.eye(features.shape[1], dtype=float) * self.ridge
        regularizer[0, 0] = 0.0
        normal = features.T @ features + regularizer
        right_hand = features.T @ residuals
        try:
            self._coefficients = np.linalg.solve(normal, right_hand)
        except np.linalg.LinAlgError:
            self._coefficients = np.linalg.pinv(normal) @ right_hand

        affine_predictions = self._predicted + (
            features @ self._coefficients
        ) * self._target_scale
        vertical_residuals = (
            self._targets[:, 1] - affine_predictions[:, 1]
        ) / self._target_scale[1]
        (
            self._vertical_x,
            self._vertical_values,
            self._vertical_slopes,
        ) = self._fit_pchip(
            affine_predictions[:, 1] / self._target_scale[1],
            vertical_residuals,
        )

    def add(
        self,
        predicted: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[Sequence[float]] | np.ndarray,
    ) -> "CumulativeCorrection":
        """Add one calibration session and refit the cumulative correction."""

        predicted_values = np.asarray(predicted, dtype=float)
        target_values = np.asarray(targets, dtype=float)
        if (
            predicted_values.ndim != 2
            or target_values.ndim != 2
            or predicted_values.shape != target_values.shape
            or predicted_values.shape[1] != 2
            or len(predicted_values) == 0
        ):
            raise ValueError("predicted and targets must both have shape (n>=1, 2)")
        if not np.all(np.isfinite(predicted_values)) or not np.all(
            np.isfinite(target_values)
        ):
            raise ValueError("predicted and targets must be finite")

        self._predicted = np.vstack((self._predicted, predicted_values))
        self._targets = np.vstack((self._targets, target_values))
        self._fit()
        return self

    def replace(
        self,
        predicted: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[Sequence[float]] | np.ndarray,
    ) -> "CumulativeCorrection":
        """Replace observations after a mapper is refit.

        Correction samples belong to a mapper's output coordinate system.  A
        recalibration changes that system, so old output-space samples must not
        be blended with residuals from the new mapper.  The application calls
        this with leave-one-out predictions from the jointly refit dataset.
        """

        predicted_values = np.asarray(predicted, dtype=float)
        target_values = np.asarray(targets, dtype=float)
        if (
            predicted_values.ndim != 2
            or target_values.ndim != 2
            or predicted_values.shape != target_values.shape
            or predicted_values.shape[1] != 2
            or len(predicted_values) == 0
        ):
            raise ValueError("predicted and targets must both have shape (n>=1, 2)")
        if not np.all(np.isfinite(predicted_values)) or not np.all(
            np.isfinite(target_values)
        ):
            raise ValueError("predicted and targets must be finite")
        self._predicted = predicted_values.copy()
        self._targets = target_values.copy()
        self._fit()
        return self

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible accumulated observations and parameters."""

        return {
            "screen_size": list(self.screen_size),
            "ridge": self.ridge,
            "predicted": self._predicted.tolist(),
            "targets": self._targets.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        expected_screen_size: Optional[tuple[int, int]] = None,
    ) -> "CumulativeCorrection":
        """Restore a correction after validating its screen and finite data."""

        if not isinstance(payload, dict):
            raise ValueError("correction state must be an object")
        dimensions = np.asarray(payload.get("screen_size"), dtype=float)
        if (
            dimensions.shape != (2,)
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions <= 0)
            or not np.all(dimensions == np.floor(dimensions))
        ):
            raise ValueError("correction state has invalid screen_size")
        screen_size = (int(dimensions[0]), int(dimensions[1]))
        if expected_screen_size is not None and screen_size != expected_screen_size:
            raise ValueError("correction state screen_size does not match camera")
        try:
            ridge = float(payload.get("ridge", 1e-2))
        except (TypeError, ValueError) as exc:
            raise ValueError("correction state has invalid ridge") from exc

        correction = cls(screen_size=screen_size, ridge=ridge)
        predicted = np.asarray(payload.get("predicted", []), dtype=float)
        targets = np.asarray(payload.get("targets", []), dtype=float)
        if predicted.size == 0 and targets.size == 0:
            return correction
        correction.add(predicted, targets)
        return correction

    def apply(self, point: Sequence[float]) -> tuple[float, float]:
        """Apply affine correction followed by the smooth vertical residual."""

        values = np.asarray(point, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("point must contain two finite values")
        if self._coefficients is None:
            return float(values[0]), float(values[1])

        normalized = values / self._target_scale
        delta = self._affine_features(normalized) @ self._coefficients
        corrected = values + delta[0] * self._target_scale
        if (
            self._vertical_x is not None
            and self._vertical_values is not None
            and self._vertical_slopes is not None
        ):
            vertical_delta = self._pchip_eval(
                corrected[1] / self._target_scale[1],
                self._vertical_x,
                self._vertical_values,
                self._vertical_slopes,
            )
            corrected[1] += vertical_delta * self._target_scale[1]
        if not np.all(np.isfinite(corrected)):
            raise RuntimeError("cumulative correction produced a non-finite point")
        return float(corrected[0]), float(corrected[1])
