"""Regularized gaze mapping models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .correction import CumulativeCorrection


def polynomial_features(points: np.ndarray) -> np.ndarray:
    """Return [1, x, y, x², x*y, y²] for one or more 2D points."""

    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    x = values[:, 0]
    y = values[:, 1]
    return np.column_stack((np.ones(len(values)), x, y, x * x, x * y, y * y))


def gaze_features(points: np.ndarray) -> np.ndarray:
    """Build model features for iris position and compact face context.

    Two-column inputs remain supported for the original V2 tests.  Inputs with
    five or more columns contain iris x/y followed by face context.  Only the
    iris coordinates receive quadratic terms; the whole-face context stays
    linear to avoid overfitting a small personal calibration set.
    """

    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or (values.shape[1] != 2 and values.shape[1] < 5):
        raise ValueError("points must have shape (n, 2) or (n, >=5)")
    if values.shape[1] == 2:
        return polynomial_features(values)
    iris = polynomial_features(values[:, :2])
    return np.column_stack((iris, values[:, 2:]))

def linear_gaze_features(points: np.ndarray) -> np.ndarray:
    """Return a bias plus every raw mapper input as a linear feature.

    V2.6 deliberately keeps the personal model linear. The controlled
    fixed-target head-pose phase supplies the independent variation needed to
    identify pose compensation without relying on polynomial extrapolation.
    """

    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("points must have shape (n, >=2)")
    return np.column_stack((np.ones(len(values)), values))



class RidgePolynomial2D:
    """A small ridge-regularized 2D quadratic regressor.

    The intercept is not regularized.  Keeping this model local avoids adding
    a heavy machine-learning dependency and makes calibration deterministic.
    """

    def __init__(self, ridge: float = 1e-3) -> None:
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.ridge = float(ridge)
        self.coefficients: Optional[np.ndarray] = None
        self.rmse: Optional[float] = None
        self.input_dimension: Optional[int] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None

    model_kind = "polynomial_iris"

    def _feature_matrix(self, points: np.ndarray) -> np.ndarray:
        return gaze_features(points)

    @property
    def ready(self) -> bool:
        return self.coefficients is not None

    def fit(self, points: np.ndarray, targets: np.ndarray) -> "RidgePolynomial2D":
        point_values = np.asarray(points, dtype=float)
        if point_values.ndim == 1:
            point_values = point_values.reshape(1, -1)
        features = self._feature_matrix(point_values)
        values = np.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("targets must have shape (n, 2)")
        if len(features) != len(values):
            raise ValueError("points and targets must have the same length")
        if len(features) < 6:
            raise ValueError("at least 6 calibration points are required")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(values)):
            raise ValueError("calibration data contains non-finite values")

        self.input_dimension = int(point_values.shape[1])
        self.feature_mean = np.zeros(features.shape[1], dtype=float)
        self.feature_scale = np.ones(features.shape[1], dtype=float)
        if features.shape[1] > 1:
            self.feature_mean[1:] = np.mean(features[:, 1:], axis=0)
            self.feature_scale[1:] = np.std(features[:, 1:], axis=0)
            self.feature_scale[1:] = np.maximum(self.feature_scale[1:], 1e-6)
        normalized_features = (features - self.feature_mean) / self.feature_scale

        regularizer = np.eye(normalized_features.shape[1], dtype=float) * self.ridge
        regularizer[0, 0] = 0.0
        normal = normalized_features.T @ normalized_features + regularizer
        right_hand = normalized_features.T @ values
        try:
            self.coefficients = np.linalg.solve(normal, right_hand)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(normal) @ right_hand

        residual = normalized_features @ self.coefficients - values
        self.rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        return self

    def predict(self, point: Sequence[float]) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("mapping model has not been fitted")
        features = self._feature_matrix(np.asarray(point, dtype=float))
        if features.shape[1] != self.coefficients.shape[0]:
            raise ValueError(
                "prediction feature dimension does not match the fitted model"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("prediction point contains non-finite values")
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("mapping feature normalization is missing")
        normalized_features = (features - self.feature_mean) / self.feature_scale
        prediction = (normalized_features @ self.coefficients)[0]
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("mapping model produced a non-finite prediction")
        return prediction

class RidgeLinear2D(RidgePolynomial2D):
    """Standardized linear ridge model used by the V2.6 mapper."""

    model_kind = "linear"

    def _feature_matrix(self, points: np.ndarray) -> np.ndarray:
        return linear_gaze_features(points)



@dataclass(frozen=True)
class GazePrediction:
    """Predictions from the two eye-specific models."""

    left: tuple[float, float]
    right: tuple[float, float]
    gaze: tuple[float, float]


class DualEyeMapper:
    """Fit one small model per eye and average their predictions."""

    def __init__(
        self,
        ridge: float = 1e-3,
        screen_size: Optional[tuple[int, int]] = None,
        model_kind: str = "polynomial_iris",
    ) -> None:
        self.ridge = float(ridge)
        if model_kind not in {"linear", "polynomial_iris"}:
            raise ValueError("model_kind must be 'linear' or 'polynomial_iris'")
        self.model_kind = model_kind
        model_type = RidgeLinear2D if model_kind == "linear" else RidgePolynomial2D
        self.left_model = model_type(ridge)
        self.right_model = model_type(ridge)
        if screen_size is not None:
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
            self.screen_size: Optional[tuple[int, int]] = (width, height)
            self._target_scale = np.array([float(width), float(height)])
        else:
            self.screen_size = None
            self._target_scale = None
        self.output_correction: Optional["CumulativeCorrection"] = None

    def set_output_correction(
        self, correction: Optional["CumulativeCorrection"]
    ) -> None:
        """Attach a calibration-only cumulative correction to predictions."""

        if correction is not None and correction.screen_size != self.screen_size:
            raise ValueError("correction and mapper screen sizes must match")
        self.output_correction = correction

    @property
    def ready(self) -> bool:
        return self.left_model.ready and self.right_model.ready

    @property
    def input_dimension(self) -> Optional[int]:
        if self.left_model.input_dimension != self.right_model.input_dimension:
            return None
        return self.left_model.input_dimension

    def fit(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        targets: np.ndarray,
    ) -> "DualEyeMapper":
        if len(left_points) != len(right_points) or len(left_points) != len(targets):
            raise ValueError("left, right, and target calibration rows must match")
        target_values = np.asarray(targets, dtype=float)
        if target_values.ndim != 2 or target_values.shape[1] != 2:
            raise ValueError("targets must have shape (n, 2)")
        if self._target_scale is not None:
            target_values = target_values / self._target_scale
        self.left_model.fit(left_points, target_values)
        self.right_model.fit(right_points, target_values)
        if self._target_scale is not None:
            self.left_model.rmse = self._pixel_rmse(
                self.left_model, left_points, target_values
            )
            self.right_model.rmse = self._pixel_rmse(
                self.right_model, right_points, target_values
            )
        return self

    def leave_one_out_predictions(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Predict every calibration point with a model that omitted it.

        These out-of-fold predictions provide useful residuals for the
        cumulative correction. Using in-sample predictions would make a
        sufficiently flexible calibration model report almost no bias.
        """

        left_values = np.asarray(left_points, dtype=float)
        right_values = np.asarray(right_points, dtype=float)
        target_values = np.asarray(targets, dtype=float)
        if (
            left_values.ndim != 2
            or right_values.ndim != 2
            or target_values.ndim != 2
            or left_values.shape != right_values.shape
            or target_values.shape != (len(left_values), 2)
            or len(left_values) < 7
        ):
            raise ValueError("leave-one-out data must contain at least 7 matched rows")
        if not np.all(np.isfinite(left_values)) or not np.all(
            np.isfinite(right_values)
        ) or not np.all(np.isfinite(target_values)):
            raise ValueError("leave-one-out data must be finite")

        predictions = []
        for omitted in range(len(left_values)):
            mask = np.ones(len(left_values), dtype=bool)
            mask[omitted] = False
            candidate = DualEyeMapper(
                ridge=self.ridge,
                screen_size=self.screen_size,
                model_kind=self.model_kind,
            ).fit(
                left_values[mask],
                right_values[mask],
                target_values[mask],
            )
            predictions.append(
                candidate.predict(
                    left_values[omitted], right_values[omitted]
                ).gaze
            )
        return np.asarray(predictions, dtype=float)

    def _pixel_rmse(
        self,
        model: RidgePolynomial2D,
        points: np.ndarray,
        normalized_targets: np.ndarray,
    ) -> float:
        predictions = np.asarray(
            [model.predict(point) for point in np.asarray(points, dtype=float)],
            dtype=float,
        )
        residual = (predictions - normalized_targets) * self._target_scale
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    def _to_pixels(self, point: np.ndarray) -> np.ndarray:
        if self._target_scale is None:
            return point
        return point * self._target_scale

    def predict(
        self, left_point: Sequence[float], right_point: Sequence[float]
    ) -> GazePrediction:
        left = self._to_pixels(self.left_model.predict(left_point))
        right = self._to_pixels(self.right_model.predict(right_point))
        gaze = (left + right) / 2.0
        if self.output_correction is not None:
            gaze = np.asarray(self.output_correction.apply(gaze), dtype=float)
        return GazePrediction(
            left=(float(left[0]), float(left[1])),
            right=(float(right[0]), float(right[1])),
            gaze=(float(gaze[0]), float(gaze[1])),
        )


class HeadCompensatedMapper:
    """Two-stage mapper: eye-only base mapping plus linear head residual."""

    model_kind = "two_stage_head"
    # Five relative movement values plus three session-reference values.
    head_dimension = 8

    def __init__(
        self, ridge: float = 0.05, screen_size: Optional[tuple[int, int]] = None
    ) -> None:
        self.ridge = float(ridge)
        self.screen_size = screen_size
        self.base_mapper = DualEyeMapper(
            ridge=ridge, screen_size=screen_size, model_kind="polynomial_iris"
        )
        self.compensation_model = RidgeLinear2D(ridge)
        self.output_correction: Optional["CumulativeCorrection"] = None
        self.head_min: Optional[np.ndarray] = None
        self.head_max: Optional[np.ndarray] = None

    @property
    def left_model(self) -> RidgePolynomial2D:
        return self.base_mapper.left_model

    @property
    def right_model(self) -> RidgePolynomial2D:
        return self.base_mapper.right_model

    @property
    def input_dimension(self) -> int:
        return 2 + self.head_dimension

    @property
    def ready(self) -> bool:
        return self.base_mapper.ready and self.compensation_model.ready and self.head_min is not None and self.head_max is not None

    def set_output_correction(
        self, correction: Optional["CumulativeCorrection"]
    ) -> None:
        if correction is not None and correction.screen_size != self.screen_size:
            raise ValueError("correction and mapper screen sizes must match")
        self.output_correction = correction

    def fit(
        self,
        grid_left: np.ndarray,
        grid_right: np.ndarray,
        grid_targets: np.ndarray,
        anchor_left: np.ndarray,
        anchor_right: np.ndarray,
        anchor_heads: np.ndarray,
        anchor_targets: np.ndarray,
    ) -> "HeadCompensatedMapper":
        self.base_mapper.fit(grid_left, grid_right, grid_targets)
        left_values = np.asarray(anchor_left, dtype=float)
        right_values = np.asarray(anchor_right, dtype=float)
        head_values = np.asarray(anchor_heads, dtype=float)
        target_values = np.asarray(anchor_targets, dtype=float)
        if (
            left_values.ndim != 2
            or left_values.shape[1] != 2
            or right_values.shape != left_values.shape
            or head_values.shape != (len(left_values), self.head_dimension)
            or target_values.shape != (len(left_values), 2)
        ):
            raise ValueError("anchor data dimensions do not match the two-stage mapper")
        if not np.all(np.isfinite(head_values)):
            raise ValueError("head calibration data contains non-finite values")
        spans = np.ptp(head_values, axis=0)
        required_spans = np.array([0.25, 0.25, 0.12, 0.12, 0.10])
        feature_names = (
            "yaw", "pitch", "horizontal position", "vertical position", "distance"
        )
        missing = [
            name
            for name, span, required in zip(
                feature_names, spans[:5], required_spans
            )
            if span < required
        ]
        if missing:
            raise ValueError(
                "head calibration coverage is insufficient: " + ", ".join(missing)
            )
        # A single session keeps its three reference-context values constant.
        # Require the five relative axes to be identifiable; later sessions
        # add the variation needed to learn reference position/distance terms.
        if np.linalg.matrix_rank(linear_gaze_features(head_values)) < 6:
            raise ValueError("head calibration features are not independently identifiable")
        self.head_min = np.min(head_values, axis=0)
        self.head_max = np.max(head_values, axis=0)
        base_predictions = np.asarray(
            [
                self.base_mapper.predict(left, right).gaze
                for left, right in zip(left_values, right_values)
            ],
            dtype=float,
        )
        residual_targets = target_values - base_predictions
        self.compensation_model.fit(head_values, residual_targets)
        return self

    def predict(
        self,
        left_eye: Sequence[float],
        right_eye: Sequence[float],
        head: Sequence[float],
    ) -> GazePrediction:
        base = self.base_mapper.predict(left_eye, right_eye)
        if self.head_min is None or self.head_max is None:
            raise RuntimeError("head calibration range is missing")
        head_value = np.asarray(head, dtype=float)
        if head_value.shape != (self.head_dimension,) or not np.all(
            np.isfinite(head_value)
        ):
            raise ValueError(
                f"head features must be a finite {self.head_dimension}-value vector"
            )
        bounded_head = np.clip(head_value, self.head_min, self.head_max)
        correction = self.compensation_model.predict(bounded_head)
        left = np.asarray(base.left, dtype=float) + correction
        right = np.asarray(base.right, dtype=float) + correction
        gaze = (left + right) / 2.0
        if self.output_correction is not None:
            gaze = np.asarray(self.output_correction.apply(gaze), dtype=float)
        return GazePrediction(
            left=(float(left[0]), float(left[1])),
            right=(float(right[0]), float(right[1])),
            gaze=(float(gaze[0]), float(gaze[1])),
        )
