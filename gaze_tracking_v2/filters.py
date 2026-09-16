"""Small, explicit filters used by the V2 runtime."""

from __future__ import annotations

from typing import Optional, Sequence


class ExponentialSmoother:
    """EMA filter where alpha directly controls responsiveness."""

    def __init__(self, alpha: float = 0.45) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.value: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self.value = None

    def update(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        if self.value is None:
            self.value = (x, y)
        else:
            previous_x, previous_y = self.value
            self.value = (
                self.alpha * x + (1.0 - self.alpha) * previous_x,
                self.alpha * y + (1.0 - self.alpha) * previous_y,
            )
        return self.value

