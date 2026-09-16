"""Head-pose quality gating, kept separate from gaze-coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    tx: Optional[float] = None
    ty: Optional[float] = None
    tz: Optional[float] = None
    reprojection_error_px: Optional[float] = None


@dataclass(frozen=True)
class PoseGateState:
    pose: Optional[HeadPose]
    relative_pose: Optional[HeadPose]
    reference_ready: bool
    candidate_allowed: bool
    allowed: bool
    started: bool
    ended: bool


class HeadPoseGate:
    """Accept only stable head poses relative to a neutral reference.

    This is deliberately a gate, not a gaze model feature.  If a user turns
    too far from the reference pose, the application reports an invalid frame
    instead of interpreting head motion as a change in visual focus.  A
    reference is collected after reset because approximate PnP coordinates do
    not guarantee that a frontal face has absolute angles of exactly zero.
    """

    def __init__(
        self,
        *,
        max_abs_yaw: float = 25.0,
        max_abs_pitch: float = 20.0,
        max_abs_roll: float = 20.0,
        open_frames: int = 2,
        close_frames: int = 2,
        reference_frames: int = 1,
    ) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (max_abs_yaw, max_abs_pitch, max_abs_roll)
            )
            or min(max_abs_yaw, max_abs_pitch, max_abs_roll) <= 0
        ):
            raise ValueError("pose limits must be positive")
        try:
            open_limit = int(open_frames)
            close_limit = int(close_frames)
            reference_limit = int(reference_frames)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frame counts must be positive integers") from exc
        if (
            open_limit != open_frames
            or close_limit != close_frames
            or reference_limit != reference_frames
            or open_limit < 1
            or close_limit < 1
        ):
            raise ValueError("open_frames and close_frames must be positive")
        if reference_limit < 1:
            raise ValueError("reference_frames must be positive")
        self.max_abs_yaw = float(max_abs_yaw)
        self.max_abs_pitch = float(max_abs_pitch)
        self.max_abs_roll = float(max_abs_roll)
        self.open_frames = open_limit
        self.close_frames = close_limit
        self.reference_frames = reference_limit
        self.reset()

    def reset(self) -> None:
        self.allowed = False
        self._good_count = 0
        self._bad_count = 0
        self._reference_samples: list[HeadPose] = []
        self.reference: Optional[HeadPose] = None

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _circular_mean(values: list[float]) -> float:
        radians = [math.radians(value) for value in values]
        sine = sum(math.sin(value) for value in radians)
        cosine = sum(math.cos(value) for value in radians)
        return float(math.degrees(math.atan2(sine, cosine)))

    @staticmethod
    def _median_optional(values: list[Optional[float]]) -> Optional[float]:
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return None
        finite.sort()
        middle = len(finite) // 2
        if len(finite) % 2:
            return finite[middle]
        return (finite[middle - 1] + finite[middle]) / 2.0

    @staticmethod
    def _reference_is_stable(values: list[HeadPose]) -> bool:
        """Reject a neutral reference collected while the user is moving."""

        for name, limit in (("yaw", 2.0), ("pitch", 2.0), ("roll", 2.5)):
            base = float(getattr(values[0], name))
            series = [
                HeadPoseGate._wrap_degrees(float(getattr(item, name)) - base)
                for item in values
            ]
            if max(series) - min(series) > limit:
                return False
        for name, limit in (("tx", 12.0), ("ty", 12.0), ("tz", 20.0)):
            series = [
                float(value)
                for item in values
                for value in (getattr(item, name),)
                if value is not None
            ]
            if len(series) == len(values) and max(series) - min(series) > limit:
                return False
        return True

    def _relative_pose(self, pose: Optional[HeadPose]) -> Optional[HeadPose]:
        if pose is None or self.reference is None:
            return None
        translation = tuple(
            None
            if current is None or reference is None
            else float(current - reference)
            for current, reference in (
                (pose.tx, self.reference.tx),
                (pose.ty, self.reference.ty),
                (pose.tz, self.reference.tz),
            )
        )
        return HeadPose(
            yaw=self._wrap_degrees(pose.yaw - self.reference.yaw),
            pitch=self._wrap_degrees(pose.pitch - self.reference.pitch),
            roll=self._wrap_degrees(pose.roll - self.reference.roll),
            tx=translation[0],
            ty=translation[1],
            tz=translation[2],
            reprojection_error_px=pose.reprojection_error_px,
        )

    def _within_limits(self, pose: Optional[HeadPose]) -> bool:
        relative = self._relative_pose(pose)
        if relative is None:
            return False
        return (
            abs(relative.yaw) <= self.max_abs_yaw
            and abs(relative.pitch) <= self.max_abs_pitch
            and abs(relative.roll) <= self.max_abs_roll
        )

    def update(self, pose: Optional[HeadPose]) -> PoseGateState:
        if pose is not None:
            pose_values = [pose.yaw, pose.pitch, pose.roll]
            pose_values.extend(
                value
                for value in (
                    pose.tx,
                    pose.ty,
                    pose.tz,
                    pose.reprojection_error_px,
                )
                if value is not None
            )
            if not all(math.isfinite(value) for value in pose_values):
                pose = None
        if self.reference is None:
            if pose is None:
                self._reference_samples.clear()
            else:
                self._reference_samples.append(pose)
                self._reference_samples = self._reference_samples[-self.reference_frames:]
                if (
                    len(self._reference_samples) >= self.reference_frames
                    and self._reference_is_stable(self._reference_samples)
                ):
                    values = self._reference_samples
                    self.reference = HeadPose(
                        yaw=self._circular_mean([item.yaw for item in values]),
                        pitch=self._circular_mean([item.pitch for item in values]),
                        roll=self._circular_mean([item.roll for item in values]),
                        tx=self._median_optional([item.tx for item in values]),
                        ty=self._median_optional([item.ty for item in values]),
                        tz=self._median_optional([item.tz for item in values]),
                    )

        candidate_allowed = self._within_limits(pose)
        started = False
        ended = False
        if candidate_allowed:
            self._good_count += 1
            self._bad_count = 0
            if not self.allowed and self._good_count >= self.open_frames:
                self.allowed = True
                started = True
        else:
            self._bad_count += 1
            self._good_count = 0
            if self.allowed and self._bad_count >= self.close_frames:
                self.allowed = False
                ended = True
        return PoseGateState(
            pose=pose,
            relative_pose=self._relative_pose(pose),
            reference_ready=self.reference is not None,
            candidate_allowed=candidate_allowed,
            allowed=self.allowed,
            started=started,
            ended=ended,
        )
