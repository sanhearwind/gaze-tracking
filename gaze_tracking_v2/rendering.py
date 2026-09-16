"""Render overlays at display resolution while retaining camera-space coordinates."""
from __future__ import annotations

import cv2


class DisplayCanvas:
    def __init__(self, frame, display_size):
        height, width = frame.shape[:2]
        self.scale = max(0.1, min(display_size[0] / width, display_size[1] / height))
        self.image = cv2.resize(frame, (max(1, round(width * self.scale)),
                                       max(1, round(height * self.scale))),
                                interpolation=cv2.INTER_LINEAR)
        # Layout and model coordinates remain in the original camera space.
        self.shape = frame.shape
        self.drawing = _ScaledDrawing(self)


class _ScaledDrawing:
    def __init__(self, canvas):
        self.canvas = canvas

    def __getattr__(self, name):
        return getattr(cv2, name)

    def point(self, value):
        return tuple(round(v * self.canvas.scale) for v in value)

    def thickness(self, value):
        return value if value < 0 else max(1, round(value * self.canvas.scale))

    def circle(self, image, center, radius, color, thickness, lineType=cv2.LINE_AA):
        return cv2.circle(image.image, self.point(center),
                          max(1, round(radius * image.scale)), color,
                          self.thickness(thickness), lineType)

    def line(self, image, start, end, color, thickness, lineType=cv2.LINE_AA):
        return cv2.line(image.image, self.point(start), self.point(end), color,
                        self.thickness(thickness), lineType)

    def rectangle(self, image, start, end, color, thickness):
        return cv2.rectangle(image.image, self.point(start), self.point(end), color,
                             self.thickness(thickness))

    def ellipse(self, image, center, axes, angle, start, end, color, thickness,
                lineType=cv2.LINE_AA):
        return cv2.ellipse(image.image, self.point(center), self.point(axes),
                           angle, start, end, color, self.thickness(thickness), lineType)

    def putText(self, image, text, origin, font, scale, color, thickness,
                lineType=cv2.LINE_AA):
        return cv2.putText(image.image, text, self.point(origin), font,
                           scale * image.scale, color, self.thickness(thickness), lineType)

    def getTextSize(self, text, font, scale, thickness):
        size, baseline = cv2.getTextSize(text, font, scale * self.canvas.scale,
                                        self.thickness(thickness))
        return (tuple(round(v / self.canvas.scale) for v in size),
                round(baseline / self.canvas.scale))
