"""Data types shared across the pose estimation domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    """A single keypoint with location, confidence, and COCO visibility.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        confidence: Model confidence score (0.0–1.0). For ground truth, set to 1.0.
        visibility: COCO convention — 0: unlabeled, 1: labeled but not visible, 2: labeled and visible.
    """

    x: float
    y: float
    confidence: float = 1.0
    visibility: int = 2

    def to_coco_triple(self) -> tuple[float, float, int]:
        """Return (x, y, visibility) as used in COCO JSON."""
        return (self.x, self.y, self.visibility)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in COCO format [x, y, width, height].

    All values are in pixels, with (x, y) at the top-left corner.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return (x, y, w, h) — COCO format."""
        return (self.x, self.y, self.w, self.h)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    @classmethod
    def from_keypoints(cls, keypoints: list[Keypoint], padding: float = 0.1) -> BBox:
        """Compute a bounding box that encloses all visible keypoints.

        Args:
            keypoints: List of keypoints. Only those with visibility > 0 are used.
            padding: Fractional padding added to each side.
        """
        visible = [(kp.x, kp.y) for kp in keypoints if kp.visibility > 0]
        if not visible:
            return cls(0.0, 0.0, 0.0, 0.0)
        xs, ys = zip(*visible)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        pad_x = w * padding
        pad_y = h * padding
        return cls(
            x=x_min - pad_x,
            y=y_min - pad_y,
            w=w + 2 * pad_x,
            h=h + 2 * pad_y,
        )


@dataclass
class PoseResult:
    """Pose estimation result for a single person/character instance.

    Attributes:
        keypoints: Exactly 17 keypoints in COCO17 order.
        bbox: Bounding box in COCO [x, y, w, h] format.
        score: Overall detection/pose confidence.
        metadata: Arbitrary metadata for evaluation (body_type, frame_type, render_type, etc.).
    """

    keypoints: list[Keypoint]
    bbox: BBox
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_visible(self) -> int:
        """Number of keypoints with visibility == 2 (labeled and visible)."""
        return sum(1 for kp in self.keypoints if kp.visibility == 2)

    def keypoints_array(self) -> np.ndarray:
        """Return keypoints as (17, 3) array of [x, y, visibility]."""
        return np.array([kp.to_coco_triple() for kp in self.keypoints], dtype=np.float32)

    def to_coco_keypoints(self) -> list[float]:
        """Return flat list of [x1, y1, v1, x2, y2, v2, ...] for COCO JSON."""
        flat: list[float] = []
        for kp in self.keypoints:
            flat.extend(kp.to_coco_triple())
        return flat
