from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import cv2
import numpy as np


_UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


@dataclass(frozen=True)
class LetteringPlane:
    camera: np.ndarray
    surface: np.ndarray
    size: tuple[int, int]
    depth_ratio: float

    def projection(self) -> np.ndarray:
        width, height = self.size
        return (self.camera @ self.surface) / np.array([width - 1, height - 1, 1])

    def pixels_per_texel(self) -> float:
        matrix = self.projection()
        width, height = self.size
        point = matrix @ np.array([(width - 1) / 2, (height - 1) / 2, 1])
        derivative = (matrix[:2, :2] - np.outer(point[:2] / point[2], matrix[2, :2])) / point[2]
        return sqrt(abs(float(np.linalg.det(derivative))))


def reconstruct_lettering_plane(quad, image_size, texture_size) -> LetteringPlane | None:
    """Lift a tapered image quadrilateral into a plausible rectangular camera-facing surface."""
    homography = cv2.getPerspectiveTransform(_UNIT_SQUARE, quad)
    if not np.isfinite(homography).all():
        return None
    depth = _UNIT_SQUARE @ homography[2, :2] + homography[2, 2]
    near, far = float(depth.min()), float(depth.max())
    if near <= 0 or not 1.10 <= far / near <= 1.8:
        return None

    width, height = image_size
    cx, cy = (width - 1) / 2, (height - 1) / 2
    focal = float(max(width, height))
    vanishing = homography[:2, :2] - np.outer((cx, cy), homography[2, :2])
    du, dv = homography[2, :2]
    if abs(du) >= .02 and abs(dv) >= .02:
        squared = -float(vanishing[:, 0] @ vanishing[:, 1]) / (du * dv)
        if not (.5 * focal) ** 2 <= squared <= (3 * focal) ** 2:
            return None
        focal = sqrt(squared)
    camera = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
    surface = np.linalg.solve(camera, homography)
    across, down = surface[:, 0], surface[:, 1]
    physical_width, physical_height = np.linalg.norm(across), np.linalg.norm(down)
    if min(physical_width, physical_height) <= 1e-6:
        return None
    if abs(across @ down) > .14 * physical_width * physical_height:
        return None

    texture_width, texture_height = texture_size
    metric_width = round(1 + (texture_height - 1) * physical_width / physical_height)
    if not max(24, .65 * texture_width) <= metric_width <= 1.5 * texture_width:
        return None
    return LetteringPlane(camera, surface, (metric_width, texture_height), far / near)
