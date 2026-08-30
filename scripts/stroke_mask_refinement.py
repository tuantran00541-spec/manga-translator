"""Research-only stroke-aware expansion for verified text masks.

This module intentionally lives under scripts/ so the experiment is not part of
the production app import graph. Detector segmentation remains the seed
authority; this helper only tests conservative bounded expansion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class StrokeRefinementStats:
    components: int
    source_pixels: int
    refined_pixels: int
    max_radius_used: int
    mean_radius_used: float

    @property
    def growth_ratio(self) -> float:
        if self.source_pixels <= 0:
            return 1.0
        return self.refined_pixels / self.source_pixels


def _as_gray(image: np.ndarray | None) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported crop image shape: {image.shape}")


def _component_radius(component: np.ndarray, min_radius: int, max_radius: int) -> int:
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    values = distance[component > 0]
    if values.size == 0:
        return min_radius
    half_width = float(np.percentile(values, 70.0))
    radius = int(math.ceil(max(1.0, half_width * 0.80)))
    return int(np.clip(radius, min_radius, max_radius))


def _guard_radius_for_artwork(
    gray_roi: np.ndarray | None,
    component: np.ndarray,
    radius: int,
    min_radius: int,
) -> int:
    if gray_roi is None or radius <= min_radius:
        return radius
    outer_r = radius + 2
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_r * 2 + 1, outer_r * 2 + 1)
    )
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outer = cv2.dilate(component, outer_kernel, iterations=1) > 0
    inner = cv2.dilate(component, inner_kernel, iterations=1) > 0
    ring = outer & ~inner
    if int(np.count_nonzero(ring)) < 16:
        return radius
    ring_values = gray_roi[ring]
    ring_std = float(ring_values.std())
    edges = cv2.Canny(gray_roi, 64, 128, L2gradient=True) > 0
    edge_density = float(edges[ring].mean())
    if ring_std >= 38.0 or edge_density >= 0.14:
        return max(min_radius, radius - 2)
    if ring_std >= 24.0 or edge_density >= 0.08:
        return max(min_radius, radius - 1)
    return radius


def refine_stroke_mask(
    mask: np.ndarray,
    crop_img: np.ndarray | None = None,
    *,
    safe_envelope: np.ndarray | None = None,
    min_radius: int = 1,
    max_radius: int = 6,
    complexity_guard: bool = True,
) -> tuple[np.ndarray, StrokeRefinementStats]:
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D numpy array")
    if min_radius < 0 or max_radius < min_radius:
        raise ValueError("invalid dilation radius bounds")

    source = (mask > 127).astype(np.uint8) * 255
    source_pixels = int(np.count_nonzero(source))
    if source_pixels == 0:
        return source, StrokeRefinementStats(0, 0, 0, 0, 0.0)

    gray = _as_gray(crop_img)
    if gray is not None and gray.shape != source.shape:
        raise ValueError("crop_img and mask must have the same height/width")

    envelope_bool: np.ndarray | None = None
    if safe_envelope is not None:
        if safe_envelope.shape != source.shape:
            raise ValueError("safe_envelope and mask must have the same shape")
        envelope_bool = safe_envelope > 127

    num_labels, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        source, connectivity=8
    )
    refined = source.copy()
    radii: list[int] = []
    pad = max_radius + 3
    image_h, image_w = source.shape

    for label in range(1, num_labels):
        x, y, width, height, area = (int(v) for v in component_stats[label])
        if area <= 0 or width <= 0 or height <= 0:
            continue
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_w, x + width + pad)
        y2 = min(image_h, y + height + pad)
        component = (labels[y1:y2, x1:x2] == label).astype(np.uint8)
        radius = _component_radius(component, min_radius, max_radius)
        if complexity_guard and gray is not None:
            radius = _guard_radius_for_artwork(
                gray[y1:y2, x1:x2], component, radius, min_radius
            )
        radii.append(radius)
        if radius <= 0:
            continue
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        expanded = cv2.dilate(component, kernel, iterations=1) > 0
        if envelope_bool is not None:
            expanded &= envelope_bool[y1:y2, x1:x2]
        refined[y1:y2, x1:x2][expanded] = 255

    refined[source > 0] = 255
    refined_pixels = int(np.count_nonzero(refined))
    return refined, StrokeRefinementStats(
        components=len(radii),
        source_pixels=source_pixels,
        refined_pixels=refined_pixels,
        max_radius_used=max(radii, default=0),
        mean_radius_used=float(np.mean(radii)) if radii else 0.0,
    )
