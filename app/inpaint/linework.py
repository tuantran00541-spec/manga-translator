from __future__ import annotations

import cv2
import numpy as np

_ROI_PAD_PX = 64
_MAX_EDGE_SEEDS = 3000
_MAX_RAY_LENGTH = 48
_MIN_NEAR_PIXELS = 24
_MIN_GRADIENT = 17.0
_MIN_CONTRAST = 30.0
_MAX_MISMATCH = 14.0
_MAX_MISMATCH_RATIO = 0.2


def _pixel_coordinates(points: np.ndarray, shape: tuple[int, int]):
    rounded = np.rint(points).astype(np.int32)
    x, y = rounded[..., 0], rounded[..., 1]
    valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    return x.clip(0, shape[1] - 1), y.clip(0, shape[0] - 1), valid


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)


def continue_linework(source, painted, write_mask, hole=None):
    """Interpolate narrow gaps where the same oriented ink profile is visible on both sides."""
    if (
        source is None
        or painted is None
        or write_mask is None
        or source.ndim not in (2, 3)
        or source.shape != painted.shape
        or write_mask.ndim != 2
        or write_mask.shape != source.shape[:2]
    ):
        return painted
    if hole is not None and (hole.ndim != 2 or hole.shape != source.shape[:2]):
        return painted

    write_full = (
        write_mask if write_mask.dtype == np.bool_ else (write_mask > 127)
    )
    wy, wx = np.nonzero(write_full)
    if wy.size == 0:
        return painted

    full_h, full_w = source.shape[:2]
    y0 = max(0, int(wy.min()) - _ROI_PAD_PX)
    y1 = min(full_h, int(wy.max()) + 1 + _ROI_PAD_PX)
    x0 = max(0, int(wx.min()) - _ROI_PAD_PX)
    x1 = min(full_w, int(wx.max()) + 1 + _ROI_PAD_PX)

    src = source[y0:y1, x0:x1]
    pnt = painted[y0:y1, x0:x1]
    write = write_full[y0:y1, x0:x1]
    if hole is None:
        hidden = write
    else:
        hole_roi = hole[y0:y1, x0:x1]
        hidden = write | (hole_roi if hole_roi.dtype == np.bool_ else (hole_roi > 127))

    erode_border = (
        cv2.BORDER_CONSTANT
        if (x0 == 0 or y0 == 0 or x1 == full_w or y1 == full_h)
        else cv2.BORDER_REPLICATE
    )
    known = (
        cv2.erode(
            (~hidden).astype(np.uint8),
            np.ones((5, 5), np.uint8),
            borderType=erode_border,
            borderValue=0,
        )
        != 0
    )
    if x0 > 0:
        known[:, :2] &= ~hidden[:, :2]
    if y0 > 0:
        known[:2, :] &= ~hidden[:2, :]
    if x1 < full_w:
        known[:, -2:] &= ~hidden[:, -2:]
    if y1 < full_h:
        known[-2:, :] &= ~hidden[-2:, :]

    near = (
        cv2.dilate(
            write.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        )
        > 0
    ) & known
    if np.count_nonzero(near) < _MIN_NEAR_PIXELS:
        return painted

    gray_src = _to_gray(src)
    smooth = cv2.GaussianBlur(gray_src, (3, 3), 0.65)
    edges = cv2.Canny(smooth, 50, 100, L2gradient=True) > 0
    y, x = np.nonzero(edges & near)
    if x.size <= 1:
        return painted

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=0.125)
    magnitude = cv2.magnitude(gx, gy)
    if x.size > _MAX_EDGE_SEEDS:
        top_idx = np.argpartition(magnitude[y, x], -_MAX_EDGE_SEEDS)[-_MAX_EDGE_SEEDS:]
        y, x = y[top_idx], x[top_idx]

    normal = np.column_stack((gx[y, x], gy[y, x])) / magnitude[y, x, None].clip(min=1.0)
    start = np.column_stack((x, y)).astype(np.float32)
    forward = normal[:, ::-1].copy()
    forward[:, 0] *= -1
    probe = (
        start[:, None, :]
        + forward[:, None, :]
        * np.array([-10, -7, -4, 4, 7, 10], np.float32)[None, :, None]
    )
    qx, qy, in_bounds = _pixel_coordinates(probe, write.shape)
    crosses = hidden[qy, qx] & in_bounds
    left, right = crosses[:, :3].any(axis=1), crosses[:, 3:].any(axis=1)
    keep = (left ^ right) & (magnitude[y, x] >= _MIN_GRADIENT)
    if not keep.any():
        return painted
    start, normal = start[keep], normal[keep]
    forward = forward[keep] * np.where(left[keep, None], -1.0, 1.0).astype(np.float32)

    lengths = np.arange(1, _MAX_RAY_LENGTH + 1, dtype=np.float32)
    rays = start[:, None, :] + forward[:, None, :] * lengths[None, :, None]
    ex, ey, in_bounds = _pixel_coordinates(rays, write.shape)
    crosses_write = np.maximum.accumulate(write[ey, ex], axis=1)
    crosses_hole = np.cumsum(hidden[ey, ex], axis=1) >= 2
    near_edge = (
        cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        )
        > 0
    )
    facing = gx[ey, ex] * normal[:, :1] + gy[ey, ex] * normal[:, 1:]
    reachable = in_bounds & known[ey, ex] & crosses_hole & crosses_write
    matches = (
        reachable
        & near_edge[ey, ex]
        & (facing >= np.maximum(_MIN_GRADIENT, 0.96 * magnitude[ey, ex]))
    )
    # Search only the first span of real artwork on the far side of the gap.
    matches &= np.cumsum(reachable, axis=1) <= 4
    paired = matches.any(axis=1)
    if not paired.any():
        return painted
    indices = matches[paired].argmax(axis=1)
    end = np.column_stack((ex[paired, indices], ey[paired, indices])).astype(np.float32)
    start, normal, forward = start[paired], normal[paired], forward[paired]

    # Compare complete color profiles and verify that both edges persist away from the hole.
    sections = np.arange(-3, 4, dtype=np.float32)[None, :, None] * normal[:, None, :]
    a = (start[:, None, :] - 3 * forward[:, None, :] + sections).astype(np.float32)
    b = (end[:, None, :] + 3 * forward[:, None, :] + sections).astype(np.float32)
    ax, ay, a_bound = _pixel_coordinates(a, write.shape)
    bx, by, b_bound = _pixel_coordinates(b, write.shape)
    supported = (a_bound & b_bound & known[ay, ax] & known[by, bx]).all(axis=1)
    profile_a = cv2.remap(
        src, a, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    profile_b = cv2.remap(
        src, b, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    if profile_a.ndim == 2:
        contrast = np.minimum(
            np.abs(profile_a[:, 0] - profile_a[:, -1]),
            np.abs(profile_b[:, 0] - profile_b[:, -1]),
        )
        mismatch = np.abs(profile_a - profile_b).mean(axis=1)
    else:
        contrast = np.minimum(
            np.max(np.abs(profile_a[:, 0] - profile_a[:, -1]), axis=1),
            np.max(np.abs(profile_b[:, 0] - profile_b[:, -1]), axis=1),
        )
        mismatch = np.abs(profile_a - profile_b).mean(axis=(1, 2))
    good = (
        supported
        & (contrast >= _MIN_CONTRAST)
        & (mismatch <= np.minimum(_MAX_MISMATCH, _MAX_MISMATCH_RATIO * contrast))
    )
    if not good.any():
        return painted
    start, end, normal = start[good], end[good], normal[good]
    profile_a, profile_b = profile_a[good], profile_b[good]

    length = np.linalg.norm(end - start, axis=1)
    confidence = np.clip(1.0 - mismatch[good] / 20.0, 0.0, 1.0) * np.clip(
        1.4 - length / 50.0, 0.0, 1.0
    )
    n_steps = min(_MAX_RAY_LENGTH + 1, max(2, int(np.ceil(float(length.max()))) + 1))
    centres = (
        start[:, None, :]
        + (end - start)[:, None, :]
        * np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[None, :, None]
    )
    # Cover the same cross-section on the canonical pixel grid, including diagonal edges.
    offsets = np.linspace(-3.0, 3.0, 13, dtype=np.float32)
    points = np.rint(
        centres[:, :, None, :] + normal[:, None, None, :] * offsets[None, None, :, None]
    )
    px, py, valid = _pixel_coordinates(points, write.shape)
    distance = ((points - centres[:, :, None, :]) * normal[:, None, None, :]).sum(axis=-1)
    valid &= write[py, px] & (np.abs(distance) <= 2.8)
    if not valid.any():
        return painted

    ray, step, _ = np.nonzero(valid)
    x, y = px[valid], py[valid]
    offset = distance[valid]
    sample = np.clip(offset + 3.0, 0.0, 5.9999)
    i = sample.astype(np.int32)
    fraction = (sample - i) if profile_a.ndim == 2 else (sample - i)[:, None]
    denom = float(n_steps - 1)
    along = (step / denom) if profile_a.ndim == 2 else (step / denom)[:, None]
    profile = (
        (1.0 - along)
        * ((1.0 - fraction) * profile_a[ray, i] + fraction * profile_a[ray, i + 1])
        + along
        * ((1.0 - fraction) * profile_b[ray, i] + fraction * profile_b[ray, i + 1])
    )
    strength = confidence[ray] * np.minimum(1.0, (2.8 - np.abs(offset)) / 1.35)
    pixel = y.astype(np.int64) * write.shape[1] + x
    order = np.lexsort((strength, pixel))
    winners = order[np.r_[pixel[order[1:]] != pixel[order[:-1]], True]]
    x, y = x[winners], y[winners]
    alpha = strength[winners] if pnt.ndim == 2 else strength[winners, None]
    restored = painted.copy()
    restored[y0:y1, x0:x1][y, x] = np.clip(
        np.rint(pnt[y, x] * (1.0 - alpha) + profile[winners] * alpha), 0, 255
    ).astype(np.uint8)
    return restored
