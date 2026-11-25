from __future__ import annotations

import cv2
import numpy as np


def _pixel_coordinates(points, shape):
    rounded = np.rint(points).astype(np.int32)
    x, y = rounded[..., 0], rounded[..., 1]
    valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    return x.clip(0, shape[1] - 1), y.clip(0, shape[0] - 1), valid


def continue_linework(source, painted, write_mask, hole=None):
    """Interpolate narrow gaps where the same oriented ink profile is visible on both sides."""
    write = write_mask > 127
    if not write.any() or source.shape[:2] != painted.shape[:2]:
        return painted
    hidden = write if hole is None else write | (hole > 127)
    known = cv2.erode((~hidden).astype(np.uint8), np.ones((5, 5), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    near = (cv2.dilate(write_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 127) & known
    if np.count_nonzero(near) < 24:
        return painted

    gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    smooth = cv2.GaussianBlur(gray_src, (3, 3), .65)
    edges = cv2.Canny(smooth, 50, 100, L2gradient=True) > 0
    y, x = np.nonzero(edges & near)
    if not 1 < x.size <= 3000:
        return painted
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=.125)
    magnitude = cv2.magnitude(gx, gy)
    normal = np.column_stack((gx[y, x], gy[y, x])) / magnitude[y, x, None].clip(min=1.)
    start = np.column_stack((x, y)).astype(np.float32)
    forward = normal[:, ::-1].copy()
    forward[:, 0] *= -1
    probe = start[:, None, :] + forward[:, None, :] * np.array([-10, -7, -4, 4, 7, 10], np.float32)[None, :, None]
    qx, qy, in_bounds = _pixel_coordinates(probe, write.shape)
    crosses = hidden[qy, qx] & in_bounds
    left, right = crosses[:, :3].any(axis=1), crosses[:, 3:].any(axis=1)
    keep = (left ^ right) & (magnitude[y, x] >= 17.)
    if not keep.any():
        return painted
    start, normal = start[keep], normal[keep]
    forward = forward[keep] * np.where(left[keep, None], -1., 1.).astype(np.float32)

    lengths = np.arange(1, 49, dtype=np.float32)
    rays = start[:, None, :] + forward[:, None, :] * lengths[None, :, None]
    ex, ey, in_bounds = _pixel_coordinates(rays, write.shape)
    crosses_write = np.maximum.accumulate(write[ey, ex], axis=1)
    crosses_hole = np.cumsum(hidden[ey, ex], axis=1) >= 2
    near_edge = cv2.dilate(edges.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    facing = gx[ey, ex] * normal[:, :1] + gy[ey, ex] * normal[:, 1:]
    reachable = in_bounds & known[ey, ex] & crosses_hole & crosses_write
    matches = reachable & near_edge[ey, ex] & (facing >= np.maximum(17., .96 * magnitude[ey, ex]))
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
    a = start[:, None, :] - 3 * forward[:, None, :] + sections
    b = end[:, None, :] + 3 * forward[:, None, :] + sections
    ax, ay, a_bound = _pixel_coordinates(a, write.shape)
    bx, by, b_bound = _pixel_coordinates(b, write.shape)
    supported = (a_bound & b_bound & known[ay, ax] & known[by, bx]).all(axis=1)
    profile_a = cv2.remap(source, a, None, cv2.INTER_LINEAR).astype(np.float32)
    profile_b = cv2.remap(source, b, None, cv2.INTER_LINEAR).astype(np.float32)
    if profile_a.ndim == 2:
        contrast = np.minimum(np.abs(profile_a[:, 0] - profile_a[:, -1]),
                              np.abs(profile_b[:, 0] - profile_b[:, -1]))
        mismatch = np.abs(profile_a - profile_b).mean(axis=1)
    else:
        contrast = np.minimum(np.max(np.abs(profile_a[:, 0] - profile_a[:, -1]), axis=1),
                              np.max(np.abs(profile_b[:, 0] - profile_b[:, -1]), axis=1))
        mismatch = np.abs(profile_a - profile_b).mean(axis=(1, 2))
    good = supported & (contrast >= 30) & (mismatch <= np.minimum(14, .2 * contrast))
    if not good.any():
        return painted
    start, end, normal = start[good], end[good], normal[good]
    profile_a, profile_b = profile_a[good], profile_b[good]

    length = np.linalg.norm(end - start, axis=1)
    confidence = np.clip(1 - mismatch[good] / 20, .0, 1.) * np.clip(1.4 - length / 50, .0, 1.)
    centres = start[:, None, :] + (end - start)[:, None, :] * np.linspace(0, 1, 49, dtype=np.float32)[None, :, None]
    # Cover the same cross-section on the canonical pixel grid, including diagonal edges.
    offsets = np.linspace(-3, 3, 13, dtype=np.float32)
    points = np.rint(centres[:, :, None, :] + normal[:, None, None, :] * offsets[None, None, :, None])
    px, py, valid = _pixel_coordinates(points, write.shape)
    distance = ((points - centres[:, :, None, :]) * normal[:, None, None, :]).sum(axis=-1)
    valid &= write[py, px] & (np.abs(distance) <= 2.8)
    if not valid.any():
        return painted

    ray, step, _ = np.nonzero(valid)
    x, y = px[valid], py[valid]
    offset = distance[valid]
    sample = offset + 3
    i = sample.astype(np.int32)
    fraction = (sample - i) if profile_a.ndim == 2 else (sample - i)[:, None]
    along = (step / 48) if profile_a.ndim == 2 else (step / 48)[:, None]
    profile = ((1 - along) * ((1 - fraction) * profile_a[ray, i] + fraction * profile_a[ray, i + 1])
               + along * ((1 - fraction) * profile_b[ray, i] + fraction * profile_b[ray, i + 1]))
    strength = confidence[ray] * np.minimum(1., (2.8 - np.abs(offset)) / 1.35)
    pixel = y * write.shape[1] + x
    order = np.lexsort((strength, pixel))
    winners = order[np.r_[pixel[order[1:]] != pixel[order[:-1]], True]]
    x, y = x[winners], y[winners]
    alpha = strength[winners] if painted.ndim == 2 else strength[winners, None]
    restored = painted.copy()
    restored[y, x] = np.rint(painted[y, x] * (1 - alpha) + profile[winners] * alpha).astype(np.uint8)
    return restored
