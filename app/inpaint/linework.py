from __future__ import annotations

import cv2
import numpy as np

_ROI_PAD_PX = 96
_MAX_EDGE_SEEDS = 3000
_MAX_RAY_LENGTH = 96
_MIN_NEAR_PIXELS = 24
_MIN_GRADIENT = 17.0
_MIN_CONTRAST = 30.0
_MAX_MISMATCH = 14.0
_MAX_MISMATCH_RATIO = 0.2
_CURVE_PAIR_MAX_DIST = 96.0


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
    all_start, all_normal, all_forward = start, normal, forward
    s_start = np.empty((0, 2), np.float32)
    s_end = np.empty((0, 2), np.float32)
    s_norm = np.empty((0, 2), np.float32)
    s_pa = np.empty((0, 7) if src.ndim == 2 else (0, 7, src.shape[2]), np.float32)
    s_pb = s_pa
    s_len = np.empty(0, np.float32)
    s_mis = np.empty(0, np.float32)
    s_half = np.empty(0, np.float32)
    unpaired_mask = np.ones(len(all_start), dtype=bool)

    if paired.any():
        indices = matches[paired].argmax(axis=1)
        end = np.column_stack((ex[paired, indices], ey[paired, indices])).astype(np.float32)
        p_start, p_normal, p_forward = start[paired], normal[paired], forward[paired]

        # Compare complete color profiles and verify that both edges persist away from the hole.
        # For thick strokes (>6px), test +-6px cross-section if +-3px falls entirely inside solid ink.
        sections3 = np.arange(-3, 4, dtype=np.float32)[None, :, None] * p_normal[:, None, :]
        a = (p_start[:, None, :] - 3 * p_forward[:, None, :] + sections3).astype(np.float32)
        b = (end[:, None, :] + 3 * p_forward[:, None, :] + sections3).astype(np.float32)
        ax, ay, a_bound = _pixel_coordinates(a, write.shape)
        bx, by, b_bound = _pixel_coordinates(b, write.shape)
        supported = (a_bound & b_bound & known[ay, ax] & known[by, bx]).all(axis=1)
        profile_a = cv2.remap(
            src, a, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        ).astype(np.float32)
        profile_b = cv2.remap(
            src, b, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        ).astype(np.float32)

        def _contrast_and_mismatch(pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if pa.ndim == 2:
                c = np.minimum(np.abs(pa[:, 0] - pa[:, -1]), np.abs(pb[:, 0] - pb[:, -1]))
                m = np.abs(pa - pb).mean(axis=1)
            else:
                c = np.minimum(np.max(np.abs(pa[:, 0] - pa[:, -1]), axis=1), np.max(np.abs(pb[:, 0] - pb[:, -1]), axis=1))
                m = np.abs(pa - pb).mean(axis=(1, 2))
            return c, m

        contrast, mismatch = _contrast_and_mismatch(profile_a, profile_b)
        half_width = np.full(len(p_start), 3.0, dtype=np.float32)
        low_c = supported & (contrast < _MIN_CONTRAST)
        if np.any(low_c):
            wide_steps = np.linspace(-6.0, 6.0, 7, dtype=np.float32)[None, :, None] * p_normal[low_c, None, :]
            aw = (p_start[low_c, None, :] - 4 * p_forward[low_c, None, :] + wide_steps).astype(np.float32)
            bw = (end[low_c, None, :] + 4 * p_forward[low_c, None, :] + wide_steps).astype(np.float32)
            axw, ayw, abw = _pixel_coordinates(aw, write.shape)
            bxw, byw, bbw = _pixel_coordinates(bw, write.shape)
            supw = (abw & bbw & known[ayw, axw] & known[byw, bxw]).all(axis=1)
            paw = cv2.remap(src, aw, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            pbw = cv2.remap(src, bw, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            cw, mw = _contrast_and_mismatch(paw, pbw)
            idx_low = np.flatnonzero(low_c)
            better = supw & (cw >= _MIN_CONTRAST)
            profile_a[idx_low[better]] = paw[better]
            profile_b[idx_low[better]] = pbw[better]
            contrast[idx_low[better]] = cw[better]
            mismatch[idx_low[better]] = mw[better]
            half_width[idx_low[better]] = 6.0
        good = (
            supported
            & (contrast >= _MIN_CONTRAST)
            & (mismatch <= np.minimum(_MAX_MISMATCH, _MAX_MISMATCH_RATIO * contrast))
        )
        length = np.linalg.norm(end - p_start, axis=1)
        if good.any():
            s_start = p_start[good]
            s_end = end[good]
            s_norm = p_normal[good]
            s_pa = profile_a[good]
            s_pb = profile_b[good]
            s_len = length[good]
            s_mis = mismatch[good]
            s_half = half_width[good]
            unpaired_mask[np.flatnonzero(paired)[good]] = False

    # Also evaluate smooth curved Hermite arcs between boundary entry/exit endpoints not joined straight.
    arc_centres_list: list[np.ndarray] = []
    arc_norms_list: list[np.ndarray] = []
    arc_pa_list: list[np.ndarray] = []
    arc_pb_list: list[np.ndarray] = []
    arc_conf_list: list[float] = []
    if int(np.count_nonzero(unpaired_mask)) >= 2:
        u_start = all_start[unpaired_mask]
        u_fwd = all_forward[unpaired_mask]
        u_norm = all_normal[unpaired_mask]
        # Cluster closely adjacent same-direction edge points into sparse stroke anchors.
        anchor_idx: list[int] = []
        taken = np.zeros(len(u_start), dtype=bool)
        for k in range(len(u_start)):
            if taken[k]:
                continue
            anchor_idx.append(k)
            d2 = np.sum((u_start - u_start[k]) ** 2, axis=1)
            same_dir = (u_fwd @ u_fwd[k]) > 0.70
            taken |= (d2 <= 9.0) & same_dir
            if len(anchor_idx) >= 64:
                break
        if len(anchor_idx) >= 2:
            pts_a = u_start[anchor_idx]
            fwds_a = u_fwd[anchor_idx]
            norms_a = u_norm[anchor_idx]
            prof_probes = (pts_a[:, None, :] - 3 * fwds_a[:, None, :] + (np.arange(-3, 4, dtype=np.float32)[None, :, None] * norms_a[:, None, :])).astype(np.float32)
            pax, pay, pab = _pixel_coordinates(prof_probes, write.shape)
            pa_sup = (pab & known[pay, pax]).all(axis=1)
            pa_vals = cv2.remap(src, prof_probes, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            used_j: set[int] = set()
            for ia in range(len(pts_a)):
                if not pa_sup[ia]:
                    continue
                best_pair = None
                best_cost = 1e9
                for jb in range(ia + 1, len(pts_a)):
                    if jb in used_j or not pa_sup[jb]:
                        continue
                    chord = pts_a[jb] - pts_a[ia]
                    dist_ab = float(np.linalg.norm(chord))
                    if not (10.0 <= dist_ab <= _CURVE_PAIR_MAX_DIST):
                        continue
                    uchord = chord / dist_ab
                    da = float(np.dot(fwds_a[ia], uchord))
                    db = float(np.dot(fwds_a[jb], -uchord))
                    if da < 0.55 or db < 0.55:
                        continue
                    ang_a = float(np.arccos(np.clip(da, -1.0, 1.0)))
                    ang_b = float(np.arccos(np.clip(db, -1.0, 1.0)))
                    if abs(ang_a - ang_b) > 0.62:  # ~35 deg symmetry for circular/smooth arcs
                        continue
                    # Reject nearly collinear pairs unless they have high profile contrast match
                    pa_i, pb_j = pa_vals[ia], pa_vals[jb]
                    # Normal reversal across opposite ray directions maps -normal -> reverse profile
                    flip = float(np.dot(norms_a[ia], norms_a[jb])) < 0.0
                    pb_aligned = pb_j[::-1] if flip else pb_j
                    if pa_i.ndim == 1:
                        ca = float(np.abs(pa_i[0] - pa_i[-1]))
                        cb = float(np.abs(pb_aligned[0] - pb_aligned[-1]))
                        mis = float(np.mean(np.abs(pa_i - pb_aligned)))
                    else:
                        ca = float(np.max(np.abs(pa_i[0] - pa_i[-1])))
                        cb = float(np.max(np.abs(pb_aligned[0] - pb_aligned[-1])))
                        mis = float(np.mean(np.abs(pa_i - pb_aligned)))
                    cmin = min(ca, cb)
                    if cmin < _MIN_CONTRAST or mis > min(_MAX_MISMATCH, _MAX_MISMATCH_RATIO * cmin):
                        continue
                    cost = mis + 8.0 * abs(ang_a - ang_b) + 0.05 * dist_ab
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (jb, pts_a[ia], pts_a[jb], fwds_a[ia] * dist_ab, -fwds_a[jb] * dist_ab, pa_i, pb_aligned, mis, dist_ab)
                if best_pair is not None:
                    jb, p0, p1, m0, m1, pa_i, pb_al, mis, dist_ab = best_pair
                    used_j.add(ia)
                    used_j.add(jb)
                    n_arc = max(4, min(_MAX_RAY_LENGTH + 1, int(np.ceil(dist_ab)) + 1))
                    ts = np.linspace(0.0, 1.0, n_arc, dtype=np.float32)[:, None]
                    h00 = 2 * ts**3 - 3 * ts**2 + 1
                    h10 = ts**3 - 2 * ts**2 + ts
                    h01 = -2 * ts**3 + 3 * ts**2
                    h11 = ts**3 - ts**2
                    pts_curve = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                    # Verify curve mostly traverses write_mask / hidden (not cutting across visible known artwork)
                    cx, cy, c_in = _pixel_coordinates(pts_curve, write.shape)
                    if c_in.all() and float(hidden[cy, cx].mean()) >= 0.70 and float(write[cy, cx].mean()) >= 0.35:
                        dpts = np.gradient(pts_curve, axis=0)
                        dlen = np.maximum(np.linalg.norm(dpts, axis=1, keepdims=True), 1e-4)
                        tnorms = np.column_stack((-dpts[:, 1], dpts[:, 0])) / dlen
                        # Orient curve normal to match p0's gradient normal
                        if float(np.dot(tnorms[0], norms_a[ia])) < 0.0:
                            tnorms = -tnorms
                        arc_centres_list.append(pts_curve)
                        arc_norms_list.append(tnorms.astype(np.float32))
                        arc_pa_list.append(pa_i)
                        arc_pb_list.append(pb_al)
                        arc_conf_list.append(float(np.clip(1.0 - mis / 20.0, 0.0, 1.0) * np.clip(1.1 - dist_ab / 180.0, 0.72, 1.0)))

    if not s_start.size and not arc_centres_list:
        return painted

    restored = painted.copy()

    if s_start.size:
        confidence = np.clip(1.0 - s_mis / 20.0, 0.0, 1.0) * np.clip(1.15 - s_len / 160.0, 0.72, 1.0)
        n_steps = min(_MAX_RAY_LENGTH + 1, max(2, int(np.ceil(float(s_len.max()))) + 1))
        centres = (
            s_start[:, None, :]
            + (s_end - s_start)[:, None, :]
            * np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[None, :, None]
        )
        for hw_val in (3.0, 6.0):
            sel = np.isclose(s_half, hw_val)
            if not np.any(sel):
                continue
            offsets = np.linspace(-hw_val, hw_val, 25 if hw_val > 3.5 else 13, dtype=np.float32)
            points = np.rint(
                centres[sel][:, :, None, :] + s_norm[sel][:, None, None, :] * offsets[None, None, :, None]
            )
            px, py, valid = _pixel_coordinates(points, write.shape)
            distance = ((points - centres[sel][:, :, None, :]) * s_norm[sel][:, None, None, :]).sum(axis=-1)
            lim = hw_val - 0.2
            valid &= write[py, px] & (np.abs(distance) <= lim)
            if not valid.any():
                continue
            ray, step, _ = np.nonzero(valid)
            x, y = px[valid], py[valid]
            offset = distance[valid]
            step_scale = hw_val / 3.0
            sample = np.clip(offset / step_scale + 3.0, 0.0, 5.9999)
            i = sample.astype(np.int32)
            pa_sel, pb_sel, conf_sel = s_pa[sel], s_pb[sel], confidence[sel]
            fraction = (sample - i) if pa_sel.ndim == 2 else (sample - i)[:, None]
            denom = float(n_steps - 1)
            along = (step / denom) if pa_sel.ndim == 2 else (step / denom)[:, None]
            profile = (
                (1.0 - along)
                * ((1.0 - fraction) * pa_sel[ray, i] + fraction * pa_sel[ray, i + 1])
                + along
                * ((1.0 - fraction) * pb_sel[ray, i] + fraction * pb_sel[ray, i + 1])
            )
            strength = conf_sel[ray] * np.minimum(1.0, (lim - np.abs(offset)) / (1.35 * step_scale))
            pixel = y.astype(np.int64) * write.shape[1] + x
            order = np.lexsort((strength, pixel))
            winners = order[np.r_[pixel[order[1:]] != pixel[order[:-1]], True]]
            x, y = x[winners], y[winners]
            alpha = strength[winners] if pnt.ndim == 2 else strength[winners, None]
            restored[y0:y1, x0:x1][y, x] = np.clip(
                np.rint(restored[y0:y1, x0:x1][y, x] * (1.0 - alpha) + profile[winners] * alpha), 0, 255
            ).astype(np.uint8)

        # Fill narrow hollow core strictly between opposite-facing parallel edges of thick (5..14px) strokes.
        if len(s_start) >= 2:
            # Sample coarse distinct anchors (~40 max) to avoid O(N^2) on huge rays
            picked_idx: list[int] = []
            seen_mask = np.zeros(len(s_start), dtype=bool)
            for k in range(len(s_start)):
                if seen_mask[k]:
                    continue
                picked_idx.append(k)
                d2 = np.sum((s_start - s_start[k]) ** 2, axis=1)
                seen_mask |= (d2 <= 4.0) & ((s_norm @ s_norm[k]) > 0.85)
                if len(picked_idx) >= 48:
                    break
            fill_img = restored[y0:y1, x0:x1]
            used_thick: set[int] = set()
            for ii, ia in enumerate(picked_idx):
                if ia in used_thick:
                    continue
                p_sa, p_ea, na = s_start[ia], s_end[ia], s_norm[ia]
                for jb in picked_idx[ii + 1:]:
                    if jb in used_thick or float(np.dot(na, s_norm[jb])) > -0.94:
                        continue
                    p_sb, p_eb = s_start[jb], s_end[jb]
                    sep_s = float(np.linalg.norm(p_sb - p_sa))
                    sep_e = float(np.linalg.norm(p_eb - p_ea))
                    if not (3.0 <= sep_s <= 14.0 and 3.0 <= sep_e <= 14.0):
                        continue
                    # Probe solid ink color just outside the hole along the midpoint between the two edges
                    dir_a = (p_ea - p_sa) / max(1.0, float(np.linalg.norm(p_ea - p_sa)))
                    probe_a = p_sa - 3.0 * dir_a + 0.5 * (p_sb - p_sa)
                    probe_b = p_ea + 3.0 * dir_a + 0.5 * (p_eb - p_ea)
                    pxa = int(np.clip(round(probe_a[0]), 0, write.shape[1] - 1))
                    pya = int(np.clip(round(probe_a[1]), 0, write.shape[0] - 1))
                    pxb = int(np.clip(round(probe_b[0]), 0, write.shape[1] - 1))
                    pyb = int(np.clip(round(probe_b[1]), 0, write.shape[0] - 1))
                    if not (known[pya, pxa] and known[pyb, pxb]):
                        continue
                    ink_a = src[pya, pxa].astype(np.float32)
                    ink_b = src[pyb, pxb].astype(np.float32)
                    # Reject if midpoint outside hole is white paper (meaning two separate thin lines, not one thick line)
                    bg_a = float(s_pa[ia].max()) if s_pa[ia].ndim == 1 else float(np.mean(np.max(s_pa[ia], axis=0)))
                    if float(np.mean(np.abs(ink_a - ink_b))) > 22.0:
                        continue
                    if float(np.max(np.abs(ink_a - bg_a))) < _MIN_CONTRAST * 0.8:
                        continue
                    ink_fill = np.clip(np.rint(0.5 * (ink_a + ink_b)), 0, 255).astype(np.uint8)
                    poly = np.rint(np.array([p_sa, p_ea, p_eb, p_sb])).astype(np.int32)
                    core_mask = np.zeros(write.shape, dtype=np.uint8)
                    cv2.fillConvexPoly(core_mask, poly, 255)
                    core_fill = (core_mask > 0) & write
                    if np.any(core_fill):
                        fill_img[core_fill] = ink_fill
                        used_thick.add(ia)
                        used_thick.add(jb)
                    break

    if arc_centres_list:
        offsets = np.linspace(-3.0, 3.0, 13, dtype=np.float32)
        for pts_curve, tnorms, pa_i, pb_al, conf in zip(
            arc_centres_list, arc_norms_list, arc_pa_list, arc_pb_list, arc_conf_list
        ):
            n_arc = len(pts_curve)
            points = np.rint(pts_curve[:, None, :] + tnorms[:, None, :] * offsets[None, :, None])
            px, py, valid = _pixel_coordinates(points, write.shape)
            distance = ((points - pts_curve[:, None, :]) * tnorms[:, None, :]).sum(axis=-1)
            valid &= write[py, px] & (np.abs(distance) <= 2.8)
            if not valid.any():
                continue
            step_idx, _ = np.nonzero(valid)
            ax, ay = px[valid], py[valid]
            offset = distance[valid]
            sample = np.clip(offset + 3.0, 0.0, 5.9999)
            i = sample.astype(np.int32)
            fraction = (sample - i) if pa_i.ndim == 1 else (sample - i)[:, None]
            along = (step_idx / float(max(1, n_arc - 1))) if pa_i.ndim == 1 else (step_idx / float(max(1, n_arc - 1)))[:, None]
            profile = (
                (1.0 - along) * ((1.0 - fraction) * pa_i[i] + fraction * pa_i[i + 1])
                + along * ((1.0 - fraction) * pb_al[i] + fraction * pb_al[i + 1])
            )
            strength = conf * np.minimum(1.0, (2.8 - np.abs(offset)) / 1.35)
            pixel = ay.astype(np.int64) * write.shape[1] + ax
            order = np.lexsort((strength, pixel))
            winners = order[np.r_[pixel[order[1:]] != pixel[order[:-1]], True]]
            ax, ay = ax[winners], ay[winners]
            alpha = strength[winners] if pnt.ndim == 2 else strength[winners, None]
            restored[y0:y1, x0:x1][ay, ax] = np.clip(
                np.rint(restored[y0:y1, x0:x1][ay, ax] * (1.0 - alpha) + profile[winners] * alpha), 0, 255
            ).astype(np.uint8)

    return restored
