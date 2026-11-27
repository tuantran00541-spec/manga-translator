from __future__ import annotations

import cv2
import numpy as np

_PERIOD_MIN = 2.5
_PERIOD_MAX = 18.0
_FFT_PEAK_RATIO_MIN = 20.0
_NCC_MIN = 0.78
_SEAM_NCC_MIN = 0.68


def _is_low_chroma(crop: np.ndarray, mask: np.ndarray) -> bool:
    if crop.ndim != 3 or crop.shape[2] < 3 or not np.any(mask):
        return False
    pts = crop[mask].astype(np.float32)
    spread = float(np.max(pts.max(axis=1) - pts.min(axis=1)))
    if spread <= 28.0:
        return True
    ab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[mask, 1:3].astype(np.float32)
    return float(np.max(ab.std(axis=0))) <= 6.5


def _fft_dominant_shifts(gray: np.ndarray, context: np.ndarray) -> list[tuple[int, int]]:
    """Find candidate spatial lattice periods (dx, dy) from 2D FFT magnitude peaks."""
    h, w = gray.shape[:2]
    if min(h, w) < 32 or float(context.mean()) < 0.35:
        return []
    mean_val = float(gray[context].mean())
    signal = np.where(context, gray.astype(np.float32) - mean_val, 0.0)
    win_y = np.hanning(h).astype(np.float32)
    win_x = np.hanning(w).astype(np.float32)
    spec = np.fft.fft2(signal * (win_y[:, None] * win_x[None, :]))
    mag = np.abs(np.fft.fftshift(spec))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    fy = (yy - cy) / float(h)
    fx = (xx - cx) / float(w)
    freq = np.hypot(fx, fy)
    valid = (freq >= 1.0 / _PERIOD_MAX) & (freq <= 1.0 / _PERIOD_MIN)
    if not np.any(valid):
        return []
    band_vals = mag[valid]
    med = max(float(np.median(band_vals)), 1e-3)
    masked = np.where(valid, mag, 0.0)
    peak_idx = int(np.argmax(masked))
    py, px = divmod(peak_idx, w)
    if float(masked[py, px]) / med < _FFT_PEAK_RATIO_MIN:
        return []

    candidates: list[tuple[int, int]] = []
    for _ in range(3):
        py, px = divmod(int(np.argmax(masked)), w)
        val = float(masked[py, px])
        if val / med < _FFT_PEAK_RATIO_MIN:
            break
        dfy, dfx = (py - cy) / float(h), (px - cx) / float(w)
        f2 = dfx * dfx + dfy * dfy
        if f2 > 1e-6:
            # Wavevector gives period along normal; refine around integer multiples
            period = 1.0 / np.sqrt(f2)
            ux, uy = dfx / np.sqrt(f2), dfy / np.sqrt(f2)
            for k in (1, 2):
                dx = int(round(ux * period * k))
                dy = int(round(uy * period * k))
                if (dx, dy) != (0, 0) and (dx, dy) not in candidates:
                    candidates.append((dx, dy))
        y0, y1 = max(0, py - 2), min(h, py + 3)
        x0, x1 = max(0, px - 2), min(w, px + 3)
        masked[y0:y1, x0:x1] = 0.0
    return candidates


def _ncc_shift(gray_f: np.ndarray, context: np.ndarray, dx: int, dy: int) -> float:
    h, w = gray_f.shape
    if abs(dx) >= w - 4 or abs(dy) >= h - 4:
        return 0.0
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    tx0, tx1 = sx0 + dx, sx1 + dx
    ty0, ty1 = sy0 + dy, sy1 + dy
    both = context[sy0:sy1, sx0:sx1] & context[ty0:ty1, tx0:tx1]
    if int(np.count_nonzero(both)) < 96:
        return 0.0
    a = gray_f[sy0:sy1, sx0:sx1][both]
    b = gray_f[ty0:ty1, tx0:tx1][both]
    a_c = a - float(a.mean())
    b_c = b - float(b.mean())
    denom = float(np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c)))
    if denom < 1e-3:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


def _best_lattice_vectors(gray: np.ndarray, context: np.ndarray) -> list[tuple[int, int]]:
    raw = _fft_dominant_shifts(gray, context)
    if not raw:
        return []
    gray_f = gray.astype(np.float32)
    scored: list[tuple[float, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for base_dx, base_dy in raw:
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cand = (base_dx + ox, base_dy + oy)
                if cand == (0, 0) or cand in seen:
                    continue
                seen.add(cand)
                score = _ncc_shift(gray_f, context, cand[0], cand[1])
                if score >= _NCC_MIN:
                    scored.append((score, cand))
    scored.sort(key=lambda item: (-item[0], item[1][0] ** 2 + item[1][1] ** 2))
    picked: list[tuple[int, int]] = []
    for _score, vec in scored:
        if all(abs(vec[0] * p[1] - vec[1] * p[0]) >= max(4, abs(vec[0] * p[0] + vec[1] * p[1]) // 4) for p in picked):
            picked.append(vec)
        if len(picked) >= 2:
            break
    if not picked and scored:
        picked.append(scored[0][1])
    return picked


def has_periodic_screentone(crop: np.ndarray, context_mask: np.ndarray) -> bool:
    """Return True when clean context shows regular B&W periodic halftone/hatching structure."""
    if crop.shape[0] < 32 or crop.shape[1] < 32 or float(context_mask.mean()) < 0.30:
        return False
    if not _is_low_chroma(crop, context_mask):
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    hp = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0))
    if float(np.percentile(hp[context_mask], 85)) < 4.0:
        return False
    return bool(_best_lattice_vectors(gray, context_mask))


def _has_crossing_linework(crop: np.ndarray, mask_bool: np.ndarray, context: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    hp = gray.astype(np.float32) - cv2.medianBlur(gray, 15).astype(np.float32)
    dark_strokes = (hp < -42.0) & context
    if not np.any(dark_strokes):
        return False
    rim = (cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~mask_bool
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_strokes.astype(np.uint8), connectivity=8)
    for cid in range(1, count):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        if max(w, h) >= 24 and area >= 30:
            rx, ry = int(stats[cid, cv2.CC_STAT_LEFT]), int(stats[cid, cv2.CC_STAT_TOP])
            comp = labels[ry:ry + h, rx:rx + w] == cid
            if np.any(comp & rim[ry:ry + h, rx:rx + w]):
                return True
    return False


def try_screentone_fill(
    crop: np.ndarray,
    local_mask: np.ndarray,
    hole: np.ndarray | None = None,
) -> np.ndarray | None:
    """Fill holes over regular manga screentones using phase-locked lattice shifting."""
    mask_bool = local_mask > 127
    if (
        crop.ndim != 3
        or crop.shape[2] != 3
        or int(np.count_nonzero(mask_bool)) < 32
        or float((~mask_bool).mean()) < 0.35
    ):
        return None
    hidden = mask_bool if hole is None else (mask_bool | (hole > 127))
    clean_ctx = cv2.erode((~hidden).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if float(clean_ctx.mean()) < 0.30 or not _is_low_chroma(crop, clean_ctx):
        return None
    if _has_crossing_linework(crop, mask_bool, clean_ctx):
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    vectors = _best_lattice_vectors(gray, clean_ctx)
    if not vectors:
        return None

    h, w = crop.shape[:2]
    all_shifts: list[tuple[int, int]] = []
    v1 = vectors[0]
    v2 = vectors[1] if len(vectors) > 1 else (0, 0)
    max_k1 = max(2, min(16, int(np.ceil(max(h, w) / max(1.0, np.hypot(*v1))))))
    max_k2 = max(1, min(12, int(np.ceil(max(h, w) / max(1.0, np.hypot(*v2)))))) if v2 != (0, 0) else 0
    for k1 in range(-max_k1, max_k1 + 1):
        for k2 in range(-max_k2, max_k2 + 1):
            if k1 == 0 and k2 == 0:
                continue
            all_shifts.append((k1 * v1[0] + k2 * v2[0], k1 * v1[1] + k2 * v2[1]))
    all_shifts.sort(key=lambda s: s[0] * s[0] + s[1] * s[1])

    my, mx = np.nonzero(mask_bool)
    filled = crop.copy()
    votes_sum = np.zeros((my.size, 3), dtype=np.float32)
    votes_wt = np.zeros(my.size, dtype=np.float32)
    samples_kept = np.zeros(my.size, dtype=np.int32)

    for dx, dy in all_shifts:
        remaining = samples_kept < 5
        if not np.any(remaining):
            break
        sx = mx + dx
        sy = my + dy
        ok = remaining & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if not np.any(ok):
            continue
        valid_src = np.zeros(my.size, dtype=bool)
        valid_src[ok] = clean_ctx[sy[ok], sx[ok]]
        if not np.any(valid_src):
            continue
        dist2 = float(dx * dx + dy * dy)
        weight = 1.0 / max(4.0, dist2 ** 0.5)
        votes_sum[valid_src] += crop[sy[valid_src], sx[valid_src]].astype(np.float32) * weight
        votes_wt[valid_src] += weight
        samples_kept[valid_src] += 1

    if not np.all(votes_wt > 0):
        return None

    synth = np.clip(np.rint(votes_sum / votes_wt[:, None]), 0, 255).astype(np.uint8)
    filled[my, mx] = synth

    # Seam continuity verification: shifted context across outer rim should stay coherent.
    gray_filled = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY).astype(np.float32)
    seam_band = (cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & clean_ctx
    if int(np.count_nonzero(seam_band)) >= 32:
        d1x, d1y = v1
        sy_i, sx_i = np.nonzero(mask_bool)
        tx = sx_i + d1x
        ty = sy_i + d1y
        in_bounds = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        if np.any(in_bounds):
            sy_b, sx_b = sy_i[in_bounds], sx_i[in_bounds]
            ty_b, tx_b = ty[in_bounds], tx[in_bounds]
            ref = clean_ctx[ty_b, tx_b]
            if int(np.count_nonzero(ref)) >= 32:
                a = gray_filled[sy_b[ref], sx_b[ref]]
                b = gray_filled[ty_b[ref], tx_b[ref]]
                a_c, b_c = a - float(a.mean()), b - float(b.mean())
                denom = float(np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c)))
                if denom > 1e-3 and float(np.dot(a_c, b_c) / denom) < _SEAM_NCC_MIN:
                    return None

    return filled
