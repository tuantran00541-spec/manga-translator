from __future__ import annotations

import cv2
import numpy as np

from app.parameters import (
    MANUAL_FEATHER_RADIUS,
    SMART_FILL_BLACK_EDGE_DENSITY_MAX,
    SMART_FILL_BLACK_LEVEL,
    SMART_FILL_BLACK_RATIO_MIN,
    SMART_FILL_BLACK_STD_MAX,
    SMART_FILL_CANNY_HIGH,
    SMART_FILL_CANNY_LOW,
    SMART_FILL_CHROMA_STD_MAX,
    SMART_FILL_CLEAN_RING_MARGIN,
    SMART_FILL_CONTEXT_MARGIN_FACTOR,
    SMART_FILL_EDGE_DENSITY_MAX,
    SMART_FILL_FULL_STD_MAX,
    SMART_FILL_MIDTONE_MAX,
    SMART_FILL_MIDTONE_MIN,
    SMART_FILL_MIDTONE_STD_MAX,
    SMART_FILL_RING_AGREE,
    SMART_FILL_RING_NEAR_PX,
    SMART_FILL_RING_PIXELS_MIN,
    SMART_FILL_RING_TOLERANCE,
    SMART_FILL_SURFACE_MIN_RANGE,
    SMART_FILL_SURFACE_RESIDUAL_MAX,
    SMART_FILL_SURFACE_SEED_TOL,
    SMART_FILL_WHITE_LEVEL,
    SMART_FILL_WHITE_RATIO_MIN,
    SMART_FILL_WHITE_STD_MAX,
)

_TONE_MATCH_MAX_SHIFT = 18.0
_TONE_MATCH_VALID_DRIFT_MAX = 24.0
_TONE_MATCH_MIN_SHIFT = 0.5
_GRAIN_MIN_AMBIENT_STD = 0.75
_GRAIN_MAX_STD = 10.0


def smart_fill_color(crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
    """Return a uniform BGR background fill color if the mask surround is flat."""
    mask_bool = local_mask > 127
    if not np.any(mask_bool):
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    non_mask = ~mask_bool
    if not np.any(non_mask):
        return None

    margin = max(1, int(SMART_FILL_CLEAN_RING_MARGIN))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & non_mask
    if int(np.count_nonzero(ring)) < SMART_FILL_RING_PIXELS_MIN:
        return None

    context_margin = margin * SMART_FILL_CONTEXT_MARGIN_FACTOR
    context_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (context_margin * 2 + 1, context_margin * 2 + 1),
    )
    context = (cv2.dilate(mask_bool.astype(np.uint8), context_kernel) > 0) & non_mask

    ring_gray = gray[ring]
    context_gray = gray[context]
    ring_pixels = crop[ring]
    context_std = float(context_gray.std())

    chroma_safe = True
    if crop.ndim == 3 and crop.shape[2] == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
        context_ab = lab[context, 1:3].astype(np.float32, copy=False)
        ring_chroma_std = float(np.max(ring_ab.std(axis=0))) if ring_ab.size else 0.0
        context_chroma_std = float(np.max(context_ab.std(axis=0))) if context_ab.size else 0.0
        chroma_safe = bool(
            ring_chroma_std <= SMART_FILL_CHROMA_STD_MAX
            and context_chroma_std <= SMART_FILL_CHROMA_STD_MAX
        )

    edges = cv2.Canny(
        gray,
        SMART_FILL_CANNY_LOW,
        SMART_FILL_CANNY_HIGH,
        L2gradient=True,
    ) > 0
    context_edge_density = float(edges[context].mean())

    white_ratio = float((ring_gray > SMART_FILL_WHITE_LEVEL).mean())
    black_ratio = float((ring_gray < SMART_FILL_BLACK_LEVEL).mean())
    ring_std = float(ring_gray.std())
    median_gray = float(np.median(ring_gray))

    if (
        chroma_safe
        and white_ratio >= SMART_FILL_WHITE_RATIO_MIN
        and ring_std <= SMART_FILL_WHITE_STD_MAX
        and context_std <= SMART_FILL_FULL_STD_MAX
        and context_edge_density <= SMART_FILL_EDGE_DENSITY_MAX
    ):
        white_pixels = ring_pixels[ring_gray > SMART_FILL_WHITE_LEVEL]
        if len(white_pixels):
            return np.median(white_pixels, axis=0).astype(np.uint8)

    if (
        chroma_safe
        and black_ratio >= SMART_FILL_BLACK_RATIO_MIN
        and ring_std <= SMART_FILL_BLACK_STD_MAX
        and context_std <= SMART_FILL_BLACK_STD_MAX
        and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
    ):
        black_pixels = ring_pixels[ring_gray < SMART_FILL_BLACK_LEVEL]
        if len(black_pixels):
            return np.median(black_pixels, axis=0).astype(np.uint8)

    if (
        chroma_safe
        and SMART_FILL_MIDTONE_MIN <= median_gray <= SMART_FILL_MIDTONE_MAX
        and ring_std <= SMART_FILL_MIDTONE_STD_MAX
        and context_std <= SMART_FILL_MIDTONE_STD_MAX
        and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
    ):
        return np.median(ring_pixels, axis=0).astype(np.uint8)

    return None


def ring_fill_color(crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
    """Return a clean ring median color when strokes do not cross the immediate boundary."""
    hole = local_mask > 127
    if crop.ndim != 3 or not hole.any():
        return None
    hole_u8 = hole.astype(np.uint8)

    def band(px: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(hole_u8, kernel) > 0

    grown = band(SMART_FILL_CLEAN_RING_MARGIN)
    ring = grown & ~hole
    if int(np.count_nonzero(ring)) < SMART_FILL_RING_PIXELS_MIN:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab - np.median(lab[ring], axis=0)
    off = ring & (np.einsum("...k,...k->...", diff, diff) > SMART_FILL_RING_TOLERANCE ** 2)
    near = band(SMART_FILL_RING_NEAR_PX) & ~hole
    if np.count_nonzero(near & off) > (1.0 - SMART_FILL_RING_AGREE) * np.count_nonzero(near):
        return None
    touching = cv2.dilate(hole_u8, np.ones((3, 3), np.uint8)) > 0
    outer = grown & ~(cv2.erode(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    count, parts = cv2.connectedComponents(off.astype(np.uint8))
    for label in range(1, count):
        part = parts == label
        if (part & touching).any() and (part & outer).any():
            return None
    return np.median(crop[ring & ~off], axis=0).astype(np.uint8)


def quadratic_design(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Build normalized quadratic polynomial design matrix over crop coordinates."""
    x = (xs.astype(np.float32) + 0.5) / max(1.0, float(width))
    y = (ys.astype(np.float32) + 0.5) / max(1.0, float(height))
    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=1)


def smart_fill_surface(
    crop: np.ndarray,
    local_mask: np.ndarray,
    fill_color: np.ndarray,
) -> np.ndarray | None:
    """Fit a smooth quadratic BGR surface over the mask when ring shading varies smoothly."""
    mask_bool = local_mask > 127
    margin = max(1, int(SMART_FILL_CLEAN_RING_MARGIN))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & ~mask_bool
    ys, xs = np.nonzero(ring)
    if xs.size < max(SMART_FILL_RING_PIXELS_MIN, 24) or crop.ndim != 3:
        return None
    values = crop[ring].astype(np.float32)
    height, width = crop.shape[:2]
    design = quadratic_design(xs, ys, width, height)
    inliers = np.max(np.abs(values - fill_color.astype(np.float32)), axis=1) <= SMART_FILL_SURFACE_SEED_TOL
    coef = None
    for _ in range(3):
        if int(np.count_nonzero(inliers)) < max(24, design.shape[1] * 4):
            return None
        coef, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)
        residual = np.max(np.abs(design @ coef - values), axis=1)
        refined = residual <= SMART_FILL_SURFACE_RESIDUAL_MAX * 2.0
        if np.array_equal(refined, inliers):
            break
        inliers = refined
    fit = design[inliers] @ coef
    rms = float(np.sqrt(np.mean((fit - values[inliers]) ** 2)))
    if rms > SMART_FILL_SURFACE_RESIDUAL_MAX:
        return None
    my, mx = np.nonzero(mask_bool)
    surface = quadratic_design(mx, my, width, height) @ coef
    low = values[inliers].min(axis=0) - 1.0
    high = values[inliers].max(axis=0) + 1.0
    surface = np.clip(surface, low, high)
    if float(np.max(np.ptp(surface, axis=0))) < SMART_FILL_SURFACE_MIN_RANGE:
        return None
    return np.clip(np.rint(surface), 0, 255).astype(np.uint8)


def _clean_context_mask(
    crop: np.ndarray,
    mask_bool: np.ndarray,
    radius_px: int,
    hole: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer_ring, low_gradient_ring) excluding mask, pending holes, and strong strokes."""
    exclude = mask_bool if hole is None else (mask_bool | (hole > 127))
    k = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & ~exclude
    if int(np.count_nonzero(ring)) < 16:
        return ring, ring

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy) / 4.0
    cap = min(20.0, float(np.percentile(grad[ring], 60)) + 1e-3)
    smooth = ring & (grad <= cap)
    if int(np.count_nonzero(smooth)) >= 24:
        return ring, smooth
    return ring, ring


def match_boundary_tone(
    original_crop: np.ndarray,
    painted: np.ndarray,
    local_mask: np.ndarray,
    hole: np.ndarray | None = None,
) -> np.ndarray:
    """Align per-channel brightness/color offset of neural inpaint output to clean ring."""
    mask_bool = local_mask > 127
    if (
        not np.any(mask_bool)
        or original_crop.shape != painted.shape
        or original_crop.ndim != 3
        or original_crop.shape[2] != 3
    ):
        return painted

    ring, smooth = _clean_context_mask(original_crop, mask_bool, radius_px=5, hole=hole)
    if int(np.count_nonzero(smooth)) < 16:
        return painted

    orig_vals = original_crop[smooth].astype(np.float32)
    paint_vals = painted[smooth].astype(np.float32)
    delta = np.median(orig_vals - paint_vals, axis=0)
    peak = float(np.max(np.abs(delta)))
    if peak < _TONE_MATCH_MIN_SHIFT or peak > _TONE_MATCH_VALID_DRIFT_MAX:
        return painted

    delta = np.clip(delta, -_TONE_MATCH_MAX_SHIFT, _TONE_MATCH_MAX_SHIFT)
    aligned = painted.copy()
    shifted = painted[mask_bool].astype(np.float32) + delta
    aligned[mask_bool] = np.clip(np.rint(shifted), 0, 255).astype(np.uint8)
    return aligned


def restore_ambient_grain(
    original_crop: np.ndarray,
    painted: np.ndarray,
    local_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    hole: np.ndarray | None = None,
) -> np.ndarray:
    """Restore natural high-frequency paper/scan micro-grain inside smooth filled regions."""
    mask_bool = local_mask > 127
    if (
        not np.any(mask_bool)
        or original_crop.shape != painted.shape
        or original_crop.ndim != 3
        or original_crop.shape[2] != 3
    ):
        return painted

    ring, smooth = _clean_context_mask(original_crop, mask_bool, radius_px=12, hole=hole)
    if int(np.count_nonzero(smooth)) < 48:
        return painted

    orig_f = original_crop.astype(np.float32)
    hp_orig = orig_f - cv2.GaussianBlur(orig_f, (5, 5), 1.0)
    ctx_hp = hp_orig[smooth]
    med = np.median(ctx_hp, axis=0)
    ambient_std = 1.4826 * np.median(np.abs(ctx_hp - med), axis=0)
    if float(np.max(ambient_std)) < _GRAIN_MIN_AMBIENT_STD:
        return painted

    paint_f = painted.astype(np.float32)
    hp_paint = paint_f - cv2.GaussianBlur(paint_f, (5, 5), 1.0)
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    interior = eroded if int(np.count_nonzero(eroded)) >= 16 else mask_bool
    p_hp = hp_paint[interior]
    paint_med = np.median(p_hp, axis=0)
    paint_std = 1.4826 * np.median(np.abs(p_hp - paint_med), axis=0)

    target_std = np.minimum(ambient_std * 0.9, _GRAIN_MAX_STD)
    deficit = np.sqrt(np.maximum(0.0, target_std ** 2 - paint_std ** 2))
    if float(np.max(deficit)) < 0.6:
        return painted

    # Separate luminance grain vs chroma grain so B&W scans don't receive color speckle.
    luma_hp = ctx_hp.mean(axis=1)
    luma_mad = 1.4826 * float(np.median(np.abs(luma_hp - np.median(luma_hp))))
    chroma_hp = ctx_hp - luma_hp[:, None]
    chroma_mad = 1.4826 * np.median(np.abs(chroma_hp - np.median(chroma_hp, axis=0)), axis=0)

    total_sq = np.maximum(luma_mad ** 2 + chroma_mad ** 2, 1e-6)
    luma_share = np.clip((luma_mad ** 2) / total_sq, 0.0, 1.0)
    luma_sigma = float(np.median(deficit * np.sqrt(luma_share)))
    chroma_sigma = deficit * np.sqrt(np.maximum(0.0, 1.0 - luma_share))

    seed = (hash((tuple(int(v) for v in crop_box), int(np.count_nonzero(mask_bool)))) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    h, w = painted.shape[:2]

    raw_luma = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    corr_luma = 0.72 * raw_luma + 0.28 * cv2.GaussianBlur(raw_luma, (3, 3), 0.7)
    corr_luma /= max(float(corr_luma.std()), 1e-6)

    noise = corr_luma[:, :, None] * luma_sigma
    if float(np.max(chroma_sigma)) >= 0.35:
        raw_chroma = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
        corr_chroma = 0.7 * raw_chroma + 0.3 * cv2.GaussianBlur(raw_chroma, (3, 3), 0.7)
        corr_chroma -= corr_chroma.mean(axis=2, keepdims=True)
        for ch in range(3):
            c_std = float(corr_chroma[:, :, ch].std())
            if c_std > 1e-6:
                corr_chroma[:, :, ch] /= c_std
        noise = noise + corr_chroma * chroma_sigma[None, None, :]

    ctx_vals = orig_f[smooth]
    low = np.percentile(ctx_vals, 1, axis=0) - 4.0
    high = np.percentile(ctx_vals, 99, axis=0) + 4.0

    out = painted.copy()
    noisy_vals = np.clip(paint_f[mask_bool] + noise[mask_bool], low, high)
    out[mask_bool] = np.clip(np.rint(noisy_vals), 0, 255).astype(np.uint8)
    return out


def blend_painted(
    image: np.ndarray,
    painted: np.ndarray,
    local_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    feather: bool,
    hole: np.ndarray | None = None,
) -> np.ndarray:
    """Composite inpainted crop back into image with seam-safe boundary blending."""
    cx1, cy1, cx2, cy2 = crop_box
    original_crop = image[cy1:cy2, cx1:cx2]
    if feather:
        core = local_mask > 127
        alpha = core.astype(np.float32)
        if MANUAL_FEATHER_RADIUS > 0:
            k = MANUAL_FEATHER_RADIUS * 2 + 1
            feathered = cv2.GaussianBlur(alpha, (k, k), 0)
            alpha = np.where(core, 1.0, feathered)
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
        blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
        image[cy1:cy2, cx1:cx2] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return image

    mask_bool = local_mask > 127
    if not np.any(mask_bool):
        return image

    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
    if float(dist.max()) >= 3.0 and original_crop.ndim == 3 and original_crop.shape == painted.shape:
        # Soften only the outermost 1.5px rim where original pixel is already background-like,
        # preserving full alpha=1.0 over any stroke/fringe at the perimeter while never altering ~mask.
        alpha = mask_bool.astype(np.float32)
        rim = mask_bool & (dist <= 1.5)
        if hole is not None:
            rim &= ~(hole > 127)
        if np.any(rim):
            _, smooth = _clean_context_mask(original_crop, mask_bool, radius_px=6, hole=hole)
            if int(np.count_nonzero(smooth)) >= 16:
                orig_f = original_crop.astype(np.float32)
                paint_f = painted.astype(np.float32)
                ctx_vals = orig_f[smooth]
                ctx_mad = float(np.max(1.4826 * np.median(np.abs(ctx_vals - np.median(ctx_vals, axis=0)), axis=0)))
                rim_tol = max(18.0, min(42.0, ctx_mad * 4.5 + 12.0))
                diff = np.max(np.abs(orig_f - paint_f), axis=2)
                soft = rim & (diff <= rim_tol)
                alpha[soft] = np.clip(0.45 + 0.40 * np.clip(dist[soft] - 1.0, 0.0, 0.5) / 0.5, 0.45, 0.85)
        alpha_3d = alpha[:, :, None]
        blended = painted.astype(np.float32) * alpha_3d + original_crop.astype(np.float32) * (1.0 - alpha_3d)
        image[cy1:cy2, cx1:cx2] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    else:
        mask_3d = mask_bool[:, :, None]
        image[cy1:cy2, cx1:cx2] = np.where(mask_3d, painted, original_crop)
    return image
