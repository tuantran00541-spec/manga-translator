from __future__ import annotations

import cv2
import numpy as np

from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.smart_fill import match_boundary_tone, restore_ambient_grain, restore_inpaint_sharpness


def _border_jump(image: np.ndarray, mask: np.ndarray) -> float:
    """Measure median RGB jump across the mask perimeter between inside rim and outer ring."""
    m = mask > 127
    inner_rim = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    outer_rim = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~m
    img_f = image.astype(np.float32)
    return float(np.max(np.abs(img_f[inner_rim].mean(axis=0) - img_f[outer_rim].mean(axis=0))))


def _high_pass_mad(patch: np.ndarray, region: np.ndarray) -> float:
    f = patch.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (5, 5), 1.0)
    vals = hp[region]
    return float(np.max(1.4826 * np.median(np.abs(vals - np.median(vals, axis=0)), axis=0)))


def test_boundary_tone_alignment_removes_lama_brightness_drift():
    crop = np.full((120, 160, 3), 210, np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[35:85, 45:115] = 255

    # Simulate neural model output shifted darker by 10 levels across the crop.
    painted = np.full_like(crop, 200)
    aligned = match_boundary_tone(crop, painted, mask)

    assert np.array_equal(aligned[mask <= 127], painted[mask <= 127])
    assert abs(float(aligned[mask > 127].mean()) - 210.0) <= 1.0


def test_rim_feathering_reduces_seam_step_while_keeping_dark_ink_erased():
    image = np.full((140, 180, 3), (205, 212, 220), np.uint8)
    mask = np.zeros((140, 180), np.uint8)
    mask[30:110, 40:140] = 255
    painted = np.full_like(image, (195, 202, 210))

    blended = Inpainter._blend_painted(image.copy(), painted, mask, (0, 0, 180, 140), feather=False)
    assert np.array_equal(blended[mask <= 127], image[mask <= 127])
    assert _border_jump(blended, mask) < 7.0

    # If dark text ink sits directly on the mask's 1px edge, alpha must remain 1.0 (fully erased).
    dirty = image.copy()
    dirty[30, 60:90] = (10, 10, 10)
    erased = Inpainter._blend_painted(dirty, np.full_like(dirty, (205, 212, 220)), mask, (0, 0, 180, 140), feather=False)
    assert int(erased[30, 75].min()) >= 200


def test_ambient_grain_restoration_matches_noisy_paper_and_skips_flat():
    rng = np.random.default_rng(42)
    mask = np.zeros((160, 220), np.uint8)
    mask[45:115, 55:165] = 255
    m = mask > 127

    noisy = np.clip(185.0 + rng.normal(0.0, 5.0, (160, 220, 3)), 0, 255).astype(np.uint8)
    # Drawn black line outside the mask must not blow up grain estimate.
    cv2.line(noisy, (10, 10), (210, 150), (0, 0, 0), 2)
    flat_patch = noisy.copy()
    flat_patch[m] = 185

    restored = restore_ambient_grain(noisy, flat_patch, mask, (0, 0, 220, 160))
    assert np.array_equal(restored[~m], flat_patch[~m])
    eroded = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ctx = (cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~m
    ctx = ctx & (cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY) > 40)
    assert _high_pass_mad(restored, eroded) >= 0.5 * _high_pass_mad(noisy, ctx)

    # Pure flat canvas remains untouched bit-for-bit.
    pure = np.full((160, 220, 3), 240, np.uint8)
    assert np.array_equal(restore_ambient_grain(pure, pure.copy(), mask, (0, 0, 220, 160)), pure)


def test_boundary_tone_alignment_fixes_spatially_varying_drift():
    crop = np.full((140, 200, 3), 180, np.uint8)
    mask = np.zeros((140, 200), np.uint8)
    mask[35:105, 45:155] = 255

    # Simulate split-lit LaMa drift: left side -10 darker, right side +10 brighter (global median = 0).
    x_ramp = np.linspace(-10.0, 10.0, 200, dtype=np.float32)[None, :, None]
    painted = np.clip(crop.astype(np.float32) + x_ramp, 0, 255).astype(np.uint8)

    aligned = match_boundary_tone(crop, painted, mask)
    assert np.array_equal(aligned[mask <= 127], painted[mask <= 127])
    left_err = abs(float(aligned[70, 50].mean()) - 180.0)
    right_err = abs(float(aligned[70, 150].mean()) - 180.0)
    assert left_err < 3.0 and right_err < 3.0, f"expected seamless left/right match, got {left_err}, {right_err}"


def test_inpaint_sharpness_restores_blurred_linework_and_skips_flat():
    art = np.full((180, 220, 3), 220, np.uint8)
    for x in range(10, 210, 14):
        cv2.line(art, (x, 0), (x + 25, 179), (40, 45, 55), 2)
    mask = np.zeros((180, 220), np.uint8)
    mask[50:130, 60:160] = 255
    m = mask > 127

    soft = art.copy()
    blurred = cv2.GaussianBlur(art, (7, 7), 1.8)
    soft[m] = blurred[m]

    restored = restore_inpaint_sharpness(art, soft, mask)
    assert np.array_equal(restored[~m], soft[~m])

    def edge_energy(img):
        l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        d = np.abs(l - cv2.GaussianBlur(l, (0, 0), 1.2))
        return float(np.percentile(d[m], 90))

    assert edge_energy(restored) > edge_energy(soft) * 1.15

    # Flat/smooth gradient panel must stay untouched bit-for-bit.
    flat = np.full((180, 220, 3), 215, np.uint8)
    assert np.array_equal(restore_inpaint_sharpness(flat, flat.copy(), mask), flat)


def test_rim_feathering_remains_active_with_pending_hole():
    image = np.full((140, 180, 3), (205, 212, 220), np.uint8)
    mask = np.zeros((140, 180), np.uint8)
    mask[30:110, 40:140] = 255
    # Pending neighbor box elsewhere plus own mask in `hole` must NOT disable rim blending on `mask`.
    hole = mask.copy()
    hole[115:135, 40:140] = 255
    painted = np.full_like(image, (195, 202, 210))

    blended = Inpainter._blend_painted(image.copy(), painted, mask, (0, 0, 180, 140), feather=False, hole=hole)
    assert np.array_equal(blended[mask <= 127], image[mask <= 127])
    assert _border_jump(blended, mask) < 6.0


def test_lama_context_dilation_stays_tight_on_detailed_artwork():
    rng = np.random.default_rng(17)
    art = rng.integers(20, 235, (420, 420, 3), dtype=np.uint8)
    mask = np.zeros((420, 420), np.uint8)
    mask[50:370, 50:370] = 255  # 320x320 mask (3% would have been 10px without textured cap)

    seen = []
    inpainter = Inpainter()
    inpainter.dynamic_lama = True
    inpainter._ensure_session = lambda: None
    inpainter._lama_fill_single = lambda crop, model_mask: (seen.append(model_mask.copy()), crop.copy())[1]
    inpainter._begin_metrics()

    inpainter._lama_fill(art.copy(), art.copy(), mask, (0, 0, 420, 420))
    assert seen
    dilated_ring = (seen[0] > 127) & (mask <= 127)
    assert int(np.count_nonzero(dilated_ring)) > 0
    # 7px cap leaves row 40 (10px outside mask) completely unmasked as real artwork context!
    assert seen[0][40, 210] == 0
