from __future__ import annotations

import cv2
import numpy as np

from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.linework import continue_linework
from app.inpaint.screentone import try_screentone_fill
from app.inpaint.smart_fill import match_boundary_tone, restore_ambient_grain, smart_fill_gradient_surface


def test_native_512_tiling_runs_without_downscale_and_propagates():
    inp = Inpainter()
    inp.dynamic_lama = True
    inp._ensure_session = lambda: None
    seen_shapes = []
    # Mock LaMa returns dark center replacement so progressive propagation is observable
    def fake_single(tile_img, tile_mask):
        seen_shapes.append(tile_img.shape[:2])
        out = tile_img.copy()
        out[tile_mask > 127] = (200, 205, 210)
        return out

    inp._lama_fill_single = fake_single
    crop = np.full((600, 900, 3), 240, np.uint8)
    mask = np.zeros((600, 900), np.uint8)
    mask[200:350, 200:700] = 255
    result = inp._lama_fill_tiled(crop, mask)
    assert all(max(s) <= 512 for s in seen_shapes)
    assert result.shape == crop.shape
    assert np.all(np.abs(result[250:300, 300:600].astype(int) - (200, 205, 210)) <= 2)


def test_multibox_lama_still_dilates_model_context_without_expanding_write():
    image = np.random.default_rng(12).integers(50, 220, (200, 240, 3), dtype=np.uint8)
    local_mask = np.zeros((200, 240), np.uint8)
    local_mask[60:110, 70:140] = 255
    pending_hole = np.zeros((200, 240), np.uint8)
    pending_hole[118:135, 70:140] = 255
    seen_masks = []

    inp = Inpainter()
    inp.dynamic_lama = True
    inp._ensure_session = lambda: None
    inp._lama_fill_single = lambda crop, m: (seen_masks.append(m.copy()), np.zeros_like(crop))[1]
    result = inp._lama_fill(image.copy(), image.copy(), local_mask, (0, 0, 240, 200), hole=pending_hole)

    expected_union = cv2.bitwise_or(local_mask, pending_hole)
    assert len(seen_masks) == 1
    # Verify trimmed model mask still includes both local dilation AND pending_hole without modifying write boundary
    mh, mw = seen_masks[0].shape[:2]
    assert mh >= 75 and mw >= 70
    assert np.count_nonzero(seen_masks[0]) > np.count_nonzero(local_mask) + np.count_nonzero(pending_hole)
    assert np.array_equal(result[local_mask == 0], image[local_mask == 0])


def test_decoupled_gradient_surface_fills_steep_sky_without_lama():
    h, w = 260, 360
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    b = 240.0 - 0.30 * yy - 0.10 * xx
    g = 210.0 - 0.25 * yy + 0.05 * xx
    r = 160.0 - 0.15 * yy + 0.12 * xx
    sky = np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)

    mask = np.zeros((h, w), np.uint8)
    mask[80:170, 90:260] = 255
    inp = Inpainter()
    inp._ensure_session = lambda: None
    inp._lama_fill = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should use decoupled gradient fill"))
    inp._begin_metrics()

    out = inp._smart_paint_region(sky.copy(), mask, (0, 0, w, h))
    metrics = inp.last_metrics()
    assert metrics.get("smart_fill_gradient_regions") == 1
    err = np.abs(out[mask > 127].astype(np.float32) - sky[mask > 127].astype(np.float32))
    assert float(err.mean()) < 2.0


def test_screentone_phase_locked_synthesis_and_grain_guard():
    h, w = 160, 200
    yy, xx = np.mgrid[0:h, 0:w]
    # Crisp 6x6 halftone dot pattern
    dot = (((yy % 6) < 2) & ((xx % 6) < 2))
    img = np.repeat(np.where(dot, np.uint8(90), np.uint8(230))[..., None], 3, axis=2)
    mask = np.zeros((h, w), np.uint8)
    mask[50:110, 60:140] = 255
    corrupted = img.copy()
    corrupted[mask > 127] = 180

    filled = try_screentone_fill(corrupted, mask)
    assert filled is not None
    mae = float(np.mean(np.abs(filled[mask > 127].astype(np.float32) - img[mask > 127].astype(np.float32))))
    assert mae < 4.0, f"expected crisp halftone reconstruction, got MAE={mae}"

    # Grain restoration must NOT sprinkle random noise on top of periodic screentone
    after_grain = restore_ambient_grain(img, filled, mask, (0, 0, w, h))
    assert np.array_equal(after_grain, filled)


def test_curved_bubble_arc_and_thick_linework_continuation():
    img = np.full((220, 260, 3), 245, np.uint8)
    # Draw a curved speech-bubble arc across the canvas
    cv2.ellipse(img, (130, 180), (110, 110), 0, 210, 330, (20, 20, 20), 4)
    truth = img.copy()

    # Erase a 60px gap along the top of the arc
    write_mask = np.zeros((220, 260), np.uint8)
    write_mask[55:95, 100:160] = 255
    painted = img.copy()
    painted[write_mask > 127] = 245

    repaired = continue_linework(img, painted, write_mask)
    assert np.array_equal(repaired[write_mask == 0], painted[write_mask == 0])
    # Reconstructed curve inside the center gap must restore dark ink pixels close to truth
    center_gap = repaired[62:85, 118:142, 0]
    assert int(center_gap.min()) < 100, f"curved arc was not continued: min={center_gap.min()}"

    # Thick 8px straight border continuation across a gap
    thick = np.full((160, 220, 3), 245, np.uint8)
    cv2.line(thick, (10, 80), (210, 80), (15, 15, 15), 8)
    w2 = np.zeros((160, 220), np.uint8)
    w2[68:92, 85:135] = 255
    p2 = thick.copy()
    p2[w2 > 127] = 245
    rep2 = continue_linework(thick, p2, w2)
    assert int(rep2[78:83, 105:115, 0].min()) < 100


def test_boundary_tone_alignment_catches_seam_jump_and_contrast_washout():
    h, w = 140, 180
    # Half textured art with dark shadows & highlights; paint is washed-out inside mask
    rng = np.random.default_rng(7)
    orig = np.full((h, w, 3), 220, np.uint8)
    orig[:, :w // 2] = 60
    mask = np.zeros((h, w), np.uint8)
    mask[35:105, 45:135] = 255

    # Paint matches orig OUTSIDE mask (like multi-box LaMa seeing context), shifted & low-contrast INSIDE
    painted = orig.copy()
    painted[mask > 127] = np.clip(
        140 + 0.80 * (orig[mask > 127].astype(np.float32) - 140.0) - 8.0, 0, 255
    ).astype(np.uint8)

    aligned = match_boundary_tone(orig, painted, mask)
    assert np.array_equal(aligned[mask <= 127], painted[mask <= 127])
    err_before = float(np.mean(np.abs(painted[mask > 127].astype(np.float32) - orig[mask > 127].astype(np.float32))))
    err_after = float(np.mean(np.abs(aligned[mask > 127].astype(np.float32) - orig[mask > 127].astype(np.float32))))
    assert err_after < err_before
