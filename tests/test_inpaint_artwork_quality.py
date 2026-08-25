import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.inpaint.lama_inpainter import Inpainter


def _mask_rect(h=96, w=160, x1=55, y1=36, x2=105, y2=60):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def test_flat_white_region_keeps_smart_fill_shortcut():
    image = np.full((96, 160, 3), 255, dtype=np.uint8)
    mask = _mask_rect()
    image[mask > 127] = 20

    inpainter = Inpainter.__new__(Inpainter)
    calls = []

    def fake_lama(*args, **kwargs):
        calls.append(True)
        return args[0]

    inpainter._lama_fill = fake_lama
    result = inpainter._smart_paint_region(image.copy(), mask, (0, 0, 160, 96))

    assert not calls
    assert np.all(result[mask > 127] >= 250)


def test_art_edge_near_mask_routes_to_lama_instead_of_flat_fill():
    image = np.full((96, 160, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 48), (159, 48), (20, 20, 20), 2)
    mask = _mask_rect(y1=38, y2=58)
    image[mask > 127] = 245

    inpainter = Inpainter.__new__(Inpainter)
    calls = []

    def fake_lama(full_image, crop, local_mask, crop_box, feather=False):
        calls.append(True)
        return full_image

    inpainter._lama_fill = fake_lama
    inpainter._smart_paint_region(image.copy(), mask, (0, 0, 160, 96))

    assert calls == [True]


def test_nearly_black_textured_region_routes_to_lama():
    image = np.zeros((96, 160, 3), dtype=np.uint8)
    cv2.line(image, (0, 48), (159, 48), (18, 18, 18), 3)
    mask = _mask_rect(y1=38, y2=58)
    image[mask > 127] = 235

    inpainter = Inpainter.__new__(Inpainter)
    calls = []

    def fake_lama(full_image, crop, local_mask, crop_box, feather=False):
        calls.append(True)
        return full_image

    inpainter._lama_fill = fake_lama
    inpainter._smart_paint_region(image.copy(), mask, (0, 0, 160, 96))

    assert calls == [True]


def test_missing_detector_mask_is_not_expanded_to_rectangle():
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    box = BubbleBox(30, 20, 90, 55, 0.9, None)
    mask = build_mask((80, 120), [box], image)
    assert not np.any(mask > 127)


def test_geometry_override_missing_mask_fails_safe_instead_of_erasing_art():
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    cv2.line(image, (10, 38), (110, 38), (20, 20, 20), 2)
    # Detector-origin boxes keep model confidence after a geometry override. If
    # the old pipeline drops their segmentation mask, cleanup must now no-op
    # instead of converting the entire edited rectangle into a destructive mask.
    overridden = BubbleBox(25, 18, 95, 58, 0.86, None)
    mask = build_mask((80, 120), [overridden], image)
    assert not np.any(mask > 127)


def test_manual_box_keeps_explicit_rectangle_fallback_contract():
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    # Manual boxes are persisted by the current pipeline with confidence=1.0.
    manual = BubbleBox(30, 20, 90, 55, 1.0, None)
    mask = build_mask((80, 120), [manual], image)
    assert np.any(mask > 127)


def test_fixed_long_crop_uses_tiled_path():
    inpainter = Inpainter.__new__(Inpainter)
    inpainter.dynamic_lama = False
    calls = []

    def tiled(crop, mask):
        calls.append("tiled")
        return crop.copy()

    def single(crop, mask):
        calls.append("single")
        return crop.copy()

    inpainter._lama_fill_tiled = tiled
    inpainter._lama_fill_single = single

    image = np.zeros((160, 760, 3), dtype=np.uint8)
    mask = np.zeros((160, 760), dtype=np.uint8)
    mask[50:110, 80:680] = 255
    inpainter._lama_fill(image.copy(), image.copy(), mask, (0, 0, 760, 160))
    assert calls == ["tiled"]


def test_fixed_near_square_crop_stays_single_call():
    inpainter = Inpainter.__new__(Inpainter)
    inpainter.dynamic_lama = False
    calls = []

    def tiled(crop, mask):
        calls.append("tiled")
        return crop.copy()

    def single(crop, mask):
        calls.append("single")
        return crop.copy()

    inpainter._lama_fill_tiled = tiled
    inpainter._lama_fill_single = single

    image = np.zeros((560, 560, 3), dtype=np.uint8)
    mask = np.zeros((560, 560), dtype=np.uint8)
    mask[180:380, 180:380] = 255
    inpainter._lama_fill(image.copy(), image.copy(), mask, (0, 0, 560, 560))
    assert calls == ["single"]


def test_auto_composite_never_changes_pixels_outside_effective_mask():
    inpainter = Inpainter.__new__(Inpainter)
    inpainter.dynamic_lama = False

    def single(crop, mask):
        return np.full_like(crop, 255)

    inpainter._lama_fill_single = single
    image = np.full((80, 120, 3), 37, dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[25:45, 40:70] = 255
    result = inpainter._lama_fill(image.copy(), image.copy(), mask, (0, 0, 120, 80))

    assert np.array_equal(result[mask <= 127], image[mask <= 127])
    assert np.all(result[mask > 127] == 255)


def test_roi_metric_exposes_local_line_art_loss_hidden_by_global_error():
    target = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.line(target, (250, 300), (650, 300), (20, 20, 20), 3)
    mask = np.zeros(target.shape[:2], dtype=np.uint8)
    mask[285:315, 420:480] = 255

    damaged = target.copy()
    damaged[mask > 127] = 255

    global_mae = float(np.abs(damaged.astype(np.float32) - target.astype(np.float32)).mean())
    masked_mae = float(
        np.abs(damaged.astype(np.float32) - target.astype(np.float32))[mask > 127].mean()
    )

    assert global_mae < 1.0
    assert masked_mae > 5.0
