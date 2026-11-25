import cv2
import numpy as np
import pytest

from app.detector.boxes import BubbleBox
from app.detector.ink_mask import PANEL_INK_MASK
from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.panel_fill import fit_panel_grid
from app.page_processing import _prepare_panel_fill


def _panels(count, textured=False):
    image = np.full((168, 540, 3), (30, 46, 92), np.uint8)
    foreground = np.zeros(image.shape[:2], np.uint8)
    frames = np.zeros_like(foreground)
    vertices = np.array([(0, 25), (25, 0), (75, 0), (100, 25),
                         (100, 75), (75, 100), (25, 100), (0, 75)], np.int32)
    ys, xs = np.indices(foreground.shape)
    for i in range(count):
        x = 24 + i * 128
        polygon = vertices + [x, 30]
        cell = cv2.fillPoly(np.zeros_like(foreground), [polygon], 255) > 0
        value = 160 + 4 * i + ys[cell] // 8 + xs[cell] // 20
        if textured:
            value = np.where((xs[cell] % 12) < 6, 66, 213)
        image[cell] = np.stack((value, value + 6, value + 9), axis=1)
        cv2.polylines(frames, [polygon], True, 255, 5, cv2.LINE_AA)
        cv2.putText(foreground, str(88 - i), (x + 23, 96), cv2.FONT_HERSHEY_DUPLEX, 1.35, 255, 5)
    image[frames > 0] = (13, 14, 18)
    original_surface = image.copy()
    image[foreground > 0] = (250, 250, 250)
    return image, foreground, frames, original_surface


def test_repeated_panels_use_only_local_surfaces_and_preserve_frames(monkeypatch):
    image, letters, frames, expected = _panels(4)
    box = BubbleBox(0, 0, image.shape[1], image.shape[0], .91,
                    np.full(letters.shape, 255, np.uint8),
                    semantic_type="free_text", mask_source="text_segmenter", safe_to_inpaint=True)
    assert _prepare_panel_fill(image, box, letters)
    assert box.mask_source == PANEL_INK_MASK

    monkeypatch.setattr("app.inpaint.lama_inpainter.load_inpaint_settings",
                        lambda: {"enabled": True, "hybrid_auto": True})
    monkeypatch.setattr("app.inpaint.lama_inpainter.inpaint_regions_via_api",
                        lambda *args, **kwargs: pytest.fail("sent panels to the hybrid provider"))
    inpainter = Inpainter()
    monkeypatch.setattr(inpainter, "_lama_fill", lambda *args, **kwargs: pytest.fail("sent panels to LaMa"))
    cleaned = inpainter.inpaint(image, [box], protected_regions=[{"x1": 0, "x2": 140, "y1": 0, "y2": 168}])

    assert np.array_equal(cleaned[:, :140], image[:, :140])
    assert np.array_equal(cleaned[frames > 0], image[frames > 0])
    assert np.array_equal(cleaned[box.mask == 0], image[box.mask == 0])
    restored = (letters > 0) & (np.indices(letters.shape)[1] >= 140)
    errors = np.abs(cleaned[restored].astype(int) - expected[restored].astype(int))
    assert np.max(errors) <= 3
    assert inpainter.last_metrics()["smart_fill_regions"] == 1


@pytest.mark.parametrize("count,textured", [(2, False), (4, True)])
def test_panel_fill_declines_weakly_supported_surfaces(count, textured):
    image, letters, _, _ = _panels(count, textured)
    assert fit_panel_grid(image, letters) is None


def test_panel_fill_declines_text_that_is_separate_from_the_frames():
    image, letters, _, _ = _panels(4)
    image[4:13, 210:223] = 250
    letters[4:13, 210:223] = 255
    assert fit_panel_grid(image, letters) is None
