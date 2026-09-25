from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.inpaint.lama_inpainter import Inpainter
from app.model_contracts import decode_lama_output


def _inpainter() -> Inpainter:
    inpainter = Inpainter()
    inpainter._ensure_session = lambda: None

    def no_lama(*_args, **_kwargs):
        raise AssertionError("these backgrounds must stay on the Smart Fill route")

    inpainter._lama_fill = no_lama
    return inpainter


def _paint(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    inpainter = _inpainter()
    inpainter._begin_metrics()
    out = inpainter._smart_paint_region(image.copy(), mask, (0, 0, image.shape[1], image.shape[0]))
    return out, inpainter.last_metrics()


def _delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
    lb = cv2.cvtColor(b.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
    la[:, 0] *= 100 / 255
    lb[:, 0] *= 100 / 255
    return np.sqrt(((la - lb) ** 2).sum(axis=1))


def _text_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (100, 110), (600, 290), 255, -1)
    return mask


def _vertical_gradient(top, bottom, h=400, w=700) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    top, bottom = np.array(top, np.float64), np.array(bottom, np.float64)
    return np.broadcast_to(top + (bottom - top) * t, (h, w, 3)).round().astype(np.uint8).copy()


@pytest.mark.parametrize("bgr", [(255, 255, 255), (5, 5, 5), (128, 128, 128), (40, 40, 210), (200, 245, 250)])
def test_flat_backgrounds_keep_the_exact_flat_colour(bgr):
    image = np.full((400, 700, 3), bgr, np.uint8)
    mask = _text_mask(400, 700)
    out, metrics = _paint(image, mask)
    assert np.array_equal(out, image)
    assert metrics.get("smart_fill_regions") == 1
    assert not metrics.get("smart_fill_gradient_regions")


def test_white_to_cream_gradient_follows_the_background():
    # Measured on a real chapter: a caption on white fading to cream was
    # filled with one colour and left a visible rounded patch.
    image = _vertical_gradient((253, 255, 255), (222, 244, 250))
    mask = _text_mask(*image.shape[:2])
    m = mask > 127
    flat = image.copy()
    flat[m] = Inpainter._smart_fill_color(image, mask)

    out, metrics = _paint(image, mask)

    assert metrics.get("smart_fill_gradient_regions") == 1
    new_error, old_error = _delta_e(out[m], image[m]), _delta_e(flat[m], image[m])
    assert new_error.mean() < 0.3 < old_error.mean()
    assert new_error.max() < old_error.max()
    assert np.array_equal(out[~m], image[~m])


def test_gradient_fit_ignores_outline_and_glyph_pixels_in_the_ring():
    image = _vertical_gradient((250, 252, 255), (228, 238, 250))
    truth = image.copy()
    mask = _text_mask(*image.shape[:2])
    # A dark bubble outline and stray glyph pixels just outside the mask.
    cv2.rectangle(image, (96, 106), (604, 294), (20, 20, 20), 1)
    image[292:295, 300:320] = (30, 30, 30)
    m = mask > 127
    fill = np.median(truth[m], axis=0).astype(np.uint8)
    surface = Inpainter._smart_fill_surface(image, mask, fill)
    assert surface is not None
    assert _delta_e(surface, truth[m]).mean() < 0.5


def test_grainy_background_is_not_turned_into_a_fake_gradient():
    rng = np.random.default_rng(0)
    image = np.clip(np.full((400, 700, 3), 150.0) + rng.normal(0, 6, (400, 700, 3)), 0, 255).astype(np.uint8)
    mask = _text_mask(400, 700)
    fill = np.median(image[mask == 0], axis=0).astype(np.uint8)
    assert Inpainter._smart_fill_surface(image, mask, fill) is None


def test_lama_output_rounds_to_nearest_instead_of_truncating():
    class NormalizedContract:
        output_range = "zero_to_one"

    near_white = decode_lama_output(np.full((1, 3, 2, 2), 0.999), NormalizedContract)
    assert near_white[0, 0].tolist() == [255, 255, 255]
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 255, (1, 3, 64, 64))
    decoded = decode_lama_output(values / 255.0, NormalizedContract).astype(np.float64)
    assert abs(float((decoded - values[0].transpose(1, 2, 0)).mean())) < 0.05
