from __future__ import annotations

from pathlib import Path


FAST = Path("app/inpaint/fast_lama_inpainter.py")
TESTS = Path("tests/test_fast_inpaint_paths.py")

fast = FAST.read_text(encoding="utf-8")

knob_old = '''_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n'''
knob_new = '''_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0\n)\n# Recover outlined/anti-aliased glyph halos from either side of the local\n# background luminance. Growth is connectivity-limited, spatially bounded,\n# clipped to erase authority, and still subject to the authority-fraction gate.\n_STROKE_REFINE_HALO_RADIUS = _env_int(\n    "MANGA_STROKE_REFINE_HALO_RADIUS", 2, 1, 4\n)\n_STROKE_REFINE_HALO_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_HALO_DELTA_MIN", 8.0, 2.0, 48.0\n)\n_STROKE_REFINE_HALO_STD_SCALE = _env_float(\n    "MANGA_STROKE_REFINE_HALO_STD_SCALE", 1.25, 0.5, 4.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n'''
if fast.count(knob_old) != 1:
    raise RuntimeError("dual-polarity knob marker mismatch")
fast = fast.replace(knob_old, knob_new, 1)

refine_old = '''        authority_pixels = int(np.count_nonzero(authority))\n        model_pixels = int(np.count_nonzero(keep))\n        if authority_pixels <= 0 or model_pixels < 24:\n            return None\n        if model_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:\n            return None\n\n        fringe = cv2.dilate(keep, np.ones((3, 3), dtype=np.uint8), iterations=1)\n        refined = np.where(authority, fringe, 0).astype(np.uint8)\n        refined_pixels = int(np.count_nonzero(refined))\n        if refined_pixels <= 0:\n            return None\n        if refined_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:\n            return None\n        return refined\n'''
refine_new = '''        authority_pixels = int(np.count_nonzero(authority))\n        model_pixels = int(np.count_nonzero(keep))\n        if authority_pixels <= 0 or model_pixels < 24:\n            return None\n        if model_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:\n            return None\n\n        # The dominant-polarity component pass above finds the glyph core, but\n        # manga/manhwa lettering often has an opposite-polarity outline (e.g.\n        # black glyph + white stroke) plus a low-contrast anti-alias fringe.\n        # Grow only through contrast-bearing pixels connected to the confirmed\n        # core; using absolute contrast recovers both bright and dark halos.\n        halo_delta = max(\n            _STROKE_REFINE_HALO_DELTA_MIN,\n            ring_std * _STROKE_REFINE_HALO_STD_SCALE,\n        )\n        halo_candidate = authority & (np.abs(gray_f - background) >= halo_delta)\n        grown = keep > 127\n        halo_kernel = np.ones((3, 3), dtype=np.uint8)\n        for _ in range(_STROKE_REFINE_HALO_RADIUS):\n            adjacent = cv2.dilate(grown.astype(np.uint8), halo_kernel, iterations=1) > 0\n            next_grown = grown | (halo_candidate & adjacent)\n            if np.array_equal(next_grown, grown):\n                break\n            grown = next_grown\n\n        # One final one-pixel support fringe catches outer anti-alias values that\n        # are intentionally close to the fitted background. There is no unbounded\n        # morphology: the candidate can advance at most HALO_RADIUS pixels from\n        # verified support, then one final anti-alias pixel, and never outside the\n        # detector authority. The existing authority-fraction gate remains the\n        # final size safety check.\n        fringe = cv2.dilate(grown.astype(np.uint8), halo_kernel, iterations=1)\n        refined = np.where(authority, fringe, 0).astype(np.uint8)\n        refined_pixels = int(np.count_nonzero(refined))\n        if refined_pixels <= 0:\n            return None\n        if refined_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:\n            return None\n        return refined\n'''
if fast.count(refine_old) != 1:
    raise RuntimeError("dual-polarity refine marker mismatch")
fast = fast.replace(refine_old, refine_new, 1)
FAST.write_text(fast, encoding="utf-8")


tests = TESTS.read_text(encoding="utf-8")
anchor = '''def test_authority_ring_reconstruction_rejects_textured_region():\n'''
new_test = r'''def test_dual_polarity_halo_recovers_bright_outline_around_dark_strokes():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(210 + xx * 0.040 + yy * 0.022, 0, 255)
    background[..., 1] = np.clip(216 + xx * 0.036 + yy * 0.020, 0, 255)
    background[..., 2] = np.clip(224 + xx * 0.030 + yy * 0.017, 0, 255)
    image = background.copy()

    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)

    core = np.zeros((h, w), dtype=np.uint8)
    core[76:84, 84:236] = 255
    core[106:114, 102:220] = 255
    core[136:144, 126:202] = 255
    outline = cv2.dilate(
        core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    outline_only = (outline > 0) & (core == 0)
    image[outline_only] = 252
    image[core > 0] = 4

    authority = build_mask(image.shape[:2], [box], image) > 127
    inpainter = FastInpainter()
    output = inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_authority_gradient_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert not np.any(changed & (~authority))
    # Regression for the real p045 failure: a one-pixel same-polarity fringe
    # leaves the outer white outline visible. The dual-polarity halo must cover
    # almost all of the 2px bright outline before background reconstruction.
    assert float(changed[outline_only].mean()) > 0.95

    text_support = outline > 0
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[text_support].mean()
    )
    assert mae < 6.0


'''
if tests.count(anchor) != 1:
    raise RuntimeError("dual-polarity test anchor mismatch")
tests = tests.replace(anchor, new_test + anchor, 1)
TESTS.write_text(tests, encoding="utf-8")

print("dual-polarity halo candidate applied")
