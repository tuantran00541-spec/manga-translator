from __future__ import annotations

from pathlib import Path


FAST = Path("app/inpaint/fast_lama_inpainter.py")
TESTS = Path("tests/test_fast_inpaint_paths.py")

fast = FAST.read_text(encoding="utf-8")

knob_old = '''_STROKE_REFINE_HALO_STD_SCALE = _env_float(\n    "MANGA_STROKE_REFINE_HALO_STD_SCALE", 1.25, 0.5, 4.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n'''
knob_new = '''_STROKE_REFINE_HALO_STD_SCALE = _env_float(\n    "MANGA_STROKE_REFINE_HALO_STD_SCALE", 1.25, 0.5, 4.0\n)\n# A global median misses pale outlines on gradient bubbles. Fit the local smooth\n# background from the authority ring and recover only residual pixels connected\n# to the already-confirmed glyph support. This remains bounded and authority-only.\n_STROKE_REFINE_SURFACE_HALO_RADIUS = _env_int(\n    "MANGA_STROKE_REFINE_SURFACE_HALO_RADIUS", 6, 1, 12\n)\n_STROKE_REFINE_SURFACE_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_SURFACE_DELTA_MIN", 2.5, 1.0, 16.0\n)\n_STROKE_REFINE_SURFACE_RMSE_SCALE = _env_float(\n    "MANGA_STROKE_REFINE_SURFACE_RMSE_SCALE", 2.5, 1.0, 8.0\n)\n_STROKE_REFINE_SURFACE_RMSE_MAX = _env_float(\n    "MANGA_STROKE_REFINE_SURFACE_RMSE_MAX", 4.0, 1.0, 12.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n'''
if fast.count(knob_old) != 1:
    raise RuntimeError("surface-residual knob marker mismatch")
fast = fast.replace(knob_old, knob_new, 1)

halo_old = '''        halo_candidate = authority & (np.abs(gray_f - background) >= halo_delta)\n        grown = keep > 127\n        halo_kernel = np.ones((3, 3), dtype=np.uint8)\n        for _ in range(_STROKE_REFINE_HALO_RADIUS):\n            adjacent = cv2.dilate(grown.astype(np.uint8), halo_kernel, iterations=1) > 0\n            next_grown = grown | (halo_candidate & adjacent)\n            if np.array_equal(next_grown, grown):\n                break\n            grown = next_grown\n\n        # One final one-pixel support fringe catches outer anti-alias values that\n'''
halo_new = '''        halo_candidate = authority & (np.abs(gray_f - background) >= halo_delta)\n        grown = keep > 127\n        halo_kernel = np.ones((3, 3), dtype=np.uint8)\n        for _ in range(_STROKE_REFINE_HALO_RADIUS):\n            adjacent = cv2.dilate(grown.astype(np.uint8), halo_kernel, iterations=1) > 0\n            next_grown = grown | (halo_candidate & adjacent)\n            if np.array_equal(next_grown, grown):\n                break\n            grown = next_grown\n\n        # Refine the low-contrast tail against a local quadratic background, not\n        # the global ring median. This is the important gradient-background case:\n        # a white anti-alias pixel may be only +3 locally while being almost equal\n        # to the median sampled elsewhere. Fit robustly from clean ring samples,\n        # then allow bounded connectivity growth only through pixels whose color\n        # residual exceeds the ring noise floor.\n        edge_margin = cv2.dilate(\n            edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1\n        ) > 0\n        surface_ring = ring & (~edge_margin)\n        if int(np.count_nonzero(surface_ring)) >= 96:\n            sample_y, sample_x = np.nonzero(surface_ring)\n            design = self._quadratic_design(\n                sample_x, sample_y, crop.shape[1], crop.shape[0]\n            )\n            values = crop[sample_y, sample_x].astype(np.float32)\n            if values.ndim == 1:\n                values = values[:, None]\n\n            fit_keep = np.ones(len(sample_x), dtype=bool)\n            coeff = None\n            for _ in range(3):\n                if int(np.count_nonzero(fit_keep)) < 64:\n                    coeff = None\n                    break\n                coeff, *_ = np.linalg.lstsq(\n                    design[fit_keep], values[fit_keep], rcond=None\n                )\n                predicted = design @ coeff\n                residual = np.sqrt(np.mean((predicted - values) ** 2, axis=1))\n                active = residual[fit_keep]\n                median_residual = float(np.median(active))\n                mad = float(np.median(np.abs(active - median_residual)))\n                limit = median_residual + max(1.5, 3.0 * 1.4826 * mad)\n                next_keep = residual <= limit\n                if int(np.count_nonzero(next_keep)) < 64:\n                    break\n                if np.array_equal(next_keep, fit_keep):\n                    fit_keep = next_keep\n                    break\n                fit_keep = next_keep\n\n            if coeff is not None and int(np.count_nonzero(fit_keep)) >= 64:\n                fitted = design[fit_keep] @ coeff\n                fit_rmse = float(\n                    np.sqrt(np.mean((fitted - values[fit_keep]) ** 2))\n                )\n                if np.isfinite(fit_rmse) and fit_rmse <= _STROKE_REFINE_SURFACE_RMSE_MAX:\n                    target_y, target_x = np.nonzero(authority)\n                    target_design = self._quadratic_design(\n                        target_x, target_y, crop.shape[1], crop.shape[0]\n                    )\n                    target_expected = target_design @ coeff\n                    target_actual = crop[target_y, target_x].astype(np.float32)\n                    if target_actual.ndim == 1:\n                        target_actual = target_actual[:, None]\n                    target_residual = np.sqrt(\n                        np.mean((target_actual - target_expected) ** 2, axis=1)\n                    )\n                    surface_delta = max(\n                        _STROKE_REFINE_SURFACE_DELTA_MIN,\n                        fit_rmse * _STROKE_REFINE_SURFACE_RMSE_SCALE,\n                    )\n                    surface_candidate = np.zeros_like(authority, dtype=bool)\n                    surface_candidate[target_y, target_x] = (\n                        target_residual >= surface_delta\n                    )\n                    for _ in range(_STROKE_REFINE_SURFACE_HALO_RADIUS):\n                        adjacent = (\n                            cv2.dilate(\n                                grown.astype(np.uint8), halo_kernel, iterations=1\n                            )\n                            > 0\n                        )\n                        next_grown = grown | (surface_candidate & adjacent)\n                        if np.array_equal(next_grown, grown):\n                            break\n                        grown = next_grown\n\n        # One final one-pixel support fringe catches outer anti-alias values that\n'''
if fast.count(halo_old) != 1:
    raise RuntimeError("surface-residual halo marker mismatch")
fast = fast.replace(halo_old, halo_new, 1)
FAST.write_text(fast, encoding="utf-8")


tests = TESTS.read_text(encoding="utf-8")
anchor = '''def test_authority_ring_reconstruction_rejects_textured_region():\n'''
new_test = r'''def test_surface_residual_halo_recovers_subtle_outline_on_gradient():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(188 + xx * 0.16 + yy * 0.025, 0, 255)
    background[..., 1] = np.clip(194 + xx * 0.15 + yy * 0.022, 0, 255)
    background[..., 2] = np.clip(202 + xx * 0.13 + yy * 0.018, 0, 255)
    image = background.copy()

    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)

    core = np.zeros((h, w), dtype=np.uint8)
    core[77:84, 86:234] = 255
    core[107:114, 104:218] = 255
    core[137:144, 128:200] = 255
    outline = cv2.dilate(
        core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    outline_only = (outline > 0) & (core == 0)
    # Deliberately subtle: only +4 over the *local* gradient. A single global
    # median plus the old 2px halo cannot reliably recover the full 4px outline.
    lifted = np.clip(background.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    image[outline_only] = lifted[outline_only]
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
    assert float(changed[outline_only].mean()) > 0.97

    text_support = outline > 0
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[text_support].mean()
    )
    assert mae < 4.0


'''
if tests.count(anchor) != 1:
    raise RuntimeError("surface-residual test anchor mismatch")
tests = tests.replace(anchor, new_test + anchor, 1)
TESTS.write_text(tests, encoding="utf-8")

print("surface-residual halo candidate applied")
