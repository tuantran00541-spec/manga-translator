from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one marker, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Dimension alone is not a sufficient memory guard for arbitrary dynamic input.
# Pair it with a pixel budget so elongated manga crops can stay native while
# truly large 2-D areas still fall back to overlapping tiles.
replace_once(
    "app/parameters.py",
    '''DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM = _env_int(\n    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 1024, minimum=128, maximum=4096\n)\n''',
    '''DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM = _env_int(\n    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 1024, minimum=128, maximum=4096\n)\nDYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS = _env_int(\n    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS",\n    1024 * 1024,\n    minimum=128 * 128,\n    maximum=4096 * 4096,\n)\n''',
)

replace_once(
    "app/inpaint/lama_inpainter.py",
    '''from app.parameters import (\n    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,\n''',
    '''from app.parameters import (\n    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,\n    DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS,\n''',
)

replace_once(
    "app/inpaint/lama_inpainter.py",
    '''            "lama_model_ms": 0,\n            "session_lock_wait_ms": 0,\n''',
    '''            "lama_model_ms": 0,\n            "lama_native_single_regions": 0,\n            "lama_tiled_regions": 0,\n            "session_lock_wait_ms": 0,\n''',
)

old = '''        # Wide/tall free text loses background detail when a dynamic LaMa crop\n        # is squeezed to 512px just as it does with the fixed model. Preserve\n        # native detail with overlapping tiles for both backends. Small and\n        # near-square regions retain the single-call fast path.\n        if long_crop or texture_tiling or (feather and max_dim > INPAINT_SIZE):\n            painted = self._lama_fill_tiled(crop, local_mask)\n        else:\n            painted = self._lama_fill_single(crop, local_mask)\n'''
new = '''        # Dynamic LaMa accepts arbitrary native dimensions. Do not split a\n        # medium elongated text ROI merely because its aspect ratio is large:\n        # grid tiling destroys the global context LaMa's Fourier path is meant\n        # to use and is a known source of polygon/facet seams. A pixel budget\n        # keeps this safe on CPU; genuinely large crops still use tiles.\n        crop_pixels = int(crop_h * crop_w)\n        dynamic_native_ok = bool(\n            self.dynamic_lama\n            and not feather\n            and max_dim <= DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM\n            and crop_pixels <= DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS\n        )\n        if dynamic_native_ok:\n            self._metric_add("lama_native_single_regions")\n            painted = self._lama_fill_single(crop, local_mask)\n        elif long_crop or texture_tiling or (feather and max_dim > INPAINT_SIZE):\n            self._metric_add("lama_tiled_regions")\n            painted = self._lama_fill_tiled(crop, local_mask)\n        else:\n            painted = self._lama_fill_single(crop, local_mask)\n'''
replace_once("app/inpaint/lama_inpainter.py", old, new)

path = Path("tests/test_fast_inpaint_paths.py")
text = path.read_text(encoding="utf-8")
if "test_dynamic_long_crop_uses_native_single_call_within_pixel_budget" in text:
    raise RuntimeError("dynamic native budget test already exists")
text += r'''


def test_dynamic_long_crop_uses_native_single_call_within_pixel_budget():
    # Old policy tiled this crop solely because 900/400 >= 2 and max_dim > 512.
    # Dynamic LaMa should preserve one global-context call when the area is safe.
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(400, 900, 3), dtype=np.uint8)
    mask = np.zeros((400, 900), dtype=np.uint8)
    mask[30:370, 40:860] = 255

    inpainter = FastInpainter()
    inpainter._begin_metrics()
    inpainter.dynamic_lama = True
    inpainter._ensure_session = lambda: None
    shapes = []

    def fake_run_lama(canvas, mask_canvas):
        shapes.append(canvas.shape[:2])
        return canvas.copy()

    inpainter._run_lama = fake_run_lama
    output = inpainter._lama_fill(
        image.copy(),
        image.copy(),
        mask,
        (0, 0, 900, 400),
    )
    metrics = inpainter.last_metrics()

    assert output.shape == image.shape
    assert len(shapes) == 1
    assert shapes[0][0] >= 400
    assert shapes[0][1] >= 900
    assert metrics["lama_native_single_regions"] == 1
    assert metrics["lama_tiled_regions"] == 0
'''
path.write_text(text, encoding="utf-8")
