import numpy as np
import cv2

from app.inpaint.lama_inpainter import Inpainter


def _dynamic_test_inpainter(calls):
    inp = Inpainter.__new__(Inpainter)
    inp.dynamic_lama = True

    def fake_run(canvas, mask):
        calls.append((canvas.shape[:2], mask.shape))
        return canvas.copy()

    inp._run_lama = fake_run
    return inp


def test_dynamic_lama_preserves_native_resolution_and_pads_to_stride_8():
    calls = []
    inp = _dynamic_test_inpainter(calls)
    crop = np.zeros((218, 481, 3), dtype=np.uint8)
    mask = np.zeros((218, 481), dtype=np.uint8)
    mask[40:100, 80:180] = 255

    output = inp._lama_fill_single(crop, mask)

    assert output.shape == crop.shape
    assert calls == [((224, 488), (224, 488))]


def test_dynamic_lama_only_downscales_when_crop_exceeds_legacy_budget():
    calls = []
    inp = _dynamic_test_inpainter(calls)
    crop = np.zeros((700, 900, 3), dtype=np.uint8)
    mask = np.zeros((700, 900), dtype=np.uint8)
    mask[200:300, 300:500] = 255

    output = inp._lama_fill_single(crop, mask)

    assert output.shape == crop.shape
    h, w = calls[0][0]
    assert h % 8 == 0 and w % 8 == 0
    assert max(h, w) <= 512


def test_fixed_path_uses_area_when_resizing_enlarged_inference_back_down(monkeypatch):
    inp = Inpainter.__new__(Inpainter)
    inp.dynamic_lama = False
    inp._run_lama = lambda canvas, mask: canvas.copy()
    crop = np.zeros((120, 300, 3), dtype=np.uint8)
    mask = np.zeros((120, 300), dtype=np.uint8)
    mask[20:80, 40:180] = 255

    real_resize = cv2.resize
    calls = []

    def recording_resize(src, dsize, *args, **kwargs):
        calls.append((dsize, kwargs.get("interpolation")))
        return real_resize(src, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", recording_resize)
    inp._lama_fill_single(crop, mask)

    assert calls[-1] == ((300, 120), cv2.INTER_AREA)


def test_tile_overlap_weight_actually_tapers():
    weight = Inpainter._tile_weight(512, 64, True, True)
    assert weight[0] == 0.0
    assert weight[-1] == 0.0
    assert weight[63] == 1.0
    assert weight[-64] == 1.0
    assert weight[256] == 1.0
