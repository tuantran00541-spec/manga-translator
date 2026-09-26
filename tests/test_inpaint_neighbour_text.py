import numpy as np

import app.parameters as params
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter


def _box(x1, y1, x2, y2):
    return BubbleBox(x1, y1, x2, y2, 0.9, np.full((y2 - y1, x2 - x1), 255, np.uint8),
                     source_role="text_segmenter", safe_to_inpaint=True)


def _run(monkeypatch, hide: bool):
    monkeypatch.setattr(params, "INPAINT_HIDE_NEIGHBOUR_TEXT", hide)
    image = np.zeros((400, 300, 3), np.uint8)
    image[:, :, 1] = np.arange(300, dtype=np.uint8)[None, :] // 2   # textured, so Smart Fill stays out
    image[60:125, 40:120] = 255                                       # upper bubble text
    image[162:222, 40:120] = 255                                      # lower bubble, 37 px below
    calls = []
    inpainter = Inpainter()
    inpainter.dynamic_lama = True
    inpainter._ensure_session = lambda: None

    def fake_lama(canvas, mask_canvas):
        calls.append((canvas.copy(), mask_canvas.copy()))
        out = canvas.copy()
        out[mask_canvas > 127] = 7
        return out

    inpainter._run_lama = fake_lama
    # Lower box first: the detector orders by confidence, not position. Bubble-shaped
    # clusters get square crops, so each crop reaches into the other bubble.
    result = inpainter.inpaint(image, [_box(40, 162, 120, 222), _box(40, 60, 120, 125)])
    return image, result, calls


def test_neighbour_text_is_hidden_and_only_the_current_cluster_is_written(monkeypatch):
    image, result, calls = _run(monkeypatch, hide=True)
    assert len(calls) == 2
    first_canvas, first_mask = calls[0]
    # The first call is the upper cluster, and the lower text is not visible to it.
    upper_rows = np.nonzero((first_mask > 127).any(axis=1))[0]
    assert upper_rows.min() < 100, "clusters are cleaned top to bottom"
    visible = (first_mask <= 127) & (first_canvas.min(axis=2) == 255)
    assert not visible.any(), "no unmasked white text pixel reaches LaMa"
    assert (result[60:125, 40:120] == 7).all() and (result[162:222, 40:120] == 7).all()


def test_off_keeps_the_old_behaviour(monkeypatch):
    _, _, calls = _run(monkeypatch, hide=False)
    first_canvas, first_mask = calls[0]
    visible = (first_mask <= 127) & (first_canvas.min(axis=2) == 255)
    assert visible.any(), "without the option the other cluster's text is visible context"
