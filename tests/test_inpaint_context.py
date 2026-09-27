import numpy as np

from app.detector.boxes import BubbleBox
from app.inpaint.lama_inpainter import Inpainter


def _box(x1, y1, x2, y2):
    return BubbleBox(x1, y1, x2, y2, 0.9, np.full((y2 - y1, x2 - x1), 255, np.uint8), safe_to_inpaint=True)


def test_text_not_erased_yet_is_hidden_from_lama():
    image = np.random.default_rng(3).integers(0, 255, (1000, 1000, 3), dtype=np.uint8)  # art, so no flat fill
    inpainter = Inpainter()
    inpainter._ensure_session = lambda: None
    inpainter.dynamic_lama = True
    seen = []
    inpainter._lama_fill_single = lambda crop, mask: (seen.append(mask.copy()), crop)[1]
    # Two texts far enough apart to be inpainted one after the other, close enough to share a crop.
    inpainter.inpaint(image, [_box(100, 100, 400, 300), _box(100, 380, 400, 440)])
    first, second = seen
    assert first[410, 250] == 255, "the second text is a hole while the first is inpainted"
    assert second[5, 250] == 0, "the first text, already erased, is context again"
