import numpy as np
import pytest

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


def test_tiles_holding_only_hidden_text_are_not_painted():
    inpainter = Inpainter()
    inpainter.dynamic_lama = True
    runs = []
    inpainter._lama_fill_single = lambda crop, mask: (runs.append(1), crop)[1]
    crop = np.zeros((600, 3000, 3), np.uint8)
    hole, own = np.zeros((600, 3000), np.uint8), np.zeros((600, 3000), np.uint8)
    own[100:200, 100:300] = 255
    hole[100:200, 2600:2800] = 255  # another text, far right, only hidden
    inpainter._lama_fill_tiled(crop, np.maximum(hole, own), own=own)
    assert len(runs) == 1


def test_lama_never_sees_the_pixels_under_the_hole():
    inpainter = Inpainter()
    inpainter._ensure_session = lambda: None
    inpainter.dynamic_lama = True
    inpainter.image_input, inpainter.mask_input, inpainter.output_name = "image", "mask", "output"
    fed = {}

    class Session:
        def run(self, _names, feed):
            fed.update(feed)
            raise StopIteration

    inpainter.session = Session()
    canvas = np.full((16, 16, 3), 250, np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    with pytest.raises(StopIteration):
        inpainter._run_lama(canvas, mask)
    assert fed["image"][0, :, 4:12, 4:12].max() == 0 and fed["image"][0, :, 0, 0].min() > 0.9
