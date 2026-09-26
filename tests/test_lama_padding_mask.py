import numpy as np
import pytest

from app.inpaint.lama_inpainter import Inpainter


def _inpainter(dynamic: bool):
    inpainter = Inpainter()
    inpainter.dynamic_lama = dynamic
    seen = {}

    def run(canvas, mask_canvas):
        seen["canvas"], seen["mask"] = canvas, mask_canvas
        return canvas.copy()

    inpainter._run_lama = run
    return inpainter, seen


@pytest.mark.parametrize("dynamic", [True, False])
def test_padding_marks_text_cut_by_the_crop_edge_as_masked(dynamic):
    # White text touching the top edge, as in a slice cut through an SFX.
    crop = np.full((45, 61, 3), 30, np.uint8)
    crop[:10, 20:40] = 255
    mask = np.zeros((45, 61), np.uint8)
    mask[:12, 18:42] = 255
    inpainter, seen = _inpainter(dynamic)
    fill = inpainter._lama_fill_single_dynamic if dynamic else inpainter._lama_fill_single_fixed
    fill(crop, mask)

    canvas, mask_canvas = seen["canvas"], seen["mask"]
    white = canvas.min(axis=2) > 200
    assert white.any()
    assert not np.any(white & (mask_canvas <= 127)), "no known pixel may carry the text being erased"
