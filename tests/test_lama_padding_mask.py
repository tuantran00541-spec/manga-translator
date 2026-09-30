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


@pytest.mark.parametrize("size", [(600, 216), (700, 250), (45, 61), (512, 200)])
def test_the_dynamic_model_only_sees_sides_that_are_multiples_of_16(size):
    # A 600x216 crop became a 184-wide canvas and the model raised on it, failing the whole page.
    inpainter, seen = _inpainter(True)
    crop = np.full((*size, 3), 90, np.uint8)
    mask = np.zeros(size, np.uint8)
    mask[size[0] // 3:size[0] // 2, size[1] // 4:3 * size[1] // 4] = 255
    result = inpainter._lama_fill_single_dynamic(crop, mask)
    assert result.shape == crop.shape
    assert seen["canvas"].shape[0] % 16 == 0 and seen["canvas"].shape[1] % 16 == 0
