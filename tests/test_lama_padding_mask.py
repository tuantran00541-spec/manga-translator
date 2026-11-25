import cv2
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
    inpainter, seen = _inpainter(True)
    crop = np.full((*size, 3), 90, np.uint8)
    mask = np.zeros(size, np.uint8)
    mask[size[0] // 3:size[0] // 2, size[1] // 4:3 * size[1] // 4] = 255
    result = inpainter._lama_fill_single_dynamic(crop, mask)
    assert result.shape == crop.shape
    assert seen["canvas"].shape[0] % 16 == 0 and seen["canvas"].shape[1] % 16 == 0


def test_dynamic_lama_uses_384_for_smooth_crops_and_512_for_textured_crops():
    inpainter, seen = _inpainter(True)
    mask = np.zeros((512, 512), np.uint8)
    mask[180:330, 180:330] = 255

    smooth = np.full((512, 512, 3), 230, np.uint8)
    inpainter._lama_fill_single_dynamic(smooth, mask)
    assert max(seen["canvas"].shape[:2]) == 384

    textured = np.random.default_rng(11).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    inpainter._lama_fill_single_dynamic(textured, mask)
    assert max(seen["canvas"].shape[:2]) == 512


def test_lama_fill_tightens_sparse_mask_crop_before_inference():
    inpainter, _ = _inpainter(True)
    inpainter._ensure_session = lambda: None
    seen_shapes = []
    inpainter._lama_fill_single = lambda c, m: (seen_shapes.append(c.shape[:2]), c.copy())[1]

    image = np.random.default_rng(5).integers(0, 255, (800, 800, 3), dtype=np.uint8)
    crop = image[100:600, 100:600]
    local_mask = np.zeros((500, 500), np.uint8)
    local_mask[220:260, 200:280] = 255  # small 40x80 mask inside 500x500 crop
    inpainter._lama_fill(image, crop, local_mask, (100, 100, 600, 600))
    assert seen_shapes and seen_shapes[0][0] < 250 and seen_shapes[0][1] < 250


def test_border_protected_display_inpaints_art_notches_per_component():
    inpainter, _ = _inpainter(True)
    seen_shapes = []
    inpainter._lama_fill_single = lambda c, m: (seen_shapes.append(c.shape[:2]), c.copy())[1]

    img = np.full((600, 800, 3), 255, np.uint8)
    # Colored panel art strip along bottom
    img[480:600, :] = (20, 40, 200)
    mask = np.zeros((600, 800), np.uint8)
    # Two small widely separated protrusions touching the colored art
    mask[460:500, 80:120] = 255
    mask[460:500, 680:720] = 255
    inpainter._paint_border_protected_display(img, mask, (0, 0, 800, 600))
    assert len(seen_shapes) == 2
    assert all(w < 200 and h < 200 for h, w in seen_shapes)


def test_border_protected_display_uses_original_context_and_conceals_full_ink():
    inpainter, _ = _inpainter(True)
    seen = []
    inpainter._lama_fill_single = lambda crop, hole: (seen.append((crop.copy(), hole.copy())), crop.copy())[1]
    image = np.full((240, 260, 3), 253, np.uint8)
    image[140:] = (35, 90, 160)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[113:166, 122:146] = 255
    image[mask > 0] = (16, 16, 16)

    inpainter._paint_border_protected_display(image, mask, (0, 0, 260, 240))
    assert len(seen) == 1
    crop, model_mask = seen[0]
    assert np.count_nonzero(model_mask) == np.count_nonzero(mask)
    assert np.all(crop[model_mask > 0] == 16)


def test_sparse_contour_at_the_hole_keeps_native_resolution():
    inpainter, seen = _inpainter(True)
    image = np.full((512, 512, 3), 215, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[230:282, 230:282] = 255
    cv2.line(image, (165, 212), (344, 296), (26, 32, 42), 2, cv2.LINE_AA)

    inpainter._lama_fill_single_dynamic(image, mask)
    assert max(seen["canvas"].shape[:2]) == 512


def test_mask_repair_does_not_claim_art_next_to_confirmed_text():
    from app.detector.boxes import BubbleBox
    from app.detector.mask_builder import build_mask

    canvas = np.full((120, 200, 3), (220, 180, 140), np.uint8)
    y, x = np.indices(canvas.shape[:2])
    canvas[(x >= 100) & (y > .77 * (x - 103) + 52)] = (125, 181, 175)
    cv2.line(canvas, (101, 51), (169, 103), (45, 35, 25), 3)
    canvas[44:60, 72:92] = (30, 30, 30)
    mask = np.zeros((100, 170), np.uint8)
    mask[34:50, 57:77] = 255
    box = BubbleBox(15, 10, 185, 110, .95, mask, safe_to_inpaint=True)

    expanded = build_mask(canvas.shape[:2], [box], canvas)
    assert expanded[51, 82] == 255
    assert expanded[81, 140] == 0


def test_build_mask_can_skip_ink_repair_and_runtime_threads_stay_balanced():
    from app.detector.boxes import BubbleBox
    from app.detector import mask_builder
    from app.runtime_responsiveness import recommended_ort_intra_threads, recommended_workers

    crop = np.full((120, 200, 3), 255, np.uint8)
    crop[40:80, 40:160] = 0
    box = BubbleBox(40, 40, 160, 80, 0.9, np.zeros((40, 120), np.uint8), safe_to_inpaint=True)
    box.mask[10:20, 10:30] = 255
    m_skip = mask_builder.build_mask((120, 200), [box], crop, repair_ink=False)
    m_repair = mask_builder.build_mask((120, 200), [box], crop, repair_ink=True)
    assert int((m_skip > 0).sum()) < int((m_repair > 0).sum())

    for cpus in (4, 8, 12, 16):
        workers = recommended_workers(cpus)
        threads = recommended_ort_intra_threads(cpus)
        assert threads >= 2
        assert workers * threads <= cpus
