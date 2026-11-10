import numpy as np

from app.detector.boxes import BubbleBox
from app.inpaint.mask_geometry import remap_local_mask_page_space
from app.page_processing import _contained, _fold_nested
from app.pipeline import ChapterPipeline


def test_a_mask_that_only_partly_covers_a_stroke_is_repaired_from_the_ink():
    """Rerun-4 Mount Hua p075: the letter model read the middle of the
    giant SFX and missed its ends, so the mask covered a third of the
    strokes and the clean left 47% of the ink smeared on the page. Ink
    that touches the mask and belongs to the box joins it; a shape
    bigger than the box, or one the mask never touched, stays out."""
    import cv2

    from app.detector.mask_builder import build_mask

    image = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(image, (100, 80), (130, 220), (0, 0, 0), -1)  # a stroke, mask covers only its middle
    cv2.rectangle(image, (135, 100), (160, 280), (0, 0, 0), -1)  # art beside it, running past the box's bottom
    cv2.rectangle(image, (280, 80), (310, 220), (0, 0, 0), -1)  # a stroke the mask never touched
    mask = np.zeros((200, 300), np.uint8)
    mask[70:110, 50:80] = 255  # the segmenter found only the stroke's middle
    box = BubbleBox(50, 50, 350, 250, 0.9, mask, source_model="text_segmenter",
                    class_name="text", semantic_type="free_text", mask_source="text_segmenter",
                    safe_to_inpaint=True, ocr_eligible=True, source_role="text_segmenter")
    built = build_mask((300, 400), [box], image)
    assert built[100, 115] > 127 and built[200, 115] > 127, "the whole stroke joins the mask"
    assert built[150, 147] <= 127, "the art bar runs past the box, so it stays out"
    assert built[150, 295] <= 127, "ink the mask never touched is not the mask's to take"


def test_remap_local_mask_page_space_copies_intersection():
    source_box = {"x1": 0, "y1": 0, "x2": 4, "y2": 4}
    target_box = {"x1": 2, "y1": 2, "x2": 6, "y2": 6}
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255

    result = remap_local_mask_page_space(mask, source_box, target_box)

    assert result is not None
    assert result.shape == (4, 4)
    assert result[0, 0] == 255


def test_map_seam_detection_to_page_clips_mask_and_coordinates():
    mask = np.arange(1, 13, dtype=np.uint8).reshape(4, 3)
    box = BubbleBox(1, 0, 4, 4, 1.0, mask.copy())

    mapped = ChapterPipeline._map_seam_detection_to_page(
        box,
        seam_source_y1=10,
        page_source_y1=12,
        page_height=3,
    )

    assert mapped is not None
    assert (mapped.y1, mapped.y2) == (0, 2)
    assert mapped.mask is not None
    assert mapped.mask.shape == (2, 3)


def test_contained_uses_half_area_threshold():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask[:10, :10].copy())

    assert _contained(small, large) is True


def test_fold_nested_merges_a_small_box_into_its_containing_box():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask.copy())

    folded = _fold_nested([large, small])

    assert len(folded) == 1


def _lettered_box(x1, y1, x2, y2, rows, score, core, cols=None):
    """A box whose letters fill ``rows`` (row ranges in box coords) and
    ``cols`` (column ranges; all columns when omitted)."""
    letters = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for r0, r1 in rows:
        if cols is None:
            letters[r0:r1, :] = True
        else:
            for c0, c1 in cols:
                letters[r0:r1, c0:c1] = True
    return BubbleBox(x1, y1, x2, y2, score, np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255,
                     letters=letters, core=core)


def test_fold_nested_keeps_two_texts_with_separate_letter_rows_apart():
    """A small cloud tucking under a big oval is a second text, not a
    piece of it: its letter rows, inside its own core, never meet the
    oval's, and its columns sit beside the oval's, so the fold must not
    join them even though the rect is 65% inside the oval's."""
    # Real column ranges: the oval's letters run x 210-514, the cloud's
    # x 515-634, so the core-clipped letter boxes are x-disjoint.
    oval = _lettered_box(135, 1346, 644, 1677, [(75, 300)], 0.88, (187, 1398, 525, 1608),
                         cols=[(75, 379)])
    cloud = _lettered_box(468, 1569, 677, 1708, [(43, 96)], 0.82, (491, 1592, 654, 1685),
                          cols=[(47, 166)])

    folded = _fold_nested([oval, cloud])

    assert len(folded) == 2
    assert {(box.x1, box.y1, box.x2, box.y2) for box in folded} == {
        (135, 1346, 644, 1677), (468, 1569, 677, 1708)}


def test_fold_nested_still_merges_a_text_flowing_into_the_next_lobe():
    """One text flowing down into the next lobe is not two texts: its
    letter rows never meet inside the cores, but the lower lines sit
    under the upper ones, so their columns overlap and the fold joins."""
    upper = _lettered_box(0, 1959, 371, 2330, [(65, 216)], 0.88, (7, 2006, 183, 2196),
                          cols=[(12, 159)])
    lower = _lettered_box(21, 2147, 506, 2382, [(59, 172)], 0.82, (60, 2186, 467, 2343),
                          cols=[(63, 435)])

    folded = _fold_nested([upper, lower])

    assert len(folded) == 1
    assert (folded[0].x1, folded[0].y1, folded[0].x2, folded[0].y2) == (0, 1959, 506, 2382)


def test_fold_nested_still_merges_letter_rows_that_share_a_line():
    """Two halves of one line read as one text: their core-clipped letter
    rows meet, so the fold still joins them."""
    left = _lettered_box(0, 0, 400, 100, [(20, 80)], 0.9, (0, 0, 400, 100))
    right = _lettered_box(200, 0, 500, 100, [(20, 80)], 0.8, (200, 0, 500, 100))

    folded = _fold_nested([left, right])

    assert len(folded) == 1
    assert (folded[0].x1, folded[0].y1, folded[0].x2, folded[0].y2) == (0, 0, 500, 100)
