"""Grouping respects bubble masks: two lobes of one waist stay two objects.

The waist split (chunk 1) divides one double bubble into a box per lobe,
each with its own lobe mask. Every later stage that joins boxes or objects
back together must refuse the join when the two sides' bubble masks are
different lobes -- otherwise the split is undone and render lays one block
across the waist again (rerun-5 ch42 p038: one object, five lines).
"""
import numpy as np

from app.detector.boxes import BubbleBox, apply_final_nms
from app.page_processing import _fold_nested


def _separate(a, b):
    from app.detector.boxes import bubble_masks_are_separate_lobes

    return bubble_masks_are_separate_lobes(a, b)


def _lobe_mask(bbox, lo, hi, kind="bubble"):
    """A mask whose every band is one bbox-local interval [lo, hi]."""
    return {"bbox": list(bbox), "cx": (bbox[0] + bbox[2]) // 2,
            "cy": (bbox[1] + bbox[3]) // 2,
            "rows": [[[lo, hi]] for _ in range(24)], "kind": kind}


def _box(x1, y1, x2, y2, mask, letters_rows=(), letters_cols=(0, 1), conf=0.9,
         model="kiuyha", role="text_segmenter", sem="text_segmenter"):
    letters = np.zeros((y2 - y1, x2 - x1), bool)
    for r0, r1 in letters_rows:
        letters[r0:r1, letters_cols[0]:letters_cols[1]] = True
    return BubbleBox(x1, y1, x2, y2, conf, mask=np.full((y2 - y1, x2 - x1), 255, np.uint8),
                     letters=letters, bubble_mask=mask, core=(x1, y1, x2, y2),
                     source_model=model, source_role=role, semantic_type=sem)



def test_separate_lobes_side_by_side():
    a = _lobe_mask((0, 0, 100, 200), 10, 90)
    b = _lobe_mask((100, 0, 200, 200), 10, 90)
    assert _separate(a, b)


def test_separate_lobes_stacked():
    a = _lobe_mask((111, 55, 428, 241), 5, 312)
    b = _lobe_mask((38, 264, 592, 614), 5, 549)
    assert _separate(a, b)


def test_same_mask_twice_is_not_separate():
    a = _lobe_mask((0, 0, 100, 200), 10, 90)
    assert not _separate(a, a)
    assert not _separate(a, _lobe_mask((0, 0, 100, 200), 10, 90))


def test_overlapping_masks_of_one_bubble_are_not_separate():
    a = _lobe_mask((0, 0, 200, 200), 10, 190)
    b = _lobe_mask((100, 0, 300, 200), 10, 190)
    assert not _separate(a, b)


def test_small_overlap_below_threshold_is_separate_above_is_not():
    a = _lobe_mask((0, 0, 100, 200), 10, 90)
    assert _separate(a, _lobe_mask((85, 0, 185, 200), 0, 80))
    assert not _separate(a, _lobe_mask((78, 0, 178, 200), 0, 80))


def test_multi_interval_rows_count_only_real_overlap():
    a = {"bbox": [0, 0, 100, 200], "cx": 50, "cy": 100,
         "rows": [[[10, 40], [60, 90]] for _ in range(24)], "kind": "bubble"}
    b = _lobe_mask((0, 0, 100, 200), 45, 55)
    assert _separate(a, b)


def test_rect_masks_and_missing_masks_never_veto():
    a = _lobe_mask((0, 0, 100, 200), 10, 90, kind="rect")
    b = _lobe_mask((100, 0, 200, 200), 10, 90, kind="rect")
    assert not _separate(a, b)
    assert not _separate(a, None)
    assert not _separate(None, b)
    empty = _lobe_mask((0, 0, 100, 200), 10, 90)
    empty["rows"] = [[] for _ in range(24)]
    assert not _separate(empty, b)



def test_fold_refuses_a_box_whose_mask_is_the_other_lobe():
    head = _box(0, 0, 200, 200, _lobe_mask((0, 0, 100, 200), 10, 90),
                letters_rows=[(50, 60)], letters_cols=(20, 180))
    small = _box(100, 40, 190, 70, _lobe_mask((100, 0, 200, 200), 10, 90),
                 letters_rows=[(10, 20)], letters_cols=(10, 80))
    assert len(_fold_nested([head, small])) == 2


def test_fold_still_merges_a_fragment_of_the_same_bubble():
    head = _box(0, 0, 200, 200, _lobe_mask((0, 0, 100, 200), 10, 90),
                letters_rows=[(50, 60)], letters_cols=(20, 180))
    small = _box(100, 40, 190, 70, _lobe_mask((0, 0, 100, 200), 10, 90),
                 letters_rows=[(10, 20)], letters_cols=(10, 80))
    assert len(_fold_nested([head, small])) == 1



def test_nms_keeps_a_second_lobe_instead_of_bucketing_it():
    a = _box(0, 0, 100, 100, _lobe_mask((0, 0, 60, 100), 5, 55))
    b = _box(40, 0, 140, 100, _lobe_mask((80, 0, 140, 100), 5, 55), conf=0.85)
    assert len(apply_final_nms([a, b])) == 2


def test_nms_still_merges_the_same_detection_twice():
    a = _box(0, 0, 100, 100, _lobe_mask((0, 0, 60, 100), 5, 55))
    b = _box(40, 0, 140, 100, _lobe_mask((0, 0, 60, 100), 5, 55), conf=0.85)
    merged = apply_final_nms([a, b])
    assert len(merged) == 1
    assert (merged[0].x1, merged[0].x2) == (0, 140)



def _mh068_objects(separate=True):
    from app.ai_mode.seams import join_stacked_lines  # noqa: F401 (import check)
    upper_mask = _lobe_mask((111, 55, 428, 241), 5, 312)
    lower_mask = _lobe_mask((38, 264, 592, 614), 5, 549) if separate else upper_mask
    page = {"text_objects": [
        {"id": "text_upper", "region": {"x1": 106, "y1": 61, "x2": 428, "y2": 268},
         "letter_bounds": {"x1": 148, "y1": 92, "x2": 396, "y2": 224},
         "bubble_mask": upper_mask, "translation": ""},
        {"id": "text_lower", "region": {"x1": 45, "y1": 261, "x2": 574, "y2": 606},
         "letter_bounds": {"x1": 112, "y1": 280, "x2": 533, "y2": 400},
         "bubble_mask": lower_mask, "translation": ""},
    ]}
    return {"pages": [page]}, page


def test_join_refuses_two_stacked_lobes():
    from app.ai_mode.seams import join_stacked_lines

    manifest, page = _mh068_objects(separate=True)
    assert join_stacked_lines(manifest, [0]) == 0
    assert all("joined_into" not in obj for obj in page["text_objects"])


def test_join_still_joins_two_lines_of_one_bubble():
    from app.ai_mode.seams import join_stacked_lines

    manifest, page = _mh068_objects(separate=False)
    assert join_stacked_lines(manifest, [0]) == 1
    assert page["text_objects"][1]["joined_into"] == "text_upper"



def _dup_objects(separate=True):
    keeper_mask = _lobe_mask((0, 0, 200, 100), 5, 195)
    small_mask = _lobe_mask((200, 0, 400, 100), 5, 195) if separate else keeper_mask
    text = "Tôi không muốn đi về nhà"
    page = {"text_objects": [
        {"id": "text_keeper", "region": {"x1": 0, "y1": 0, "x2": 200, "y2": 100},
         "letter_bounds": {"x1": 0, "y1": 0, "x2": 200, "y2": 100},
         "translation": text, "bubble_mask": keeper_mask},
        {"id": "text_small", "region": {"x1": 100, "y1": 0, "x2": 250, "y2": 100},
         "letter_bounds": {"x1": 100, "y1": 0, "x2": 250, "y2": 100},
         "translation": text, "bubble_mask": small_mask},
    ]}
    return {"pages": [page]}, page


def test_drop_keeps_the_lobe_region_out_of_the_keeper():
    from app.ai_mode.seams import drop_overlapping_letters

    manifest, page = _dup_objects(separate=True)
    dropped = drop_overlapping_letters(manifest)
    assert dropped == [(0, "text_small")]
    assert page["text_objects"][0]["region"] == {"x1": 0, "y1": 0, "x2": 200, "y2": 100}
    assert page["text_objects"][0]["letter_bounds"] == {"x1": 0, "y1": 0, "x2": 200, "y2": 100}


def test_drop_still_absorbs_a_half_of_the_same_bubble():
    from app.ai_mode.seams import drop_overlapping_letters

    manifest, page = _dup_objects(separate=False)
    assert drop_overlapping_letters(manifest) == [(0, "text_small")]
    assert page["text_objects"][0]["region"] == {"x1": 0, "y1": 0, "x2": 250, "y2": 100}



def _ch111_pieces():
    left = _box(46, 81, 319, 338, _lobe_mask((50, 83, 319, 322), 5, 264),
                letters_rows=[(42, 60), (90, 108), (140, 158)], letters_cols=(40, 220))
    right = _box(319, 187, 560, 338, _lobe_mask((319, 117, 565, 357), 5, 241),
                 letters_rows=[(60, 78), (100, 118)], letters_cols=(10, 200))
    return [left, right]


def test_split_lobes_survive_grouping_end_to_end():
    from app.detector.kiuyha_detector import split_waist_boxes
    from app.text_objects import ensure_page_text_objects
    from app.ai_mode.seams import join_stacked_lines

    pieces = _ch111_pieces()
    assert len(apply_final_nms(list(pieces))) == 2
    assert len(_fold_nested(list(pieces))) == 2
    assert len(split_waist_boxes(list(pieces))) == 2
    page = {"boxes": [
        {"id": "box_00", "x1": 46, "y1": 81, "x2": 319, "y2": 338,
         "ocr_text": "left", "bubble_mask": pieces[0].bubble_mask},
        {"id": "box_01", "x1": 319, "y1": 187, "x2": 560, "y2": 338,
         "ocr_text": "right", "bubble_mask": pieces[1].bubble_mask},
    ], "text_objects": []}
    created, _ = ensure_page_text_objects(page)
    assert created == 2
    page["text_objects"][0]["letter_bounds"] = {"x1": 86, "y1": 123, "x2": 266, "y2": 239}
    page["text_objects"][1]["letter_bounds"] = {"x1": 329, "y1": 247, "x2": 519, "y2": 305}
    assert join_stacked_lines({"pages": [page]}, [0]) == 0
    assert all("joined_into" not in obj for obj in page["text_objects"])
    assert page["text_objects"][0]["bubble_mask"]["bbox"] == [50, 83, 319, 322]
    assert page["text_objects"][1]["bubble_mask"]["bbox"] == [319, 117, 565, 357]


def test_two_lines_of_one_bubble_still_join_end_to_end():
    from app.text_objects import ensure_page_text_objects
    from app.ai_mode.seams import join_stacked_lines

    mask = _lobe_mask((90, 90, 410, 215), 5, 315)
    page = {"boxes": [
        {"id": "box_00", "x1": 100, "y1": 100, "x2": 400, "y2": 150,
         "ocr_text": "upper", "bubble_mask": mask},
        {"id": "box_01", "x1": 100, "y1": 155, "x2": 400, "y2": 205,
         "ocr_text": "lower", "bubble_mask": mask},
    ], "text_objects": []}
    created, _ = ensure_page_text_objects(page)
    assert created == 2
    page["text_objects"][0]["letter_bounds"] = {"x1": 110, "y1": 110, "x2": 390, "y2": 140}
    page["text_objects"][1]["letter_bounds"] = {"x1": 110, "y1": 160, "x2": 390, "y2": 195}
    assert join_stacked_lines({"pages": [page]}, [0]) == 1
    assert page["text_objects"][1]["joined_into"] == "text_00"
