from app.ai_mode.seams import seam_mirror_ids, sync_seam_mirrors


def _slice(source_y1, core_top, core_bottom, objects):
    return {"source_page": 0, "stitch_core": {"source_y1": source_y1, "core_source_y1": core_top,
                                              "core_source_y2": core_bottom, "core_y1": core_top - source_y1,
                                              "core_y2": core_bottom - source_y1},
            "text_objects": objects}


def _obj(obj_id, y1, y2, translation="", **extra):
    return {"id": obj_id, "region": {"x1": 100, "y1": y1, "x2": 500, "y2": y2}, "translation": translation, **extra}


def _chapter():
    # The cut is at source y=2000; the bubble spans 1900-2200 with its centre (2050) in slice 1's core.
    top = _slice(0, 0, 2000, [_obj("a", 1900, 2200), _obj("solo", 100, 300, "Chào")])
    bottom = _slice(1232, 2000, 4000, [_obj("b", 668, 968, "Mẹ tôi có tâm hồn thi sĩ.",
                                            typography_role="dialogue", font_ai_id="dialogue.mac-dinh-3",
                                            style={"font": "auto"}, lettering_color="#101010")])
    return {"pages": [top, bottom]}


def test_the_slice_holding_the_centre_owns_a_bubble_across_the_cut():
    manifest = _chapter()
    assert seam_mirror_ids(manifest, 0) == {"a"}
    assert seam_mirror_ids(manifest, 1) == set()


def test_the_other_slice_letters_an_exact_copy_at_the_same_place():
    manifest = _chapter()
    assert sync_seam_mirrors(manifest) == [0]
    mirror = manifest["pages"][0]["text_objects"][0]
    assert mirror["translation"] == "Mẹ tôi có tâm hồn thi sĩ."
    assert (mirror["font_ai_id"], mirror["lettering_color"], mirror["typography_role"]) == (
        "dialogue.mac-dinh-3", "#101010", "dialogue")
    assert mirror["region"] == {"x1": 100, "y1": 1900, "x2": 500, "y2": 2200}
    assert mirror["seam_owner"] == {"page": 1, "id": "b"}
    assert sync_seam_mirrors(manifest) == [], "nothing changes a second time"


def test_a_bubble_only_one_slice_has_is_translated_there():
    manifest = _chapter()
    manifest["pages"][1]["text_objects"] = []
    assert seam_mirror_ids(manifest, 0) == set()
    manifest = _chapter()
    manifest["pages"][1]["skipped"] = True
    assert seam_mirror_ids(manifest, 0) == set()


def test_of_two_letterings_on_the_same_text_only_the_larger_stays():
    from app.ai_mode.seams import drop_overlapping_letters

    manifest = {"pages": [{"text_objects": [_obj("small", 120, 160, "Có khi cậu phải giết chúng"),
                                            _obj("big", 100, 300, "Có khi cậu phải giết chúng đấy, nhóc."),
                                            _obj("other", 400, 500, "Ừm.")]}]}
    assert drop_overlapping_letters(manifest) == [(0, "small")]
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts == {"small": "", "big": "Có khi cậu phải giết chúng đấy, nhóc.", "other": "Ừm."}


def test_a_core_box_cut_by_the_slice_edge_gives_way_to_the_whole_seam_box():
    from app.detector.boxes import BubbleBox
    from app.page_processing import _contained

    seam = BubbleBox(100, 1800, 500, 2200, 0.9, None)
    assert _contained(BubbleBox(110, 1900, 490, 2000, 0.8, None), seam), "the top quarter the core saw"
    assert not _contained(BubbleBox(110, 100, 490, 300, 0.8, None), seam)


def test_review_boxes_are_masked_by_letters_not_rectangles():
    import cv2
    import numpy as np

    from app.pipeline_editing import _paint_residue, _stroke_box

    image = np.full((400, 600, 3), (60, 90, 40), np.uint8)
    cv2.putText(image, "LEFT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (250, 250, 250), 3)
    box = _stroke_box(image, (150, 150, 450, 260))
    assert box["mask"] is not None and box["manual"]
    mask = np.zeros((400, 600), np.uint8)
    _paint_residue(mask, image, (150, 150, 450, 260))
    assert mask[200, 210] == 255 or mask[205, 205] == 255
    assert (mask[150:260, 150:450] == 255).mean() < 0.6, "the art round the letters is kept"
    flat = np.full((400, 600, 3), 200, np.uint8)
    _paint_residue(mask := np.zeros((400, 600), np.uint8), flat, (10, 10, 60, 60))
    assert (mask[10:60, 10:60] == 255).all(), "nothing stands out: the rectangle is repainted"


def test_both_slices_agree_on_the_owner_even_when_one_sees_the_text_cut():
    # Slice 0 sees only the top of the bubble (1900-2050, centre in its own core); slice 1 sees it whole.
    top = _slice(0, 0, 2000, [_obj("a", 1900, 2050, "")])
    bottom = _slice(1232, 2000, 4000, [_obj("b", 668, 1168, "Chào")])
    manifest = {"pages": [top, bottom]}
    assert seam_mirror_ids(manifest, 0) == {"a"} and seam_mirror_ids(manifest, 1) == set()
