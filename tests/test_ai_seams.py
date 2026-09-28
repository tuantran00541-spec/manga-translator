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


def test_a_seam_line_inside_the_core_box_joins_it_as_one_text():
    import numpy as np

    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_nested

    core = BubbleBox(52, 3777, 1600, 4320, 0.9, np.ones((543, 1548), np.uint8))
    line = BubbleBox(209, 4026, 1431, 4300, 0.8, np.ones((274, 1222), np.uint8))
    apart = BubbleBox(100, 100, 400, 200, 0.8, np.ones((100, 300), np.uint8))
    folded = _fold_nested([line, apart, core])
    assert sorted((b.x1, b.y1, b.x2, b.y2) for b in folded) == [(52, 3777, 1600, 4320), (100, 100, 400, 200)]


def test_review_boxes_are_masked_by_letters_not_rectangles():
    import cv2
    import numpy as np

    from app.detector import ctd_mask
    from app.pipeline_editing import _paint_residue, _stroke_box
    from ctd_fake import InkModel

    ctd_mask._session = InkModel(light_above=235)
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


def test_a_core_box_and_the_seam_box_holding_it_join_their_masks():
    import numpy as np

    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_nested

    seam = np.zeros((718, 1593), np.uint8)
    seam[:, :900] = 255  # the seam pass missed the end of the line
    core = np.zeros((535, 1310), np.uint8)
    core[:, 900:] = 255  # the core pass has it
    folded = _fold_nested([BubbleBox(0, 501, 1593, 1219, 0.6, seam, source_role="text_segmenter"),
                           BubbleBox(128, 768, 1438, 1303, 0.5, core, source_role="text_segmenter")])
    assert len(folded) == 1
    box = folded[0]
    assert box.mask[995 - box.y1, 1186 - box.x1] and box.mask[600 - box.y1, 100 - box.x1]


def test_lines_split_from_one_title_become_one_object_before_translation():
    from app.ai_mode.seams import join_stacked_lines

    def obj(obj_id, x1, y1, x2, y2):
        return {"id": obj_id, "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "translation": ""}

    # Shadow Slave 1, slice 48: BY THE / NIGHTMARE / SPELL as three detector boxes.
    title = [obj("by", 441, 2857, 1100, 3136), obj("nightmare", 81, 3147, 1600, 3420), obj("spell", 521, 3456, 1092, 3720)]
    # Two bubbles one above the other are two texts.
    bubbles = [obj("top", 200, 100, 700, 300), obj("bottom", 220, 380, 680, 560)]
    manifest = {"pages": [{"text_objects": title + bubbles}]}
    assert join_stacked_lines(manifest, [0]) == 2
    by, nightmare, spell, top, bottom = manifest["pages"][0]["text_objects"]
    assert by["region"] == {"x1": 81, "y1": 2857, "x2": 1600, "y2": 3720}
    assert by["joined_lines"] == ["nightmare", "spell"]
    assert nightmare["joined_into"] == spell["joined_into"] == "by"
    assert "joined_into" not in top and "joined_into" not in bottom and "joined_lines" not in top
    from app.routers.translation import _vision_candidates

    sent = [candidate["id"] for candidate in _vision_candidates(manifest["pages"][0], force=False)]
    assert "nightmare" not in sent and "spell" not in sent and "by" in sent, "joined lines are not translated alone"


def test_a_line_joined_into_another_does_not_block_export():
    from app.editorial_gate import editorial_preflight

    def box(box_id, y):
        return {"id": box_id, "x1": 0, "y1": y, "x2": 100, "y2": y + 50, "confidence": 0.9,
                "source_role": "text_segmenter", "origin": "detector", "safe_to_inpaint": True}

    joined = {"id": "ob", "source_boxes": ["b"], "region": {"x1": 0, "y1": 55, "x2": 100, "y2": 105},
              "ocr_text": "", "translation": "", "joined_into": "oa"}
    page = {"boxes": [box("a", 0), box("b", 55)], "text_objects": [
        {"id": "oa", "source_boxes": ["a"], "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 105},
         "ocr_text": "", "translation": "Bởi Chú thuật Ác Mộng", "joined_lines": ["ob"]}, joined]}
    assert editorial_preflight({"pages": [page]})["blocker_count"] == 0
    joined["joined_into"] = "gone"
    assert editorial_preflight({"pages": [page]})["blocker_count"] == 1, "a join to nothing is still a blank line"
