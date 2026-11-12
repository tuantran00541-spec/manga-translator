from app.ai_mode.seams import seam_mirror_ids, sync_seam_mirrors


def _slice(source_y1, core_top, core_bottom, objects):
    return {"source_page": 0, "stitch_core": {"source_y1": source_y1, "core_source_y1": core_top,
                                              "core_source_y2": core_bottom, "core_y1": core_top - source_y1,
                                              "core_y2": core_bottom - source_y1},
            "text_objects": objects}


def _obj(obj_id, y1, y2, translation="", **extra):
    return {"id": obj_id, "region": {"x1": 100, "y1": y1, "x2": 500, "y2": y2}, "translation": translation, **extra}


def _chapter():
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


def test_a_duplicate_far_from_the_keeper_keeps_its_own_region():
    """N3: the keeper absorbs a dropped duplicate's region only when the
    two regions touch. The dropped object's lettering overlaps the
    keeper's (that is why it is dropped), but its region sits far away —
    a stale box, not the keeper's bubble — so the keeper must not grow
    to cover it."""
    from app.ai_mode.seams import drop_overlapping_letters

    keeper = _obj("keeper", 100, 300, "Có khi cậu phải giết chúng đấy, nhóc.")
    near = _obj("near", 120, 160, "Có khi cậu phải giết chúng")
    far = _obj("far", 400, 460, "Có khi cậu phải giết chúng")
    far["letter_bounds"] = {"x1": 100, "y1": 120, "x2": 500, "y2": 160}
    manifest = {"pages": [{"text_objects": [keeper, near, far]}]}
    assert drop_overlapping_letters(manifest) == [(0, "near"), (0, "far")]
    assert keeper["region"] == {"x1": 100, "y1": 100, "x2": 500, "y2": 300}
    assert far["region"] == {"x1": 100, "y1": 400, "x2": 500, "y2": 460}
    assert far.get("overlap_dropped") is True


def test_a_seam_owner_is_never_dropped_as_a_duplicate():
    """N16: a seam mirror is a second lettering of the same text by
    design; dropping it as a duplicate would erase the mirror copy."""
    from app.ai_mode.seams import drop_overlapping_letters

    manifest = {"pages": [{"text_objects": [_obj("owner", 100, 300, "Chào bạn",
                                                  seam_owner={"page": 1, "id": "b"}),
                                            _obj("big", 100, 300, "Chào bạn")]}]}
    assert drop_overlapping_letters(manifest) == []


def test_sync_seam_mirrors_does_not_copy_an_untranslated_owner():
    """P2-6: a mirror may only carry a translation the gate would keep;
    an English owner stays untranslated on both slices."""
    manifest = _chapter()
    manifest["pages"][1]["text_objects"][0]["translation"] = "SENIOR BROTHER!"
    assert sync_seam_mirrors(manifest) == []
    assert manifest["pages"][0]["text_objects"][0]["translation"] == ""


def test_sync_seam_mirrors_copies_the_gate_filtered_owner_text():
    """P2-6: a mixed owner copies its Vietnamese line only, not the
    English line the gate drops."""
    manifest = _chapter()
    manifest["pages"][1]["text_objects"][0]["translation"] = "Đi thôi\nSENIOR BROTHER!"
    assert sync_seam_mirrors(manifest) == [0]
    assert manifest["pages"][0]["text_objects"][0]["translation"] == "Đi thôi"


def test_of_two_letterings_on_the_same_text_only_the_larger_stays():
    from app.ai_mode.seams import drop_overlapping_letters

    manifest = {"pages": [{"text_objects": [_obj("small", 120, 160, "Có khi cậu phải giết chúng"),
                                            _obj("big", 100, 300, "Có khi cậu phải giết chúng đấy, nhóc."),
                                            _obj("other", 400, 500, "Ừm.")]}]}
    assert drop_overlapping_letters(manifest) == [(0, "small")]
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts == {"small": "", "big": "Có khi cậu phải giết chúng đấy, nhóc.", "other": "Ừm."}


def test_overlapping_lettering_with_different_words_both_stay():
    """Rerun-4 ch42 p012: lobe 1's "ANYHOW," lettering sits 91% inside
    the double bubble's lettering, but it is different words, not the
    same text twice — dropping on geometry alone emptied lobe 1."""
    from app.ai_mode.seams import drop_overlapping_letters

    big = {"id": "bubble", "region": {"x1": 196, "y1": 366, "x2": 719, "y2": 648},
           "letter_bounds": {"x1": 196, "y1": 374, "x2": 668, "y2": 590},
           "translation": "Sống chung với kẻ nóng nảy đó, cậu sắp khổ dài dài rồi."}
    small = {"id": "lobe1", "region": {"x1": 145, "y1": 340, "x2": 386, "y2": 450},
             "letter_bounds": {"x1": 181, "y1": 375, "x2": 349, "y2": 420},
             "translation": "Dù sao đi nữa."}
    manifest = {"pages": [{"text_objects": [big, small]}]}
    assert drop_overlapping_letters(manifest) == []
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts["lobe1"] == "Dù sao đi nữa."
    assert texts["bubble"].startswith("Sống chung")


def test_overlapping_cloud_against_its_oval_both_stay():
    """Rerun-4 ch42 p008: the "HE..." cloud tucks against the big oval,
    so its lettering shares 0.59 with the oval's — different words, so
    both stay."""
    from app.ai_mode.seams import drop_overlapping_letters

    big = {"id": "oval", "region": {"x1": 134, "y1": 1346, "x2": 644, "y2": 1677},
           "letter_bounds": {"x1": 208, "y1": 1418, "x2": 636, "y2": 1668},
           "translation": "Tan học, tôi sẽ đi thẳng đến phòng huấn luyện riêng."}
    small = {"id": "cloud", "region": {"x1": 468, "y1": 1569, "x2": 677, "y2": 1708},
             "letter_bounds": {"x1": 489, "y1": 1607, "x2": 639, "y2": 1708},
             "translation": "Hê..."}
    manifest = {"pages": [{"text_objects": [big, small]}]}
    assert drop_overlapping_letters(manifest) == []
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts["cloud"] == "Hê..."


def test_a_reworded_duplicate_still_drops():
    """Same text read twice need not come back word-identical; most
    words shared is still a duplicate."""
    from app.ai_mode.seams import drop_overlapping_letters

    manifest = {"pages": [{"text_objects": [_obj("small", 120, 160, "Chuyện đó có gì không ổn?"),
                                            _obj("big", 100, 300, "Chuyện đó có vấn đề gì không?")]}]}
    assert drop_overlapping_letters(manifest) == [(0, "small")]
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts["small"] == ""


def test_a_dropped_duplicate_lends_its_geometry_to_the_keeper():
    """Rerun-4 ch42 p010: the detector split one bubble into left and
    right halves; dropping the twin must not shrink the survivor to its
    half — the keeper's region and lettering grow to the union, so the
    text is fitted to the whole bubble."""
    from app.ai_mode.seams import drop_overlapping_letters

    left = {"id": "left", "region": {"x1": 78, "y1": 36, "x2": 395, "y2": 352},
            "letter_bounds": {"x1": 145, "y1": 46, "x2": 395, "y2": 311},
            "translation": "Chuyện đó có vấn đề gì không?"}
    right = {"id": "right", "region": {"x1": 195, "y1": 0, "x2": 517, "y2": 253},
             "letter_bounds": {"x1": 212, "y1": 46, "x2": 463, "y2": 251},
             "translation": "Chuyện đó có vấn đề gì không?"}
    manifest = {"pages": [{"text_objects": [left, right]}]}
    assert drop_overlapping_letters(manifest) == [(0, "right")]
    kept = manifest["pages"][0]["text_objects"][0]
    assert kept["id"] == "left"
    assert kept["region"] == {"x1": 78, "y1": 0, "x2": 517, "y2": 352}
    assert kept["letter_bounds"] == {"x1": 145, "y1": 46, "x2": 463, "y2": 311}
    assert kept["translation"] == "Chuyện đó có vấn đề gì không?"
    assert manifest["pages"][0]["text_objects"][1]["translation"] == ""


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


def test_review_repaint_with_original_leaves_a_clean_bubble_alone(monkeypatch):
    import cv2
    import numpy as np

    from app.detector import ctd_mask
    from app.pipeline_editing import _paint_residue
    from ctd_fake import InkModel

    monkeypatch.setattr(ctd_mask, "_session", InkModel())
    original = np.full((400, 600, 3), 160, np.uint8)
    cv2.putText(original, "LEFT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    clean = np.full((400, 600, 3), 160, np.uint8)
    mask = np.zeros((400, 600), np.uint8)
    _paint_residue(mask, clean, (150, 150, 450, 260), original=original)
    assert not mask.any(), "a bubble whose letters are already gone needs no repaint"


def test_review_repaint_with_original_erases_only_ghost_ink(monkeypatch):
    import cv2
    import numpy as np

    from app.detector import ctd_mask
    from app.pipeline_editing import _paint_residue
    from ctd_fake import InkModel

    monkeypatch.setattr(ctd_mask, "_session", InkModel())
    original = np.full((400, 600, 3), 160, np.uint8)
    cv2.putText(original, "LEFT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    clean = np.full((400, 600, 3), 160, np.uint8)
    cv2.putText(clean, "LEFT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 50, 50), 3)
    mask = np.zeros((400, 600), np.uint8)
    _paint_residue(mask, clean, (150, 150, 450, 260), original=original)
    assert mask[200, 210] == 255 or mask[205, 205] == 255
    assert (mask[150:260, 150:450] == 255).mean() < 0.6, "the bubble fill round the ghost is kept"


def test_review_repaint_with_original_ignores_isolated_specks(monkeypatch):
    import cv2
    import numpy as np

    from app.detector import ctd_mask
    from app.pipeline_editing import _paint_residue
    from ctd_fake import InkModel

    monkeypatch.setattr(ctd_mask, "_session", InkModel())
    original = np.full((400, 600, 3), 160, np.uint8)
    cv2.putText(original, "LEFT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    clean = np.full((400, 600, 3), 160, np.uint8)
    ink_y, ink_x = np.where(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) < 30)
    for y, x in list(zip(ink_y, ink_x))[::37]:
        clean[y, x] = (50, 50, 50)
    mask = np.zeros((400, 600), np.uint8)
    _paint_residue(mask, clean, (150, 150, 450, 260), original=original)
    assert not mask.any(), "single-pixel specks must not become repaint runs"


def test_both_slices_agree_on_the_owner_even_when_one_sees_the_text_cut():
    top = _slice(0, 0, 2000, [_obj("a", 1900, 2050, "")])
    bottom = _slice(1232, 2000, 4000, [_obj("b", 668, 1168, "Chào")])
    manifest = {"pages": [top, bottom]}
    assert seam_mirror_ids(manifest, 0) == {"a"} and seam_mirror_ids(manifest, 1) == set()


def test_a_core_box_and_the_seam_box_holding_it_join_their_masks():
    import numpy as np

    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_nested

    seam = np.zeros((718, 1593), np.uint8)
    seam[:, :900] = 255
    core = np.zeros((535, 1310), np.uint8)
    core[:, 900:] = 255
    folded = _fold_nested([BubbleBox(0, 501, 1593, 1219, 0.6, seam, source_role="text_segmenter"),
                           BubbleBox(128, 768, 1438, 1303, 0.5, core, source_role="text_segmenter")])
    assert len(folded) == 1
    box = folded[0]
    assert box.mask[995 - box.y1, 1186 - box.x1] and box.mask[600 - box.y1, 100 - box.x1]


def test_lines_split_from_one_title_become_one_object_before_translation():
    from app.ai_mode.seams import join_stacked_lines

    def obj(obj_id, x1, y1, x2, y2):
        return {"id": obj_id, "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "translation": ""}

    title = [obj("by", 441, 2857, 1100, 3136), obj("nightmare", 81, 3147, 1600, 3420), obj("spell", 521, 3456, 1092, 3720)]
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


def _cut_slice(source_y1, core_y1, core_y2, objects):
    core = {"source_y1": source_y1, "core_y1": core_y1, "core_y2": core_y2,
            "core_source_y1": source_y1 + core_y1, "core_source_y2": source_y1 + core_y2}
    return {"stitch_core": core, "text_objects": [
        {"id": oid, "region": dict(zip(("x1", "y1", "x2", "y2"), rect)), "translation": ""} for oid, rect in objects]}


def test_two_bubbles_of_one_balloon_stay_two_texts():
    from app.ai_mode.seams import join_stacked_lines

    lobes = (((477, 537, 1419, 1115), (219, 1221, 1182, 1824)), ((128, 533, 952, 1073), (375, 1135, 1459, 1711)),
             ((195, 324, 1021, 936), (272, 1047, 1435, 1726)))
    manifest = {"pages": [{"text_objects": [
        {"id": f"line{n}", "region": dict(zip(("x1", "y1", "x2", "y2"), rect)), "translation": ""}
        for n, rect in enumerate(pair)]} for pair in lobes]}
    assert join_stacked_lines(manifest, [0, 1, 2]) == 0


def test_two_centred_bubbles_with_a_wide_letter_band_stay_two_texts():
    from app.ai_mode.seams import join_stacked_lines

    manifest = {"pages": [{"text_objects": [
        {"id": "upper", "region": {"x1": 0, "y1": 1902, "x2": 766, "y2": 2433}, "translation": "",
         "letter_bounds": {"x1": 120, "y1": 2018, "x2": 750, "y2": 2326}},
        {"id": "lower", "region": {"x1": 154, "y1": 2478, "x2": 834, "y2": 2955}, "translation": "",
         "letter_bounds": {"x1": 260, "y1": 2582, "x2": 761, "y2": 2855}},
    ]}]}
    assert join_stacked_lines(manifest, [0]) == 0


def test_fragments_of_one_block_with_a_narrow_letter_band_still_join():
    from app.ai_mode.seams import join_stacked_lines

    manifest = {"pages": [{"text_objects": [
        {"id": "upper", "region": {"x1": 0, "y1": 1902, "x2": 766, "y2": 2433}, "translation": "",
         "letter_bounds": {"x1": 120, "y1": 2018, "x2": 750, "y2": 2400}},
        {"id": "lower", "region": {"x1": 154, "y1": 2478, "x2": 834, "y2": 2955}, "translation": "",
         "letter_bounds": {"x1": 260, "y1": 2500, "x2": 761, "y2": 2855}},
    ]}]}
    assert join_stacked_lines(manifest, [0]) == 1


def test_a_sentence_split_by_a_slice_cut_is_lettered_once():
    from app.ai_mode.seams import join_stacked_lines, seam_mirror_ids, sync_seam_mirrors

    manifest = {"pages": [
        _cut_slice(6369, 768, 3399, [("before", (443, 3132, 1150, 3471)), ("nightmares", (80, 3377, 1515, 3696))]),
        _cut_slice(9000, 768, 2857, [("before6", (447, 508, 1135, 840)), ("nightmares6", (8, 746, 1562, 1371))]),
    ]}
    assert join_stacked_lines(manifest, [0, 1]) == 2
    assert seam_mirror_ids(manifest, 0) == {"before"} and seam_mirror_ids(manifest, 1) == set(), \
        "the slice holding most of the sentence owns it"
    owner = manifest["pages"][1]["text_objects"][0]
    owner["translation"] = "Trước khi ác mộng thành hiện thực"
    sync_seam_mirrors(manifest)
    lettered = [[obj["id"] for obj in page["text_objects"] if obj["translation"]] for page in manifest["pages"]]
    assert lettered == [["before"], ["before6"]], "one lettering per slice, the same sentence on both"
    assert manifest["pages"][0]["text_objects"][0]["translation"] == owner["translation"]


def test_a_mirror_of_a_joined_line_letters_the_block_it_joined():
    from app.ai_mode.seams import join_stacked_lines, sync_seam_mirrors

    manifest = {"pages": [
        _cut_slice(0, 0, 1000, [("low", (380, 880, 1220, 1100)), ("top", (400, 700, 1200, 900))]),
        _cut_slice(700, 300, 2000, [("low_mirror", (380, 180, 1220, 400))]),
    ]}
    join_stacked_lines(manifest, [0, 1])
    low, top = manifest["pages"][0]["text_objects"]
    assert low["joined_into"] == "top"
    top["translation"] = "Một câu"
    sync_seam_mirrors(manifest)
    mirror = manifest["pages"][1]["text_objects"][0]
    assert mirror["translation"] == "Một câu" and mirror["seam_owner"] == {"page": 0, "id": "top"}
    assert mirror["region"] == {"x1": 380, "y1": 0, "x2": 1220, "y2": 400}


def test_the_seam_copy_and_a_joined_block_letter_where_the_source_lettering_was():
    from app.ai_mode.seams import join_stacked_lines, sync_seam_mirrors

    manifest = {"pages": [
        _cut_slice(0, 0, 1000, [("low", (380, 880, 1220, 1100)), ("top", (400, 700, 1200, 900))]),
        _cut_slice(700, 300, 2000, [("low_mirror", (380, 180, 1220, 400))]),
    ]}
    low, top = manifest["pages"][0]["text_objects"]
    low["letter_bounds"] = {"x1": 450, "y1": 920, "x2": 1150, "y2": 1060}
    top["letter_bounds"] = {"x1": 470, "y1": 730, "x2": 1130, "y2": 870}
    join_stacked_lines(manifest, [0, 1])
    assert top["letter_bounds"] == {"x1": 450, "y1": 730, "x2": 1150, "y2": 1060}
    top["translation"] = "Một câu"
    sync_seam_mirrors(manifest)
    mirror = manifest["pages"][1]["text_objects"][0]
    assert mirror["letter_bounds"] == {"x1": 450, "y1": 30, "x2": 1150, "y2": 360}, "shifted to the copy's slice"


def test_the_lettering_area_is_the_extent_the_erase_mask_covers():
    import numpy as np

    from app.ai_mode.seams import set_letter_bounds
    from app.image_io import encode_mask

    mask = np.zeros((300, 1171), np.uint8)
    mask[40:260, 172:1201 - 165] = 255
    page = {"boxes": [{"id": "box_a", "x1": 165, "y1": 827, "x2": 1336, "y2": 1127, "mask": encode_mask(mask)}],
            "text_objects": [{"id": "text_a", "source_boxes": ["box_a"], "region": {"x1": 165, "y1": 827, "x2": 1336, "y2": 1127}},
                             {"id": "manual", "source_boxes": [], "region": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]}
    set_letter_bounds(page)
    assert page["text_objects"][0]["letter_bounds"] == {"x1": 337, "y1": 867, "x2": 1201, "y2": 1087}
    assert "letter_bounds" not in page["text_objects"][1], "a hand-drawn object letters its whole region"


def test_a_found_box_grown_over_most_of_the_slice_keeps_its_own_rectangle(monkeypatch):
    import numpy as np

    from app.detector import kiuyha_detector
    from app.pipeline_editing import _stroke_box

    monkeypatch.setattr(kiuyha_detector, "letter_mask",
                        lambda image, rect: ((0, 0, 600, 400), np.ones((400, 600), bool)))
    box = _stroke_box(np.zeros((400, 600, 3), np.uint8), (150, 150, 450, 260))
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (150, 150, 450, 260) and box["mask"] is None


def _luna_21_22():
    """Former Supreme 1, slices 21-22: the narration box spans the cut.

    Slice 21 (page 20) holds only the bottom sliver as an overlap-context
    box; slice 22 (page 21) holds the whole box and translates it.
    """
    top = _slice(0, 0, 2103, [])
    top["boxes"] = [{"id": "box_89", "x1": 89, "y1": 2698, "x2": 779, "y2": 2871,
                     "overlap_context_only": True}]
    bottom = _slice(1335, 2103, 6083, [
        _obj("text_ca3c", 1310, 1803, "Trọng tâm của sức mạnh Võ Lâm Minh,\nlực lượng tấn công bí mật, “Ẩn Các”."),
    ])
    bottom["text_objects"][0]["region"] = {"x1": 26, "y1": 1310, "x2": 809, "y2": 1803}
    return {"pages": [top, bottom]}


def test_an_overlap_context_box_gets_a_mirror_from_the_neighbour_that_owns_it():
    from app.ai_mode.seams import create_seam_mirrors

    manifest = _luna_21_22()
    assert create_seam_mirrors(manifest) == [(0, "text_89")]
    mirror = manifest["pages"][0]["text_objects"][0]
    assert mirror["region"] == {"x1": 89, "y1": 2698, "x2": 779, "y2": 2871}
    assert mirror["source_boxes"] == ["box_89"] and mirror["translation"] == ""
    assert mirror["seam_owner"] == {"page": 1, "id": "text_ca3c"}
    assert create_seam_mirrors(manifest) == [], "a second pass creates nothing"
    assert sync_seam_mirrors(manifest) == [0]
    assert mirror["translation"].startswith("Trọng tâm của sức mạnh Võ Lâm Minh")
    assert mirror["region"] == {"x1": 26, "y1": 2645, "x2": 809, "y2": 3138}


def test_an_overlap_box_with_no_matching_neighbour_gets_no_mirror():
    from app.ai_mode.seams import create_seam_mirrors

    manifest = _luna_21_22()
    manifest["pages"][1]["text_objects"][0]["region"] = {"x1": 26, "y1": 100, "x2": 809, "y2": 300}
    assert create_seam_mirrors(manifest) == []


def test_a_mirror_is_never_created_from_another_mirror():
    from app.ai_mode.seams import _owner

    owner = _obj("o", 100, 300, "Chào")
    mirror = _obj("m", 100, 300, "Chào", seam_owner={"page": 0, "id": "o"})
    manifest = {"pages": [_slice(0, 0, 2000, [owner]),
                          _slice(1232, 1000, 4000, [mirror]),
                          _slice(1000, 3000, 5000, [])]}
    probe = {"region": {"x1": 100, "y1": 332, "x2": 500, "y2": 532}}
    assert _owner(manifest["pages"], 2, probe) == (1, mirror)
    assert _owner(manifest["pages"], 2, probe, skip_mirrors=True) is None


def test_a_seam_mirror_is_not_marked_source_missing():
    from app.ai_mode.seams import create_seam_mirrors
    from app.text_objects import ensure_page_text_objects

    manifest = _luna_21_22()
    manifest["pages"][0]["boxes"][0]["ocr_eligible"] = False
    assert create_seam_mirrors(manifest) == [(0, "text_89")]
    created, _ = ensure_page_text_objects(manifest["pages"][0])
    assert created == 0, "the overlap box makes no ordinary object"
    mirror = manifest["pages"][0]["text_objects"][0]
    assert "source_missing" not in mirror, "its anchor box is outside the core by design"


def test_a_seam_mirror_does_not_block_export_on_its_own():
    from app.editorial_gate import editorial_preflight

    def box(box_id):
        return {"id": box_id, "x1": 0, "y1": 0, "x2": 100, "y2": 50,
                "confidence": 0.95, "semantic_type": "speech_bubble"}

    def obj(obj_id, boxes, translation, **extra):
        return {"id": obj_id, "source_boxes": boxes, "ocr_text": "hello world",
                "translation": translation, "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}, **extra}

    mirror = obj("m", ["box_m"], "", seam_owner={"page": 1, "id": "o"})
    owner = obj("o", ["box_o"], "Chào thế giới")
    manifest = {"pages": [{"boxes": [box("box_m")], "text_objects": [mirror]},
                          {"boxes": [box("box_o")], "text_objects": [owner]}]}
    assert editorial_preflight(manifest)["blocker_count"] == 0
    owner["translation"] = ""
    blockers = editorial_preflight(manifest)["blockers"]
    assert [item["kind"] for item in blockers] == ["untranslated_story_object"], \
        "the owner still blocks once; the copy never double-counts it"


def _stepped_narration():
    """Run 4, slices 22/23: one stepped narration box split by the cut.

    The upper fragment carries the first sentence; the lower fragment's
    translation re-reads the whole box and repeats it. Mirrors start
    unlettered, as the sync leaves them.
    """
    upper = {"id": "upper", "region": {"x1": 31, "y1": 4414, "x2": 662, "y2": 4733},
             "translation": "Lý Diệp chưa từng thấy mặt\nCác Chủ Ẩn Các."}
    lower = {"id": "lower", "region": {"x1": 187, "y1": 765, "x2": 833, "y2": 1154},
             "translation": "Lý Diệp chưa từng thấy mặt\nCác Chủ Ẩn Các, mà có thấy,\n"
                            "đó cũng là kẻ hắn không dám\nbắt chuyện."}
    mirror_upper = {"id": "upper_mirror", "region": {"x1": 31, "y1": 434, "x2": 662, "y2": 753},
                    "translation": ""}
    mirror_lower = {"id": "lower_mirror", "region": {"x1": 187, "y1": 4745, "x2": 833, "y2": 5134},
                    "translation": ""}
    return {"pages": [
        _slice(1335, 2103, 6083, [upper, mirror_lower]),
        _slice(5315, 6083, 8948, [mirror_upper, lower]),
    ]}


def test_a_stepped_box_repeating_its_first_sentence_is_lettered_once():
    from app.ai_mode.seams import drop_duplicated_leading_text, sync_seam_mirrors

    manifest = _stepped_narration()
    assert sync_seam_mirrors(manifest) == [0, 1]
    assert drop_duplicated_leading_text(manifest) == [0, 1]
    assert drop_duplicated_leading_text(manifest) == [], "trimming twice changes nothing"
    first, second = manifest["pages"]
    assert first["text_objects"][0]["translation"] == "Lý Diệp chưa từng thấy mặt\nCác Chủ Ẩn Các."
    assert first["text_objects"][1]["translation"] == "Mà có thấy,\nđó cũng là kẻ hắn không dám\nbắt chuyện."
    assert second["text_objects"][0]["translation"] == "Lý Diệp chưa từng thấy mặt\nCác Chủ Ẩn Các."
    assert second["text_objects"][1]["translation"] == "Mà có thấy,\nđó cũng là kẻ hắn không dám\nbắt chuyện."
    assert first["text_objects"][1]["seam_owner"] == {"page": 1, "id": "lower"}
    assert second["text_objects"][0]["seam_owner"] == {"page": 0, "id": "upper"}


def test_a_hyphenated_prefix_is_trimmed_by_words_not_whitespace_tokens():
    from app.ai_mode.seams import drop_duplicated_leading_text

    manifest = {"pages": [{"text_objects": [
        {"id": "upper", "region": {"x1": 100, "y1": 100, "x2": 500, "y2": 300},
         "translation": "It's a well-known fact."},
        {"id": "lower", "region": {"x1": 100, "y1": 310, "x2": 500, "y2": 510},
         "translation": "It's a well-known fact that he left."},
    ]}]}
    assert drop_duplicated_leading_text(manifest) == [0]
    assert manifest["pages"][0]["text_objects"][1]["translation"] == "That he left."


def test_a_lower_fragment_that_does_not_repeat_is_left_alone():
    from app.ai_mode.seams import drop_duplicated_leading_text

    manifest = {"pages": [{"text_objects": [
        {"id": "upper", "region": {"x1": 31, "y1": 100, "x2": 662, "y2": 400},
         "translation": "Lý Diệp chưa từng thấy mặt Các Chủ Ẩn Các."},
        {"id": "lower", "region": {"x1": 187, "y1": 412, "x2": 833, "y2": 700},
         "translation": "Hắn làm Phó Các Chủ chỉ nửa năm sau khi gia nhập."},
    ]}]}
    assert drop_duplicated_leading_text(manifest) == []
    assert manifest["pages"][0]["text_objects"][1]["translation"].startswith("Hắn làm Phó")


def test_a_mirror_without_a_local_duplicate_keeps_its_owners_text():
    from app.ai_mode.seams import drop_duplicated_leading_text, sync_seam_mirrors

    manifest = _chapter()
    assert sync_seam_mirrors(manifest) == [0]
    assert drop_duplicated_leading_text(manifest) == []
    assert manifest["pages"][0]["text_objects"][0]["translation"] == "Mẹ tôi có tâm hồn thi sĩ."


def test_distinct_short_dialogue_sharing_words_is_not_a_duplicate():
    """Audit N2: a word *set* Jaccard >= 0.6 called all six of these
    distinct pairs duplicates (3/5 words shared, or one a word-prefix
    of the other). Manhwa dialogue is typically 2-5 words, exactly
    where the set cannot tell a reworded duplicate from a different
    sentence sharing function words. The sequence predicate keeps all
    six; the text is lost if any of them drops."""
    from app.ai_mode.seams import drop_overlapping_letters

    pairs = [
        ("Tôi không muốn đi", "Tôi không muốn về"),
        ("Anh nói gì vậy?", "Anh nói gì cơ?"),
        ("Đi đi", "Đi đi mau"),
        ("Anh ta...", "Anh ta đến rồi"),
        ("Đừng đi", "Đừng đi mà"),
        ("Của tôi", "Của tôi!"),
    ]
    for small_text, big_text in pairs:
        manifest = {"pages": [{"text_objects": [_obj("small", 120, 160, small_text),
                                                _obj("big", 100, 300, big_text)]}]}
        assert drop_overlapping_letters(manifest) == [], (small_text, big_text)
        texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
        assert texts == {"small": small_text, "big": big_text}


def test_a_three_word_line_read_twice_still_drops():
    """The sequence bar rises as the text shortens, but a 3-word line
    read twice word-for-word is still one text, not two utterances."""
    from app.ai_mode.seams import drop_overlapping_letters

    manifest = {"pages": [{"text_objects": [_obj("small", 120, 160, "Đi đi mau"),
                                            _obj("big", 100, 300, "Đi đi mau")]}]}
    assert drop_overlapping_letters(manifest) == [(0, "small")]


def test_a_geometric_mirror_match_with_different_source_text_is_not_a_mirror():
    """Audit N8: a small free-text box sitting mostly inside a big
    bubble's rect matched at share 1.0 on geometry alone and was
    translated as the neighbour's text. When both sides carry a source
    read and they share no word, they are different texts."""
    from app.ai_mode.seams import seam_mirror_ids, sync_seam_mirrors

    manifest = _chapter()
    manifest["pages"][0]["text_objects"][0]["ocr_text"] = "가지 마라"
    manifest["pages"][1]["text_objects"][0]["ocr_text"] = "빨리 와라"
    assert seam_mirror_ids(manifest, 0) == set()
    assert sync_seam_mirrors(manifest) == []


def test_a_geometric_mirror_match_with_the_same_source_text_still_mirrors():
    from app.ai_mode.seams import seam_mirror_ids

    manifest = _chapter()
    manifest["pages"][0]["text_objects"][0]["ocr_text"] = "가지 마라"
    manifest["pages"][1]["text_objects"][0]["ocr_text"] = "가지 마라"
    assert seam_mirror_ids(manifest, 0) == {"a"}


def test_a_geometric_mirror_match_without_source_text_still_mirrors():
    """No source read on either side is no contradiction: geometry
    alone still decides, as before."""
    from app.ai_mode.seams import seam_mirror_ids

    manifest = _chapter()
    manifest["pages"][0]["text_objects"][0]["ocr_text"] = "가지 마라"
    assert seam_mirror_ids(manifest, 0) == {"a"}


def test_an_overlap_box_whose_source_differs_from_the_owner_gets_no_mirror():
    from app.ai_mode.seams import create_seam_mirrors

    manifest = _luna_21_22()
    manifest["pages"][0]["boxes"][0]["ocr_text"] = "가지 마라"
    manifest["pages"][1]["text_objects"][0]["ocr_text"] = "빨리 와라"
    assert create_seam_mirrors(manifest) == []


def test_overlapping_regions_with_separate_letters_both_stay():
    """Run 42539948198 slice 91: ANHUI's region overlaps SHAANXI's by
    0.65, so the region-only drop deleted a real map label — but ANHUI's
    letters sit below SHAANXI's, in different places. The drop must
    compare the lettering extent, not the loose detector boxes."""
    from app.ai_mode.seams import drop_overlapping_letters

    shaanxi = {"id": "shaanxi", "region": {"x1": 742, "y1": 1269, "x2": 873, "y2": 1360},
               "letter_bounds": {"x1": 742, "y1": 1269, "x2": 873, "y2": 1310},
               "translation": "Thiểm Tây"}
    anhui = {"id": "anhui", "region": {"x1": 719, "y1": 1316, "x2": 784, "y2": 1358},
             "letter_bounds": {"x1": 719, "y1": 1330, "x2": 784, "y2": 1358},
             "translation": "An Huy"}
    manifest = {"pages": [{"text_objects": [shaanxi, anhui]}]}
    assert drop_overlapping_letters(manifest) == []
    texts = {obj["id"]: obj["translation"] for obj in manifest["pages"][0]["text_objects"]}
    assert texts == {"shaanxi": "Thiểm Tây", "anhui": "An Huy"}
