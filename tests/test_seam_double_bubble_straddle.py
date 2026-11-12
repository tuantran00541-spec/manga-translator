from app.ai_mode.seams import seam_mirror_ids, sync_seam_mirrors

GOODBYE = "Vĩnh biệt\nngươi nhé."
TOP_LOBE = "Tối Cường Kiếm Đạo...\nĐạt đến đỉnh cao của Kiếm Đạo\nđể thấu hiểu Kiếm Ý của nó..."
BOTTOM_LOBE = "Hoàn toàn không\nphù hợp với ngươi."


def _slice(source_y1, core_top, core_bottom, objects):
    return {"source_page": 0,
            "stitch_core": {"source_y1": source_y1, "core_source_y1": core_top,
                            "core_source_y2": core_bottom,
                            "core_y1": core_top - source_y1, "core_y2": core_bottom - source_y1},
            "text_objects": objects}


def _obj(obj_id, y1, y2, translation="", x1=100, x2=500, **extra):
    return {"id": obj_id, "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "translation": translation, **extra}


def _chapter():
    s2 = _slice(4112, 4112, 5732, [
        _obj("goodbye2", 1454, 1629, GOODBYE, x1=121, x2=425, source_read="IT'S A GOODBYE."),
    ])
    s3 = _slice(4964, 5732, 7215, [
        _obj("goodbye3", 602, 777, "", x1=121, x2=425),
        _obj("top3", 1906, 2236, TOP_LOBE, x1=242, x2=858, source_read="ULTIMATE SWORD MASTERY..."),
        _obj("bottom3", 2275, 2443, "", x1=88, x2=521),
    ])
    s4 = _slice(6447, 7215, 10509, [
        _obj("top4", 423, 753, "", x1=243, x2=854),
        _obj("bottom4", 792, 960, BOTTOM_LOBE, x1=88, x2=523, source_read="IT'S NOT FITTING FOR YOU AT ALL."),
    ])
    return {"pages": [s2, s3, s4]}


def test_straddling_goodbye_mirrors_its_owner_not_the_double_bubble():
    manifest = _chapter()
    sync_seam_mirrors(manifest)
    goodbye3 = manifest["pages"][1]["text_objects"][0]
    assert goodbye3["seam_owner"] == {"page": 0, "id": "goodbye2"}
    assert goodbye3["translation"] == GOODBYE


def test_double_bubble_lobes_keep_their_own_text():
    manifest = _chapter()
    sync_seam_mirrors(manifest)
    top3, bottom3 = manifest["pages"][1]["text_objects"][1], manifest["pages"][1]["text_objects"][2]
    assert top3.get("seam_owner") in (None, {"page": 1, "id": "top3"})
    assert top3["translation"] == TOP_LOBE
    assert bottom3["seam_owner"] == {"page": 2, "id": "bottom4"}
    assert bottom3["translation"] == BOTTOM_LOBE
    top4 = manifest["pages"][2]["text_objects"][0]
    assert top4["seam_owner"] == {"page": 1, "id": "top3"}
    assert top4["translation"] == TOP_LOBE


def test_mirror_ids_name_the_straddlers_only():
    manifest = _chapter()
    assert seam_mirror_ids(manifest, 1) == {"goodbye3", "bottom3"}
    assert seam_mirror_ids(manifest, 2) == {"top4"}
    assert seam_mirror_ids(manifest, 0) == set()