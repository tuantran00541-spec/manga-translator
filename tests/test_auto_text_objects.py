from app.text_objects import ensure_page_text_objects


def box(box_id, x1=10, y1=20, x2=110, y2=70, text=""):
    return {
        "id": box_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "ocr_text": text,
    }


def test_detected_boxes_create_stable_editable_objects():
    page = {"boxes": [box("box_abc", text="SOURCE")], "text_objects": []}
    created, changed = ensure_page_text_objects(page)

    assert created == 1
    assert changed is True
    obj = page["text_objects"][0]
    assert obj["id"] == "text_abc"
    assert obj["source_boxes"] == ["box_abc"]
    assert obj["ocr_text"] == "SOURCE"
    assert obj["translation"] == ""
    assert obj["auto_generated"] is True


def test_sync_is_idempotent_and_does_not_duplicate_objects():
    page = {"boxes": [box("box_abc")], "text_objects": []}
    ensure_page_text_objects(page)
    created, changed = ensure_page_text_objects(page)

    assert created == 0
    assert changed is False
    assert len(page["text_objects"]) == 1


def test_machine_fields_follow_box_until_user_edits_them():
    page = {"boxes": [box("box_abc", text="OLD")], "text_objects": []}
    ensure_page_text_objects(page)
    obj = page["text_objects"][0]

    page["boxes"] = [box("box_abc", x1=15, y1=25, x2=115, y2=75, text="NEW")]
    ensure_page_text_objects(page)
    assert obj["region"] == {"x1": 15, "y1": 25, "x2": 115, "y2": 75}
    assert obj["ocr_text"] == "NEW"

    obj["region"] = {"x1": 30, "y1": 35, "x2": 130, "y2": 85}
    obj["ocr_text"] = "USER FIX"
    page["boxes"] = [box("box_abc", x1=20, y1=30, x2=120, y2=80, text="MACHINE AGAIN")]
    ensure_page_text_objects(page)

    assert obj["region"] == {"x1": 30, "y1": 35, "x2": 130, "y2": 85}
    assert obj["ocr_text"] == "USER FIX"


def test_manual_grouped_object_prevents_duplicate_auto_object():
    page = {
        "boxes": [box("box_a"), box("box_b")],
        "text_objects": [
            {
                "id": "manual",
                "source_boxes": ["box_a", "box_b"],
                "region": {"x1": 1, "y1": 1, "x2": 200, "y2": 100},
                "ocr_text": "",
                "translation": "",
            }
        ],
    }
    created, changed = ensure_page_text_objects(page)

    assert created == 0
    assert changed is False
    assert len(page["text_objects"]) == 1
