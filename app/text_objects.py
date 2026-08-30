from __future__ import annotations

import copy


DEFAULT_TEXT_OBJECT_STYLE = {
    "color": "auto",
    "font": "default",
    "fontSize": "auto",
    "bold": False,
    "strokeWidth": "auto",
    "strokeColor": "auto",
    "bgColor": "transparent",
    "cornerRadius": "0",
    "horizontalAlign": "center",
    "verticalAlign": "middle",
}

OCR_METADATA_FIELDS = (
    "ocr_confidence",
    "ocr_model",
    "ocr_orientation",
    "ocr_region_count",
    "ocr_quality",
    "ocr_quality_reason",
)
TRANSLATION_MACHINE_FIELDS = (
    "translation_source",
    "translation_model",
    "translation_input_text",
    "auto_translation",
)


def _region_from_box(box: dict) -> dict | None:
    try:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _auto_object_id(box_id: str) -> str:
    suffix = box_id[4:] if box_id.startswith("box_") else box_id
    return f"text_{suffix}"


def _source_box_ids(obj: dict) -> set[str]:
    refs = obj.get("source_boxes")
    if not isinstance(refs, list):
        return set()
    return {str(ref) for ref in refs if isinstance(ref, str) and ref}


def _sync_ocr_metadata(obj: dict, box: dict) -> bool:
    changed = False
    for key in OCR_METADATA_FIELDS:
        if key in box:
            value = copy.deepcopy(box[key])
            if obj.get(key) != value:
                obj[key] = value
                changed = True
        elif obj.pop(key, None) is not None:
            changed = True
    return changed


def invalidate_stale_machine_translation(obj: dict, source_text: str) -> bool:
    """Clear only an untouched generated translation whose OCR source changed.

    Ownership is intentionally prospective. Legacy manifests without the
    ``auto_translation`` snapshot are left untouched because a user may already
    have edited the translated text while older versions still labelled it as
    coming from DeepSeek.
    """
    if obj.get("translation_source") != "deepseek" or "auto_translation" not in obj:
        return False
    translation_input = str(obj.get("translation_input_text") or "").strip()
    if translation_input == str(source_text or "").strip():
        return False
    current_translation = str(obj.get("translation") or "")
    auto_translation = str(obj.get("auto_translation") or "")
    if current_translation != auto_translation:
        return False
    obj["translation"] = ""
    for key in TRANSLATION_MACHINE_FIELDS:
        obj.pop(key, None)
    return True


def _sync_existing_auto_object(obj: dict, box: dict, region: dict) -> bool:
    changed = False
    previous_auto_geometry = obj.get("auto_geometry")
    current_region = obj.get("region")
    if previous_auto_geometry is None or current_region == previous_auto_geometry:
        if current_region != region:
            obj["region"] = copy.deepcopy(region)
            changed = True
    if obj.get("auto_geometry") != region:
        obj["auto_geometry"] = copy.deepcopy(region)
        changed = True

    box_text = str(box.get("ocr_text") or "")
    previous_auto_text = str(obj.get("auto_ocr_text") or "")
    current_text = str(obj.get("ocr_text") or "")
    # Auto-generated objects follow machine OCR only while the displayed text
    # still equals the last machine-owned value. This lets an empty OCR rerun
    # clear stale text, while preserving explicit user edits including a manual
    # clear to the empty string.
    follows_machine_text = current_text == previous_auto_text
    effective_source = box_text if follows_machine_text else current_text
    changed = invalidate_stale_machine_translation(obj, effective_source) or changed
    if follows_machine_text:
        if current_text != box_text:
            obj["ocr_text"] = box_text
            changed = True
        if previous_auto_text != box_text:
            obj["auto_ocr_text"] = box_text
            changed = True
        changed = _sync_ocr_metadata(obj, box) or changed

    if obj.pop("source_missing", None) is not None:
        changed = True
    return changed


def sync_existing_auto_text_object(page: dict, box: dict) -> bool:
    """Sync existing auto-generated text objects for one committed detector box.

    This deliberately does not create objects for unrelated boxes, making it safe
    to call from the per-box OCR commit path without changing text-object creation
    lifecycle for the rest of the page.
    """
    box_id = str(box.get("id") or "")
    region = _region_from_box(box)
    if not box_id or region is None:
        return False
    changed = False
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("auto_generated"):
            continue
        if box_id not in _source_box_ids(obj):
            continue
        changed = _sync_existing_auto_object(obj, box, region) or changed
    return changed


def ensure_page_text_objects(page: dict) -> tuple[int, bool]:
    """Ensure detected text boxes have editable text objects without overwriting user work.

    Existing objects that already reference a detector box win, including manually
    grouped objects. Auto-generated objects only follow detector geometry/OCR while
    the user has not changed those fields since the previous automatic sync.
    """
    objects = page.setdefault("text_objects", [])
    if not isinstance(objects, list):
        objects = []
        page["text_objects"] = objects

    boxes = page.get("boxes") or []
    active_box_ids: set[str] = set()
    covered: dict[str, dict] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for box_id in _source_box_ids(obj):
            covered.setdefault(box_id, obj)

    created = 0
    changed = False
    for box in boxes:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        if box.get("ocr_eligible") is False:
            continue
        box_id = str(box.get("id") or "")
        if not box_id:
            continue
        region = _region_from_box(box)
        if region is None:
            continue
        active_box_ids.add(box_id)

        existing = covered.get(box_id)
        if existing is not None:
            if existing.get("auto_generated"):
                changed = _sync_existing_auto_object(existing, box, region) or changed
            continue

        box_text = str(box.get("ocr_text") or "")
        obj = {
            "id": _auto_object_id(box_id),
            "shape": "rectangle",
            "region": copy.deepcopy(region),
            "source_boxes": [box_id],
            "ocr_text": box_text,
            "translation": "",
            "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
            "origin": "detector",
            "auto_generated": True,
            "auto_geometry": copy.deepcopy(region),
            "auto_ocr_text": box_text,
        }
        for key in OCR_METADATA_FIELDS:
            if key in box:
                obj[key] = copy.deepcopy(box[key])
        objects.append(obj)
        covered[box_id] = obj
        created += 1
        changed = True

    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("auto_generated"):
            continue
        refs = _source_box_ids(obj)
        missing = bool(refs) and refs.isdisjoint(active_box_ids)
        if bool(obj.get("source_missing")) != missing:
            if missing:
                obj["source_missing"] = True
            else:
                obj.pop("source_missing", None)
            changed = True

    return created, changed
