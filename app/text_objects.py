from __future__ import annotations

import copy

from app.region_policy import geometry_center_in_regions, overlap_share, page_preserve_regions


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
    "ocr_coverage",
    "ocr_target_mode",
    "ocr_retry_applied",
    "ocr_text_color",
    "ocr_color_profile",
    "ocr_font_size",
    "ocr_text_region",
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


def auto_object_id(box_id: str) -> str:
    suffix = box_id[4:] if box_id.startswith("box_") else box_id
    return f"text_{suffix}"


def source_box_ids(obj: dict) -> set[str]:
    refs = obj.get("source_boxes")
    if not isinstance(refs, list):
        return set()
    return {str(ref) for ref in refs if isinstance(ref, str) and ref}


def _sync_ocr_metadata(obj: dict, box: dict, *, skip_geometry: bool = False) -> bool:
    from app.ocr.visual_metadata import has_manual_color_profile

    changed = False
    keep_color = has_manual_color_profile(obj.get("ocr_color_profile"))
    for key in OCR_METADATA_FIELDS:
        if skip_geometry and key == "ocr_text_region":
            continue
        if keep_color and key in ("ocr_text_color", "ocr_color_profile"):
            continue
        if key in box:
            value = copy.deepcopy(box[key])
            if obj.get(key) != value:
                obj[key] = value
                changed = True
        elif obj.pop(key, None) is not None:
            changed = True
    return changed


def invalidate_stale_machine_translation(obj: dict, source_text: str) -> bool:
    if not obj.get("translation_source") or "auto_translation" not in obj:
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
    merged = bool(obj.get("joined_lines") or obj.get("overlap_merged"))
    previous_auto_geometry = obj.get("auto_geometry")
    current_region = obj.get("region")
    if not merged:
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
        changed = _sync_ocr_metadata(obj, box, skip_geometry=merged) or changed
    else:
        from app.ocr.visual_metadata import sync_raw_source_color

        changed = sync_raw_source_color(obj, [box]) or changed
        for color_key in ("ocr_text_color", "ocr_color_profile"):
            if color_key in box and not obj.get(color_key):
                obj[color_key] = copy.deepcopy(box[color_key])
                changed = True
    if not merged:
        if isinstance(box.get("bubble_mask"), dict):
            value = copy.deepcopy(box["bubble_mask"])
            if obj.get("bubble_mask") != value:
                obj["bubble_mask"] = value
                changed = True
        elif obj.pop("bubble_mask", None) is not None:
            changed = True
        if box.get("letters_mask"):
            value = copy.deepcopy(box["letters_mask"])
            if obj.get("letters_mask") != value:
                obj["letters_mask"] = value
                changed = True
        elif obj.pop("letters_mask", None) is not None:
            changed = True

    if obj.pop("source_missing", None) is not None:
        changed = True
    return changed


def sync_existing_auto_text_object(page: dict, box: dict) -> bool:
    box_id = str(box.get("id") or "")
    region = _region_from_box(box)
    if not box_id or region is None:
        return False
    changed = False
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("auto_generated"):
            continue
        if box_id not in source_box_ids(obj):
            continue
        if obj.get("seam_owner"):
            continue
        if obj.get("verified_balloon") or len(source_box_ids(obj)) > 1:
            box_by_id = {str(b["id"]): b for b in page.get("boxes") or []
                         if isinstance(b, dict) and b.get("id") and not b.get("removed")}
            members = [box_by_id[ref] for ref in obj["source_boxes"] if ref in box_by_id]
            changed = _sync_balloon_ocr(obj, members) or changed
            continue
        changed = _sync_existing_auto_object(obj, box, region) or changed
    return changed


def _new_auto_object(box_id, box, region):
    text = str(box.get("ocr_text") or "")
    obj = {
        "id": auto_object_id(box_id), "shape": "rectangle",
        "region": copy.deepcopy(region), "source_boxes": [box_id],
        "ocr_text": text, "translation": "", "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
        "font_selection_mode": "auto", "font_match": None, "font_ai_id": None,
        "origin": "detector", "auto_generated": True, "auto_geometry": copy.deepcopy(region),
        "auto_ocr_text": text,
    }
    for key in OCR_METADATA_FIELDS + ("bubble_mask", "letters_mask"):
        if box.get(key) is not None:
            obj[key] = copy.deepcopy(box[key])
    return obj


def _update_fields(obj, updates):
    changed = False
    for key, value in updates.items():
        if obj.get(key) != value:
            obj[key] = copy.deepcopy(value)
            changed = True
    return changed


def _sync_balloon_ocr(obj, members):
    if not members:
        return False
    if str(obj.get("ocr_text") or "") != str(obj.get("auto_ocr_text") or ""):
        from app.ocr.visual_metadata import sync_raw_source_color

        return sync_raw_source_color(obj, members)
    if len(members) == 1:
        text = str(members[0].get("ocr_text") or "")
        changed = _sync_ocr_metadata(obj, members[0])
    else:
        from app.ocr.visual_metadata import sync_group_visual_metadata

        languages = {str(box.get("ocr_lang") or "").lower() for box in members if box.get("ocr_text")}
        separator = "" if languages and languages <= {"ja", "japan"} else "\n"
        text = separator.join(str(box["ocr_text"]).strip() for box in members if str(box.get("ocr_text") or "").strip())
        visual_keys = ("ocr_text_color", "ocr_color_profile", "ocr_font_size", "ocr_text_region")
        before = tuple(obj.get(key) for key in visual_keys)
        sync_group_visual_metadata(obj, {"boxes": members}, [str(box["id"]) for box in members])
        changed = before != tuple(obj.get(key) for key in visual_keys)
    changed = invalidate_stale_machine_translation(obj, text) or changed
    return _update_fields(obj, {"ocr_text": text, "auto_ocr_text": text}) or changed


def _can_absorb(obj):
    style = obj.get("style") or {}
    return (obj.get("region") == obj.get("auto_geometry")
            and obj.get("shape", "rectangle") == "rectangle"
            and str(obj.get("ocr_text") or "") == str(obj.get("auto_ocr_text") or "")
            and obj.get("font_selection_mode") in (None, "auto", "default", "ai")
            and all(
                k == "font"
                or value == DEFAULT_TEXT_OBJECT_STYLE.get(k)
                or (k == "cornerRadius" and str(value) == str(DEFAULT_TEXT_OBJECT_STYLE.get(k)))
                for k, value in style.items()
            ))


def _combined_machine_translation(ids, covered):
    fragments = [covered.get(ref) for ref in ids]
    if any(not obj or not str(obj.get("translation") or "").strip()
           or obj.get("translation") != obj.get("auto_translation") for obj in fragments):
        return ""
    roles = {obj.get("typography_role") or "dialogue" for obj in fragments}
    speakers = {obj["speaker"] for obj in fragments if obj.get("speaker")}
    if len(roles) != 1 or not roles <= {"dialogue", "whisper", "thought", "shout", "dark_threat"} or len(speakers) > 1:
        return ""
    return " ".join(obj["translation"].strip() for obj in fragments)


def _apply_balloon_plans(page, objects, active_boxes, covered, preserve_regions, regroup_machine):
    created = 0
    changed = False
    confirmed = set()
    for plan in page.get("balloons") or []:
        ids = plan["source_order"]
        if not ids or any(
            ref not in active_boxes or active_boxes[ref].get("geometry_overridden")
            or list(_region_from_box(active_boxes[ref]).values()) != plan["sources"][ref]
            for ref in ids
        ):
            continue
        if any(overlap_share(plan["bubble_mask"]["bbox"], region) for region in preserve_regions):
            continue
        existing = {id(covered[ref]): covered[ref] for ref in ids if ref in covered}
        members = [active_boxes[ref] for ref in ids]
        refs = set(ids)
        if any(not obj.get("auto_generated") or obj.get("seam_owner") or obj.get("joined_lines")
               or obj.get("joined_into") or obj.get("overlap_merged") or source_box_ids(obj) - refs
               for obj in existing.values()):
            continue
        obj = next(iter(existing.values())) if len(existing) == 1 else None
        combined = ""
        if obj is None or source_box_ids(obj) != refs:
            if any(not _can_absorb(candidate) for candidate in existing.values()):
                continue
            if any(str(candidate.get("translation") or "").strip() for candidate in existing.values()):
                combined = _combined_machine_translation(ids, covered) if regroup_machine else ""
                if not combined:
                    continue
            obj = covered.get(ids[0]) or next(iter(existing.values()), None)
        region = {key: (min if key.endswith("1") else max)(int(box[key]) for box in members)
                  for key in ("x1", "y1", "x2", "y2")}
        if obj is None:
            obj = _new_auto_object(ids[0], members[0], region)
            objects.append(obj)
            created += 1
            changed = True
        for other in existing.values():
            if other is not obj:
                objects.remove(other)
                changed = True
        updates = {
            "source_boxes": ids, "auto_geometry": region, "letter_bounds": plan["letter_bounds"],
            "bubble_mask": plan["bubble_mask"], "verified_balloon": True,
        }
        if obj.get("region") == obj.get("auto_geometry") or obj.get("auto_geometry") is None:
            updates["region"] = region
        if not obj.get("container"):
            updates["container"] = plan["container"]
        if plan.get("source_cap_px") is not None:
            updates["source_cap_px"] = plan["source_cap_px"]
        if len(ids) == 1 and members[0].get("letters_mask"):
            updates["letters_mask"] = members[0]["letters_mask"]
        elif obj.pop("letters_mask", None) is not None:
            changed = True
        changed = _update_fields(obj, updates) or changed
        translation_meta = {key: obj[key] for key in ("translation_source", "translation_model") if key in obj}
        changed = _sync_balloon_ocr(obj, members) or changed
        if combined:
            changed = _update_fields(obj, {
                **translation_meta, "translation": combined, "auto_translation": combined,
                "translation_input_text": str(obj.get("ocr_text") or "").strip(),
            }) or changed
        for ref in ids:
            covered[ref] = obj
            confirmed.add(ref)
    return created, changed, confirmed


def ensure_page_text_objects(page: dict, *, regroup_machine: bool = False) -> tuple[int, bool]:
    objects = page.setdefault("text_objects", [])
    if not isinstance(objects, list):
        objects = []
        page["text_objects"] = objects

    boxes = page.get("boxes") or []
    preserve_regions = page_preserve_regions(page)
    active_box_ids: set[str] = set()
    active_boxes: dict[str, dict] = {}
    covered: dict[str, dict] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for box_id in source_box_ids(obj):
            covered.setdefault(box_id, obj)

    for box in boxes:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        box_id = str(box.get("id") or "")
        if box.get("ocr_eligible") is False:
            anchor = covered.get(box_id) if box_id else None
            if isinstance(anchor, dict) and anchor.get("seam_owner"):
                active_box_ids.add(box_id)
            continue
        if not box_id:
            continue
        region = _region_from_box(box)
        if region is None:
            continue
        if geometry_center_in_regions(region, preserve_regions):
            continue
        active_box_ids.add(box_id)
        active_boxes[box_id] = box

    created, changed, confirmed = _apply_balloon_plans(
        page, objects, active_boxes, covered, preserve_regions, regroup_machine,
    )
    for box_id, box in active_boxes.items():
        existing = covered.get(box_id)
        if existing is not None:
            if existing.get("auto_generated") and box_id not in confirmed:
                if existing.pop("verified_balloon", None):
                    existing.pop("bubble_mask", None)
                    existing.pop("letter_bounds", None)
                    existing.pop("letters_mask", None)
                    existing["needs_review"] = True
                    changed = True
                if len(source_box_ids(existing)) == 1:
                    changed = _sync_existing_auto_object(existing, box, _region_from_box(box)) or changed
            continue

        obj = _new_auto_object(box_id, box, _region_from_box(box))
        objects.append(obj)
        covered[box_id] = obj
        created += 1
        changed = True

    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("auto_generated"):
            continue
        refs = source_box_ids(obj)
        missing = bool(refs) and (refs.isdisjoint(active_box_ids)
                                  or (len(refs) > 1 and not refs.issubset(active_box_ids)))
        if bool(obj.get("source_missing")) != missing:
            if missing:
                obj["source_missing"] = True
            else:
                obj.pop("source_missing", None)
            changed = True

    return created, changed


def clear_page_object_translations(page: dict, object_ids: set[str]) -> bool:
    """Clear translations on restored objects so re-renders never paint over original pixels."""
    if not object_ids or not isinstance(page, dict):
        return False
    changed = False
    for item in page.get("text_objects") or []:
        if not (isinstance(item, dict) and str(item.get("id")) in object_ids):
            continue
        if str(item.get("translation") or "").strip() or "auto_translation" in item:
            item["translation"] = ""
            item.pop("auto_translation", None)
            item["translation_attempted"] = True
            changed = True
    return changed
