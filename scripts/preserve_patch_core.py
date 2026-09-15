from pathlib import Path


def rep(path, old, new, count=1):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"{path}: expected {count} x {old[:90]!r}, found {found}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")


def splice(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{path}: missing start marker {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{path}: missing end marker {end_marker!r}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


# Canonical policy. A preserve region means source content wins end-to-end.
Path("app/region_policy.py").write_text('''from __future__ import annotations

import numpy as np


def page_preserve_regions(page: dict | None) -> list[dict]:
    if not isinstance(page, dict):
        return []
    value = page.get("preserve_regions")
    if isinstance(value, list):
        return value
    legacy = page.get("excluded_regions")
    return legacy if isinstance(legacy, list) else []


def geometry_center_in_regions(geometry: dict | None, regions) -> bool:
    if not isinstance(geometry, dict) or not regions:
        return False
    try:
        x1, y1, x2, y2 = (float(geometry[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for region in regions:
        if not isinstance(region, dict):
            continue
        try:
            rx1, ry1, rx2, ry2 = (float(region[k]) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        if rx2 > rx1 and ry2 > ry1 and rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
    return False


def geometry_in_preserve_regions(page: dict, geometry: dict | None) -> bool:
    return geometry_center_in_regions(geometry, page_preserve_regions(page))


def text_object_in_preserve_region(page: dict, obj: dict) -> bool:
    return geometry_in_preserve_regions(page, obj.get("region") if isinstance(obj, dict) else None)


def subtract_regions_from_mask(mask: np.ndarray | None, regions) -> np.ndarray | None:
    if mask is None:
        return None
    result = np.ascontiguousarray(mask.copy())
    h, w = result.shape[:2]
    for region in regions or []:
        if not isinstance(region, dict):
            continue
        try:
            x1 = max(0, min(w, int(region["x1"])))
            y1 = max(0, min(h, int(region["y1"])))
            x2 = max(0, min(w, int(region["x2"])))
            y2 = max(0, min(h, int(region["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 0
    return result
''', encoding="utf-8")

# Manifest v4: old excluded_regions migrates once; process_required prevents RAW/stale state use.
rep("app/manifest_utils.py", "MANIFEST_SCHEMA_VERSION = 3\n", "MANIFEST_SCHEMA_VERSION = 4\n")
rep("app/manifest_utils.py", '        if page.get("width") is None or page.get("height") is None:\n', '''        legacy_regions = page.pop("excluded_regions", None)
        preserve_regions = page.get("preserve_regions")
        if not isinstance(preserve_regions, list):
            page["preserve_regions"] = (
                copy.deepcopy(legacy_regions) if isinstance(legacy_regions, list) else []
            )
            changed = True
        elif isinstance(legacy_regions, list) and legacy_regions and not preserve_regions:
            page["preserve_regions"] = copy.deepcopy(legacy_regions)
            changed = True
        elif legacy_regions is not None:
            changed = True
        if page.get("process_required") is None:
            page["process_required"] = bool(not page.get("skipped") and not page.get("clean"))
            changed = True

        if page.get("width") is None or page.get("height") is None:
''')
rep("app/manifest_utils.py", '''        "skipped": page.get("skipped", False),
        "excluded_regions": copy.deepcopy(page.get("excluded_regions", [])),
        "boxes": copy.deepcopy(page.get("boxes", [])),
''', '''        "skipped": page.get("skipped", False),
        "process_required": bool(page.get("process_required", False)),
        "preserve_regions": copy.deepcopy(page.get("preserve_regions", [])),
        "boxes": copy.deepcopy(page.get("boxes", [])),
''')

# Pipeline canonical state.
rep("app/pipeline.py", '''                    "skipped": False,
                    "excluded_regions": [],
                    "source_page": source_index,
''', '''                    "skipped": False,
                    "process_required": True,
                    "preserve_regions": [],
                    "source_page": source_index,
''')
rep("app/pipeline.py", '''                    target_page["processing_metrics"] = dict(
                        page_data.get("processing_metrics") or {}
                    )
                    bump_page_revision(target_page, "process_revision")
''', '''                    target_page["processing_metrics"] = dict(
                        page_data.get("processing_metrics") or {}
                    )
                    target_page["process_required"] = False
                    bump_page_revision(target_page, "process_revision")
''')
rep("app/pipeline.py", '                            copy.deepcopy(page.get("excluded_regions", [])),\n', '                            copy.deepcopy(page.get("preserve_regions", [])),\n')
rep("app/pipeline.py", '                excluded,\n                existing_boxes,\n', '                preserve_regions,\n                existing_boxes,\n')
rep("app/pipeline.py", '                    excluded_regions=excluded,\n', '                    preserve_regions=preserve_regions,\n')
rep("app/pipeline.py", '''                    page["skipped"] = skipped
                    if skipped:
                        if page.get("clean") is not None or page.get("boxes"):
                            changed = True
                        page["clean"] = None
                        page["boxes"] = []
''', '''                    page["skipped"] = skipped
                    if skipped:
                        if page.get("clean") is not None or page.get("boxes"):
                            changed = True
                        page["clean"] = None
                        page["boxes"] = []
                        page["process_required"] = False
                    else:
                        page["process_required"] = True
''')

# Process clean: preserve beats auto + manual inpaint and intentional source text is not residue noise.
rep("app/page_processing.py", 'from app.manifest_utils import assign_stable_detector_box_ids\n', 'from app.manifest_utils import assign_stable_detector_box_ids\nfrom app.region_policy import geometry_center_in_regions, subtract_regions_from_mask\n')
rep("app/page_processing.py", "excluded_regions", "preserve_regions", 3)
rep("app/page_processing.py", '''            manual_mask = self._read_manual_mask(mask_path, clean_image.shape[:2])
            if manual_mask is None:
                continue
''', '''            manual_mask = self._read_manual_mask(mask_path, clean_image.shape[:2])
            manual_mask = subtract_regions_from_mask(manual_mask, preserve_regions)
            if manual_mask is None or not cv2.countNonZero(manual_mask):
                continue
''')
rep("app/page_processing.py", '''            residue_boxes = self.detector.verify_post_inpaint_residue(
                clean_image,
                effective_boxes,
            )
''', '''            residue_boxes = self.detector.verify_post_inpaint_residue(
                clean_image,
                effective_boxes,
            )
            residue_boxes = [
                box for box in residue_boxes
                if not geometry_center_in_regions(
                    {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                    preserve_regions,
                )
            ]
''')
rep("app/page_processing.py", '            if record.get("needs_review") or not record.get("safe_to_inpaint")\n        ]\n', '            if (record.get("needs_review") or not record.get("safe_to_inpaint"))\n            and not geometry_center_in_regions(record, preserve_regions)\n        ]\n')
rep("app/page_processing.py", '            if record.get("deferred_reason")\n        ]\n', '            if record.get("deferred_reason")\n            and not geometry_center_in_regions(record, preserve_regions)\n        ]\n')

# Text-object lifecycle.
rep("app/text_objects.py", 'import copy\n', 'import copy\n\nfrom app.region_policy import geometry_center_in_regions, page_preserve_regions\n')
rep("app/text_objects.py", '    boxes = page.get("boxes") or []\n    active_box_ids: set[str] = set()\n', '    boxes = page.get("boxes") or []\n    preserve_regions = page_preserve_regions(page)\n    active_box_ids: set[str] = set()\n')
rep("app/text_objects.py", '''        region = _region_from_box(box)
        if region is None:
            continue
        active_box_ids.add(box_id)
''', '''        region = _region_from_box(box)
        if region is None:
            continue
        if geometry_center_in_regions(region, preserve_regions):
            continue
        active_box_ids.add(box_id)
''')

# OCR: plan, direct OCR, grouped OCR, and commit races all honor preserve/process state.
rep("app/ocr/service.py", 'from app.image_io import read_image\n', 'from app.image_io import read_image\nfrom app.region_policy import geometry_center_in_regions, page_preserve_regions, text_object_in_preserve_region\n')
rep("app/ocr/service.py", '''                if page.get("skipped"):
                    continue
                for box in page.get("boxes", []) or []:
''', '''                if page.get("skipped") or page.get("process_required"):
                    continue
                preserve_regions = page_preserve_regions(page)
                for box in page.get("boxes", []) or []:
''')
rep("app/ocr/service.py", '''                    if box.get("ocr_eligible") is False:
                        continue
                    if ocr_target_skip_reason(box):
''', '''                    if box.get("ocr_eligible") is False:
                        continue
                    if geometry_center_in_regions(box, preserve_regions):
                        continue
                    if ocr_target_skip_reason(box):
''')
rep("app/ocr/service.py", '''            if page.get("skipped"):
                raise ValueError("Cannot OCR a skipped page")
            box = _find_box(page, box_id)
''', '''            if page.get("skipped"):
                raise ValueError("Cannot OCR a skipped page")
            if page.get("process_required"):
                raise ValueError("Cannot OCR a page that requires processing")
            box = _find_box(page, box_id)
''')
rep("app/ocr/service.py", '''            if box.get("ocr_eligible") is False:
                raise ValueError(f"OCR target box is not eligible: {box_id}")
            original_value = page.get("original")
''', '''            if box.get("ocr_eligible") is False:
                raise ValueError(f"OCR target box is not eligible: {box_id}")
            if geometry_center_in_regions(box, page_preserve_regions(page)):
                raise ValueError(f"OCR target box is inside a preserve region: {box_id}")
            original_value = page.get("original")
''')
rep("app/ocr/service.py", '    matches: list[tuple[int, int, str, str]] = []\n    for box in page.get("boxes", []) or []:\n', '    matches: list[tuple[int, int, str, str]] = []\n    preserve_regions = page_preserve_regions(page)\n    for box in page.get("boxes", []) or []:\n')
rep("app/ocr/service.py", '        if box.get("ocr_eligible") is False:\n            continue\n        box_id = box.get("id")\n', '        if box.get("ocr_eligible") is False:\n            continue\n        if geometry_center_in_regions(box, preserve_regions):\n            continue\n        box_id = box.get("id")\n')
rep("app/ocr/service.py", '''            target = _find_box(page, box_id)
            if self._box_changed(target, box_snapshot):
''', '''            if page.get("skipped") or page.get("process_required"):
                raise OCRResultStale("Page became unavailable while OCR was running")
            target = _find_box(page, box_id)
            if target is not None and geometry_center_in_regions(target, page_preserve_regions(page)):
                raise OCRResultStale("OCR target entered a preserve region while OCR was running")
            if self._box_changed(target, box_snapshot):
''')
rep("app/ocr/service.py", '''            obj = _find_text_object(page, text_object_id)
            if obj is None:
                raise ValueError(f"Text object not found {text_object_id!r}")
''', '''            obj = _find_text_object(page, text_object_id)
            if obj is None:
                raise ValueError(f"Text object not found {text_object_id!r}")
            if page.get("skipped") or page.get("process_required"):
                raise ValueError("Cannot OCR this page before processing")
            if text_object_in_preserve_region(page, obj):
                raise ValueError("Cannot OCR a text object inside a preserve region")
''')
rep("app/ocr/service.py", '''            obj = _find_text_object(page, text_object_id)
            if obj is None:
                return manifest
            if self._group_result_stale(page, obj, snapshot, original_revision):
''', '''            obj = _find_text_object(page, text_object_id)
            if obj is None:
                return manifest
            if page.get("skipped") or page.get("process_required") or text_object_in_preserve_region(page, obj):
                return manifest
            if self._group_result_stale(page, obj, snapshot, original_revision):
''')

# Translation.
rep("app/routers/translation.py", 'from app.text_objects import ensure_page_text_objects\n', 'from app.text_objects import ensure_page_text_objects\nfrom app.region_policy import text_object_in_preserve_region\n')
rep("app/routers/translation.py", '    skipped_ocr_reject = 0\n    skipped_source_missing = 0\n', '    skipped_ocr_reject = 0\n    skipped_source_missing = 0\n    skipped_preserve_region = 0\n')
rep("app/routers/translation.py", '            if page.get("skipped"):\n                continue\n            _, changed = ensure_page_text_objects(page)\n', '            if page.get("skipped") or page.get("process_required"):\n                continue\n            _, changed = ensure_page_text_objects(page)\n')
rep("app/routers/translation.py", '''                if obj.get("source_missing"):
                    skipped_source_missing += 1
                    continue
                source = str(obj.get("ocr_text") or "").strip()
''', '''                if obj.get("source_missing"):
                    skipped_source_missing += 1
                    continue
                if text_object_in_preserve_region(page, obj):
                    skipped_preserve_region += 1
                    continue
                source = str(obj.get("ocr_text") or "").strip()
''')
rep("app/routers/translation.py", '            if page.get("skipped"):\n                stale += 1\n                continue\n            obj = _find_object(page, str(item["id"]))\n', '            if page.get("skipped") or page.get("process_required"):\n                stale += 1\n                continue\n            obj = _find_object(page, str(item["id"]))\n')
rep("app/routers/translation.py", '''            if obj is None or obj.get("source_missing"):
                stale += 1
                continue
            if str(obj.get("ocr_text") or "").strip() != str(item["text"]).strip():
''', '''            if obj is None or obj.get("source_missing"):
                stale += 1
                continue
            if text_object_in_preserve_region(page, obj):
                stale += 1
                continue
            if str(obj.get("ocr_text") or "").strip() != str(item["text"]).strip():
''')
rep("app/routers/translation.py", '            "skipped_source_missing": skipped_source_missing,\n', '            "skipped_source_missing": skipped_source_missing,\n            "skipped_preserve_region": skipped_preserve_region,\n', 2)

# Render identity + render behavior.
rep("app/render/identity.py", 'RENDER_IDENTITY_VERSION = "phase45-v1"\n', 'RENDER_IDENTITY_VERSION = "phase45-v2"\n')
rep("app/render/identity.py", '        "skipped": bool(page.get("skipped", False)),\n        "content": render_content,\n', '''        "skipped": bool(page.get("skipped", False)),
        "process_required": bool(page.get("process_required", False)),
        "preserve_regions": [
            {k: int(region.get(k, 0)) for k in ("x1", "y1", "x2", "y2")}
            for region in (page.get("preserve_regions") or []) if isinstance(region, dict)
        ],
        "content": render_content,
''')
rep("app/routers/render_commit.py", 'from app.security import validate_chapter_id\n', 'from app.security import validate_chapter_id\nfrom app.region_policy import geometry_center_in_regions, page_preserve_regions, text_object_in_preserve_region\n')
rep("app/routers/render_commit.py", '            if isinstance(obj, dict) and not obj.get("source_missing")\n', '''            if (
                isinstance(obj, dict)
                and not obj.get("source_missing")
                and not text_object_in_preserve_region(page, obj)
            )
''')
rep("app/routers/render_commit.py", '    return render_boxes_legacy(\n        image,\n        req,\n        page,\n', '''    legacy_page = copy.deepcopy(page)
    preserve_regions = page_preserve_regions(page)
    for box in legacy_page.get("boxes") or []:
        if isinstance(box, dict) and geometry_center_in_regions(box, preserve_regions):
            box["removed"] = True
    return render_boxes_legacy(
        image,
        req,
        legacy_page,
''')
rep("app/routers/render_commit.py", '    for obj in page.get("text_objects") or []:\n        if not isinstance(obj, dict) or obj.get("source_missing"):\n            continue\n', '''    for obj in page.get("text_objects") or []:
        if (
            not isinstance(obj, dict)
            or obj.get("source_missing")
            or text_object_in_preserve_region(page, obj)
        ):
            continue
''')
rep("app/routers/render_commit.py", '        page = copy.deepcopy(pages[req.page_index])\n        drafts = copy.deepcopy(manifest.get("drafts", {}))\n', '''        page = copy.deepcopy(pages[req.page_index])
        if page.get("skipped"):
            raise HTTPException(409, "Cannot render a skipped page; unskip it first")
        if page.get("process_required"):
            raise HTTPException(409, "Cannot render this page before processing")
        drafts = copy.deepcopy(manifest.get("drafts", {}))
''')
