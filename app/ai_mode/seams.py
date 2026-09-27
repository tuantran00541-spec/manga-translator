"""Text crossing a slice cut belongs to one slice; its neighbour letters an exact copy so the stitched page is seamless."""
from __future__ import annotations

import copy

OWNER_IOU = 0.5  # the same bubble seen from two slices
# Everything that decides how a line is lettered, copied from the owning slice.
LETTERING_KEYS = (
    "translation", "auto_translation", "translation_source", "translation_model", "translation_input_text",
    "typography_role", "lettering_color", "enlarge", "source_cap_px", "source_read", "font_ai_id",
    "font_selection_mode", "font_match", "style", "needs_review", "ocr_text_color", "ocr_font_size",
)


def _core(page: dict) -> tuple[int, int, int] | None:
    """(source offset of the slice, core start, core end) in source-page pixels."""
    core = page.get("stitch_core")
    if not isinstance(core, dict):
        return None
    try:
        return int(core["source_y1"]), int(core["core_source_y1"]), int(core["core_source_y2"])
    except (KeyError, TypeError, ValueError):
        return None


def _rect(obj: dict, offset: int) -> tuple[int, int, int, int] | None:
    region = obj.get("region")
    if not isinstance(region, dict):
        return None
    try:
        return int(region["x1"]), int(region["y1"]) + offset, int(region["x2"]), int(region["y2"]) + offset
    except (KeyError, TypeError, ValueError):
        return None


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _owner(pages: list, page_index: int, obj: dict):
    """(page index, object) of the slice whose core holds the object's centre, when it has the same object."""
    page = pages[page_index]
    core = _core(page)
    rect = _rect(obj, core[0]) if core else None
    if rect is None:
        return None
    center = (rect[1] + rect[3]) / 2
    if core[1] <= center < core[2]:
        return None
    for other_index in (page_index - 1, page_index + 1):
        if not 0 <= other_index < len(pages):
            continue
        other = pages[other_index]
        other_core = _core(other)
        if (other.get("skipped") or other_core is None or other.get("source_page") != page.get("source_page")
                or not other_core[1] <= center < other_core[2]):
            continue
        best = max(((candidate, _iou(rect, other_rect)) for candidate in other.get("text_objects") or []
                    if isinstance(candidate, dict) and (other_rect := _rect(candidate, other_core[0]))),
                   key=lambda item: item[1], default=(None, 0.0))
        if best[0] is not None and best[1] >= OWNER_IOU:
            return other_index, best[0]
    return None


def seam_mirror_ids(manifest: dict, page_index: int) -> set[str]:
    """Objects of this slice that another slice owns and letters."""
    pages = manifest.get("pages") or []
    if not 0 <= page_index < len(pages):
        return set()
    return {str(obj["id"]) for obj in pages[page_index].get("text_objects") or []
            if isinstance(obj, dict) and obj.get("id") and _owner(pages, page_index, obj) is not None}


def sync_seam_mirrors(manifest: dict) -> list[int]:
    """Copy each owner's lettering and region onto its mirror; returns the slices that changed."""
    pages = manifest.get("pages") or []
    changed: list[int] = []
    for page_index, page in enumerate(pages):
        if page.get("skipped"):
            continue
        core = _core(page)
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            found = _owner(pages, page_index, obj)
            if found is None or not str(found[1].get("translation") or "").strip():
                continue
            owner_page, owner = found
            shift = _core(pages[owner_page])[0] - core[0]
            region = {key: int(owner["region"][key]) + (shift if key in ("y1", "y2") else 0)
                      for key in ("x1", "y1", "x2", "y2")}
            update = {key: copy.deepcopy(owner[key]) for key in LETTERING_KEYS if key in owner}
            stale = [key for key in LETTERING_KEYS if key in obj and key not in owner]
            if all(obj.get(key) == value for key, value in update.items()) and not stale and obj.get("region") == region:
                continue
            obj.update(update)
            for key in stale:
                obj.pop(key)
            obj["region"] = region
            obj.pop("ocr_text_region", None)
            obj["seam_owner"] = {"page": owner_page, "id": str(owner.get("id"))}
            if page_index not in changed:
                changed.append(page_index)
    return changed


OVERLAP_KEEP = 0.5  # two lettered objects sharing this much of the smaller one garble each other


def drop_overlapping_letters(manifest: dict) -> list[tuple[int, str]]:
    """Of two translated objects on one slice that mostly overlap, keep the larger; returns what was dropped."""
    dropped = []
    for page_index, page in enumerate(manifest.get("pages") or []):
        objects = [obj for obj in page.get("text_objects") or [] if isinstance(obj, dict)
                   and str(obj.get("translation") or "").strip() and _rect(obj, 0) is not None]
        objects.sort(key=lambda obj: -_area(_rect(obj, 0)))
        kept: list[dict] = []
        for obj in objects:
            rect = _rect(obj, 0)
            if any(_shared(rect, _rect(other, 0)) >= OVERLAP_KEEP for other in kept):
                obj["translation"] = ""
                obj["overlap_dropped"] = True
                dropped.append((page_index, str(obj.get("id"))))
            else:
                kept.append(obj)
    return dropped


def _area(rect) -> int:
    return max(1, (rect[2] - rect[0]) * (rect[3] - rect[1]))


def _shared(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min(_area(a), _area(b))
