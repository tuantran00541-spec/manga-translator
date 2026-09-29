"""Text crossing a slice cut belongs to one slice; its neighbour letters an exact copy so the stitched page is seamless."""
from __future__ import annotations

import copy

OWNER_SHARE = 0.5  # share of the smaller box two slices' boxes must share to be the same text
# Everything that decides how a line is lettered, copied from the owning slice.
LETTERING_KEYS = (
    "translation", "auto_translation", "translation_source", "translation_model", "translation_input_text",
    "typography_role", "container", "lettering_color", "enlarge", "source_cap_px", "font_ai_id",
    "font_selection_mode", "font_match", "style", "needs_review", "ocr_text_color", "ocr_font_size",
)


def _shifted(rect: dict, shift: int) -> dict:
    return {key: int(rect[key]) + (shift if key in ("y1", "y2") else 0) for key in ("x1", "y1", "x2", "y2")}


def set_letter_bounds(page: dict) -> None:
    """Give each object the extent of the source lettering its boxes erase, where the translation is lettered."""
    from app.mask_store import decode_mask_value
    from app.text_objects import source_box_ids

    boxes = {str(box.get("id")): box for box in page.get("boxes") or [] if isinstance(box, dict)}
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict):
            continue
        rects = []
        for ref in source_box_ids(obj):
            box = boxes.get(ref)
            mask = decode_mask_value(box.get("mask")) if box else None
            try:
                x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                continue
            if mask is None or mask.shape != (y2 - y1, x2 - x1) or not mask.any():
                continue
            ys, xs = mask.nonzero()
            rects.append((x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1))
        if rects and len(rects) == len(source_box_ids(obj)):
            obj["letter_bounds"] = {"x1": min(r[0] for r in rects), "y1": min(r[1] for r in rects),
                                    "x2": max(r[2] for r in rects), "y2": max(r[3] for r in rects)}
        else:
            obj.pop("letter_bounds", None)


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


def _owner(pages: list, page_index: int, obj: dict):
    """(page index, object) of the slice whose core holds the object's centre, when it has the same object."""
    page = pages[page_index]
    core = _core(page)
    rect = _rect(obj, core[0]) if core else None
    if rect is None:
        return None
    for other_index in (page_index - 1, page_index + 1):
        if not 0 <= other_index < len(pages):
            continue
        other = pages[other_index]
        other_core = _core(other)
        if other.get("skipped") or other_core is None or other.get("source_page") != page.get("source_page"):
            continue
        # The two slices may box the same text a little differently, so containment counts, not IoU.
        # A line joined into another is lettered by that one, whose region covers it.
        best = max(((candidate, other_rect) for candidate in other.get("text_objects") or []
                    if isinstance(candidate, dict) and not candidate.get("joined_into")
                    and (other_rect := _rect(candidate, other_core[0]))),
                   key=lambda item: _shared(rect, item[1]), default=None)
        if best is None or _shared(rect, best[1]) < OWNER_SHARE:
            continue
        # Both slices judge by the same whole text, so exactly one of them owns it.
        center = (min(rect[1], best[1][1]) + max(rect[3], best[1][3])) / 2
        if other_core[1] <= center < other_core[2]:
            return other_index, best[0]
    return None


def seam_mirror_ids(manifest: dict, page_index: int) -> set[str]:
    """Objects of this slice that another slice owns and letters."""
    pages = manifest.get("pages") or []
    if not 0 <= page_index < len(pages):
        return set()
    return {str(obj["id"]) for obj in pages[page_index].get("text_objects") or []
            if isinstance(obj, dict) and obj.get("id") and not obj.get("joined_into")
            and _owner(pages, page_index, obj) is not None}


def sync_seam_mirrors(manifest: dict) -> list[int]:
    """Copy each owner's lettering and region onto its mirror; returns the slices that changed."""
    pages = manifest.get("pages") or []
    changed: list[int] = []
    for page_index, page in enumerate(pages):
        if page.get("skipped"):
            continue
        core = _core(page)
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("joined_into"):
                continue
            found = _owner(pages, page_index, obj)
            if found is None or not str(found[1].get("translation") or "").strip():
                continue
            owner_page, owner = found
            shift = _core(pages[owner_page])[0] - core[0]
            region = _shifted(owner["region"], shift)
            update = {key: copy.deepcopy(owner[key]) for key in LETTERING_KEYS if key in owner}
            # The copy letters in exactly the area the owner does, so the two halves line up.
            if isinstance(owner.get("letter_bounds"), dict):
                update["letter_bounds"] = _shifted(owner["letter_bounds"], shift)
            stale = [key for key in LETTERING_KEYS + ("letter_bounds",) if key in obj and key not in update]
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


STACK_GAP = 0.2  # widest gap between stacked lines of one block, as a share of the shorter line
STACK_TUCK = 0.5  # deepest two lines' boxes may overlap, as a share of the shorter line
STACK_OVERLAP = 0.5  # share of the narrower line two stacked lines must share across
# Lines of one block are centred on each other; two bubbles stacked in one balloon sit off to the sides.
STACK_CENTRE = 0.15


def _stacked(upper, lower) -> bool:
    """True when ``lower`` continues the block ``upper`` is part of."""
    h = min(upper[3] - upper[1], lower[3] - lower[1])
    if h <= 0:
        return False
    gap = lower[1] - upper[3]
    across = min(upper[2], lower[2]) - max(upper[0], lower[0])
    wider = max(upper[2] - upper[0], lower[2] - lower[0])
    offset = abs((upper[0] + upper[2]) - (lower[0] + lower[2])) / 2
    return (-STACK_TUCK * h <= gap <= STACK_GAP * h and offset <= STACK_CENTRE * wider
            and across >= STACK_OVERLAP * min(upper[2] - upper[0], lower[2] - lower[0]))


def join_stacked_lines(manifest: dict, page_indices: list[int]) -> int:
    """Lines the detector split from one block become one object before translation, so no word is dropped.

    Lines in the overlap with a neighbour join too: both slices then hold the same block and one of them owns it.
    """
    pages = manifest.get("pages") or []
    joined = 0
    for page_index in page_indices:
        page = pages[page_index]
        lines = sorted((obj for obj in page.get("text_objects") or []
                        if isinstance(obj, dict) and obj.get("id") and not obj.get("source_missing")
                        and not obj.get("joined_into") and not str(obj.get("translation") or "").strip()
                        and _rect(obj, 0) is not None),
                       key=lambda obj: (_rect(obj, 0)[1], _rect(obj, 0)[0]))
        # A block is every line linked to another by stacking, so words split side by side join too.
        group = list(range(len(lines)))

        def root(i: int) -> int:
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        for i, upper in enumerate(lines):
            for j in range(i + 1, len(lines)):
                a, b = _rect(upper, 0), _rect(lines[j], 0)
                if _stacked(*sorted((a, b), key=lambda r: r[1])):
                    group[root(j)] = root(i)
        blocks: dict[int, list[dict]] = {}
        for i, obj in enumerate(lines):
            blocks.setdefault(root(i), []).append(obj)
        for block in blocks.values():
            if len(block) < 2:
                continue
            rects = [_rect(obj, 0) for obj in block]
            top = block[0]
            top["region"] = {"x1": min(r[0] for r in rects), "y1": min(r[1] for r in rects),
                             "x2": max(r[2] for r in rects), "y2": max(r[3] for r in rects)}
            top["joined_lines"] = [str(obj["id"]) for obj in block[1:]]
            bounds = [obj.get("letter_bounds") for obj in block]
            if all(isinstance(b, dict) for b in bounds):
                top["letter_bounds"] = {"x1": min(b["x1"] for b in bounds), "y1": min(b["y1"] for b in bounds),
                                        "x2": max(b["x2"] for b in bounds), "y2": max(b["y2"] for b in bounds)}
            else:
                top.pop("letter_bounds", None)
            for obj in block[1:]:
                obj["joined_into"] = str(top["id"])
                obj["translation"] = ""
            joined += len(block) - 1
    return joined
