"""Text crossing a slice cut belongs to one slice; its neighbour letters an exact copy so the stitched page is seamless."""
from __future__ import annotations

import copy
import re
import unicodedata

OWNER_SHARE = 0.5  # share of the smaller box two slices' boxes must share to be the same text
# Everything that decides how a line is lettered, copied from the owning slice.
LETTERING_KEYS = (
    "translation", "auto_translation", "translation_source", "translation_model", "translation_input_text",
    "typography_role", "container", "lettering_color", "enlarge", "source_cap_px", "font_ai_id",
    "font_selection_mode", "font_match", "style", "needs_review", "ocr_text_color", "ocr_font_size",
)


def _shifted(rect: dict, shift: int) -> dict:
    return {key: int(rect[key]) + (shift if key in ("y1", "y2") else 0) for key in ("x1", "y1", "x2", "y2")}


def _shifted_letters(value, shift: int):
    """A letter bitmap moved ``shift`` rows down (negative: up), same frame.

    A seam mirror letters the owner's text on the neighbouring slice, so
    its bitmap moves with its region; rows shifted off the frame are
    lost, and the vacated rows are empty. An undecodable value stays
    undecodable -- the mirror keeps no bitmap rather than a wrong one.
    """
    import numpy as np

    from app.image_io import encode_mask
    from app.mask_store import decode_mask_value

    arr = decode_mask_value(value)
    if arr is None:
        return None
    moved = np.zeros_like(arr)
    if shift == 0:
        moved = arr
    elif 0 < shift < arr.shape[0]:
        moved[shift:] = arr[:-shift]
    elif -arr.shape[0] < shift < 0:
        moved[:shift] = arr[-shift:]
    return encode_mask(moved)


def _top_letter_group(letters: np.ndarray, x1: int, y1: int) -> tuple[int, int, int, int] | None:
    """BBox of the topmost text in a letters mask holding two texts stacked with a wide gap.

    Row runs part where the empty band between them is at least BLOCK_GAP line
    heights; one run, or runs with only narrow gaps, is a single text, so None
    keeps the caller's mask-extent behaviour for ordinary boxes.
    """
    import numpy as np

    from app.knobs import knob

    rows = letters.any(axis=1)
    if not rows.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    runs = [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]
    if len(runs) < 2:
        return None
    height = float(np.median([end - start for start, end in runs]))
    if height <= 0:
        return None
    groups = [[runs[0]]]
    for run in runs[1:]:
        if run[0] - groups[-1][-1][1] >= knob("detect.block_gap") * height:
            groups.append([run])
        else:
            groups[-1].append(run)
    if len(groups) < 2:
        return None
    top_y1, top_y2 = groups[0][0][0], groups[0][-1][1]
    cols = np.flatnonzero(letters[top_y1:top_y2].any(axis=0))
    if len(cols) == 0:
        return None
    return (x1 + int(cols[0]), y1 + top_y1, x1 + int(cols[-1]) + 1, y1 + top_y2)


def set_letter_bounds(page: dict) -> None:
    """Give each object the extent of the source lettering its boxes erase, where the translation is lettered."""
    from app.mask_store import decode_mask_value
    from app.text_objects import source_box_ids

    boxes = {str(box.get("id")): box for box in page.get("boxes") or [] if isinstance(box, dict)}
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict):
            continue
        rects = []
        box_rects = []
        for ref in source_box_ids(obj):
            box = boxes.get(ref)
            if not box:
                continue
            try:
                x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                continue
            mask = decode_mask_value(box.get("mask"))
            letters = decode_mask_value(box.get("letters_mask"))
            top = None
            if letters is not None and letters.shape == (y2 - y1, x2 - x1) and letters.any():
                top = _top_letter_group(letters, x1, y1)
            if top is not None:
                rects.append(top)
            elif mask is not None and mask.shape == (y2 - y1, x2 - x1) and mask.any():
                ys, xs = mask.nonzero()
                rects.append((x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1))
            else:
                box_rects.append((x1, y1, x2, y2))
        # One unusable box must not erase the extent the usable boxes still prove.
        # A detection box with no usable mask still bounds the lettering it
        # detected; without any bounds the fit has only the region-wide
        # attempt, so the box extent beats no bounds at all. A topmost
        # letters group wins where it exists: a merged box can hold two
        # texts, and only the topmost is the one translated. Mask extents
        # win next: they are the letters, while a box may hold outline and
        # margin round them.
        use = rects or box_rects
        if use:
            obj["letter_bounds"] = {"x1": min(r[0] for r in use), "y1": min(r[1] for r in use),
                                    "x2": max(r[2] for r in use), "y2": max(r[3] for r in use)}
        # Otherwise keep whatever bounds the object already had: no usable
        # box must erase an extent proven earlier.
        # The letter bitmap follows the same boxes, framed on the
        # object's region: the render's trim reads it to see a diagonal
        # waist, and the clean image cannot supply it. A merged object
        # unions its boxes' letters; an object with no usable letters
        # keeps whatever bitmap it already had.
        import numpy as np

        from app.image_io import encode_mask

        region = obj.get("region")
        if isinstance(region, dict):
            try:
                rx1, ry1, rx2, ry2 = (int(region[k]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                rx1 = ry1 = rx2 = ry2 = 0
            if rx2 > rx1 and ry2 > ry1:
                union = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
                usable = False
                for ref in source_box_ids(obj):
                    box = boxes.get(ref)
                    if not box:
                        continue
                    try:
                        x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
                    except (KeyError, TypeError, ValueError):
                        continue
                    letters = decode_mask_value(box.get("letters_mask"))
                    if (letters is None or letters.shape != (y2 - y1, x2 - x1)
                            or not letters.any()):
                        continue
                    r0, r1 = max(0, y1 - ry1), min(ry2 - ry1, y2 - ry1)
                    c0, c1 = max(0, x1 - rx1), min(rx2 - rx1, x2 - rx1)
                    ar0, ac0 = r0 - (y1 - ry1), c0 - (x1 - rx1)
                    union[r0:r1, c0:c1] |= letters[
                        ar0:ar0 + (r1 - r0), ac0:ac0 + (c1 - c0)].astype(np.uint8)
                    usable = True
                if usable:
                    obj["letters_mask"] = encode_mask(union)
        # The bubble interior follows the same boxes: an object over
        # several boxes gets the union of their interiors, so a double
        # bubble split into two objects keeps one lobe each.
        from app.detector.bubble_mask import merge_bubble_masks

        masks = [box.get("bubble_mask") for ref in source_box_ids(obj)
                 if (box := boxes.get(ref)) is not None
                 and isinstance(box.get("bubble_mask"), dict)]
        merged = merge_bubble_masks(masks) if masks else None
        if merged is not None:
            obj["bubble_mask"] = merged
        else:
            obj.pop("bubble_mask", None)


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


def _source_words(obj: dict) -> list[str]:
    """Words of the source text the object was read from, when known.

    The vision model's read wins where it exists; the detector OCR is
    the fallback. An object with neither carries no source evidence.
    """
    for key in ("source_read", "ocr_text", "auto_ocr_text"):
        words = _fold_words(obj.get(key))
        if words:
            return words
    return []


def _source_texts_agree(a: dict, b: dict) -> bool:
    """True when nothing known contradicts two objects being the same text.

    Mirror matching is geometry-only today: a small free-text box sitting
    mostly inside a big bubble's rect matches at share 1.0 and is then
    translated as the neighbour's text, losing its own. When both sides
    carry a source read and they share no word at all, they are different
    texts -- geometry notwithstanding. When either side has no source
    text there is nothing to contradict, so geometry alone still decides.
    """
    words_a = _source_words(a)
    words_b = _source_words(b)
    if not words_a or not words_b:
        return True
    return bool(set(words_a) & set(words_b))


def _separate_lobes(a: dict, b: dict) -> bool:
    """Two objects in different lobes of one waist are never one block.

    The geometry and letter gates cannot tell a stacked double bubble
    from one block's lines: the lobes sit centred and their lettering
    flows on. The masks can -- the lobes' interiors are disjoint.
    """
    from app.detector.boxes import bubble_masks_are_separate_lobes

    return bubble_masks_are_separate_lobes(a.get("bubble_mask"), b.get("bubble_mask"))


def _owner(pages: list, page_index: int, obj: dict, *, skip_mirrors: bool = False):
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
        # A mirror is a copy, not a source: creating from one would chain copies
        # across slices instead of lettering from the slice that owns the text.
        best = max(((candidate, other_rect) for candidate in other.get("text_objects") or []
                    if isinstance(candidate, dict) and not candidate.get("joined_into")
                    and not (skip_mirrors and candidate.get("seam_owner"))
                    and (other_rect := _rect(candidate, other_core[0]))),
                   key=lambda item: _shared(rect, item[1]), default=None)
        if best is None or _shared(rect, best[1]) < OWNER_SHARE:
            continue
        if not _source_texts_agree(obj, best[0]):
            continue
        # Both slices judge by the same whole text, so exactly one of them owns it.
        center = (min(rect[1], best[1][1]) + max(rect[3], best[1][3])) / 2
        if other_core[1] <= center < other_core[2]:
            return other_index, best[0]
    return None


def create_seam_mirrors(manifest: dict) -> list[tuple[int, str]]:
    """Give each overlap-context box its own mirror object; returns what was created.

    A detector box lying entirely outside its slice's stitch core is kept
    only as context: clean never erases it and ``ensure_page_text_objects``
    makes no object for it, so the text it holds stays in English on that
    slice even though the neighbouring slice translates it. The mirror
    anchors the fragment here with the owner's id; the translate step
    already skips such mirrors (``lettered_elsewhere``) and the existing
    seam sync then letters each one exactly as its owning slice does.
    A box with no matching neighbour object gets nothing: with no owner
    there is no lettering to copy.
    """
    from app.text_objects import DEFAULT_TEXT_OBJECT_STYLE, _auto_object_id, source_box_ids

    pages = manifest.get("pages") or []
    created: list[tuple[int, str]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict) or page.get("skipped"):
            continue
        covered = {ref for obj in page.get("text_objects") or [] if isinstance(obj, dict)
                   for ref in source_box_ids(obj)}
        # One text, one mirror per slice: two fragment boxes of the same text
        # would letter it twice, so only the larger fragment anchors it.
        by_owner: dict[tuple[int, str], tuple[dict, dict]] = {}
        for box in page.get("boxes") or []:
            if not isinstance(box, dict) or not box.get("overlap_context_only") or box.get("removed"):
                continue
            box_id = str(box.get("id") or "")
            if not box_id or box_id in covered:
                continue
            try:
                region = {"x1": int(box["x1"]), "y1": int(box["y1"]),
                          "x2": int(box["x2"]), "y2": int(box["y2"])}
            except (KeyError, TypeError, ValueError):
                continue
            if region["x2"] <= region["x1"] or region["y2"] <= region["y1"]:
                continue
            found = _owner(pages, page_index,
                           {"region": region, "ocr_text": str(box.get("ocr_text") or "")},
                           skip_mirrors=True)
            if found is None:
                continue
            owner_page, owner = found
            key = (owner_page, str(owner.get("id")))
            if key in by_owner and _area((region["x1"], region["y1"], region["x2"], region["y2"])) <= _area(
                    (by_owner[key][0]["x1"], by_owner[key][0]["y1"],
                     by_owner[key][0]["x2"], by_owner[key][0]["y2"])):
                continue
            by_owner[key] = (box, owner)
        for (owner_page, owner_id), (box, owner) in by_owner.items():
            region = {"x1": int(box["x1"]), "y1": int(box["y1"]),
                      "x2": int(box["x2"]), "y2": int(box["y2"])}
            box_id = str(box.get("id"))
            ocr_text = str(box.get("ocr_text") or "")
            page.setdefault("text_objects", []).append({
                "id": _auto_object_id(box_id),
                "shape": "rectangle",
                "region": copy.deepcopy(region),
                "source_boxes": [box_id],
                "ocr_text": ocr_text,
                "translation": "",
                "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
                "font_selection_mode": "auto",
                "font_match": None,
                "font_ai_id": None,
                "origin": "detector",
                "auto_generated": True,
                "auto_geometry": copy.deepcopy(region),
                "auto_ocr_text": ocr_text,
                "seam_owner": {"page": owner_page, "id": owner_id},
            })
            created.append((page_index, _auto_object_id(box_id)))
    return created


_WORD_SEPARATORS = ".,;:!?\"'“”‘’…()[]{}—–-"


def _norm_words(text: str) -> list[str]:
    """Words of a translation, punctuation and case aside, for prefix comparison."""
    lowered = " ".join(str(text or "").split()).lower()
    return lowered.translate(str.maketrans({c: " " for c in _WORD_SEPARATORS})).split()


def _trim_leading_words(text: str, count: int) -> str:
    """``text`` without its first ``count`` words; the continuation becomes sentence-initial."""
    # The same punctuation-aware boundaries as _norm_words, so a contraction or hyphen counts as its parts.
    matches = list(re.finditer(r"[^\s" + re.escape(_WORD_SEPARATORS) + r"]+", text))
    if len(matches) <= count:
        return ""
    rest = text[matches[count - 1].end():].lstrip(" \t\r\n,.!?;:\"'“”‘’…—–-")
    for i, ch in enumerate(rest):
        if ch.isalpha():
            return rest[:i] + ch.upper() + rest[i + 1:]
    return rest


def _dup_stacked(upper_rect, lower_rect) -> bool:
    """True when ``lower`` sits directly under ``upper``, as fragments of one stepped box do.

    The centre rule of :func:`_stacked` is left out on purpose: a stepped
    box's fragments are offset sideways by the step, which is exactly why
    they were never joined. The verbatim word match below is the real gate;
    geometry only confirms the two are neighbours.
    """
    h = min(upper_rect[3] - upper_rect[1], lower_rect[3] - lower_rect[1])
    if h <= 0:
        return False
    gap = lower_rect[1] - upper_rect[3]
    across = min(upper_rect[2], lower_rect[2]) - max(upper_rect[0], lower_rect[0])
    return (-STACK_TUCK * h <= gap <= STACK_GAP * h
            and across >= STACK_OVERLAP * min(upper_rect[2] - upper_rect[0], lower_rect[2] - lower_rect[0]))


def drop_duplicated_leading_text(manifest: dict) -> list[int]:
    """Strip a sentence a lower stacked fragment repeats from the fragment above it.

    A stepped box split into two fragments is translated twice: the upper
    fragment carries its own sentence, while the lower fragment's
    translation re-reads the whole box and repeats it. The repeat is
    verbatim, so only an exact word-prefix match trims, and only between
    stacked neighbours; mirrors and owners alike are trimmed, since the
    sync has already copied the owner's wording onto its mirror. Returns
    the slices that changed.
    """
    pages = manifest.get("pages") or []
    changed: list[int] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict) or page.get("skipped"):
            continue
        objs = [obj for obj in page.get("text_objects") or []
                if isinstance(obj, dict) and not obj.get("joined_into")
                and not obj.get("overlap_dropped") and not obj.get("source_missing")
                and str(obj.get("translation") or "").strip()
                and _rect(obj, 0) is not None]
        objs.sort(key=lambda obj: (_rect(obj, 0)[1], _rect(obj, 0)[0]))
        for lower_index, lower in enumerate(objs):
            lower_words = _norm_words(lower.get("translation"))
            if len(lower_words) < 4:
                continue
            for upper in objs[:lower_index]:
                upper_words = _norm_words(upper.get("translation"))
                if len(upper_words) < 3 or len(upper_words) >= len(lower_words):
                    continue
                if lower_words[:len(upper_words)] != upper_words:
                    continue
                if not _dup_stacked(_rect(upper, 0), _rect(lower, 0)):
                    continue
                trimmed = _trim_leading_words(str(lower.get("translation")), len(upper_words))
                if not trimmed or _norm_words(trimmed) == lower_words:
                    continue
                old = str(lower.get("translation") or "")
                lower["translation"] = trimmed
                if str(lower.get("auto_translation") or "") == old:
                    lower["auto_translation"] = trimmed
                lower_words = _norm_words(trimmed)
                if page_index not in changed:
                    changed.append(page_index)
    return changed


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
            # An object dropped for overlapping another stays empty, or the two would garble each other again.
            if not isinstance(obj, dict) or obj.get("joined_into") or obj.get("overlap_dropped"):
                continue
            found = _owner(pages, page_index, obj)
            if found is None or not str(found[1].get("translation") or "").strip():
                continue
            owner_page, owner = found
            shift = _core(pages[owner_page])[0] - core[0]
            region = _shifted(owner["region"], shift)
            update = {key: copy.deepcopy(owner[key]) for key in LETTERING_KEYS if key in owner}
            # The owner's translation was gated when it was committed, but a
            # later polish or manual edit can leave English (or CJK) in it;
            # the copy passes the same gate, or the mirror letters the
            # untranslated text the gate would have blanked. A copy that
            # gates to nothing is skipped, leaving the mirror as it was.
            from app.translation.vision import VisionTranslationResult, _drop_foreign_script

            owner_id = str(owner.get("id"))
            probe = VisionTranslationResult(
                {owner_id: str(owner.get("translation") or "")}, "", {}, None)
            probe_data = {"translations": [{"id": owner_id,
                                            "source": str(owner.get("source_read") or owner.get("ocr_text") or ""),
                                            "translated_text": str(owner.get("translation") or "")}]}
            gated = _drop_foreign_script(probe, probe_data)[0].translations.get(owner_id, "")
            if not gated.strip():
                continue
            if gated != str(owner.get("translation") or ""):
                update["translation"] = gated
                if "auto_translation" in update:
                    update["auto_translation"] = gated
            # The copy letters in exactly the area the owner does, so the two halves line up.
            if isinstance(owner.get("letter_bounds"), dict):
                update["letter_bounds"] = _shifted(owner["letter_bounds"], shift)
            if isinstance(owner.get("bubble_mask"), dict):
                from app.detector.bubble_mask import shift_bubble_mask

                update["bubble_mask"] = shift_bubble_mask(owner["bubble_mask"], 0, shift)
            if owner.get("letters_mask"):
                shifted_letters = _shifted_letters(owner["letters_mask"], shift)
                if shifted_letters is not None:
                    update["letters_mask"] = shifted_letters
            stale = [key for key in LETTERING_KEYS + ("letter_bounds", "bubble_mask", "letters_mask") if key in obj and key not in update]
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


def _letter_rect(obj: dict) -> tuple[int, int, int, int] | None:
    """The object's lettering extent when known, else its region.

    Detector regions are loose — two map labels' boxes can overlap by
    more than half while their letters sit in different places (slice
    91: ANHUI's box overlaps SHAANXI's by 0.65, but ANHUI's letters are
    below SHAANXI's). Dropping on the region alone deletes a real label.
    Letter bounds are set by set_letter_bounds before the sync runs.
    """
    bounds = obj.get("letter_bounds")
    if isinstance(bounds, dict):
        try:
            rect = (int(bounds["x1"]), int(bounds["y1"]), int(bounds["x2"]), int(bounds["y2"]))
        except (KeyError, TypeError, ValueError):
            rect = None
        if rect is not None and rect[2] > rect[0] and rect[3] > rect[1]:
            return rect
    return _rect(obj, 0)


def _fold_words(text: str) -> list[str]:
    """Words of a translation, case- and punctuation-folded, for comparison."""
    folded = unicodedata.normalize("NFKC", str(text or "")).lower()
    folded = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in folded)
    return folded.split()


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common word subsequence, in order."""
    # Dialogue is short; the quadratic table is cheaper than a fix-up.
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            current[j] = previous[j - 1] + 1 if word == other else max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def _translations_duplicate(a: str, b: str) -> bool:
    """True when two translations are the same text read twice.

    Overlapping lettering is usually one text detected twice — but two
    bubbles can overlap on the page (a cloud tucked against an oval, a
    lobe against its double bubble) while holding different words, and
    dropping on geometry alone deletes the smaller bubble's text (ch42
    p008 "HE..." cloud, p012 lobe 1 "ANYHOW,"). The smaller object is
    only a duplicate when its words are the larger one's words *in the
    same order*: an exact match, an ordered word-prefix, or most words
    shared in sequence.

    A word *set* cannot make that call on manhwa dialogue, typically
    2-5 words: "Tôi không muốn đi" vs "Tôi không muốn về" share 3 of
    5 words as a set but differ in the one word that matters. So the
    bar rises as the text shortens — a 2-word line is cheap, two
    bubbles can each say "Của tôi", so only a longer shared run proves
    one text read twice.
    """
    words_a, words_b = _fold_words(a), _fold_words(b)
    if not words_a or not words_b:
        return False
    short, long = sorted((words_a, words_b), key=len)
    n = len(short)
    if short == long:
        # Same words in the same order; below 3 words that is not
        # proof of one text read twice ("Của tôi" vs "Của tôi!").
        return n >= 3
    if long[:n] == short:
        # The shorter line is how the longer one starts: a truncated
        # re-read — but only when the shared run is long enough to
        # mean something. "Đi đi" starting "Đi đi mau" does not.
        return n >= 4
    shared = _lcs_len(short, long)
    return shared >= 5 and shared / n >= 0.8


def _union_rect(a: dict | None, b: dict | None) -> dict | None:
    """Bounding-box union of two {x1,y1,x2,y2} dicts, or the one that exists."""
    if not isinstance(a, dict):
        return dict(b) if isinstance(b, dict) else None
    if not isinstance(b, dict):
        return dict(a)
    try:
        return {"x1": min(int(a["x1"]), int(b["x1"])), "y1": min(int(a["y1"]), int(b["y1"])),
                "x2": max(int(a["x2"]), int(b["x2"])), "y2": max(int(a["y2"]), int(b["y2"]))}
    except (KeyError, TypeError, ValueError):
        return dict(a)


def _regions_touch(a: dict | None, b: dict | None, gap_px: int = 8) -> bool:
    """True when two region rects overlap or sit within a few pixels of each
    other: the halves of one bubble the detector split. A dropped object
    whose region sits apart is a different bubble that merely shares
    lettering, and the keeper must not absorb its region."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    try:
        ax1, ay1, ax2, ay2 = (int(a[k]) for k in ("x1", "y1", "x2", "y2"))
        bx1, by1, bx2, by2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False
    gap_x = max(bx1 - ax2, ax1 - bx2)
    gap_y = max(by1 - ay2, ay1 - by2)
    return gap_x <= gap_px and gap_y <= gap_px


def drop_overlapping_letters(manifest: dict) -> list[tuple[int, str]]:
    """Of two translated objects on one slice that mostly overlap, keep the larger; returns what was dropped.

    Only a duplicate is dropped: the smaller object keeps its translation
    when it says something the larger one does not, however much their
    lettering overlaps. And a dropped duplicate is not just deleted —
    the detector often splits one bubble into two overlapping halves
    (ch42 p010: the bubble's left and right halves), so the keeper
    absorbs the dropped half's region and lettering, and the survivor
    is fitted to the whole bubble instead of the half it happened to be.
    A seam mirror is never dropped here, nor a keeper: its text is owned
    by another slice, and the sync, not the drop, decides what it says.
    """
    dropped = []
    for page_index, page in enumerate(manifest.get("pages") or []):
        objects = [obj for obj in page.get("text_objects") or [] if isinstance(obj, dict)
                   and str(obj.get("translation") or "").strip() and _letter_rect(obj) is not None
                   and not obj.get("seam_owner")]
        objects.sort(key=lambda obj: -_area(_letter_rect(obj)))
        kept: list[dict] = []
        for obj in objects:
            rect = _letter_rect(obj)
            overlapped = [other for other in kept
                          if _shared(rect, _letter_rect(other)) >= OVERLAP_KEEP]
            dup_of = next((other for other in overlapped
                           if _translations_duplicate(str(obj.get("translation") or ""),
                                                      str(other.get("translation") or ""))), None)
            if dup_of is not None:
                obj["translation"] = ""
                obj["overlap_dropped"] = True
                dropped.append((page_index, str(obj.get("id"))))
                # The dropped half belonged to the keeper's bubble only when
                # their regions touch: the detector often splits one bubble
                # into two overlapping halves (ch42 p010), so the keeper's
                # fit box grows to the union. A dropped object whose region
                # sits apart is a different bubble that merely shares
                # lettering, and the keeper must not absorb its region.
                # Nor may it absorb the other lobe of a waist: the two
                # halves hold different texts, and the union would letter
                # the keeper across the waist.
                if _regions_touch(dup_of.get("region"), obj.get("region")) and not _separate_lobes(dup_of, obj):
                    dup_of["region"] = _union_rect(dup_of.get("region"), obj.get("region"))
                    dup_of["letter_bounds"] = _union_rect(dup_of.get("letter_bounds"),
                                                           obj.get("letter_bounds"))
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
STACK_LETTER_GAP = 0.75  # widest empty band between two fragments' lettering, as a share of the shorter lettering


def _letters_continue(upper: dict, lower: dict) -> bool:
    """One block's lettering flows on; a second bubble restarts after the band holding both outlines.

    Box geometry alone cannot tell the two apart when the bubbles are the same width and
    centred, so the letter extents decide where they are known.
    """
    a, b = upper.get("letter_bounds"), lower.get("letter_bounds")
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return True
    try:
        gap = int(b["y1"]) - int(a["y2"])
        shorter = min(int(a["y2"]) - int(a["y1"]), int(b["y2"]) - int(b["y1"]))
    except (KeyError, TypeError, ValueError):
        return True
    return shorter <= 0 or gap <= STACK_LETTER_GAP * shorter


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
                pair = sorted(((a, upper), (b, lines[j])), key=lambda item: item[0][1])
                if (_stacked(pair[0][0], pair[1][0]) and _letters_continue(pair[0][1], pair[1][1])
                        and not _separate_lobes(pair[0][1], pair[1][1])):
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
