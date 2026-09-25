from __future__ import annotations

import hashlib

from app.parameters import EDITORIAL_STORY_CANDIDATE_CONFIDENCE
from app.region_policy import (
    geometry_center_in_regions,
    page_preserve_regions,
    text_object_in_preserve_region,
)
from app.text_objects import ensure_page_text_objects, source_box_ids


EDITORIAL_DISPOSITIONS = frozenset(
    {"promote", "duplicate", "non_story", "noise", "preserve"}
)
EDITORIAL_DROP_DISPOSITIONS = frozenset(
    {"duplicate", "non_story", "noise", "preserve"}
)
CLEANUP_DISPOSITIONS = frozenset(
    {"manual_cleaned", "preserve_source", "false_positive", "covered_by_duplicate"}
)
STORY_SEMANTIC_TYPES = frozenset(
    {
        "speech_bubble",
        "dialogue",
        "free_text",
        "narration",
        "caption",
        "text",
        "system",
        "system_text",
        "title",
    }
)
STORY_CLASS_NAMES = frozenset(
    {"text_comic", "text_free", "speech_bubble", "dialogue"}
)


def _norm(value) -> str:
    return str(value or "").strip().lower()


def _confidence(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def script_review_fingerprint(obj: dict) -> str:
    source = str(obj.get("ocr_text") or "")
    translation = str(obj.get("translation") or "")
    payload = f"{source}\0{translation}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def script_review_is_current(obj: dict) -> bool:
    return bool(
        obj.get("script_reviewed")
        and str(obj.get("script_review_fingerprint") or "")
        == script_review_fingerprint(obj)
    )


def apply_script_review(page: dict, object_id: str, *, reviewed: bool) -> bool:
    obj = next(
        (
            item
            for item in (page.get("text_objects") or [])
            if isinstance(item, dict) and str(item.get("id") or "") == str(object_id)
        ),
        None,
    )
    if obj is None or obj.get("source_missing"):
        raise ValueError(f"Active text object not found: {object_id}")
    if reviewed:
        fingerprint = script_review_fingerprint(obj)
        changed = bool(
            obj.get("script_reviewed") is not True
            or obj.get("script_review_fingerprint") != fingerprint
        )
        obj["script_reviewed"] = True
        obj["script_review_fingerprint"] = fingerprint
        return changed
    changed = False
    if obj.pop("script_reviewed", None) is not None:
        changed = True
    if obj.pop("script_review_fingerprint", None) is not None:
        changed = True
    return changed


def apply_final_review(page: dict, *, approved: bool) -> bool:
    if approved:
        render_revision = int(page.get("render_revision") or 0)
        if render_revision <= 0 or not page.get("rendered"):
            raise ValueError("Page must be rendered before final approval")
        changed = (
            int(page.get("final_review_approved_render_revision") or 0)
            != render_revision
        )
        page["final_review_approved_render_revision"] = render_revision
        return changed
    return page.pop("final_review_approved_render_revision", None) is not None


def is_story_candidate(box: dict | None) -> bool:
    if not isinstance(box, dict):
        return False
    if box.get("manual") or _norm(box.get("origin")) == "manual":
        return False
    if box.get("overlap_context_only"):
        return False
    if _confidence(box.get("confidence")) < EDITORIAL_STORY_CANDIDATE_CONFIDENCE:
        return False

    semantic = _norm(box.get("semantic_type"))
    class_name = _norm(box.get("class_name"))
    role = _norm(box.get("source_role"))
    model = _norm(box.get("source_model"))
    return bool(
        semantic in STORY_SEMANTIC_TYPES
        or class_name in STORY_CLASS_NAMES
        or role == "text_segmenter"
        or "text_segmenter" in model
    )


def _reviewed_disposition(box: dict, refs: list[dict]) -> str | None:
    for item in [box, *refs]:
        if not isinstance(item, dict):
            continue
        disposition = _norm(item.get("editorial_disposition"))
        reviewed = bool(item.get("editorial_reviewed") or item.get("human_reviewed"))
        if disposition in EDITORIAL_DISPOSITIONS and reviewed:
            return disposition
    return None


def _cleanup_disposition(box: dict, refs: list[dict]) -> str | None:
    for item in [box, *refs]:
        if not isinstance(item, dict):
            continue
        disposition = _norm(item.get("cleanup_disposition"))
        reviewed = bool(item.get("cleanup_reviewed") or item.get("human_reviewed"))
        if disposition in CLEANUP_DISPOSITIONS and reviewed:
            return disposition
    return None


def _object_is_story_like(obj: dict, candidate_ids: set[str]) -> bool:
    semantic = _norm(obj.get("semantic_type"))
    if semantic in STORY_SEMANTIC_TYPES:
        return True
    return bool(source_box_ids(obj) & candidate_ids)


def _overlap_fraction(a: dict, b: dict) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
        bx1, by1, bx2, by2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return 0.0
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller if smaller > 0 else 0.0


def _blocker(
    kind: str,
    page_index: int,
    *,
    box: dict | None = None,
    obj: dict | None = None,
    reason: str | None = None,
) -> dict:
    item = {"kind": kind, "page_index": int(page_index)}
    if isinstance(box, dict):
        if box.get("id"):
            item["box_id"] = str(box["id"])
        item["region"] = {
            key: box.get(key) for key in ("x1", "y1", "x2", "y2")
        }
        item["confidence"] = _confidence(box.get("confidence"))
        item["semantic_type"] = box.get("semantic_type")
        item["deferred_reason"] = box.get("deferred_reason")
    if isinstance(obj, dict) and obj.get("id"):
        item["object_id"] = str(obj["id"])
    if reason:
        item["reason"] = reason
    return item


def _append_blocker(blockers: list[dict], seen: set[tuple], item: dict) -> None:
    key = (
        item.get("kind"),
        item.get("page_index"),
        item.get("box_id"),
        item.get("object_id"),
        item.get("reason"),
    )
    if key not in seen:
        seen.add(key)
        blockers.append(item)


def apply_review_disposition(
    page: dict,
    box_id: str,
    *,
    editorial_disposition: str | None = None,
    cleanup_disposition: str | None = None,
) -> bool:
    boxes = page.get("boxes") or []
    box = next(
        (
            item
            for item in boxes
            if isinstance(item, dict) and str(item.get("id") or "") == str(box_id)
        ),
        None,
    )
    if box is None:
        raise ValueError(f"Detector box not found: {box_id}")

    refs = [
        obj
        for obj in (page.get("text_objects") or [])
        if isinstance(obj, dict) and str(box_id) in source_box_ids(obj)
    ]
    changed = False

    if editorial_disposition is not None:
        disposition = _norm(editorial_disposition)
        if disposition == "unresolved":
            for item in [box, *refs]:
                if item.pop("editorial_disposition", None) is not None:
                    changed = True
                if item.pop("editorial_reviewed", None) is not None:
                    changed = True
        elif disposition not in EDITORIAL_DISPOSITIONS:
            raise ValueError(f"Invalid editorial disposition: {editorial_disposition}")
        else:
            if box.get("editorial_disposition") != disposition:
                box["editorial_disposition"] = disposition
                changed = True
            if box.get("editorial_reviewed") is not True:
                box["editorial_reviewed"] = True
                changed = True
            box["human_reviewed"] = True

            if disposition == "promote":
                if box.get("ocr_eligible") is not True:
                    box["ocr_eligible"] = True
                    changed = True
                for key in ("human_review_drop", "human_review_origin", "story_role"):
                    if box.pop(key, None) is not None:
                        changed = True
                if box.pop("removed", None) is not None:
                    changed = True
                _, object_changed = ensure_page_text_objects(page)
                changed = object_changed or changed
                refs = [
                    obj
                    for obj in (page.get("text_objects") or [])
                    if isinstance(obj, dict) and str(box_id) in source_box_ids(obj)
                ]
                for obj in refs:
                    if obj.pop("source_missing", None) is not None:
                        changed = True
                    if obj.get("editorial_disposition") != "promote":
                        obj["editorial_disposition"] = "promote"
                        changed = True
                    obj["editorial_reviewed"] = True
                    obj["human_reviewed"] = True
                    if _norm(obj.get("reason")) == "human_review_non_story_duplicate_or_noise":
                        obj.pop("reason", None)
                        changed = True
            else:
                if box.get("ocr_eligible") is not False:
                    box["ocr_eligible"] = False
                    changed = True
                box["human_review_drop"] = True
                for obj in refs:
                    if obj.get("source_missing") is not True:
                        obj["source_missing"] = True
                        changed = True
                    if obj.get("editorial_disposition") != disposition:
                        obj["editorial_disposition"] = disposition
                        changed = True
                    obj["editorial_reviewed"] = True
                    obj["human_reviewed"] = True

                implied_cleanup = {
                    "duplicate": "covered_by_duplicate",
                    "non_story": "false_positive",
                    "noise": "false_positive",
                    "preserve": "preserve_source",
                }[disposition]
                if box.get("cleanup_disposition") != implied_cleanup:
                    box["cleanup_disposition"] = implied_cleanup
                    changed = True
                if box.pop("cleanup_review_clean_revision", None) is not None:
                    changed = True
                box["cleanup_reviewed"] = True

    if cleanup_disposition is not None:
        disposition = _norm(cleanup_disposition)
        if disposition == "unresolved":
            for item in [box, *refs]:
                if item.pop("cleanup_disposition", None) is not None:
                    changed = True
                if item.pop("cleanup_reviewed", None) is not None:
                    changed = True
                if item.pop("cleanup_review_clean_revision", None) is not None:
                    changed = True
        elif disposition not in CLEANUP_DISPOSITIONS:
            raise ValueError(f"Invalid cleanup disposition: {cleanup_disposition}")
        else:
            if box.get("cleanup_disposition") != disposition:
                box["cleanup_disposition"] = disposition
                changed = True
            if box.get("cleanup_reviewed") is not True:
                box["cleanup_reviewed"] = True
                changed = True
            if disposition == "manual_cleaned":
                clean_revision = int(page.get("clean_revision") or 0)
                if box.get("cleanup_review_clean_revision") != clean_revision:
                    box["cleanup_review_clean_revision"] = clean_revision
                    changed = True
            elif box.pop("cleanup_review_clean_revision", None) is not None:
                changed = True

    return changed


def editorial_preflight(
    manifest: dict,
    *,
    require_final_approval: bool = False,
) -> dict:
    blockers: list[dict] = []
    blocker_keys: set[tuple] = set()
    high_risk_regions: list[dict] = []
    story_candidate_count = 0
    resolved_candidate_count = 0
    script_review_required = bool(manifest.get("script_review_required"))
    final_review_required = bool(manifest.get("final_review_required"))

    for page_index, page in enumerate(manifest.get("pages") or []):
        if not isinstance(page, dict) or page.get("skipped"):
            continue

        objects = [
            obj for obj in (page.get("text_objects") or []) if isinstance(obj, dict)
        ]
        refs_by_box: dict[str, list[dict]] = {}
        for obj in objects:
            for box_id in source_box_ids(obj):
                refs_by_box.setdefault(box_id, []).append(obj)

        candidates = [
            box
            for box in (page.get("boxes") or [])
            if isinstance(box, dict) and is_story_candidate(box)
        ]
        candidate_ids = {
            str(box.get("id"))
            for box in candidates
            if box.get("id")
        }
        cleanup_resolved: dict[str, bool] = {}

        for box in candidates:
            story_candidate_count += 1
            box_id = str(box.get("id") or "")
            refs = refs_by_box.get(box_id, [])
            live_refs = [obj for obj in refs if not obj.get("source_missing")]
            in_preserve = geometry_center_in_regions(box, page_preserve_regions(page))
            disposition = _reviewed_disposition(box, refs)

            accounted = bool(
                in_preserve
                or disposition in EDITORIAL_DROP_DISPOSITIONS
                or live_refs
            )
            if accounted:
                resolved_candidate_count += 1
            else:
                if disposition == "promote":
                    kind = "promote_missing_object"
                    reason = "explicit promote has no live text object"
                elif refs and all(obj.get("source_missing") for obj in refs):
                    kind = "implicit_drop_without_disposition"
                    reason = "source_missing/planner drop is not an editorial disposition"
                else:
                    kind = "unaccounted_story_candidate"
                    reason = "detector story candidate has no live object or reviewed disposition"
                _append_blocker(
                    blockers,
                    blocker_keys,
                    _blocker(kind, page_index, box=box, reason=reason),
                )

            cleanup = _cleanup_disposition(box, refs)
            cleanup_ok = bool(
                in_preserve
                or disposition in EDITORIAL_DROP_DISPOSITIONS
                or cleanup in CLEANUP_DISPOSITIONS
            )
            if cleanup == "manual_cleaned":
                current_clean_revision = int(page.get("clean_revision") or 0)
                reviewed_clean_revision = box.get("cleanup_review_clean_revision")
                cleanup_ok = bool(
                    cleanup_ok
                    and (
                        (
                            current_clean_revision <= 0
                            and reviewed_clean_revision is None
                        )
                        or (
                            reviewed_clean_revision is not None
                            and int(reviewed_clean_revision)
                            == current_clean_revision
                        )
                    )
                )
            if box_id:
                cleanup_resolved[box_id] = cleanup_ok

            cleanup_risk = bool(
                box.get("deferred_reason")
                or (box.get("needs_review") and not box.get("safe_to_inpaint"))
            )
            if cleanup_risk:
                high_risk_regions.append(
                    {
                        "page_index": page_index,
                        "box_id": box_id or None,
                        "region": {
                            key: box.get(key) for key in ("x1", "y1", "x2", "y2")
                        },
                        "confidence": _confidence(box.get("confidence")),
                        "semantic_type": box.get("semantic_type"),
                        "deferred_reason": box.get("deferred_reason"),
                        "cleanup_resolved": cleanup_ok,
                        "cleanup_disposition": cleanup,
                    }
                )
                if not cleanup_ok:
                    _append_blocker(
                        blockers,
                        blocker_keys,
                        _blocker(
                            "unresolved_cleanup_review",
                            page_index,
                            box=box,
                            reason="review-only story text requires explicit cleanup disposition",
                        ),
                    )

        checked_objects: set[str] = set()
        for obj in objects:
            if obj.get("source_missing") or text_object_in_preserve_region(page, obj):
                continue
            oid = str(obj.get("id") or "")
            if oid and oid in checked_objects:
                continue
            if not _object_is_story_like(obj, candidate_ids):
                continue
            if oid:
                checked_objects.add(oid)

            source_text = str(obj.get("ocr_text") or "").strip()
            translation = str(obj.get("translation") or "").strip()
            disposition = _reviewed_disposition({}, [obj])
            if disposition == "preserve":
                continue
            if not source_text and not translation:
                _append_blocker(
                    blockers,
                    blocker_keys,
                    _blocker(
                        "story_object_missing_ocr",
                        page_index,
                        obj=obj,
                        reason="active story object has neither OCR source nor translation",
                    ),
                )
            elif source_text and not translation:
                _append_blocker(
                    blockers,
                    blocker_keys,
                    _blocker(
                        "untranslated_story_object",
                        page_index,
                        obj=obj,
                        reason="active story object has source text but no translation",
                    ),
                )
            elif source_text and translation:
                has_script_marker = bool(
                    obj.get("script_reviewed")
                    or obj.get("script_review_fingerprint")
                )
                if script_review_required or has_script_marker:
                    if not obj.get("script_reviewed"):
                        _append_blocker(
                            blockers,
                            blocker_keys,
                            _blocker(
                                "script_unreviewed",
                                page_index,
                                obj=obj,
                                reason="story translation has not been explicitly reviewed",
                            ),
                        )
                    elif not script_review_is_current(obj):
                        _append_blocker(
                            blockers,
                            blocker_keys,
                            _blocker(
                                "script_review_stale",
                                page_index,
                                obj=obj,
                                reason="OCR source or translation changed after script review",
                            ),
                        )

        if require_final_approval and final_review_required:
            render_revision = int(page.get("render_revision") or 0)
            approved_revision = int(
                page.get("final_review_approved_render_revision") or 0
            )
            if render_revision <= 0 or approved_revision != render_revision:
                _append_blocker(
                    blockers,
                    blocker_keys,
                    _blocker(
                        "final_review_stale",
                        page_index,
                        reason="final approval is missing or does not match the current render revision",
                    ),
                )

        for region in page.get("residue_regions") or []:
            if not isinstance(region, dict) or not is_story_candidate(region):
                continue
            match = max(
                candidates,
                key=lambda box: _overlap_fraction(region, box),
                default=None,
            )
            match_fraction = _overlap_fraction(region, match) if match else 0.0
            match_id = (
                str(match.get("id") or "")
                if match and match_fraction >= 0.25
                else ""
            )
            resolved = bool(match_id and cleanup_resolved.get(match_id))
            high_risk_regions.append(
                {
                    "page_index": page_index,
                    "box_id": match_id or None,
                    "region": {
                        key: region.get(key) for key in ("x1", "y1", "x2", "y2")
                    },
                    "confidence": _confidence(region.get("confidence")),
                    "semantic_type": region.get("semantic_type"),
                    "deferred_reason": "post_inpaint_text_residue",
                    "cleanup_resolved": resolved,
                    "cleanup_disposition": (
                        _cleanup_disposition(match, refs_by_box.get(match_id, []))
                        if match_id and match
                        else None
                    ),
                }
            )
            if not resolved:
                _append_blocker(
                    blockers,
                    blocker_keys,
                    _blocker(
                        "post_inpaint_text_residue",
                        page_index,
                        box=match or region,
                        reason="high-confidence story-like text residue remains unresolved",
                    ),
                )

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "ok": not blockers,
        "story_candidate_count": story_candidate_count,
        "resolved_story_candidate_count": resolved_candidate_count,
        "blocker_count": len(blockers),
        "untranslated_count": sum(
            blocker.get("kind") == "untranslated_story_object"
            for blocker in blockers
        ),
        "high_risk_region_count": len(high_risk_regions),
        "high_risk_regions": high_risk_regions,
        "script_review_required": script_review_required,
        "final_review_required": final_review_required,
        "blockers": blockers,
    }
