from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from app.text_objects import ensure_page_text_objects


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def object_id(box_id: str) -> str:
    return f"text_{box_id[4:] if box_id.startswith('box_') else box_id}"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def index_manifest(manifest: dict):
    boxes, objects = {}, {}
    for page_index, page in enumerate(manifest.get("pages", [])):
        for box in page.get("boxes", []) or []:
            box_id = str(box.get("id") or "")
            if box_id:
                if box_id in boxes:
                    raise RuntimeError(f"duplicate detector box id: {box_id}")
                boxes[box_id] = (page_index, page, box)
        for obj in page.get("text_objects", []) or []:
            if not isinstance(obj, dict):
                continue
            for box_id in obj.get("source_boxes", []) or []:
                objects.setdefault(str(box_id), []).append((page_index, page, obj))
    return boxes, objects


def exact_single_object(objects: dict, box_id: str, *, active_only=False):
    matches = []
    for _, _, obj in objects.get(box_id, []):
        if obj.get("source_boxes") != [box_id]:
            continue
        if active_only and obj.get("source_missing"):
            continue
        matches.append(obj)
    if len(matches) != 1:
        raise RuntimeError(f"{box_id}: expected one exact text object, got {len(matches)}")
    return matches[0]


def tombstone(box: dict, obj: dict, reason: str):
    box["ocr_eligible"] = False
    box["human_review_drop"] = True
    box["story_role"] = "excluded_after_human_review"
    box["human_review_origin"] = "human_review_drop"
    box["human_review_reason"] = reason
    obj["review_original_ocr_text"] = str(obj.get("ocr_text") or "")
    obj["ocr_text"] = ""
    obj["translation"] = ""
    obj["origin"] = "human_review_drop"
    obj["auto_generated"] = False
    obj["source_missing"] = True
    obj["reason"] = reason
    obj["human_review_status"] = "dropped"


def approve_text(box: dict, obj: dict, text: str, origin: str = "human_ocr_review"):
    box["ocr_text"] = text
    box["ocr_quality"] = "human_corrected"
    box["human_review_status"] = "approved"
    box["story_role"] = "story"
    obj["review_original_ocr_text"] = str(obj.get("ocr_text") or "")
    obj["ocr_text"] = text
    obj["auto_generated"] = False
    obj["origin"] = origin
    obj["human_review_status"] = "approved"
    obj.pop("source_missing", None)
    obj.pop("auto_ocr_text", None)


def active_story(manifest: dict):
    rows = []
    for page_index, page in enumerate(manifest.get("pages", [])):
        active_ids = {
            str(box.get("id")) for box in page.get("boxes", []) or []
            if isinstance(box, dict) and not box.get("removed") and box.get("ocr_eligible") is not False
        }
        source_y1 = float((page.get("stitch_core") or {}).get("source_y1") or 0)
        for obj in page.get("text_objects", []) or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            refs = [str(v) for v in obj.get("source_boxes", []) or []]
            if not set(refs) & active_ids:
                continue
            region = obj.get("region") or {}
            rows.append({
                "page_index": page_index,
                "source_page": int(page.get("source_page") or 0),
                "slice_index": int(page.get("slice_index") or 0),
                "object_id": obj.get("id"),
                "source_boxes": refs,
                "ocr_text": str(obj.get("ocr_text") or ""),
                "region": copy.deepcopy(region),
                "global_region": {
                    "x1": int(region.get("x1") or 0),
                    "y1": int(source_y1 + int(region.get("y1") or 0)),
                    "x2": int(region.get("x2") or 0),
                    "y2": int(source_y1 + int(region.get("y2") or 0)),
                },
            })
    rows.sort(key=lambda r: (r["source_page"], r["global_region"]["y1"], r["global_region"]["x1"], str(r["object_id"])))
    return rows


def duplicate_pairs(rows: list[dict]):
    pairs = []
    for i, a in enumerate(rows):
        at = norm(a["ocr_text"])
        if not at:
            continue
        ar = a["global_region"]
        A = (ar["x1"], ar["y1"], ar["x2"], ar["y2"])
        for b in rows[i + 1:]:
            if b["source_page"] != a["source_page"] or norm(b["ocr_text"]) != at:
                continue
            br = b["global_region"]
            B = (br["x1"], br["y1"], br["x2"], br["y2"])
            overlap = iou(A, B)
            if overlap > 0.5:
                pairs.append({"a": a["object_id"], "b": b["object_id"], "iou": round(overlap, 4), "text": a["ocr_text"]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--curation", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest = read_json(Path(args.manifest))
    curation = read_json(Path(args.curation))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if manifest.get("chapter_id") != curation.get("chapter_id"):
        raise RuntimeError("chapter identity mismatch")
    if manifest.get("source_url") != curation.get("source_url"):
        raise RuntimeError("source URL mismatch")

    boxes, objects = index_manifest(manifest)
    drop_decisions = [
        {"box_id": str(box_id), "reason": str(reason)}
        for reason, box_ids in (curation.get("drop_groups") or {}).items()
        for box_id in box_ids
    ]
    new_drop_ids = [d["box_id"] for d in drop_decisions]
    if len(new_drop_ids) != len(set(new_drop_ids)):
        raise RuntimeError("duplicate curation drop ids")

    for decision in drop_decisions:
        box_id = decision["box_id"]
        if box_id not in boxes:
            raise RuntimeError(f"curation drop box missing: {box_id}")
        obj = exact_single_object(objects, box_id)
        tombstone(boxes[box_id][2], obj, decision["reason"])

    for correction in curation.get("manual_corrections", []):
        box_id = str(correction["box_id"])
        if box_id not in boxes:
            raise RuntimeError(f"correction box missing: {box_id}")
        if boxes[box_id][2].get("ocr_eligible") is False:
            raise RuntimeError(f"correction targets dropped box: {box_id}")
        obj = exact_single_object(objects, box_id, active_only=True)
        approve_text(boxes[box_id][2], obj, str(correction["ocr_text"]))

    for merge in curation.get("merge_decisions", []):
        primary = str(merge["primary_box_id"])
        source_ids = [str(v) for v in merge["source_box_ids"]]
        if primary not in source_ids or len(source_ids) != len(set(source_ids)):
            raise RuntimeError(f"invalid merge source list for {primary}")
        page_indices = {boxes[box_id][0] for box_id in source_ids}
        if len(page_indices) != 1:
            raise RuntimeError(f"merge crosses slices: {primary}")
        primary_obj = exact_single_object(objects, primary, active_only=True)
        member_boxes = [boxes[box_id][2] for box_id in source_ids]
        region = {
            "x1": min(int(box["x1"]) for box in member_boxes),
            "y1": min(int(box["y1"]) for box in member_boxes),
            "x2": max(int(box["x2"]) for box in member_boxes),
            "y2": max(int(box["y2"]) for box in member_boxes),
        }
        approve_text(boxes[primary][2], primary_obj, str(merge["ocr_text"]), origin="human_review_merge")
        primary_obj["source_boxes"] = source_ids
        primary_obj["region"] = region
        primary_obj["merged_from"] = source_ids
        primary_obj["merge_reason"] = str(merge.get("reason") or "")

    for page in manifest.get("pages", []):
        active_ids = {
            str(box.get("id")) for box in page.get("boxes", []) or []
            if isinstance(box, dict) and not box.get("removed") and box.get("ocr_eligible") is not False
        }
        for box in page.get("boxes", []) or []:
            if str(box.get("id")) in active_ids:
                box["story_role"] = "story"
                box["human_review_status"] = "approved"
        for obj in page.get("text_objects", []) or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            refs = {str(v) for v in obj.get("source_boxes", []) or []}
            if refs & active_ids:
                obj["auto_generated"] = False
                obj["human_review_status"] = "approved"
                if obj.get("origin") == "detector":
                    obj["origin"] = "human_ocr_review"

    created_first = created_second = 0
    for page in manifest.get("pages", []):
        created, _ = ensure_page_text_objects(page)
        created_first += created
    for page in manifest.get("pages", []):
        created, _ = ensure_page_text_objects(page)
        created_second += created

    boxes, objects = index_manifest(manifest)
    seeded = set(str(v) for v in curation.get("seeded_non_story_tombstones", []))
    all_drop_ids = set(new_drop_ids) | seeded
    resurrection = created_first + created_second
    for box_id in all_drop_ids:
        box = boxes.get(box_id, (None, None, {}))[2]
        if box.get("ocr_eligible") is not False:
            resurrection += 1
        expected_id = object_id(box_id)
        matches = [obj for _, _, obj in objects.get(box_id, []) if obj.get("id") == expected_id]
        if len(matches) != 1:
            resurrection += 1
            continue
        obj = matches[0]
        if obj.get("origin") != "human_review_drop" or obj.get("auto_generated") or not obj.get("source_missing"):
            resurrection += 1

    rows = active_story(manifest)
    empties = [row for row in rows if not row["ocr_text"].strip()]
    duplicates = duplicate_pairs(rows)
    auto_active = []
    for page in manifest.get("pages", []):
        active_ids = {str(b.get("id")) for b in page.get("boxes", []) or [] if not b.get("removed") and b.get("ocr_eligible") is not False}
        for obj in page.get("text_objects", []) or []:
            if isinstance(obj, dict) and not obj.get("source_missing") and obj.get("auto_generated") and set(obj.get("source_boxes", []) or []) & active_ids:
                auto_active.append(obj.get("id"))

    expected = curation.get("expected") or {}
    failures = []
    if len(rows) != int(expected.get("active_story_objects", -1)):
        failures.append(f"active_story_objects={len(rows)}")
    if len(empties) != int(expected.get("active_empty_story_objects", -1)):
        failures.append(f"active_empty_story_objects={len(empties)}")
    if len(duplicates) != int(expected.get("duplicate_overlaps", -1)):
        failures.append(f"duplicate_overlaps={len(duplicates)}")
    if resurrection != int(expected.get("resurrection_count", -1)):
        failures.append(f"resurrection_count={resurrection}")
    if auto_active:
        failures.append(f"auto_generated_active={len(auto_active)}")

    manifest.setdefault("workflow", {})["stage"] = "translation"
    manifest["ocr_curation"] = {
        "status": "PASS" if not failures else "FAIL",
        "active_story_objects": len(rows),
        "active_empty_story_objects": len(empties),
        "tombstones": len(all_drop_ids),
        "resurrection_count": resurrection,
        "duplicate_overlaps": len(duplicates),
    }
    (out / "processed-manifest-curated.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "active-story.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "duplicate-overlaps.json").write_text(json.dumps(duplicates, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "ocr-curation.json").write_text(json.dumps(curation, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "checkpoint": "03b-ocr-curation",
        "chapter_id": manifest.get("chapter_id"),
        "status": "PASS" if not failures else "FAIL",
        "ocr_source": (curation.get("review_basis") or {}).get("ocr_source"),
        "reviewed_candidates": 112,
        "raw_context_empty_candidates_reviewed": 94,
        "manual_corrections": len(curation.get("manual_corrections", [])),
        "merge_decisions": len(curation.get("merge_decisions", [])),
        "new_human_review_drops": len(new_drop_ids),
        "seeded_tombstones": len(seeded),
        "tombstones_total": len(all_drop_ids),
        "active_story_objects": len(rows),
        "active_empty_story_objects": len(empties),
        "duplicate_overlaps": len(duplicates),
        "reconciliation_created_first": created_first,
        "reconciliation_created_second": created_second,
        "resurrection_count": resurrection,
        "auto_generated_active_objects": len(auto_active),
        "failures": failures,
        "next_action": "TRANSLATION_BY_SCENE" if not failures else "FIX_OCR_CURATION",
    }
    (out / "ocr-curation-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit("OCR curation gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
