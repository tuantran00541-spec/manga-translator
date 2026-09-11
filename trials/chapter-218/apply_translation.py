#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated-root", required=True)
    ap.add_argument("--translation-json", required=True)
    ap.add_argument("--review-json", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    curated_root = Path(args.curated_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = curated_root / "processed-manifest-curated.json"
    curation_summary_path = curated_root / "ocr-curation-summary.json"
    manifest = load_json(manifest_path)
    curation_summary = load_json(curation_summary_path)
    translation_doc = load_json(Path(args.translation_json))
    review = load_json(Path(args.review_json))

    failures = []
    if curation_summary.get("status") != "PASS":
        failures.append(f"curation status={curation_summary.get('status')!r}")
    if curation_summary.get("active_story_objects") != 98:
        failures.append(f"expected 98 curated active story objects, got {curation_summary.get('active_story_objects')}")
    if curation_summary.get("active_empty_story_objects") != 0:
        failures.append("curation has active empty story objects")
    if curation_summary.get("duplicate_overlaps") != 0:
        failures.append("curation has duplicate overlaps")
    if curation_summary.get("resurrection_count") != 0:
        failures.append("curation resurrection count is nonzero")

    if translation_doc.get("chapter_id") != manifest.get("chapter_id"):
        failures.append("translation chapter_id does not match manifest")
    if translation_doc.get("lang") != "vi":
        failures.append("translation lang is not vi")
    if translation_doc.get("editorial_status") != "REVIEWED":
        failures.append("translation editorial_status is not REVIEWED")
    if review.get("status") != "PASS":
        failures.append("translation human review is not PASS")
    if not review.get("scene_reviewed") or not review.get("source_order_reviewed"):
        failures.append("translation was not reviewed by scene/source order")

    translations = translation_doc.get("translations") or {}
    active = []
    tombstones = []
    for page in manifest.get("pages") or []:
        for obj in page.get("text_objects") or []:
            if obj.get("source_missing"):
                tombstones.append((page, obj))
            else:
                active.append((page, obj))

    active_ids = {obj["id"] for _, obj in active}
    translation_ids = set(translations)
    missing_ids = sorted(active_ids - translation_ids)
    unknown_ids = sorted(translation_ids - active_ids)
    if len(active) != 98:
        failures.append(f"manifest active story object count={len(active)}, expected 98")
    if missing_ids:
        failures.append(f"missing translation ids: {missing_ids}")
    if unknown_ids:
        failures.append(f"translation ids not active in curated manifest: {unknown_ids}")

    translated_count = 0
    untranslated_ids = []
    unchanged_ids = []
    for page, obj in active:
        oid = obj["id"]
        text = translations.get(oid, "")
        if not isinstance(text, str) or not text.strip():
            untranslated_ids.append(oid)
            continue
        if text.strip() == str(obj.get("ocr_text") or "").strip():
            unchanged_ids.append(oid)
        obj["translation"] = text
        obj["translation_origin"] = "human_scene_review"
        obj["translation_review_status"] = "approved"
        translated_count += 1

    tombstone_translation_ids = []
    for _, obj in tombstones:
        if str(obj.get("translation") or "").strip():
            tombstone_translation_ids.append(obj["id"])
        obj["translation"] = ""
        obj.pop("translation_origin", None)
        obj.pop("translation_review_status", None)

    if untranslated_ids:
        failures.append(f"active untranslated story objects: {untranslated_ids}")
    if tombstone_translation_ids:
        failures.append(f"tombstones unexpectedly carried translations: {tombstone_translation_ids}")

    manifest["translation_review"] = {
        "status": "PASS" if not failures else "FAIL",
        "lang": "vi",
        "source_checkpoint": "03b-ocr-curation",
        "active_story_objects": len(active),
        "translated_story_objects": translated_count,
        "active_untranslated_story_objects": len(untranslated_ids),
        "translation_origin": "human_scene_review",
        "scene_reviewed": bool(review.get("scene_reviewed")),
        "terminology_consistent": bool(review.get("terminology_consistent")),
        "character_voice_reviewed": bool(review.get("character_voice_reviewed")),
    }

    summary = {
        "checkpoint": "04-translation-review",
        "chapter_id": manifest.get("chapter_id"),
        "status": "PASS" if not failures else "FAIL",
        "language": "vi",
        "source_checkpoint": "03b-ocr-curation",
        "active_story_objects": len(active),
        "translated_story_objects": translated_count,
        "active_untranslated_story_objects": len(untranslated_ids),
        "tombstones": len(tombstones),
        "tombstone_translations": len(tombstone_translation_ids),
        "missing_translation_ids": missing_ids,
        "unknown_translation_ids": unknown_ids,
        "intentional_source_equal_translations": unchanged_ids,
        "scene_reviewed": bool(review.get("scene_reviewed")),
        "source_order_reviewed": bool(review.get("source_order_reviewed")),
        "raw_context_consulted": bool(review.get("raw_context_consulted")),
        "terminology_consistent": bool(review.get("terminology_consistent")),
        "character_voice_reviewed": bool(review.get("character_voice_reviewed")),
        "failures": failures,
        "next_action": "TYPESET_PREFLIGHT_AND_TYPOGRAPHY_PLAN" if not failures else "FIX_TRANSLATION_REVIEW",
    }

    dump_json(out / "processed-manifest-translated.json", manifest)
    dump_json(out / "translation-review-summary.json", summary)
    dump_json(out / "translation-vi.json", translation_doc)
    dump_json(out / "translation-review.json", review)

    with (out / "translation-table.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source_page", "slice_index", "object_id", "ocr_text", "translation_vi"])
        for page, obj in active:
            writer.writerow([
                page.get("source_page"),
                page.get("slice_index"),
                obj["id"],
                obj.get("ocr_text", ""),
                obj.get("translation", ""),
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
