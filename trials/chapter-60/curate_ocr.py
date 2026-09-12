from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from app.text_objects import DEFAULT_TEXT_OBJECT_STYLE, ensure_page_text_objects

CHAPTER_ID = "f1cd0121"

# Two human-reviewed skipped proposals that are real English story vocalizations.
PROMOTE = {
    (36, "box_a5ec1161458544ab"): "SQUEE!",
    (38, "box_a3c90509c54642f3"): "SQUEEE!",
}

# Explicit machine-OCR corrections confirmed against RAW proof.
CORRECTIONS = {
    (56, "box_d900bc50b4374bd9"): "I RECEIVED THE RECORDING ORB YOU SENT.",
    (56, "box_5043cba0cd1a4a5c"): "THAT WAS UNMISTAKABLY ANDROMALIUS. THE 72ND-RANKED DEMON.",
    (57, "box_cfb3d1372e1f4966"): "AND SEEING SAINT ARMIAN AND LUINA BERCHEFF STAND AGAINST HIM..",
    (57, "box_8571f5814b764f0c"): "SIR AMOD ALSO TOLD ME..",
    (60, "box_a684dac5f29045eb"): "I HAVE A FAIR IDEA OF WHAT YOU CONCEALED, SO I WON'T ASK.",
    (61, "box_22748b6c37534964"): "AS SOON AS THE BREAK BEGAN, YOU CAME HOME LIKE A MAN BEING PURSUED AND IMMEDIATELY DEVOTED ALL YOUR EFFORTS TO BUILDING A GOLEM.",
}

# Human-reviewed active boxes that must never enter story translation.
DROP_IDS = {
    # p000 title / scanlation / decorative card
    (0, "box_4d9b409ec3de4429"),
    (0, "box_ede6b33a00f94745"),
    (0, "box_b21c08e1d96d498a"),
    (0, "box_5bb018a7360241a7"),
    # p020 production credits, duplicate detector boxes
    (20, "box_cc01706c6ea64712"),
    (20, "box_e9e84335b5d540c2"),
    # confirmed duplicate owner decisions
    (21, "box_968485f59cb14b5d"),
    (49, "box_39e8ec1471384216"),
    # p052 both source fragments replaced by one curated combined object
    (52, "box_36300e8061d742e0"),
    (52, "box_771a3d7577834c28"),
    # reaction/art/noise false positives
    (43, "box_b6bd91a26f6f4533"),
    (44, "box_cdb593dcfed34114"),
    (44, "box_c14efb2c87fa442e"),
    (53, "box_2a55ae1b428242ca"),
    # p063 duplicate story detector + preserved production-credit material
    (63, "box_39c2a2f793a24eea"),
    (63, "box_43ec017503714dd9"),
    (63, "box_25b772a92acf4584"),
    (63, "box_4ecb09a9dbb24208"),
    (63, "box_26ff7288259847c2"),
    # end-card promo
    (64, "box_1a1f8c4c2daf4c77"),
}

# Source boxes replaced with curated manual objects because one detector region
# mixed distinct semantic/typographic roles.
REPLACE_WITH_MANUAL = {
    (31, "box_21b9d9a748934a8b"),
    (37, "box_f7382b1a76f24c0c"),
}

WATERMARK_PATTERNS = (
    "ASURASCANS.COM",
    "SURASCANS.COM",
    "ASURACOMIC.NET",
    "DISCORD.GG/ASURA",
    "DISCORD.GG/ASURAN",
)


def region(box: dict) -> dict:
    return {k: int(box[k]) for k in ("x1", "y1", "x2", "y2")}


def find_box(manifest: dict, page_index: int, box_id: str) -> dict:
    page = manifest["pages"][page_index]
    for box in page.get("boxes", []) or []:
        if str(box.get("id")) == box_id:
            return box
    raise KeyError(f"missing p{page_index:03d} {box_id}")


def drop_box(box: dict, reason: str) -> None:
    box["removed"] = True
    box["ocr_eligible"] = False
    box["needs_review"] = False
    box["human_review_state"] = "dropped"
    box["human_review_reason"] = reason


def human_ocr(box: dict, text: str, reason: str) -> None:
    box["removed"] = False
    box["ocr_eligible"] = True
    box["needs_review"] = False
    box["ocr_text"] = text
    box["ocr_source"] = "human_review"
    box["ocr_model"] = "human"
    box["ocr_quality"] = "good"
    box["ocr_quality_reason"] = None
    box["ocr_confidence"] = 1.0
    box["human_review_state"] = "approved"
    box["human_review_reason"] = reason


def tombstone(box: dict, page_index: int, reason: str) -> dict:
    bid = str(box.get("id") or "")
    return {
        "id": f"drop_{bid[4:] if bid.startswith('box_') else bid}",
        "shape": "rectangle",
        "region": region(box),
        "source_boxes": [bid],
        "ocr_text": str(box.get("ocr_text") or ""),
        "translation": "",
        "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
        "origin": "human_review_drop",
        "auto_generated": False,
        "source_missing": True,
        "human_review_state": "dropped",
        "human_review_reason": reason,
        "page_index": page_index,
    }


def manual_object(object_id: str, reg: dict, source_boxes: list[str], text: str,
                  semantic_role: str, reason: str) -> dict:
    return {
        "id": object_id,
        "shape": "rectangle",
        "region": copy.deepcopy(reg),
        "source_boxes": list(source_boxes),
        "ocr_text": text,
        "translation": "",
        "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
        "origin": "human_review_curated",
        "auto_generated": False,
        "ocr_source": "human_review",
        "ocr_model": "human",
        "ocr_quality": "good",
        "ocr_confidence": 1.0,
        "human_review_state": "approved",
        "human_review_reason": reason,
        "semantic_role": semantic_role,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    mp = root / "processed" / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit(f"chapter mismatch: {manifest.get('chapter_id')}")
    skipped = json.loads((root / "ocr-review" / "ocr-skipped.json").read_text(encoding="utf-8"))

    # No object lifecycle should have begun before deterministic human curation.
    pre_objects = sum(len(p.get("text_objects") or []) for p in manifest.get("pages", []))
    if pre_objects != 0:
        raise SystemExit(f"expected zero pre-curation text_objects, got {pre_objects}")

    dropped: dict[tuple[int, str], str] = {}
    promoted = []
    corrected = []

    # Resolve every scheduler-skipped proposal. Human review found exactly two
    # real English story vocalizations among the 107 non-deferred proposals;
    # deferred review regions are scaffolding and are also made non-resurrectable.
    for row in skipped:
        pi = int(row["page_index"])
        bid = str(row["box_id"])
        box = find_box(manifest, pi, bid)
        key = (pi, bid)
        if key in PROMOTE:
            human_ocr(box, PROMOTE[key], "human-reviewed skipped proposal promoted to story vocalization")
            promoted.append({"page_index": pi, "box_id": bid, "text": PROMOTE[key]})
        else:
            reason = f"human-reviewed skipped proposal resolved: {row.get('skip_reason')}"
            drop_box(box, reason)
            dropped[key] = reason

    # Explicit active non-story / duplicate decisions.
    for pi, bid in sorted(DROP_IDS):
        box = find_box(manifest, pi, bid)
        reason = "human review: non-story, noise, overlap duplicate, or superseded detector region"
        drop_box(box, reason)
        dropped[(pi, bid)] = reason

    # Remove any machine-recognized scanlation watermark missed by explicit IDs.
    for pi, page in enumerate(manifest.get("pages", [])):
        for box in page.get("boxes", []) or []:
            if box.get("removed"):
                continue
            txt = str(box.get("ocr_text") or "").upper().replace(" ", "")
            if any(p.replace(" ", "") in txt for p in WATERMARK_PATTERNS):
                bid = str(box.get("id") or "")
                reason = "human review: preserved scanlation watermark/promo, not story text"
                drop_box(box, reason)
                dropped[(pi, bid)] = reason

    # Machine OCR corrections confirmed against RAW.
    for (pi, bid), text in CORRECTIONS.items():
        box = find_box(manifest, pi, bid)
        human_ocr(box, text, "manual OCR correction confirmed against RAW")
        corrected.append({"page_index": pi, "box_id": bid, "text": text})

    # Replace mixed-role source regions before automatic text-object creation.
    for pi, bid in sorted(REPLACE_WITH_MANUAL):
        box = find_box(manifest, pi, bid)
        reason = "human review: detector region mixes distinct semantic/typographic roles"
        drop_box(box, reason)
        dropped[(pi, bid)] = reason

    # p052: two detector fragments repeat the same bookstore text. One curated
    # object spans the union so the translation renders once.
    p52_a = find_box(manifest, 52, "box_36300e8061d742e0")
    p52_b = find_box(manifest, 52, "box_771a3d7577834c28")
    p52_obj = manual_object(
        "human_p052_moonrise_bookstore",
        {"x1": 475, "y1": 141, "x2": 847, "y2": 275},
        [str(p52_a["id"]), str(p52_b["id"])],
        "MOONRISE BOOKSTORE?",
        "speech_bubble",
        "merged duplicate detector fragments after human scene review",
    )

    # p031: large uppercase dialogue and smaller explanatory text are visually
    # distinct. Conservative split follows the observed typographic break inside
    # the reviewed detector box; no artwork geometry is changed.
    p31_src = find_box(manifest, 31, "box_21b9d9a748934a8b")
    p31_main = manual_object(
        "human_p031_chancellor_dialogue",
        {"x1": 257, "y1": 2330, "x2": 900, "y2": 2548},
        [str(p31_src["id"])],
        "THE CHANCELLOR CAN'T REVEAL THE DEMON'S APPEARANCE YET, AND NEITHER CAN I.",
        "speech_bubble",
        "split mixed-size OCR region: primary dialogue",
    )
    p31_small = manual_object(
        "human_p031_nightmare_note",
        {"x1": 257, "y1": 2548, "x2": 900, "y2": 2764},
        [str(p31_src["id"])],
        "So what we encountered has become a nightmare rather than a demon.",
        "secondary_text",
        "split mixed-size OCR region: smaller explanatory text; OCR spacing corrected",
    )

    # p037: normal dialogue and SQUEAK vocalization must be independently
    # typeset. The subregions are conservative within the human-reviewed source
    # detector box and intentionally overlap slightly at the semantic boundary.
    p37_src = find_box(manifest, 37, "box_f7382b1a76f24c0c")
    p37_dialogue = manual_object(
        "human_p037_dark_horses",
        {"x1": 414, "y1": 381, "x2": 900, "y2": 625},
        [str(p37_src["id"])],
        "AND ABOUT TWO DARK HORSES...",
        "speech_bubble",
        "split combined dialogue/SFX OCR region",
    )
    p37_sfx = manual_object(
        "human_p037_squeak",
        {"x1": 414, "y1": 600, "x2": 900, "y2": 746},
        [str(p37_src["id"])],
        "SQUEAK!",
        "vocalization_sfx",
        "split combined dialogue/SFX OCR region",
    )

    # Materialize detector-owned active story objects after all source-box
    # eligibility decisions. Dropped boxes cannot create automatic objects.
    created = 0
    for page in manifest.get("pages", []):
        n, _ = ensure_page_text_objects(page)
        created += n

    # Add curated manual objects and durable tombstones.
    manifest["pages"][52].setdefault("text_objects", []).append(p52_obj)
    manifest["pages"][31].setdefault("text_objects", []).extend([p31_main, p31_small])
    manifest["pages"][37].setdefault("text_objects", []).extend([p37_dialogue, p37_sfx])

    for (pi, bid), reason in sorted(dropped.items()):
        box = find_box(manifest, pi, bid)
        manifest["pages"][pi].setdefault("text_objects", []).append(tombstone(box, pi, reason))

    # Cross-source-image sentence continuation. Keep separate render geometry,
    # but make the scene relationship durable for translation/editorial review.
    continuation = "cont_p030_p031_just_tell_it"
    for pi, bid, part in (
        (30, "box_15aed6043c954767", 1),
        (31, "box_355a8dc43c8b40ee", 2),
    ):
        for obj in manifest["pages"][pi].get("text_objects", []):
            if bid in (obj.get("source_boxes") or []) and not obj.get("source_missing"):
                obj["continuation_group"] = continuation
                obj["continuation_part"] = part
                obj["continuation_total"] = 2
                obj["continuation_full_source"] = "...JUST TELL IT AS IT HAPPENED."

    # Resurrection test: running object sync again must not recreate an active
    # detector-owned object from any human-dropped source box.
    second_created = 0
    for page in manifest.get("pages", []):
        n, _ = ensure_page_text_objects(page)
        second_created += n

    dropped_ids = {bid for _, bid in dropped}
    resurrected = []
    active = []
    empty_active = []
    for pi, page in enumerate(manifest.get("pages", [])):
        for obj in page.get("text_objects", []) or []:
            refs = {str(x) for x in (obj.get("source_boxes") or [])}
            if obj.get("source_missing"):
                continue
            active.append((pi, obj))
            if obj.get("auto_generated") and refs & dropped_ids:
                resurrected.append({"page_index": pi, "object_id": obj.get("id"), "source_boxes": sorted(refs & dropped_ids)})
            if not str(obj.get("ocr_text") or "").strip():
                empty_active.append({"page_index": pi, "object_id": obj.get("id")})

    # Guard key ownership decisions.
    active_refs = {(pi, ref) for pi, obj in active for ref in (obj.get("source_boxes") or [])}
    for key in DROP_IDS | REPLACE_WITH_MANUAL:
        if key in active_refs:
            # Manual curated replacements may intentionally retain provenance to
            # a superseded source box. Only detector-auto ownership is forbidden.
            pass

    if len(promoted) != 2:
        raise SystemExit(f"expected 2 promoted skipped proposals, got {promoted}")
    if resurrected:
        raise SystemExit(f"resurrected dropped objects: {resurrected[:10]}")
    if empty_active:
        raise SystemExit(f"active objects missing OCR: {empty_active[:10]}")
    if second_created != 0:
        raise SystemExit(f"object sync was not idempotent; second_created={second_created}")

    # Ensure p021/p049/p063 duplicate owner is singular among automatic active objects.
    duplicate_owner_checks = {
        21: {"box_f513c6697c844fd2"},
        49: {"box_d809d9a845424212"},
        63: {"box_9faef21adfdf4b20"},
    }
    for pi, expected in duplicate_owner_checks.items():
        refs = set()
        for obj in manifest["pages"][pi].get("text_objects", []) or []:
            if obj.get("source_missing"):
                continue
            refs.update(obj.get("source_boxes") or [])
        if not expected.issubset(refs):
            raise SystemExit(f"missing expected duplicate owner p{pi}: {expected} vs {refs}")

    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    active_summary = []
    for pi, obj in active:
        active_summary.append({
            "page_index": pi,
            "object_id": obj.get("id"),
            "source_boxes": obj.get("source_boxes"),
            "semantic_role": obj.get("semantic_role"),
            "ocr_text": obj.get("ocr_text"),
            "origin": obj.get("origin"),
        })
    report = {
        "checkpoint": "04b-ocr-curated",
        "chapter_id": CHAPTER_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": "03d-ocr-clean-f1cd0121",
        "raw_changed": False,
        "ocr_rerun_required": False,
        "skipped_rows_reviewed": len(skipped),
        "promoted_skipped_story_objects": promoted,
        "manual_corrections": corrected,
        "dropped_source_boxes": len(dropped),
        "automatic_objects_created": created,
        "manual_active_objects": 5,
        "active_objects": len(active),
        "tombstones": len(dropped),
        "resurrected_dropped_count": len(resurrected),
        "second_sync_created": second_created,
        "empty_active_ocr_count": len(empty_active),
        "continuation_groups": [continuation],
        "status": "PASS" if not resurrected and not empty_active and second_created == 0 else "FAIL",
        "next_action": "STOP_FOR_HUMAN_CURATED_SCENE_REVIEW",
    }
    (root / "ocr-curation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "curated-scene-index.json").write_text(json.dumps(active_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "resurrection-test.json").write_text(json.dumps({
        "dropped_source_boxes": len(dropped),
        "resurrected": resurrected,
        "resurrected_dropped_count": len(resurrected),
        "second_sync_created": second_created,
        "empty_active_ocr": empty_active,
        "pass": not resurrected and second_created == 0 and not empty_active,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
