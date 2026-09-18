from app.editorial_gate import (
    apply_final_review,
    apply_review_disposition,
    apply_script_review,
    editorial_preflight,
    is_story_candidate,
)


ROOT_BOX_ID = "box_5232e98aed3a4e1b"


def _root_box(**overrides):
    box = {
        "id": ROOT_BOX_ID,
        "origin": "detector",
        "x1": 0,
        "y1": 4193,
        "x2": 900,
        "y2": 4537,
        "confidence": 0.9258392453193665,
        "source_model": "text_segmenter.onnx",
        "source_role": "text_segmenter",
        "class_name": "text_comic",
        "semantic_type": "free_text",
        "safe_to_inpaint": False,
        "ocr_eligible": True,
        "needs_review": True,
        "deferred_reason": "box_width_limit",
    }
    box.update(overrides)
    return box


def _object(**overrides):
    obj = {
        "id": "text_5232e98aed3a4e1b",
        "source_boxes": [ROOT_BOX_ID],
        "semantic_type": "free_text",
        "region": {"x1": 0, "y1": 4193, "x2": 900, "y2": 4537},
        "ocr_text": "THE STRONGEST\nFIGHTERS FROM SIXTEEN\nWORLDS ARRIVE.",
        "translation": "NHỮNG CƯỜNG GIẢ MẠNH NHẤT\nTỪ MƯỜI SÁU THẾ GIỚI\nTỤ HỘI.",
        "source_missing": False,
    }
    obj.update(overrides)
    return obj


def _manifest(box=None, objects=None):
    return {
        "pages": [
            {
                "boxes": [box or _root_box()],
                "text_objects": list(objects or []),
                "preserve_regions": [],
                "skipped": False,
            }
        ]
    }


def _kinds(preflight):
    return {item["kind"] for item in preflight["blockers"]}


def test_real_p004_detector_candidate_is_high_risk_story_text():
    assert is_story_candidate(_root_box())


def test_real_p004_cannot_disappear_between_detector_and_final_gate():
    preflight = editorial_preflight(_manifest())

    assert preflight["status"] == "BLOCKED"
    assert preflight["story_candidate_count"] == 1
    assert "unaccounted_story_candidate" in _kinds(preflight)
    assert "unresolved_cleanup_review" in _kinds(preflight)
    assert preflight["high_risk_regions"][0]["box_id"] == ROOT_BOX_ID


def test_planner_skip_tombstone_is_not_a_human_editorial_disposition():
    box = _root_box(
        ocr_eligible=False,
        human_review_drop=True,
        human_reviewed=True,
        story_role="human_review_non_story_duplicate_or_noise",
    )
    tombstone = _object(
        ocr_text="",
        translation="",
        source_missing=True,
        origin="human_review_drop",
        human_reviewed=True,
        reason="human_review_non_story_duplicate_or_noise",
    )

    preflight = editorial_preflight(_manifest(box, [tombstone]))

    assert "implicit_drop_without_disposition" in _kinds(preflight)
    assert preflight["resolved_story_candidate_count"] == 0


def test_promoted_real_story_still_requires_cleanup_resolution():
    preflight = editorial_preflight(_manifest(objects=[_object()]))

    assert "unaccounted_story_candidate" not in _kinds(preflight)
    assert "untranslated_story_object" not in _kinds(preflight)
    assert "unresolved_cleanup_review" in _kinds(preflight)


def test_manual_cleaned_promoted_story_passes_end_to_end_preflight():
    box = _root_box(cleanup_disposition="manual_cleaned", cleanup_reviewed=True)
    preflight = editorial_preflight(_manifest(box, [_object()]))

    assert preflight["status"] == "PASS"
    assert preflight["blocker_count"] == 0
    assert preflight["resolved_story_candidate_count"] == 1


def test_explicit_non_story_disposition_is_auditable_and_resolved():
    manifest = _manifest(objects=[_object()])
    page = manifest["pages"][0]

    assert apply_review_disposition(
        page, ROOT_BOX_ID, editorial_disposition="non_story"
    )
    preflight = editorial_preflight(manifest)

    box = page["boxes"][0]
    obj = page["text_objects"][0]
    assert box["editorial_disposition"] == "non_story"
    assert box["cleanup_disposition"] == "false_positive"
    assert obj["source_missing"] is True
    assert preflight["status"] == "PASS"


def test_promote_reverses_legacy_planner_drop_without_granting_erase_authority():
    box = _root_box(
        ocr_eligible=False,
        human_review_drop=True,
        story_role="human_review_non_story_duplicate_or_noise",
    )
    tombstone = _object(source_missing=True, ocr_text="", translation="")
    manifest = _manifest(box, [tombstone])
    page = manifest["pages"][0]

    apply_review_disposition(page, ROOT_BOX_ID, editorial_disposition="promote")

    assert box["ocr_eligible"] is True
    assert box["safe_to_inpaint"] is False
    assert not box.get("human_review_drop")
    assert page["text_objects"][0].get("source_missing") is not True


def test_active_story_source_without_translation_blocks_final_export_preflight():
    box = _root_box(
        safe_to_inpaint=True,
        needs_review=False,
        deferred_reason=None,
    )
    obj = _object(translation="")

    preflight = editorial_preflight(_manifest(box, [obj]))

    assert "untranslated_story_object" in _kinds(preflight)


def test_preserve_region_is_an_explicit_resolution_surface():
    manifest = _manifest()
    manifest["pages"][0]["preserve_regions"] = [
        {"x1": 0, "y1": 4100, "x2": 900, "y2": 4540}
    ]

    preflight = editorial_preflight(manifest)

    assert preflight["status"] == "PASS"
    assert preflight["resolved_story_candidate_count"] == 1


def test_promote_repairs_missing_object_even_when_disposition_was_already_saved():
    box = _root_box(
        editorial_disposition="promote",
        editorial_reviewed=True,
        human_reviewed=True,
    )
    manifest = _manifest(box, [])
    page = manifest["pages"][0]

    changed = apply_review_disposition(
        page, ROOT_BOX_ID, editorial_disposition="promote"
    )

    assert changed is True
    assert len(page["text_objects"]) == 1
    assert page["text_objects"][0]["source_boxes"] == [ROOT_BOX_ID]


def test_manual_cleanup_resolution_expires_when_clean_revision_changes():
    box = _root_box()
    manifest = _manifest(box, [_object()])
    page = manifest["pages"][0]
    page["clean_revision"] = 3

    apply_review_disposition(
        page, ROOT_BOX_ID, cleanup_disposition="manual_cleaned"
    )
    assert editorial_preflight(manifest)["status"] == "PASS"

    page["clean_revision"] = 4
    preflight = editorial_preflight(manifest)
    assert "unresolved_cleanup_review" in _kinds(preflight)


def test_script_review_fingerprint_expires_after_translation_edit():
    box = _root_box(
        safe_to_inpaint=True,
        needs_review=False,
        deferred_reason=None,
    )
    manifest = _manifest(box, [_object()])
    manifest["script_review_required"] = True
    page = manifest["pages"][0]

    apply_script_review(
        page,
        "text_5232e98aed3a4e1b",
        reviewed=True,
    )
    assert editorial_preflight(manifest)["status"] == "PASS"

    page["text_objects"][0]["translation"] += "!"
    preflight = editorial_preflight(manifest)
    assert "script_review_stale" in _kinds(preflight)


def test_script_review_required_blocks_unreviewed_story_translation():
    box = _root_box(
        safe_to_inpaint=True,
        needs_review=False,
        deferred_reason=None,
    )
    manifest = _manifest(box, [_object()])
    manifest["script_review_required"] = True

    preflight = editorial_preflight(manifest)

    assert "script_unreviewed" in _kinds(preflight)


def test_final_review_is_bound_to_render_revision_and_only_blocks_export_stage():
    box = _root_box(
        safe_to_inpaint=True,
        needs_review=False,
        deferred_reason=None,
    )
    manifest = _manifest(box, [_object()])
    manifest["final_review_required"] = True
    page = manifest["pages"][0]
    page["rendered"] = True
    page["render_revision"] = 2

    assert editorial_preflight(manifest)["status"] == "PASS"
    assert "final_review_stale" in _kinds(
        editorial_preflight(manifest, require_final_approval=True)
    )

    apply_final_review(page, approved=True)
    assert editorial_preflight(
        manifest, require_final_approval=True
    )["status"] == "PASS"

    page["render_revision"] = 3
    preflight = editorial_preflight(
        manifest, require_final_approval=True
    )
    assert "final_review_stale" in _kinds(preflight)


def test_explicit_typography_overflow_blocks_final_preflight():
    box = _root_box(
        x1=100,
        y1=100,
        x2=170,
        y2=128,
        safe_to_inpaint=True,
        needs_review=False,
        deferred_reason=None,
    )
    obj = _object(
        region={"x1": 100, "y1": 100, "x2": 170, "y2": 128},
        translation="This is deliberately far too much text for a tiny manga balloon",
        style={
            "font": "default",
            "fontSize": 48,
            "strokeWidth": 2,
        },
    )
    manifest = _manifest(box, [obj])

    preflight = editorial_preflight(manifest)

    assert "text_overflow" in _kinds(preflight)
