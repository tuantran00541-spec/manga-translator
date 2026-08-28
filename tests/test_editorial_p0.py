from unittest.mock import patch

from app.editorial_qc import build_final_qc_report, page_editorial_issues, script_review_fingerprint
from app.editorial_layout import measure_text_layout
from app.schemas import WorkflowCheckpointRequest


def _manifest() -> dict:
    manifest = {
        "chapter_id": "deadbeef",
        "pages": [
            {
                "width": 600,
                "height": 900,
                "clean": "/tmp/clean.png",
                "rendered": True,
                "clean_revision": 2,
                "render_revision": 5,
                "needs_review": True,
                "detection_state": "needs_review",
                "detection_issues": ["unverified_regions"],
                "text_objects": [
                    {
                        "id": "text_1",
                        "region": {"x1": 100, "y1": 100, "x2": 500, "y2": 260},
                        "ocr_text": "Hello there",
                        "translation": "Xin chào",
                        "script_status": "reviewed",
                        "style": {
                            "font": "default",
                            "fontSize": "auto",
                            "strokeWidth": "auto",
                        },
                    }
                ],
            }
        ],
    }
    obj = manifest["pages"][0]["text_objects"][0]
    obj["script_review_fingerprint"] = script_review_fingerprint(obj)
    return manifest


def test_final_qc_requires_clean_ack_and_current_render_revision_approval():
    manifest = _manifest()
    with patch("app.editorial_qc.render_artifact_is_current", return_value=True):
        report = build_final_qc_report(manifest)
        assert report["ready_for_export"] is False
        assert [issue["code"] for issue in report["pages"][0]["issues"]] == ["cleanup_review"]

        page = manifest["pages"][0]
        page["clean_review_approved_revision"] = page["clean_revision"]
        page["final_qc_approved_render_revision"] = page["render_revision"]
        report = build_final_qc_report(manifest)
        assert report["ready_for_export"] is True
        assert report["summary"]["pages_approved"] == 1

        # A later render automatically makes the human approval stale.
        page["render_revision"] += 1
        report = build_final_qc_report(manifest)
        assert report["ready_for_export"] is False
        assert report["pages"][0]["approved"] is False


def test_final_qc_blocks_unreviewed_or_untranslated_script_objects():
    manifest = _manifest()
    page = manifest["pages"][0]
    page["clean_review_approved_revision"] = page["clean_revision"]
    obj = page["text_objects"][0]
    obj["script_status"] = "draft"
    obj["translation"] = ""

    with patch("app.editorial_qc.render_artifact_is_current", return_value=True):
        codes = {issue["code"] for issue in page_editorial_issues(manifest, 0)}
    assert "missing_translation" in codes
    assert "script_unreviewed" in codes

    obj["script_status"] = "skip"
    with patch("app.editorial_qc.render_artifact_is_current", return_value=True):
        codes = {issue["code"] for issue in page_editorial_issues(manifest, 0)}
    assert "missing_translation" not in codes
    assert "script_unreviewed" not in codes


def test_script_review_becomes_stale_when_text_changes():
    manifest = _manifest()
    obj = manifest["pages"][0]["text_objects"][0]
    obj["translation"] = "Xin chào lần nữa"
    manifest["pages"][0]["clean_review_approved_revision"] = 2
    with patch("app.editorial_qc.render_artifact_is_current", return_value=True):
        codes = {issue["code"] for issue in page_editorial_issues(manifest, 0)}
    assert "script_unreviewed" in codes


def test_layout_measurement_flags_explicit_text_overflow():
    result = measure_text_layout(
        "This is deliberately far too much text for a tiny manga balloon",
        (0, 0, 70, 28),
        font_size=48,
        font_name="default",
        stroke_width=2,
    )
    assert result["fits"] is False
    assert result["font_size"] == 48


def test_workflow_checkpoint_accepts_editorial_stages():
    script = WorkflowCheckpointRequest(chapter_id="deadbeef", stage="script", page_index=0)
    final_qc = WorkflowCheckpointRequest(chapter_id="deadbeef", stage="final_qc", page_index=0)
    assert script.stage == "script"
    assert final_qc.stage == "final_qc"
