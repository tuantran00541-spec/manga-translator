from pathlib import Path

from app.schemas import WorkflowCheckpointRequest


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_workflow_checkpoint_accepts_script_and_final_qc_stages():
    script = WorkflowCheckpointRequest(
        chapter_id="deadbeef",
        stage="script",
        page_index=0,
    )
    final_qc = WorkflowCheckpointRequest(
        chapter_id="deadbeef",
        stage="final_qc",
        page_index=0,
    )

    assert script.stage == "script"
    assert final_qc.stage == "final_qc"


def test_shell_exposes_script_and_final_qc_as_real_stages():
    shell = _read("app/static/js/ui-shell.js")
    template = _read("app/templates/index.html")
    api = _read("app/static/js/api.js")

    assert 'const STAGES = ["landing", "preview", "review", "script", "final_qc"];' in shell
    assert 'script: "Soát bản dịch"' in shell
    assert 'final_qc: "Kiểm tra cuối"' in shell
    assert 'renderScript' in shell
    assert 'renderFinalQC' in shell
    assert 'data-stage="script"' in template
    assert 'data-stage="final_qc"' in template
    assert '/static/js/editorial-workflow.js' in template
    assert 'script: "Soát bản dịch"' in api
    assert 'final_qc: "Kiểm tra cuối"' in api
    assert 'stage === "script"' in api
    assert 'stage === "final_qc"' in api


def test_editorial_workflow_activates_current_backend_review_gates():
    path = ROOT / "app/static/js/editorial-workflow.js"
    assert path.is_file(), "editorial workflow UI module is missing"
    source = path.read_text(encoding="utf-8")

    assert "/api/review/script" in source
    assert "/api/review/final" in source
    assert "/api/export/" in source and "/preflight" in source
    assert "/api/text_object/update" in source
    assert "window.renderScript" in source
    assert "window.renderFinalQC" in source
    assert "window.flushScriptPendingSaves" in source


def test_rendered_page_refreshes_final_qc_navigation_reachability():
    shell = _read("app/static/js/ui-shell.js")
    main = _read("app/static/js/main.js")

    assert "window.refreshWorkflowNavigation" in shell
    assert "maxReachedIndex = Math.max(maxReachedIndex, inferredReachedIndex(activeStage));" in shell
    assert "window.refreshWorkflowNavigation?.();" in main
