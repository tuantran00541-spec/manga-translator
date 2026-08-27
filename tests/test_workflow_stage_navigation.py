from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_workflow_steps_are_real_navigation_controls():
    html = read("app/templates/index.html")
    shell = read("app/static/js/ui-shell.js")
    assert html.count('data-stage="') >= 4
    assert 'id="settings-toggle"' in html
    assert 'id="app-rail"' in html
    assert 'window.navigateAppStage = navigateAppStage' in shell
    assert 'step.addEventListener("click", () => navigateAppStage(step.dataset.stage))' in shell
    assert 'targetIndex > maxReachedIndex' in shell
    assert 'step.disabled = navigationBusy || !available' in shell


def test_stage_navigation_preserves_page_and_pending_editor_changes():
    shell = read("app/static/js/ui-shell.js")
    preview = read("app/static/js/preview.js")
    api = read("app/static/js/api.js")
    assert 'await window.flushAllPendingPersists()' in shell
    assert '.review-workspace-shell.review-busy' in shell
    assert 'window.initialPreviewCanonicalPageIndex = canonicalIndex' in shell
    assert 'window.initialReviewCanonicalPageIndex = canonicalIndex' in shell
    assert 'window.editorState.activePageIndex = canonicalIndex' in shell
    assert 'window.initialPreviewCanonicalPageIndex !== undefined' in preview
    assert 'window.initialPreviewCanonicalPageIndex = pageIndex' in api
