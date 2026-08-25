from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_chapter_qc_controller_is_loaded_after_review_workspace():
    html = read("app/templates/index.html")
    workspace_pos = html.index('/static/js/review-workspace.js')
    chapter_qc_pos = html.index('/static/js/chapter-qc.js')
    assert workspace_pos < chapter_qc_pos


def test_chapter_qc_ui_wires_start_status_cancel_and_retry_endpoints():
    js = read("app/static/js/chapter-qc.js")
    assert 'requestJson("/api/visual_qc/chapter"' in js
    assert '`/api/visual_qc/chapter/${encodeURIComponent(jobId)}`' in js
    assert '/cancel`' in js
    assert '/retry`' in js
    assert 'chapter_id: window.currentChapterId' in js
    assert 'schedulePoll(state.snapshot.job_id, generation)' in js


def test_chapter_qc_results_jump_to_page_without_polluting_repaint_mask():
    js = read("app/static/js/chapter-qc.js")
    assert 'jump.dispatchEvent(new Event("change", { bubbles: true }))' in js
    assert 'marker.className = "review-qc-highlight"' in js
    assert 'wrap.appendChild(marker)' in js
    assert 'canvas.getContext' not in js
    assert 'canvas._reviewDirty' not in js


def test_chapter_qc_locks_mutating_review_controls_while_running():
    js = read("app/static/js/chapter-qc.js")
    required_controls = [
        ".brush-toggle-btn",
        ".clear-brush-btn",
        ".repaint-btn",
        ".reset-manual-btn",
        ".ai-qc-btn",
        ".brush-size-slider",
        ".review-primary-action",
    ]
    for selector in required_controls:
        assert selector in js
    assert 'workspace.classList.toggle("review-chapter-qc-running", locked)' in js


def test_chapter_qc_panel_has_progress_results_and_highlight_styles():
    css = read("app/static/css/review-workspace.css")
    assert ".chapter-qc-panel" in css
    assert ".chapter-qc-progress" in css
    assert ".chapter-qc-result" in css
    assert ".review-qc-highlight" in css
    assert ".review-chapter-qc-running" in css
