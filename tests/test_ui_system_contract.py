from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_app_shell_has_single_stage_system_and_global_settings():
    html = read("app/templates/index.html")
    assert 'id="landing-view"' in html
    assert 'id="page-view"' in html
    assert 'id="app-rail"' in html
    assert 'id="workflow-steps"' in html
    assert html.count('data-stage="') >= 4
    assert 'id="settings-toggle"' in html
    assert 'class="workbench-topbar"' in html
    assert 'id="stage-title"' in html
    assert 'data-stage="landing"' in html
    assert 'data-stage="preview"' in html
    assert 'data-stage="review"' in html
    assert 'data-stage="editor"' in html
    assert 'id="settings-drawer"' in html
    assert 'id="ai-settings-host"' in html
    assert '/static/css/ui-system.css' in html
    assert '/static/css/workbench.css' in html
    assert '/static/js/ui-shell.js' in html


def test_stage_shell_wraps_all_workspace_renderers():
    js = read("app/static/js/ui-shell.js")
    assert 'wrapRenderer("renderPreview", "preview")' in js
    assert 'wrapRenderer("renderReview", "review")' in js
    assert 'wrapRenderer("renderEditor", "editor")' in js
    assert 'landing.hidden = resolved !== "landing"' in js
    assert 'workspace.hidden = resolved === "landing"' in js


def test_review_workspace_removes_legacy_toolbar_and_preserves_controls_lifecycle():
    review = read("app/static/js/review.js")
    js = read("app/static/js/review-workspace.js")
    assert "legacyToolbar?.remove()" in js
    assert "window.mountAISettings(geminiConfig)" in js
    assert "window.REVIEW_VIRTUALIZED = true" in js
    assert "window.createReviewCard = createReviewCard" in review
    assert "captureMaskSnapshot(mountedCard)" in js
    assert "cleanupCard(mountedCard)" in js
    assert 'continueBtn.textContent = "Mở trình biên tập bản dịch"' in js


def test_review_ai_busy_state_locks_mutating_controls_and_navigation():
    js = read("app/static/js/review.js")
    workspace = read("app/static/js/review-workspace.js")
    assert "mutableControls = [brushBtn, clearBtn, submitBtn, resetManualBtn, aiQcBtn, brushSize]" in js
    assert 'control.disabled = true' in js
    assert 'aiQcBtn.textContent = "AI đang kiểm tra…"' in js
    assert 'navigator.setBusy(busy)' in workspace
    assert 'continueBtn.disabled = busy' in workspace


def test_user_facing_copy_uses_consistent_professional_terms():
    files = [
        "app/templates/index.html",
        "app/static/js/api.js",
        "app/static/js/upload.js",
        "app/static/js/preview.js",
        "app/static/js/review.js",
        "app/static/js/editor.js",
    ]
    text = "\n".join(read(path) for path in files)
    forbidden = [
        "Tải chapter",
        "Các Chapter đang dịch dở",
        "Tiếp tục dịch",
        "Ổn rồi, vào dịch",
        '"Tô lỗi"',
        '"Xóa nét vẽ"',
        '"AI rà lỗi"',
        '"Cỡ cọ "',
        "Select a text region on the image.",
        'label: "Hình ellipse"',
        '"Chèn chữ vào ảnh"',
        '"Tải ảnh này về"',
    ]
    for phrase in forbidden:
        assert phrase not in text, phrase

    required = [
        "Tải chương",
        "Chương đang xử lý",
        "Tiếp tục xử lý",
        "Đánh dấu vùng lỗi",
        "Kiểm tra bằng AI",
        "Kích thước cọ",
        "Mở trình biên tập bản dịch",
        "Kết xuất bản dịch",
    ]
    for phrase in required:
        assert phrase in text, phrase


def test_design_system_has_responsive_stage_and_settings_layouts():
    css = read("app/static/css/ui-system.css")
    workbench = read("app/static/css/workbench.css")
    assert ".app-rail" in workbench
    assert ".workbench-topbar" in workbench
    assert ".landing-source-grid" in css
    assert ".settings-drawer" in css
    assert ".review-sticky-toolbar" in css
    assert ".translation-sticky-toolbar" in css
    assert "@media (max-width: 820px)" in css
    assert "@media (max-width: 560px)" in css
