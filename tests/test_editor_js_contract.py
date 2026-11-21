from pathlib import Path

JS = Path("app/static/js")


def test_font_catalog_has_a_safe_default_before_the_api_resolves():
    source = (JS / "api.js").read_text(encoding="utf-8")
    assert "let availableFonts = [];" in source


def test_pending_edits_are_chapter_stamped_and_flushed_on_navigation():
    api = (JS / "api.js").read_text(encoding="utf-8")
    persistence = (JS / "editor-persistence.js").read_text(encoding="utf-8")
    geometry = (JS / "editor-geometry.js").read_text(encoding="utf-8")
    upload = (JS / "upload.js").read_text(encoding="utf-8")

    assert "chapterId: currentChapterId" in persistence
    assert "v.chapterId === chapterId" in persistence
    assert "entry?.chapterId === chapterId" in geometry
    assert "chapter_id: chapterId," in geometry
    assert "currentManifest.chapter_id" not in geometry, "no send-time chapter read"
    assert api.count("await flushEditsBeforeChapterChange();") == 2
    assert "window.flushEditsBeforeChapterChange()" in upload


def test_ai_mode_poll_discards_stale_generations():
    source = (JS / "ai-mode.js").read_text(encoding="utf-8")
    assert "let pollGeneration = 0;" in source
    assert "generation = ++pollGeneration;" in source
    assert source.count("if (generation !== pollGeneration) return;") == 2
    assert "poll(jobId, generation)" in source


def test_editor_upgrade_controls_and_shortcuts_are_wired():
    inspector = (JS / "editor-inspector.js").read_text(encoding="utf-8")
    persistence = (JS / "editor-persistence.js").read_text(encoding="utf-8")
    history = (JS / "editor-history.js").read_text(encoding="utf-8")
    editor = (JS / "editor.js").read_text(encoding="utf-8")
    stitch_index = (JS / "review-stitch" / "index.js").read_text(encoding="utf-8")
    stitch_tools = (JS / "review-stitch" / "tools.js").read_text(encoding="utf-8")
    stitch_overlays = (JS / "review-stitch" / "overlays.js").read_text(encoding="utf-8")

    # Inspector exposes box fill, corner radius, stroke auto reset, shape toggle, duplicate, gradient & glow
    for token in (
        "bg-toolbar",
        "corner-radius-slider",
        "stroke-auto-btn",
        "gradient-toolbar",
        "glow-toolbar",
        "duplicateTextObject",
        "Hình dạng vùng",
    ):
        assert token in inspector

    # Persistence & history carry shape + ocr_color_profile
    assert "ocr_color_profile" in persistence
    assert "shape: p.shape" in persistence
    assert "ocr_color_profile" in history
    assert "function duplicateTextObject" in editor

    # Canvas tools expose eyedropper, peek original, live preview, overlay toggle, arrow nudge & Ctrl+D
    assert 'eyedropper' in stitch_tools
    for token in (
        "setPeekOriginal",
        "sampleCanvasHex",
        "review-overlays-hidden",
        "review-live-preview-off",
        "ArrowLeft",
        "duplicateTextObject",
    ):
        assert token in stitch_index
    assert "review-overlay-preview" in stitch_overlays
