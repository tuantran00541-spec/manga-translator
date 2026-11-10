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