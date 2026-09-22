def test_user_selection_wins_over_ai_and_auto():
    from app.render.font_selection import resolve_object_font

    obj = {"style": {"font": "dialogue.roboto"}, "font_selection_mode": "user"}
    font_id, mode, metadata = resolve_object_font(
        obj,
        source_image=None,
        region=None,
        source_text="hello",
        ai_font_id="emphasis.bangers",
    )

    assert (font_id, mode) == ("dialogue.roboto", "user")
    assert metadata["reason"] == "explicit_user_selection"


def test_valid_ai_selection_is_used_when_style_is_default():
    from app.render.font_selection import resolve_object_font

    obj = {"style": {"font": "default"}}
    font_id, mode, metadata = resolve_object_font(
        obj,
        source_image=None,
        region=None,
        source_text="hello",
        ai_font_id="emphasis.bangers",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("emphasis.bangers", "ai")
    assert metadata["reason"] == "ai_selection"


def test_unknown_ai_font_is_rejected_without_path_escape():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "default"}},
        source_image=None,
        region=None,
        source_text="hello",
        ai_font_id="../../etc/passwd",
        ai_font_mode="ai",
    )

    assert font_id == "default"
    assert mode == "default"
    assert metadata["reason"] == "ai_selection_rejected"


def test_matcher_runs_for_legacy_default_objects(monkeypatch):
    from app.render import font_selection
    from app.render.font_matcher import FontMatch

    calls = []

    def fake_matcher(image, region, source_text, *, top_k):
        calls.append((image, region, source_text, top_k))
        return [FontMatch("dialogue.be-vietnam-pro", 0.91, "high", {})]

    monkeypatch.setattr(font_selection, "match_fonts", fake_matcher)
    image = object()
    font_id, mode, metadata = font_selection.resolve_object_font(
        {"style": {"font": "default"}},
        source_image=image,
        region=(0, 0, 30, 20),
        source_text="hello",
    )

    assert calls == [(image, (0, 0, 30, 20), "hello", 3)]
    assert (font_id, mode) == ("default", "default")
    assert metadata["matches"][0]["font_id"] == "dialogue.be-vietnam-pro"


def test_unsuitable_user_selection_falls_back_to_visual_match(monkeypatch):
    from app.render import font_selection
    from app.render.font_matcher import FontMatch

    monkeypatch.setattr(
        font_selection,
        "match_fonts",
        lambda *args, **kwargs: [
            FontMatch("dialogue.be-vietnam-pro", 0.91, "high", {}),
            FontMatch("dialogue.noto-sans", 0.83, "medium", {}),
        ],
    )
    font_id, mode, metadata = font_selection.resolve_object_font(
        {"style": {"font": "dialogue.roboto"}, "font_selection_mode": "user"},
        source_image=object(),
        region=(0, 0, 30, 20),
        source_text="hello",
    )

    assert (font_id, mode) == ("dialogue.be-vietnam-pro", "auto")
    assert metadata["reason"] == "user_selection_not_suitable"
    assert metadata["requested"] == "dialogue.roboto"


def test_suitable_ai_selection_is_kept_when_matcher_agrees(monkeypatch):
    from app.render import font_selection
    from app.render.font_matcher import FontMatch

    monkeypatch.setattr(
        font_selection,
        "match_fonts",
        lambda *args, **kwargs: [
            FontMatch("emphasis.bangers", 0.91, "high", {}),
            FontMatch("dialogue.noto-sans", 0.83, "medium", {}),
        ],
    )
    font_id, mode, metadata = font_selection.resolve_object_font(
        {"style": {"font": "default"}},
        source_image=object(),
        region=(0, 0, 30, 20),
        source_text="wow",
        ai_font_id="emphasis.bangers",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("emphasis.bangers", "ai")
    assert metadata["reason"] == "ai_selection"


def test_low_confidence_auto_match_falls_back_to_default(monkeypatch):
    from app.render import font_selection
    from app.render.font_matcher import FontMatch

    monkeypatch.setattr(
        font_selection,
        "match_fonts",
        lambda *args, **kwargs: [FontMatch("dialogue.be-vietnam-pro", 0.42, "low", {})],
    )
    font_id, mode, metadata = font_selection.resolve_object_font(
        {"style": {"font": "auto"}, "font_selection_mode": "auto"},
        source_image=object(),
        region=(0, 0, 30, 20),
        source_text="hello",
    )

    assert (font_id, mode) == ("default", "default")
    assert metadata["reason"] == "match_confidence_too_low"
