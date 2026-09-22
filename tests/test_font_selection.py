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
