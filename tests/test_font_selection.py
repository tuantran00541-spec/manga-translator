def test_user_selection_wins_over_ai_and_auto():
    from app.render.font_selection import resolve_object_font

    obj = {"style": {"font": "dialogue.roboto"}, "font_selection_mode": "user"}
    font_id, mode, metadata = resolve_object_font(
        obj,
        ai_font_id="emphasis.bangers",
    )

    assert (font_id, mode) == ("dialogue.roboto", "user")
    assert metadata["reason"] == "explicit_user_selection"


def test_valid_ai_selection_is_used_when_style_is_default():
    from app.render.font_selection import resolve_object_font

    obj = {"style": {"font": "default"}}
    font_id, mode, metadata = resolve_object_font(
        obj,
        ai_font_id="emphasis.bangers",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("emphasis.bangers", "ai")
    assert metadata["reason"] == "ai_selection"


def test_unknown_ai_font_is_rejected_without_path_escape():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "default"}},
        ai_font_id="../../etc/passwd",
        ai_font_mode="ai",
    )

    assert font_id == "default"
    assert mode == "default"
    assert metadata["reason"] == "ai_selection_rejected"


def test_auto_objects_render_with_the_default_font():
    from app.render.font_selection import resolve_object_font

    for obj in (
        {"style": {"font": "auto"}, "font_selection_mode": "auto"},
        {"style": {"font": "default"}, "auto_generated": True},
    ):
        font_id, mode, metadata = resolve_object_font(obj)
        assert (font_id, mode) == ("default", "default")
        assert metadata["reason"] == "auto_default"


def test_ai_may_delegate_to_auto():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "default"}},
        ai_font_id="auto",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("default", "default")
    assert metadata["reason"] == "auto_default"


def test_valid_ai_selection_is_used_for_auto_generated_objects():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "default"}, "font_selection_mode": "ai", "auto_generated": True},
        ai_font_id="emphasis.bangers",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("emphasis.bangers", "ai")
    assert metadata["reason"] == "ai_selection"


def test_unknown_user_font_falls_back_to_default():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "missing.font"}, "font_selection_mode": "user"},
    )

    assert (font_id, mode) == ("default", "default")
    assert metadata["reason"] == "user_selection_rejected"
