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
        assert (font_id, mode) == ("dialogue.mac-dinh-3", "auto")
        assert metadata["reason"] == "auto_guide"


def test_ai_may_delegate_to_auto():
    from app.render.font_selection import resolve_object_font

    font_id, mode, metadata = resolve_object_font(
        {"style": {"font": "default"}},
        ai_font_id="auto",
        ai_font_mode="ai",
    )

    assert (font_id, mode) == ("dialogue.mac-dinh-3", "auto")
    assert metadata["reason"] == "auto_guide"


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


def test_a_wide_framed_title_uses_source_proportions_when_font_is_auto():
    import base64

    import cv2
    import numpy as np

    from app.render.font_selection import resolve_object_font

    mask = np.zeros((48, 160), np.uint8)
    for start in range(8, 143, 23):
        mask[11:32, start:start + 17] = 255
    _, png = cv2.imencode(".png", mask)
    obj = {
        "style": {"font": "auto"},
        "font_selection_mode": "auto",
        "verified_balloon": True,
        "typography_role": "title",
        "container": "box",
        "source_cap_px": 21,
        "letters_mask": base64.b64encode(png).decode(),
    }
    font, mode, metadata = resolve_object_font(obj)
    assert (font, mode) == ("dialogue.be-vietnam-pro", "auto")
    assert metadata["reason"] == "source_proportions"
