import pytest


def test_font_match_request_limits_top_k_and_requires_object():
    from app.schemas import FontMatchRequest

    request = FontMatchRequest(chapter_id="chapter-1", page_index=0, object_id="text_1", top_k=5)
    assert request.top_k == 5
    with pytest.raises(ValueError):
        FontMatchRequest(chapter_id="chapter-1", page_index=0, object_id="text_1", top_k=6)


def test_font_catalog_endpoint_exposes_group_metadata():
    from app.routers.render import get_available_fonts

    fonts = get_available_fonts()
    assert fonts[0]["id"] == "default"
    assert any(item.get("category") == "sfx" for item in fonts)
