def test_font_catalog_endpoint_exposes_group_metadata():
    from app.routers.render import get_available_fonts

    fonts = get_available_fonts()
    assert fonts[0]["id"] == "default"
    assert any(item.get("category") == "sfx" for item in fonts)
