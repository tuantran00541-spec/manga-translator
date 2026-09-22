from PIL import Image, ImageDraw


def test_matcher_returns_ranked_catalog_records_for_a_crop():
    from app.render.font_matcher import match_fonts

    image = Image.new("RGB", (240, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 24), "Xin chao", fill="black")

    matches = match_fonts(image, (0, 0, 240, 96), "Xin chao", category="dialogue", top_k=3)

    assert len(matches) == 3
    assert all(0.0 <= item.score <= 1.0 for item in matches)
    assert all(item.font_id.startswith("dialogue.") for item in matches)
    assert matches[0].score >= matches[-1].score
    assert matches[0].confidence in {"low", "medium", "high"}


def test_matcher_handles_missing_crop_with_deterministic_fallback():
    from app.render.font_matcher import match_fonts

    matches = match_fonts(None, None, "fallback", category="sfx", top_k=2)

    assert [item.font_id for item in matches] == ["sfx.knewave", "sfx.luckiest-guy"]
    assert all(item.confidence == "low" for item in matches)
