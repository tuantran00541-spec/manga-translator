"""Regression tests for the three render defect families found on real chapter art."""
import json

from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.render.page_renderer import _ink_mask, _mask_bbox, render_text_objects
from app.render.text_renderer import _bubble_bands, auto_fit
from app.schemas import RenderRequest
from app.translation.vision import parse_vision_translation


def _ellipse_bubble(fill):
    image = Image.new("RGB", (700, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 650, 350], fill=fill, outline=(0, 0, 0), width=4)
    return image


def test_fragmented_bands_do_not_collapse_to_the_readable_floor(monkeypatch):
    import app.render.text_renderer as tr
    monkeypatch.setattr(tr, "_bubble_bands", lambda *a, **k: [[(10, 11)] for _ in range(24)])
    image = _ellipse_bubble("white")
    draw = ImageDraw.Draw(image)
    size, _lines, _fits, _box = auto_fit(
        draw, "Ta đã bảo ngươi đừng gọi ta là Các Chủ.", (60, 60, 640, 340),
        image.size, str(DEFAULT_FONT), source_cap_px=40, curved=True, image=image,
        container="bubble", centered=True)
    assert size >= 24, f"fragmented bands collapsed the fit to {size}px"


def _two_lobe_bubble():
    image = Image.new("RGB", (400, 600), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([40, 40, 360, 240], fill="white", outline=(0, 0, 0), width=3)
    draw.ellipse([60, 300, 340, 470], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([194, 235, 206, 305], fill="white")
    return image


def test_two_lobe_bubble_letters_only_the_source_lobe():
    image = _two_lobe_bubble()
    verdict: dict = {}
    bands = _bubble_bands(image, (40, 40, 360, 470), verdict=verdict,
                          letter_box=(90, 80, 310, 200))
    assert bands is None and verdict.get("bands_split_waist"), \
        "the column union bridged the waist instead of reporting the split"
    base = image.copy()
    obj = {"id": "t", "translation": "Vâng, vâng. Vậy họ nói gì? Họ cho ngươi ở lại Thiết Kiếm Các không?",
           "region": {"x1": 40, "y1": 40, "x2": 360, "y2": 470},
           "letter_bounds": {"x1": 90, "y1": 80, "x2": 310, "y2": 200},
           "container": "bubble", "typography_role": "dialogue", "source_cap_px": 60,
           "style": {"font": "dialogue.mac-dinh-3"}}
    req = RenderRequest(chapter_id="c", page_index=0, translations={"t": obj["translation"]})
    render_text_objects(image, req, [obj], *({} for _ in range(10)))
    box = _mask_bbox(_ink_mask(base, image))
    assert box is not None
    assert box[3] <= 240, f"ink crossed the waist into the lower lobe: {box}"


def test_placeholder_translation_is_blank_not_lettered():
    content = json.dumps({"translations": [{"id": "a", "translated_text": "N/A"}]})
    assert parse_vision_translation(content, {"a"}) == {"a": ""}
    content = json.dumps({"translations": [{"id": "a", "translated_text": "lời thoại thật"}]})
    assert parse_vision_translation(content, {"a"}) == {"a": "lời thoại thật"}