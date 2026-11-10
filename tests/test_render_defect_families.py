"""Regression tests for the three render defect families found on real chapter art."""
import json

import numpy as np
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


def test_a_fixed_size_that_crosses_a_spiky_outline_falls_back_to_the_fit():
    """Rerun-4 ch79 p021: the top spiky bubble rendered at the OCR-measured
    fixed 52px, whose lines touch the outline at 0px, while the fit caps
    at 45px with real gaps. A fixed size never saw the band machinery, so
    its lines are validated against the measured (spiky-narrowed) bands
    and fall back to the fit when they do not fit; a fixed size that
    fits is left exactly alone."""
    import math

    from app.render.text_renderer import render_text_in_box

    cx, cy, outer, inner, points = 350, 200, 170, 120, 12
    star = []
    for i in range(2 * points):
        angle = math.pi * i / points - math.pi / 2
        radius = outer if i % 2 == 0 else inner
        star.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    image = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(image).polygon(star, fill="white", outline=(0, 0, 0), width=4)
    long = "HẮN ĐÃ CHIẾN ĐẤU VỚI MỘT CON NGƯỜI ĐIÊN CUỒNG, VƯỢT QUA VÔ SỐ CẠM BẪY"
    stats: dict = {}
    render_text_in_box(image.copy(), long, (200, 60, 500, 340),
                       font_name="dialogue.mac-dinh-3", font_size=60,
                       container="spiky", stats=stats)
    assert stats["size"] == 30, f"fixed 60px should fall back to the fit, got {stats['size']}"
    stats = {}
    render_text_in_box(image.copy(), "HẮN ĐÃ CHIẾN ĐẤU", (200, 60, 500, 340),
                       font_name="dialogue.mac-dinh-3", font_size=24,
                       container="spiky", stats=stats)
    assert stats["size"] == 24, f"a fitting fixed size must stay put, got {stats['size']}"


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


def _ink_bands(base, image):
    # Diacritics sit a pixel or two above the cap line, so one line of
    # Vietnamese reads as two raw bands; merge gaps under a line pitch.
    diff = (np.abs(np.asarray(image).astype(int) - np.asarray(base).astype(int)).max(axis=2) > 0)
    rows = diff.sum(axis=1)
    raw, in_band = [], False
    for y, count in enumerate(rows):
        if count > 3 and not in_band:
            start, in_band = y, True
        elif count <= 3 and in_band:
            raw.append((start, y - 1)); in_band = False
    bands = []
    for band in raw:
        if bands and band[0] - bands[-1][1] <= 10:
            bands[-1] = (bands[-1][0], band[1])
        else:
            bands.append(band)
    return bands


def _render_free_text(image, text):
    obj = {"id": "t", "translation": text,
           "region": {"x1": 50, "y1": 100, "x2": 350, "y2": 260},
           "role": "free_text", "container": "free", "font_selection_mode": "auto"}
    req = RenderRequest(chapter_id="c", page_index=0, translations={"t": text})
    render_text_objects(image, req, [obj], {}, {"t": "default"}, *({} for _ in range(8)))
    return image


def test_doubled_translation_letters_once_not_stacked():
    image = Image.new("RGB", (400, 400), "white")
    base = image.copy()
    _render_free_text(image, "NGUYÊN TỐ\nNGUYÊN TỐ")
    assert len(_ink_bands(base, image)) == 1, "doubled translation lettered as two lines"


def test_single_line_and_differing_lines_stand():
    image = Image.new("RGB", (400, 400), "white")
    base = image.copy()
    _render_free_text(image, "NGUYÊN TỐ")
    assert len(_ink_bands(base, image)) == 1
    image = Image.new("RGB", (400, 400), "white")
    base = image.copy()
    _render_free_text(image, "NGUYÊN TỐ\nKHÁC")
    assert len(_ink_bands(base, image)) == 2, "differing lines must still letter as two lines"