import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.image_io import encode_mask
from app.render.display_lettering import render_source_sfx
from app.render.font_metrics import get_font_object, get_font_path
from app.render.identity import render_input_signature
from app.render.page_renderer import render_text_objects
from app.schemas import RenderRequest


def _sample():
    size = (460, 294)
    lettering = Image.new("L", (425, 100))
    font = get_font_object(str(get_font_path("emphasis.bungee")), 84)
    ImageDraw.Draw(lettering).text((5, 80), "SLASH", fill=255, font=font, anchor="ls")
    lettering = lettering.crop(lettering.getbbox())
    right, bottom = lettering.width - 1, lettering.height - 1
    points = np.float32([[0, 0], [right, 0], [right, bottom], [0, bottom]])
    quad = np.float32([[33, 124], [435, 11], [435, 144], [33, 219]])
    matrix = cv2.getPerspectiveTransform(points, quad)
    glyphs = cv2.warpPerspective(np.asarray(lettering), matrix, size)
    clean = Image.new("RGB", size, "white")
    raw = clean.copy()
    raw.paste((0, 0, 0), (0, 0, *size), Image.fromarray(glyphs))
    return raw, clean, glyphs


def _render(raw, clean, glyphs, reserved=None):
    return render_source_sfx(
        raw, clean, glyphs, "XOẸT", source="SLASH", font="dialogue.mac-dinh-3",
        color_profile={"fill": "#000000"},
        blocked=np.zeros(glyphs.shape, bool) if reserved is None else reserved,
    )


def test_projected_sfx_matches_slope_and_preserves_the_raster_frame():
    raw, clean, glyphs = _sample()
    result, stats = _render(raw, clean, glyphs)
    assert stats["sfx_tapered"]
    assert -21 < stats["sfx_angle"] < -8
    ink = np.any(np.asarray(result) != np.asarray(clean), axis=2)
    ys, xs = np.nonzero(ink)
    assert np.ptp(xs) > 290
    assert ys[xs < 165].mean() > ys[xs > 300].mean() + 30
    assert xs.min() > 0 and xs.max() < clean.width - 1
    assert ys.min() > 0 and ys.max() < clean.height - 1


def test_sfx_requires_clear_source_ink_and_room_to_project():
    raw, clean, glyphs = _sample()
    assert _render(raw, raw, glyphs) is None
    reserved = np.zeros(glyphs.shape, bool)
    reserved[70:160, 165:310] = True
    assert _render(raw, clean, glyphs, reserved) is None
    assert _render(raw, clean, np.zeros_like(glyphs)) is None
    assert np.all(np.asarray(clean) == 255)


def test_sfx_declines_when_an_outer_ink_band_survives_clean():
    raw, clean, glyphs = _sample()
    core = (glyphs > 0).astype(np.uint8)
    outer = cv2.dilate(core, np.ones((11, 11), np.uint8)) - cv2.dilate(core, np.ones((5, 5), np.uint8))
    original_raw = np.array(raw)
    unfinished = np.array(clean)
    original_raw[outer > 0] = unfinished[outer > 0] = 0
    assert _render(Image.fromarray(original_raw), Image.fromarray(unfinished), glyphs) is None


def test_sfx_keeps_a_source_outline_with_distinct_color():
    _, clean, glyphs = _sample()
    fill = (197, 28, 51)
    outline = (28, 36, 52)
    outer = cv2.dilate(glyphs, np.ones((7, 7), np.uint8))
    raw = clean.copy()
    raw.paste(outline, (0, 0, *raw.size), Image.fromarray(outer))
    raw.paste(fill, (0, 0, *raw.size), Image.fromarray(glyphs))
    rendered, _ = render_source_sfx(
        raw, clean, glyphs, "XOẸT", source="SLASH", font="emphasis.bungee",
        color_profile={"fill": "#c51c33", "stroke_color": "#1c2434"},
        blocked=np.zeros(glyphs.shape, bool),
    )
    pixels = np.asarray(rendered).astype(np.int16)
    for rgb in (fill, outline):
        assert np.count_nonzero(np.max(np.abs(pixels - rgb), axis=2) <= 6) > 100


@pytest.mark.parametrize("font_mode, expected", [("ai", "source_sfx"), ("user", "region_no_frame")])
def test_projected_sfx_flows_through_page_renderer_and_respects_user_font(font_mode, expected):
    raw, clean, glyphs = _sample()
    frame = dict(x1=0, y1=0, x2=raw.width, y2=raw.height)
    mask = encode_mask(glyphs)
    obj = {"id": "translated", "region": frame, "auto_geometry": frame, "auto_generated": True,
           "source_boxes": ["source"], "letters_mask": mask, "typography_role": "sfx", "container": "free",
           "translation": "XOẸT", "auto_translation": "XOẸT", "source_read": "SLASH",
           "font_selection_mode": font_mode, "style": {"font": "dialogue.mac-dinh-3"}}
    req = RenderRequest(chapter_id="chapter", page_index=0, translations={})
    verdicts = []
    assert render_text_objects(clean, req, [obj], *({} for _ in range(10)),
                               source_image_loader=lambda: raw, verdicts=verdicts,
                               source_boxes={"source": {**frame, "letters_mask": mask}}) == 1
    assert verdicts[0]["frame_source"] == expected


def test_only_sfx_render_identity_depends_on_its_source_reading(tmp_path):
    path = tmp_path / "clean.png"
    Image.new("RGB", (80, 80), "white").save(path)
    obj = {"id": "obj", "container": "free", "typography_role": "sfx", "letters_mask": "mask",
           "source_read": "SLASH", "translation": "XOẸT"}
    manifest = {"pages": [{"clean": str(path), "text_objects": [obj]}]}
    first = render_input_signature(manifest, 0)
    obj["source_read"] = "SWOOSH"
    assert render_input_signature(manifest, 0) != first
    obj["typography_role"] = "dialogue"
    first = render_input_signature(manifest, 0)
    obj["source_read"] = "OTHER"
    assert render_input_signature(manifest, 0) == first
