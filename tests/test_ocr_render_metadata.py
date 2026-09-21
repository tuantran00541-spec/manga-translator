import numpy as np
from PIL import Image

from app.ocr.paddle_v6 import OCRReadResult, PaddleV6OCR
from app.ocr.service import (
    _ocr_text_region,
    _sample_source_text_color,
    _source_font_size,
)
from app.render import page_renderer
from app.schemas import RenderRequest
from app.text_objects import ensure_page_text_objects


class _VisualMetadataPipeline:
    def predict(self, *, input):
        return [{
            "rec_texts": ["FIRST", "SECOND"],
            "rec_scores": [0.98, 0.97],
            "rec_polys": [
                [[10, 8], [90, 8], [90, 24], [10, 24]],
                [[10, 32], [90, 32], [90, 50], [10, 50]],
            ],
        }]


def test_paddle_exposes_median_line_height_as_font_size_hint():
    ocr = PaddleV6OCR()
    ocr._get_pipeline = lambda _key: _VisualMetadataPipeline()

    result = ocr.read(np.full((64, 120, 3), 255, np.uint8), "en", target_mode="all")

    assert result.font_size_hint == 17.0


def test_ocr_visual_geometry_and_size_map_back_to_original_page_coordinates():
    result = OCRReadResult(
        "HELLO",
        0.99,
        "fake",
        "horizontal",
        1,
        text_bounds=(20, 10, 80, 30),
        input_shape=(40, 100),
        font_size_hint=12,
    )
    box = {"x1": 10, "y1": 20, "x2": 110, "y2": 60}
    region = _ocr_text_region((100, 160, 3), (10, 20, 110, 60), result, box)

    assert region == {"x1": 30, "y1": 30, "x2": 90, "y2": 50}
    assert _source_font_size((10, 20, 110, 60), result, region, 1) == 12


def test_verified_text_mask_samples_source_text_color_from_original_pixels():
    image = np.full((60, 80, 3), 255, np.uint8)
    mask = np.zeros((20, 40), np.uint8)
    mask[5:15, 8:32] = 255
    image[15:25, 18:42] = (0, 0, 255)
    box = {
        "x1": 10,
        "y1": 10,
        "x2": 50,
        "y2": 30,
        "source_role": "text_segmenter",
        "mask": mask,
    }

    assert _sample_source_text_color(
        image,
        box,
        {"x1": 18, "y1": 15, "x2": 42, "y2": 25},
    ) == "#ff0000"


def test_auto_text_object_carries_ocr_visual_metadata():
    page = {
        "boxes": [{
            "id": "box_1",
            "x1": 10,
            "y1": 20,
            "x2": 110,
            "y2": 80,
            "ocr_eligible": True,
            "ocr_text": "HELLO",
            "ocr_text_color": "#123456",
            "ocr_font_size": 24,
            "ocr_text_region": {"x1": 20, "y1": 28, "x2": 98, "y2": 66},
        }]
    }

    created, changed = ensure_page_text_objects(page)

    assert (created, changed) == (1, True)
    obj = page["text_objects"][0]
    assert obj["ocr_text_color"] == "#123456"
    assert obj["ocr_font_size"] == 24
    assert obj["ocr_text_region"] == {"x1": 20, "y1": 28, "x2": 98, "y2": 66}


def test_renderer_uses_ocr_visual_defaults_until_user_overrides_geometry(monkeypatch):
    calls = []

    def fake_render(image, text, box, **kwargs):
        calls.append({"text": text, "box": box, **kwargs})
        return image

    monkeypatch.setattr(page_renderer, "render_text_in_box", fake_render)

    obj = {
        "id": "text_1",
        "translation": "",
        "region": {"x1": 10, "y1": 20, "x2": 110, "y2": 80},
        "auto_geometry": {"x1": 10, "y1": 20, "x2": 110, "y2": 80},
        "auto_generated": True,
        "ocr_text_region": {"x1": 20, "y1": 28, "x2": 98, "y2": 66},
        "ocr_text_color": "#123456",
        "ocr_font_size": 24,
        "style": {
            "color": "auto",
            "font": "default",
            "fontSize": "auto",
            "bold": False,
            "strokeWidth": "auto",
            "strokeColor": "auto",
            "bgColor": "transparent",
            "cornerRadius": "0",
            "horizontalAlign": "center",
            "verticalAlign": "middle",
        },
    }
    req = RenderRequest(
        chapter_id="chapter-test",
        page_index=0,
        translations={"text_1": "Xin chao"},
    )
    image = Image.new("RGB", (160, 120), "white")

    rendered = page_renderer.render_text_objects(
        image,
        req,
        [obj],
        *({} for _ in range(10)),
    )

    assert rendered == 1
    assert calls[0]["box"] == (20, 28, 98, 66)
    assert calls[0]["fill"] == "#123456"
    assert calls[0]["font_size"] == 24

    calls.clear()
    obj["region"] = {"x1": 30, "y1": 35, "x2": 120, "y2": 90}
    page_renderer.render_text_objects(
        image,
        req,
        [obj],
        *({} for _ in range(10)),
    )
    assert calls[0]["box"] == (30, 35, 120, 90)
