import json

import numpy as np
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import derive_bubble_mask, find_waist
from app.render.page_renderer import _ink_mask, _mask_bbox, render_text_objects
from app.render.text_renderer import _trim_mask_to_letters, auto_fit
from app.schemas import RenderRequest
from app.translation.vision import parse_vision_translation


def _ellipse_bubble(fill):
    image = Image.new("RGB", (700, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 650, 350], fill=fill, outline=(0, 0, 0), width=4)
    return image


def test_fixed_size_too_tall_for_its_box_falls_back_to_the_fit():
    from app.render.text_renderer import render_text_in_box

    image = Image.new("RGB", (700, 400), "white")
    long = "Anh đã nói với tôi rằng chuyện này không đơn giản như vậy đâu"
    stats: dict = {}
    render_text_in_box(image.copy(), long, (200, 60, 500, 140),
                       font_name="dialogue.mac-dinh-3", font_size=40,
                       stats=stats)
    assert stats.get("fixed_size_height_fallback") is True, \
        f"height overflow should fall back to the fit, got {stats}"
    assert stats["size"] < 40, f"fallback should letter smaller than 40px, got {stats['size']}"
    stats = {}
    render_text_in_box(image.copy(), "Chuyện này đơn giản", (200, 60, 500, 140),
                       font_name="dialogue.mac-dinh-3", font_size=24,
                       stats=stats)
    assert stats["size"] == 24, f"a fitting fixed size must stay put, got {stats['size']}"
    assert stats.get("fixed_size_height_fallback") is False


def test_unlabelled_bubble_with_a_mask_letters_on_the_mask():
    from app.render.text_renderer import render_text_in_box

    image = _ellipse_bubble("white")
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(),
                              (60, 60, 640, 340), seed=(300, 200))
    assert mask is not None and mask["kind"] == "bubble"
    stats: dict = {}
    render_text_in_box(image.copy(), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
                       (60, 60, 640, 340), font_name="dialogue.mac-dinh-3",
                       source_letter_box=(200, 150, 400, 250),
                       bubble_mask=mask, stats=stats)
    assert stats["frame_source"] == "mask", f"mask must drive the fit, got {stats}"
    assert stats["mask_source"] == "object"
    assert stats["lobe"] == "mask"

    stats = {}
    render_text_in_box(image.copy(), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
                       (60, 60, 640, 340), font_name="dialogue.mac-dinh-3",
                       source_letter_box=(200, 150, 400, 250), stats=stats)
    assert stats["frame_source"] == "region_no_frame", \
        "an unlabelled bubble with no mask has nothing to lay out on"
    assert stats["mask_source"] == "none"

    plain = Image.new("RGB", (700, 400), "white")
    stats = {}
    auto_fit(ImageDraw.Draw(plain), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
             (60, 60, 640, 340), plain.size, str(DEFAULT_FONT), source_cap_px=40,
             image=plain, container=None, centered=True,
             source_letter_box=(200, 150, 400, 250), stats_out=stats)
    assert stats["frame_source"] == "region_no_frame", \
        "no outline means no mask to drive the fit"

    boxed = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(boxed).rectangle([50, 50, 650, 350], fill="white",
                                    outline=(0, 0, 0), width=4)
    stats = {}
    auto_fit(ImageDraw.Draw(boxed), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
             (60, 60, 640, 340), boxed.size, str(DEFAULT_FONT), source_cap_px=40,
             image=boxed, container=None, centered=True,
             source_letter_box=(200, 150, 400, 250), stats_out=stats)
    assert stats["frame_source"] == "region_no_frame", \
        "a straight-sided box must not read as a bubble"


def test_a_fixed_size_that_crosses_a_spiky_outline_falls_back_to_the_fit():
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


def test_fragmented_mask_falls_back_to_a_rectangle_fit():
    image = _ellipse_bubble("white")
    draw = ImageDraw.Draw(image)
    fragmented = {"bbox": [60, 60, 640, 340], "cx": 350, "cy": 200,
                  "rows": [[(10, 11)] for _ in range(24)], "kind": "bubble"}
    stats: dict = {}
    size, _lines, fits, _box = auto_fit(
        draw, "Ta đã bảo ngươi đừng gọi ta là Các Chủ.", (60, 60, 640, 340),
        image.size, str(DEFAULT_FONT), source_cap_px=40, curved=True, image=image,
        container="bubble", centered=True, bubble_mask=fragmented, stats_out=stats)
    assert fits and size >= 24, f"fragmented mask collapsed the fit to {size}px"
    assert "mask_degenerate_rectangle" in stats["fallbacks"]


def _two_lobe_bubble():
    image = Image.new("RGB", (400, 600), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([40, 40, 360, 240], fill="white", outline=(0, 0, 0), width=3)
    draw.ellipse([50, 300, 350, 560], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([194, 235, 206, 305], fill="white")
    return image


def test_two_lobe_bubble_letters_only_the_source_lobe():
    image = _two_lobe_bubble()
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(),
                              (40, 40, 360, 470), seed=(200, 140))
    assert mask is not None and mask["kind"] == "bubble"
    waist = find_waist(mask)
    assert waist is not None and waist["orientation"] == "horizontal"
    trimmed = _trim_mask_to_letters(mask, (40, 40, 360, 470), (90, 80, 310, 200))
    assert trimmed["bbox"][3] <= waist["cut_y"] < mask["bbox"][3]
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


def test_stacked_second_block_uses_the_clipped_mask():
    image = _two_lobe_bubble()
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(),
                              (40, 40, 360, 560), seed=(200, 140))
    assert mask is not None and mask["kind"] == "bubble"
    first = {"id": "a", "translation": "Vâng, vâng. Vậy họ nói gì?",
             "region": {"x1": 40, "y1": 40, "x2": 360, "y2": 560},
             "letter_bounds": {"x1": 90, "y1": 80, "x2": 310, "y2": 200},
             "bubble_mask": mask,
             "container": "bubble", "typography_role": "dialogue", "source_cap_px": 60,
             "style": {"font": "dialogue.mac-dinh-3"}}
    long_text = ("Họ cho ngươi ở lại Thiết Kiếm Các không? Ngươi có nghe rõ không? "
                 "Ta đã bảo ngươi đừng gọi ta là Các Chủ.")
    second = {"id": "b", "translation": long_text,
              "region": {"x1": 40, "y1": 40, "x2": 360, "y2": 560},
              "letter_bounds": {"x1": 90, "y1": 80, "x2": 310, "y2": 480},
              "bubble_mask": mask,
              "container": "bubble", "typography_role": "dialogue", "source_cap_px": 60,
              "style": {"font": "dialogue.mac-dinh-3"}}
    req = RenderRequest(chapter_id="c", page_index=0,
                        translations={"a": first["translation"], "b": second["translation"]})
    alone = image.copy()
    render_text_objects(alone, RenderRequest(chapter_id="c", page_index=0,
                                             translations={"a": first["translation"]}),
                        [first], *({} for _ in range(10)))
    first_ink = _ink_mask(image, alone)
    assert first_ink is not None and first_ink.any()
    verdicts: list = []
    base = image.copy()
    render_text_objects(image, req, [first, second], *({} for _ in range(10)),
                        verdicts=verdicts)
    by_id = {v["id"]: v for v in verdicts}
    assert by_id["b"]["guard"] == "stacked", f"second block took {by_id['b']}"
    both_ink = _ink_mask(base, image)
    assert both_ink is not None
    second_ink = both_ink & ~first_ink
    ys, xs = np.nonzero(second_ink)
    assert len(xs) > 0, "second block lettered nothing"
    assert 40 <= xs.min() and xs.max() <= 360 and 40 <= ys.min() and ys.max() <= 560
    first_ys, _ = np.nonzero(first_ink)
    assert ys.min() > first_ys.max(), "stacked ink overlaps the first block"


def test_placeholder_translation_is_blank_not_lettered():
    content = json.dumps({"translations": [{"id": "a", "translated_text": "N/A"}]})
    assert parse_vision_translation(content, {"a"}) == {"a": ""}
    content = json.dumps({"translations": [{"id": "a", "translated_text": "lời thoại thật"}]})
    assert parse_vision_translation(content, {"a"}) == {"a": "lời thoại thật"}


def _ink_bands(base, image):
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