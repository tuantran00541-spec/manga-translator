"""Visual-regression harness: render the known defect families through the real
render path and assert the visual invariants (no empty bubble, ink stays inside
its bubble, legible floor). Emits PNGs to artifacts/visual-regression for diffing.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.detector.bubble_mask import derive_bubble_mask, find_waist
from app.parameters import RENDER_MIN_READABLE_FONT_SIZE
from app.render.page_renderer import _ink_mask, _mask_bbox, render_text_objects
from app.render.text_renderer import render_text_in_box
from app.schemas import RenderRequest

ART = Path(__file__).resolve().parent.parent / "artifacts" / "visual-regression"
FONT = "dialogue.mac-dinh-3"
TEXT = "Ta đã bảo ngươi đừng gọi ta là Các Chủ."


def _save(image, name):
    ART.mkdir(parents=True, exist_ok=True)
    image.save(ART / f"{name}.png")


def _ellipse(fill):
    image = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(image).ellipse([50, 50, 650, 350], fill=fill, outline=(0, 0, 0), width=4)
    return image


def _gradient_ellipse():
    image = Image.new("RGB", (700, 400), "white")
    grad = np.linspace(235, 160, 300).astype(np.uint8)
    band = np.repeat(grad[:, None], 600, axis=1)
    image.paste(Image.fromarray(band).convert("RGB"), (50, 50))
    mask = Image.new("L", (700, 400), 0)
    ImageDraw.Draw(mask).ellipse([50, 50, 650, 350], fill=255)
    base = Image.new("RGB", (700, 400), "white")
    base.paste(image, (0, 0), mask)
    ImageDraw.Draw(base).ellipse([50, 50, 650, 350], outline=(0, 0, 0), width=4)
    return base


def _render_one(image, name, container="bubble"):
    region = (60, 60, 640, 340)
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(), region, seed=(350, 200))
    obj = {"id": "t", "translation": TEXT, "region": dict(zip(("x1", "y1", "x2", "y2"), region)),
           "container": container, "typography_role": "dialogue", "source_cap_px": 40,
           "bubble_mask": mask, "style": {"font": FONT}}
    base = image.copy()
    req = RenderRequest(chapter_id="vr", page_index=0, translations={"t": TEXT})
    render_text_objects(image, req, [obj], *({} for _ in range(10)))
    ink = _ink_mask(base, image)
    box = _mask_bbox(ink)
    assert box is not None, f"{name}: bubble left empty (no ink)"
    ys, xs = np.nonzero(ink)
    assert bool((((xs - 350) / 300) ** 2 + ((ys - 200) / 150) ** 2 <= 1.05).all()), \
        f"{name}: ink crossed the ellipse outline: {box}"
    stats = {}
    render_text_in_box(base.copy(), TEXT, region, font_name=FONT, source_cap_px=40,
                       bubble_mask=mask, stats=stats)
    assert stats["size"] >= RENDER_MIN_READABLE_FONT_SIZE, f"{name}: below readable floor {stats}"
    _save(image, name)
    return stats


def test_white_bubble_letters_inside_and_legible():
    _render_one(_ellipse("white"), "white")


def test_grey_gradient_bubble_not_under_size():
    stats = _render_one(_gradient_ellipse(), "grey_gradient")
    assert stats["size"] >= RENDER_MIN_READABLE_FONT_SIZE


def test_spiky_bubble_letters_inside():
    import math
    cx, cy, outer, inner, points = 350, 200, 170, 120, 12
    star = [(cx + (outer if i % 2 == 0 else inner) * math.cos(math.pi * i / points - math.pi / 2),
             cy + (outer if i % 2 == 0 else inner) * math.sin(math.pi * i / points - math.pi / 2))
            for i in range(2 * points)]
    image = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(image).polygon(star, fill="white", outline=(0, 0, 0), width=4)
    region = (200, 60, 500, 340)
    obj = {"id": "t", "translation": "HẮN ĐÃ CHIẾN ĐẤU", "region": dict(zip(("x1", "y1", "x2", "y2"), region)),
           "container": "spiky", "typography_role": "shout", "source_cap_px": 40, "style": {"font": FONT}}
    base = image.copy()
    req = RenderRequest(chapter_id="vr", page_index=0, translations={"t": obj["translation"]})
    render_text_objects(image, req, [obj], *({} for _ in range(10)))
    ink = _ink_mask(base, image)
    box = _mask_bbox(ink)
    assert box is not None, "spiky bubble left empty"
    star_mask = Image.new("L", (700, 400), 0)
    ImageDraw.Draw(star_mask).polygon(star, fill=255)
    assert not (ink & (np.asarray(star_mask) == 0)).any(), \
        f"spiky ink escaped the star outline: {box}"
    _save(image, "spiky")


def test_double_lobe_each_lobe_lettered_not_crossed():
    image = Image.new("RGB", (400, 600), "black")
    d = ImageDraw.Draw(image)
    d.ellipse([40, 40, 360, 240], fill="white", outline=(0, 0, 0), width=3)
    d.ellipse([50, 300, 350, 560], fill="white", outline=(0, 0, 0), width=3)
    d.rectangle([194, 235, 206, 305], fill="white")
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(), (40, 40, 360, 560), seed=(200, 140))
    assert mask is not None and find_waist(mask) is not None
    a = {"id": "a", "translation": "Vâng, vâng. Vậy họ nói gì?",
         "region": {"x1": 40, "y1": 40, "x2": 360, "y2": 560},
         "letter_bounds": {"x1": 90, "y1": 80, "x2": 310, "y2": 200},
         "bubble_mask": mask, "container": "bubble", "typography_role": "dialogue",
         "source_cap_px": 60, "style": {"font": FONT}}
    b = {"id": "b", "translation": "Họ cho ngươi ở lại Thiết Kiếm Các không?",
         "region": {"x1": 40, "y1": 40, "x2": 360, "y2": 560},
         "letter_bounds": {"x1": 90, "y1": 320, "x2": 310, "y2": 480},
         "bubble_mask": mask, "container": "bubble", "typography_role": "dialogue",
         "source_cap_px": 60, "style": {"font": FONT}}
    base = image.copy()
    req = RenderRequest(chapter_id="vr", page_index=0, translations={"a": a["translation"], "b": b["translation"]})
    render_text_objects(image, req, [a, b], *({} for _ in range(10)))
    ink = _ink_mask(base, image)
    assert ink is not None and ink.any(), "double bubble left empty"
    ys, xs = np.nonzero(ink)
    assert ys.min() < 240 and ys.max() > 300, "both lobes must be lettered"
    upper = _mask_bbox(ink[:241, :])
    assert upper is not None and upper[3] <= 240, f"upper-lobe ink crossed the waist: {upper}"
    assert not ink[241:300, :].any(), "ink bridges the waist between the lobes"
    lower = _mask_bbox(ink[300:, :])
    assert lower is not None and 300 + lower[3] <= 560, f"lower-lobe ink outside the lower lobe: {lower}"
    _save(image, "double_lobe")
