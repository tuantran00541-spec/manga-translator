import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.ai_mode.seams import LETTERING_KEYS
from app.ocr.identity import stamp_machine_cache
from app.ocr.visual_metadata import (
    normalize_color_profile,
    sample_source_text_color_profile,
    sync_group_visual_metadata,
)
from app.render import page_renderer
from app.render.text_renderer import render_text_in_box
from app.schemas import RenderRequest
from app.text_objects import ensure_page_text_objects


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_solid_antialiased_black_on_white_has_no_false_gradient_stroke_or_glow():
    # Render crisp anti-aliased black text onto a white bubble background.
    pil = Image.new("RGB", (260, 100), (255, 255, 255))
    render_text_in_box(
        pil,
        "DIALOGUE",
        (10, 10, 250, 90),
        fill="#141414",
        font_size=36,
        stroke_width=0,
    )
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    mask = (np.asarray(pil.convert("L")) < 120).astype(np.uint8) * 255
    box = {
        "x1": 0,
        "y1": 0,
        "x2": 260,
        "y2": 100,
        "source_role": "text_segmenter",
        "mask": mask,
    }

    profile = sample_source_text_color_profile(
        bgr,
        box,
        {"x1": 10, "y1": 10, "x2": 250, "y2": 90},
    )

    assert profile is not None
    r, g, b = _hex_to_rgb(profile["fill"])
    assert max(r, g, b) < 45
    assert "gradient" not in profile
    assert "stroke_color" not in profile
    assert "glow_color" not in profile
    assert profile["lab_span"] < 20.0
    assert profile["luminance_range"][1] - profile["luminance_range"][0] < 35


def test_vertical_gradient_title_extracts_3_stops_and_luminance_amplitude():
    # Dark panel with a vertical gold -> orange -> crimson block glyph.
    h, w = 120, 240
    img = np.full((h, w, 3), (20, 18, 22), dtype=np.uint8)  # BGR dark bg
    mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1, x0, x1 = 24, 96, 40, 200
    mask[y0:y1, x0:x1] = 255
    for y in range(y0, y1):
        t = (y - y0) / max(1, (y1 - y0 - 1))
        # RGB gold (255, 235, 60) -> crimson (180, 20, 20)
        r = int(round(255 * (1 - t) + 180 * t))
        g = int(round(235 * (1 - t) + 20 * t))
        b = int(round(60 * (1 - t) + 20 * t))
        img[y, x0:x1] = (b, g, r)

    box = {
        "x1": 0,
        "y1": 0,
        "x2": w,
        "y2": h,
        "source_role": "text_segmenter",
        "mask": mask,
    }
    profile = sample_source_text_color_profile(
        img,
        box,
        {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
    )

    assert profile is not None
    assert "gradient" in profile
    assert profile["gradient"]["direction"] == "vertical"
    stops = profile["gradient"]["stops"]
    assert len(stops) == 3
    top_rgb = _hex_to_rgb(stops[0])
    bot_rgb = _hex_to_rgb(stops[-1])
    # Top is bright gold (high G), bottom is crimson (low G).
    assert top_rgb[1] > 180
    assert bot_rgb[1] < 90
    assert bot_rgb[0] > 150
    assert profile["luminance_range"][1] - profile["luminance_range"][0] >= 80
    assert profile["lab_span"] >= 40.0


def test_horizontal_gradient_extracts_left_to_right_stops():
    h, w = 100, 240
    img = np.full((h, w, 3), (18, 18, 24), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1, x0, x1 = 26, 74, 30, 210
    mask[y0:y1, x0:x1] = 255
    for x in range(x0, x1):
        t = (x - x0) / max(1, (x1 - x0 - 1))
        # RGB cyan (20, 230, 255) -> magenta (255, 30, 190)
        r = int(round(20 * (1 - t) + 255 * t))
        g = int(round(230 * (1 - t) + 30 * t))
        b = int(round(255 * (1 - t) + 190 * t))
        img[y0:y1, x] = (b, g, r)

    box = {
        "x1": 0,
        "y1": 0,
        "x2": w,
        "y2": h,
        "source_role": "text_segmenter",
        "mask": mask,
    }
    profile = sample_source_text_color_profile(
        img,
        box,
        {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
    )

    assert profile is not None
    assert "gradient" in profile
    assert profile["gradient"]["direction"] == "horizontal"
    stops = profile["gradient"]["stops"]
    assert len(stops) == 2
    left_rgb = _hex_to_rgb(stops[0])
    right_rgb = _hex_to_rgb(stops[1])
    assert left_rgb[2] > 200 and left_rgb[0] < 100
    assert right_rgb[0] > 180 and right_rgb[1] < 110


def test_colored_stroke_and_decaying_outer_glow_extracted():
    # White core glyph with a crisp crimson stroke and soft decaying crimson aura on dark bg.
    h, w = 120, 220
    canvas = Image.new("RGBA", (w, h), (15, 15, 22, 255))
    glyph = Image.new("L", (w, h), 0)
    gdraw = ImageDraw.Draw(glyph)
    gdraw.rectangle((45, 35, 175, 85), fill=255)

    # Outer glow layer (blurred crimson aura)
    glow_mask = glyph.filter(ImageFilter.GaussianBlur(radius=6))
    glow_arr = np.zeros((h, w, 4), dtype=np.uint8)
    glow_arr[:, :, 0] = 220
    glow_arr[:, :, 1] = 25
    glow_arr[:, :, 2] = 35
    glow_arr[:, :, 3] = np.clip(np.asarray(glow_mask, dtype=np.float32) * 1.4, 0, 255).astype(np.uint8)
    canvas = Image.alpha_composite(canvas, Image.fromarray(glow_arr, mode="RGBA"))

    # Crisp 2px crimson stroke directly around glyph
    mask_u8 = np.asarray(glyph, dtype=np.uint8)
    dil = cv2.dilate(mask_u8, np.ones((5, 5), dtype=np.uint8), iterations=1)
    rgb = np.asarray(canvas.convert("RGB")).copy()
    rgb[(dil > 127) & (mask_u8 <= 127)] = (210, 20, 30)
    rgb[mask_u8 > 127] = (248, 248, 248)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    box = {
        "x1": 0,
        "y1": 0,
        "x2": w,
        "y2": h,
        "source_role": "text_segmenter",
        "mask": mask_u8,
    }
    profile = sample_source_text_color_profile(
        bgr,
        box,
        {"x1": 45, "y1": 35, "x2": 175, "y2": 85},
    )

    assert profile is not None
    assert "stroke_color" in profile
    sr, sg, sb = _hex_to_rgb(profile["stroke_color"])
    assert sr > 150 and sg < 80 and sb < 80
    assert "glow_color" in profile
    gr, gg, gb = _hex_to_rgb(profile["glow_color"])
    assert gr > 70 and gr > gg + 30
    assert profile["glow_radius"] >= 2


def test_color_profile_propagates_through_identity_text_objects_and_seams():
    rich_profile = {
        "fill": "#f0a020",
        "luminance_range": [70, 225],
        "lab_span": 64.2,
        "gradient": {"direction": "vertical", "stops": ["#ffe850", "#f09020", "#b81818"]},
        "stroke_color": "#200808",
        "glow_color": "#ff4020",
        "glow_radius": 5,
    }
    box = {"id": "b1", "x1": 10, "y1": 20, "x2": 160, "y2": 90, "ocr_eligible": True}
    stamp_machine_cache(
        box,
        text="BOOM",
        lang="en",
        engine="paddle",
        source_revision=1,
        original_revision=(100, 200, 300),
        metadata={
            "confidence": 0.95,
            "text_color": "#f0a020",
            "color_profile": rich_profile,
            "font_size": 32,
            "text_region": {"x1": 14, "y1": 24, "x2": 156, "y2": 86},
        },
    )
    assert box["ocr_color_profile"] == normalize_color_profile(rich_profile)

    plain_box = {
        "id": "b2",
        "x1": 10,
        "y1": 95,
        "x2": 160,
        "y2": 130,
        "ocr_eligible": True,
        "ocr_text": "SUB",
        "ocr_text_color": "#111111",
        "ocr_color_profile": {"fill": "#111111", "luminance_range": [12, 22], "lab_span": 4.0},
    }
    page = {"boxes": [plain_box, box]}
    ensure_page_text_objects(page)
    assert any(obj.get("ocr_color_profile") == box["ocr_color_profile"] for obj in page["text_objects"])

    grouped_obj = {"id": "group_1"}
    sync_group_visual_metadata(grouped_obj, page, ["b2", "b1"])
    # Richest profile (gradient + glow + stroke) wins over plain solid profile.
    assert grouped_obj["ocr_color_profile"]["gradient"]["direction"] == "vertical"
    assert "ocr_color_profile" in LETTERING_KEYS


def test_render_text_in_box_reconstructs_gradient_stroke_and_glow():
    # Light background with bright gold->red gradient + dark stroke + warm glow.
    # Verifies stroke protects bright gradient from pale_to_auto fallback and renders actual gradient pixels.
    profile = {
        "fill": "#f0c830",
        "luminance_range": [80, 235],
        "lab_span": 72.0,
        "gradient": {"direction": "vertical", "stops": ["#fff040", "#ff8010", "#a01010"]},
        "stroke_color": "#180808",
        "glow_color": "#ff3010",
        "glow_radius": 4,
    }
    base = Image.new("RGB", (320, 120), (242, 242, 245))
    out = base.copy()
    stats: dict = {}
    render_text_in_box(
        out,
        "POWER",
        (20, 16, 300, 104),
        fill="#f0c830",
        font_size=52,
        stroke_width="auto",
        stroke_color="auto",
        color_profile=profile,
        stats=stats,
    )

    assert stats.get("fill_fallback") is None
    assert stats["color_profile_applied"] == {"gradient": True, "stroke": True, "glow": True}

    arr = np.asarray(out)
    # Find bright chromatic interior pixels (distinguished from near-grey bg and dark stroke).
    sat = arr.max(axis=2).astype(int) - arr.min(axis=2).astype(int)
    bright_chrom = (sat > 80) & (arr[:, :, 0] > 130)
    ys, _ = np.nonzero(bright_chrom)
    assert ys.size >= 80
    y_min, y_max = int(ys.min()), int(ys.max())
    cut_top = y_min + max(1, (y_max - y_min) // 3)
    cut_bot = y_max - max(1, (y_max - y_min) // 3)

    top_pixels = arr[bright_chrom & (np.arange(arr.shape[0])[:, None] <= cut_top)]
    bot_pixels = arr[bright_chrom & (np.arange(arr.shape[0])[:, None] >= cut_bot)]
    assert top_pixels.size > 0 and bot_pixels.size > 0
    # Top pixels should be golden (much higher green channel) than crimson bottom pixels.
    assert float(np.median(top_pixels[:, 1])) > float(np.median(bot_pixels[:, 1])) + 45.0

    # Dark extracted stroke pixels should exist around the glyphs.
    dark_stroke = (arr[:, :, 0] < 45) & (arr[:, :, 1] < 30) & (arr[:, :, 2] < 30)
    assert int(np.count_nonzero(dark_stroke)) >= 40


def test_page_renderer_passes_color_profile_only_when_user_color_is_auto(monkeypatch):
    calls = []

    def fake_render(image, text, box, **kwargs):
        calls.append({"text": text, "box": box, **kwargs})
        return image

    monkeypatch.setattr(page_renderer, "render_text_in_box", fake_render)

    profile = {
        "fill": "#f0a020",
        "gradient": {"direction": "vertical", "stops": ["#ffe850", "#f09020", "#b81818"]},
    }
    obj = {
        "id": "text_1",
        "translation": "",
        "region": {"x1": 10, "y1": 20, "x2": 150, "y2": 90},
        "auto_geometry": {"x1": 10, "y1": 20, "x2": 150, "y2": 90},
        "auto_generated": True,
        "ocr_text_color": "#f0a020",
        "ocr_color_profile": profile,
        "style": {"color": "auto", "font": "default", "fontSize": "auto"},
    }
    req = RenderRequest(
        chapter_id="chapter-test",
        page_index=0,
        translations={"text_1": "Xin chao"},
    )
    image = Image.new("RGB", (200, 140), "white")

    page_renderer.render_text_objects(image, req, [obj], *({} for _ in range(10)))
    assert calls[0]["color_profile"] == profile

    # When user explicitly chooses a solid color, color_profile must NOT override it.
    calls.clear()
    page_renderer.render_text_objects(
        image,
        req,
        [obj],
        {"text_1": "#0044ff"},
        *({} for _ in range(9)),
    )
    assert calls[0]["fill"] == "#0044ff"
    assert calls[0]["color_profile"] is None


def test_update_text_object_schema_accepts_and_normalizes_color_profile():
    from app.schemas import UpdateTextObjectRequest

    req = UpdateTextObjectRequest(
        chapter_id="ch-1",
        page_index=0,
        id="text_1",
        shape="ellipse",
        ocr_color_profile={
            "fill": "#FFAA00",
            "gradient": {"direction": "vertical", "stops": ["#FFE040", "#FF8010", "#B01010"]},
            "glow_color": "#FF3010",
            "glow_radius": 6,
        },
    )
    assert req.shape == "ellipse"
    assert "ocr_color_profile" in req.model_fields_set
    assert normalize_color_profile(req.ocr_color_profile)["gradient"]["direction"] == "vertical"


def test_kiuyha_process_page_automatically_captures_color_profile_without_ocr(tmp_path):
    from app.detector.boxes import BubbleBox
    from app.pipeline import ChapterPipeline

    # Build a synthetic manga page with a gold->crimson vertical gradient title on a dark panel.
    h, w = 140, 260
    img = np.full((h, w, 3), (18, 18, 22), dtype=np.uint8)
    y0, y1, x0, x1 = 30, 110, 40, 220
    # Dilated inpaint mask covers whole box; Kiuyha's exact `letters` covers only the inner glyph block.
    letters = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    letters[12:68, 16:164] = True
    dilated_mask = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    for ly in range(12, 68):
        t = (ly - 12) / max(1, (68 - 12 - 1))
        r = int(round(255 * (1 - t) + 185 * t))
        g = int(round(230 * (1 - t) + 20 * t))
        b = int(round(50 * (1 - t) + 20 * t))
        img[y0 + ly, x0 + 16:x0 + 164] = (b, g, r)

    raw_path = tmp_path / "page_000.png"
    cv2.imwrite(str(raw_path), img)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    detected_box = BubbleBox(
        x0, y0, x1, y1, 0.96, dilated_mask,
        source_model="kiuyha", class_name="text", semantic_type="free_text",
        mask_source="text_segmenter", safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter", letters=letters,
    )

    class _DummyDetector:
        def detect(self, _image):
            return [detected_box]
        def last_metrics(self):
            return {}
        def leftover_boxes(self, _clean, _boxes):
            return []

    class _DummyInpainter:
        session_loaded = False
        _prefer_dynamic = False
        serialized_inference = True
        lama_model_path = None
        def inpaint(self, image, _boxes, protected_regions=None):
            return np.full_like(image, (18, 18, 22))
        def last_metrics(self):
            return {}

    pipe = ChapterPipeline()
    pipe._detector = _DummyDetector()
    pipe._inpainter = _DummyInpainter()
    res = pipe._process_page(raw_path, processed_dir)
    assert len(res["boxes"]) == 1
    saved_box = res["boxes"][0]
    assert saved_box.get("ocr_text_color")
    prof = saved_box.get("ocr_color_profile")
    assert isinstance(prof, dict)
    assert prof["gradient"]["direction"] == "vertical"
    assert len(prof["gradient"]["stops"]) == 3

    # Propagate straight to text_objects WITHOUT running OCR, and confirm renderer uses it automatically.
    page = {"boxes": res["boxes"], "text_objects": []}
    ensure_page_text_objects(page)
    obj = page["text_objects"][0]
    assert obj["ocr_color_profile"] == prof

    # Even if OCR runs later with empty color_profile, Kiuyha's richer profile is preserved.
    stamp_machine_cache(
        saved_box,
        text="POWER",
        lang="en",
        engine="paddle",
        source_revision=1,
        original_revision=(1, 2, 3),
        metadata={"confidence": 0.9, "text_color": "", "color_profile": None},
    )
    assert saved_box["ocr_color_profile"] == prof




def test_render_commit_upgrades_raw_ink_and_reuses_exact_color(tmp_path, monkeypatch):
    import json

    import app.config
    import app.manifest_utils as manifests
    from app.image_io import encode_mask
    from app.routers import render_commit
    from app.routers.translation import _vision_candidates

    chapter_id = "d1234567"
    raw_root, processed_root = tmp_path / "raw", tmp_path / "processed"
    raw_dir, processed = raw_root / chapter_id, processed_root / chapter_id
    raw_dir.mkdir(parents=True)
    processed.mkdir(parents=True)
    monkeypatch.setattr(app.config, "RAW_DIR", raw_root)
    monkeypatch.setattr(app.config, "PROCESSED_DIR", processed_root)
    monkeypatch.setattr(manifests, "PROCESSED_DIR", processed_root)
    monkeypatch.setattr(render_commit, "OUTPUT_DIR", tmp_path / "output")
    from app import mask_store

    monkeypatch.setattr(mask_store, "PROCESSED_DIR", processed_root)
    raw_path, clean_path = raw_dir / "page.png", processed / "clean.png"
    raw_color = (232, 183, 60)
    clean = Image.new("RGB", (265, 120), "#f9f0d0")
    clean.save(clean_path)
    raw = clean.copy()
    region = {"x1": 20, "y1": 20, "x2": 245, "y2": 100}
    mask = np.zeros((80, 225), np.uint8)
    mask[20:58, 32:48] = 255
    mask[20:58, 70:86] = 255
    pixels = np.asarray(raw).copy()
    pixels[20:100, 20:245][mask != 0] = raw_color
    Image.fromarray(pixels).save(raw_path)
    boxes = [{"id": "box_1", **region, "letters_mask": encode_mask(mask),
              "ocr_color_profile": {"fill": "#927366"}}]
    obj = {"id": "text_1", "region": region, "source_boxes": ["box_1"], "container": "bubble",
           "ocr_color_profile": {"fill": "#927366"}, "lettering_color": "#332211",
           "translation": "Chào bạn", "style": {"color": "auto", "fontSize": "28"}}
    manifests.save_manifest_raw(chapter_id, {"chapter_id": chapter_id, "pages": [
        {"original": str(raw_path), "clean": str(clean_path), "boxes": boxes, "text_objects": [obj]}
    ]})
    req = RenderRequest(chapter_id=chapter_id, page_index=0, translations={"text_1": obj["translation"]})
    assert render_commit.render_page(req)["committed"]
    saved = json.loads((processed / "manifest.json").read_text())["pages"][0]
    profile = saved["text_objects"][0]["ocr_color_profile"]
    assert profile["fill"] == "#e8b73c"
    assert saved["boxes"][0]["ocr_color_profile"] == profile
    assert _vision_candidates(saved, force=True)[0]["source_ink"] == profile["fill"]
    output = tmp_path / "output" / chapter_id / "page_000.png"
    assert np.count_nonzero(np.all(np.asarray(Image.open(output)) == raw_color, axis=2)) > 30

    req.colors = {"text_1": "#0044ff"}
    assert render_commit.render_page(req)["committed"]
    pixels = np.asarray(Image.open(output))
    assert np.count_nonzero(np.all(pixels == (0, 68, 255), axis=2)) > 30
    assert not np.any(np.all(pixels == raw_color, axis=2))


def test_measured_gradient_spans_both_lines_of_a_translation():
    image = Image.new("RGB", (300, 155), "#eeeeee")
    profile = {
        "source": "raw", "sample_version": 1, "fill": "#ab6993",
        "gradient": {"direction": "vertical", "stops": ["#ee4b32", "#5849ee"]},
    }
    render_text_in_box(image, "ÉTAGE\nÉTAGE", (12, 8, 288, 147), font_size=36,
                       color_profile=profile, adapt_contrast=True)
    rgb = np.asarray(image).astype(int)
    painted = np.ptp(rgb, axis=2) > 65
    top, bottom = np.array_split(rgb[:, :, 0] - rgb[:, :, 2], 2)
    top_mask, bottom_mask = np.array_split(painted, 2)
    assert np.median(top[top_mask]) > 45
    assert np.median(bottom[bottom_mask]) < -45
