import numpy as np
import cv2
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import clip_bubble_mask, derive_bubble_mask, mask_row_spans
from app.render.composition import _word_spans, compose_text
from app.render.font_metrics import _contrast, get_font_object
from app.render.page_renderer import _ink_mask, _mask_bbox, render_text_objects
from app.routers import export, image, render_commit
from app.schemas import RenderRequest


def _region(box):
    return dict(zip(("x1", "y1", "x2", "y2"), box))


def _auto_object(oid, box, text, mask=None, **fields):
    region = _region(box)
    return {
        "id": oid,
        "region": region,
        "auto_geometry": dict(region),
        "auto_generated": True,
        "translation": text,
        "auto_translation": text,
        "container": "bubble",
        "typography_role": "dialogue",
        "bubble_mask": mask,
        "style": {"fontSize": 30, "strokeWidth": 0},
        **fields,
    }


def _oval():
    image = Image.new("RGB", (440, 320), "white")
    ImageDraw.Draw(image).ellipse((25, 25, 415, 285), outline="black", width=4)
    return image


def _contour(image, box, seed=None):
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(), box, seed=seed)
    assert mask["kind"] == "bubble"
    return mask


def _render(image, objects, translations=None, **request_styles):
    req = RenderRequest(chapter_id="sample", page_index=0,
                        translations=translations or {}, **request_styles)
    styles = render_commit._request_style_maps(req)
    verdicts = []
    count = render_text_objects(image, req, objects, *styles.values(), verdicts=verdicts)
    return count, {item["id"]: item for item in verdicts}


def test_snapshot_recovers_bubble_from_source_when_clean_outline_is_broken(monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    chapter = raw_root / "sample"
    chapter.mkdir(parents=True)
    raw = _oval()
    original = chapter / "0.png"
    raw.save(original)
    clean = raw.copy()
    ImageDraw.Draw(clean).rectangle((205, 279, 235, 290), fill="white")
    result = clean.copy()
    box = (80, 252, 360, 310)
    obj = _auto_object("one", box, "TA SẼ RA ĐÒN TRƯỚC.", {"bbox": box, "kind": "rect"})
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", raw_root)
    page = {"original": str(original), "clean": "clean.png", "text_objects": [obj]}
    count, skipped = render_commit._render_snapshot(
        result, req, page, {}, render_commit._request_style_maps(req))
    ink = _mask_bbox(_ink_mask(clean, result))
    assert (count, skipped) == (1, [])
    assert 110 < ink[1] < ink[3] < 210


def test_fixed_lettering_uses_the_outline_centre_instead_of_the_old_seed():
    base = _oval()
    box = (80, 230, 360, 280)
    mask = _contour(base, box, seed=(220, 255))
    image = base.copy()
    obj = _auto_object("one", box, "ĐI THÔI!", mask)
    count, verdicts = _render(image, [obj])
    ink = _mask_bbox(_ink_mask(base, image))
    assert count == 1 and verdicts["one"]["size"] == 30
    assert verdicts["one"]["frame_source"] == "mask"
    assert 122 < ink[1] < ink[3] < 196


def test_cuts_in_one_bubble_join_without_consuming_edited_text():
    base = _oval()
    mask = _contour(base, (105, 105, 330, 185))
    left = _auto_object("left", (105, 110, 220, 182), "ĐI",
                        clip_bubble_mask(mask, (25, 25, 220, 285)))
    right = _auto_object("right", (220, 108, 330, 179), "THÔI!",
                         clip_bubble_mask(mask, (220, 25, 416, 286)))
    image = base.copy()
    count, verdicts = _render(image, [left, right])
    ink = _mask_bbox(_ink_mask(base, image))
    assert count == 2 and verdicts["left"]["guard"] == "joined_in_bubble"
    assert verdicts["right"]["frame_source"] == "mask"
    assert abs((ink[0] + ink[2]) / 2 - 220) <= 5
    assert ink[2] - ink[0] < 180
    assert left["translation"] == "ĐI" and right["translation"] == "THÔI!"

    count, verdicts = _render(base.copy(), [left, right], translations={"left": "ĐI NGAY"})
    assert count == 2 and all(v["guard"] != "joined_in_bubble" for v in verdicts.values())


def test_stacked_fragments_use_one_block_when_the_rect_recovers_the_same_outline():
    base = _oval()
    upper = (82, 76, 358, 147)
    lower = (82, 166, 358, 242)
    first = _auto_object("one", upper, "CHÚNG TA SẼ TIÊU DIỆT\nLỰC LƯỢNG YAMAZAKI!", _contour(base, upper))
    second = _auto_object("two", lower, "ĐÃ XÂM NHẬP! SEOUL!", {"bbox": lower, "kind": "rect"})
    out = base.copy()
    count, verdicts = _render(out, [first, second])
    assert count == 2 and verdicts["one"]["frame_source"] == "mask"
    assert verdicts["two"]["guard"] == "joined_in_bubble"
    assert not np.any(np.asarray(base)[_ink_mask(base, out)] == 0)


def test_connected_balloons_keep_their_own_dialogue():
    base = Image.new("RGB", (440, 480), "black")
    draw = ImageDraw.Draw(base)
    draw.ellipse((45, 45, 395, 220), fill="white")
    draw.ellipse((45, 255, 395, 430), fill="white")
    draw.rectangle((214, 215, 226, 260), fill="white")
    upper = (78, 104, 362, 236)
    lower = (78, 264, 362, 344)
    mask = _contour(base, (45, 45, 395, 430), seed=(220, 130))
    first = _auto_object("one", upper, "CHỜ MỘT CHÚT.", mask)
    second = _auto_object("two", lower, "ĐI THÔI!", {"bbox": lower, "kind": "rect"})
    count, verdicts = _render(base.copy(), [first, second])
    assert count == 2
    assert all(verdict["guard"] != "joined_in_bubble" for verdict in verdicts.values())


def test_reflow_changes_machine_line_breaks_and_honours_user_line_breaks():
    base = _oval()
    box = (100, 105, 340, 205)
    obj = _auto_object("one", box, "CHỜ\nMỘT CHÚT.", _contour(base, box))
    _, auto = _render(base.copy(), [obj])
    assert auto["one"]["lines"] == 1 and auto["one"]["reflowed"]
    _, manual = _render(base.copy(), [obj], translations={"one": "ĐỢI\nMỘT CHÚT."})
    assert manual["one"]["lines"] == 2 and not manual["one"].get("reflowed")
    obj["translation"] = obj["auto_translation"] = "CHỜ\nMỘT CHÚT.\n\nĐI THÔI!"
    _, paragraphs = _render(base.copy(), [obj])
    assert paragraphs["one"]["lines"] == 2


def test_auto_ink_is_darkened_over_local_white_and_manual_colour_is_respected():
    base = _oval()
    box = (75, 100, 370, 215)
    obj = _auto_object("one", box, "CHỜ MỘT CHÚT.", _contour(base, box),
                       ocr_text_color="#a0a0a0")
    out = base.copy()
    _, verdicts = _render(out, [obj])
    assert verdicts["one"]["fill_fallback"] == "local_contrast"
    ink = _ink_mask(base, out)
    colour = np.asarray(out)[ink].min(axis=0)
    assert _contrast(colour, (255, 255, 255)) >= 4.5
    manual = base.copy()
    _, verdicts = _render(manual, [obj], colors={"one": "#a0a0a0"})
    assert verdicts["one"].get("fill_fallback") is None
    assert tuple(np.asarray(manual)[_ink_mask(base, manual)].min(axis=0)) == (160, 160, 160)


def test_auto_composition_stays_clear_of_a_contour_and_overlapping_art():
    base = _oval()
    region = (62, 68, 375, 245)
    mask = _contour(base, region)
    ImageDraw.Draw(base).polygon([(78, 234), (158, 228), (128, 178)], fill="black")
    words = "Tôi vẫn luôn tự hỏi liệu chúng ta có thể thay đổi được kết cục này hay không."
    obj = _auto_object("one", region, words, mask, source_cap_px=45, style={"fontSize": "auto"})
    out = base.copy()
    count, verdicts = _render(out, [obj])
    ink = _ink_mask(base, out)
    allowed = np.zeros(ink.shape, np.uint8)
    for y in range(base.height):
        for a, b in mask_row_spans(mask, y):
            allowed[y, a:b] = 1
    allowed[np.asarray(base)[..., 0] < 130] = 0
    clearance = cv2.distanceTransform(allowed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    placement = verdicts["one"]
    assert count == 1 and placement["layout_score"] is not None
    assert 2 <= placement["lines"] <= 5
    assert clearance[ink].min() > placement["edge_clearance"]


def test_auto_composition_keeps_short_dialogue_at_a_reading_size_in_a_large_bubble():
    base = _oval()
    region = (60, 65, 380, 250)
    obj = _auto_object("one", region, "Hả?", _contour(base, region), style={"fontSize": "auto"})
    count, verdicts = _render(base, [obj])
    placement = verdicts["one"]
    assert count == 1
    assert placement["lines"] == 1 and 26 <= placement["size"] <= 40


def test_composition_breaks_unspaced_scripts_without_dropping_glyphs():
    text = "「日本語、です。」\n…你好世界！"
    normalized, spans, ends = _word_spans(text)
    units = [normalized[a:b] for a, b in spans]
    assert "".join(units) == text.replace("\n", "")
    assert len(set(ends)) == 2 and len(units) > 5


def test_composition_uses_the_lobe_containing_the_source():
    base = Image.new("RGB", (700, 360), "white")
    mask = {"kind": "bubble", "bbox": [20, 30, 680, 330], "cx": 120, "cy": 170,
            "rows": [[[0, 212], [283, 660]] for _ in range(24)]}
    plan = compose_text(
        base, "Chờ tôi một chút.", tuple(mask["bbox"]), str(DEFAULT_FONT),
        preferred=36, floor=12, stroke=2, bold=False, mask=mask, anchor=(65, 130, 180, 200),
    )
    assert plan is not None
    draw = ImageDraw.Draw(base)
    font = get_font_object(str(DEFAULT_FONT), plan.size)
    for x, y, line in plan.lines:
        bounds = draw.textbbox((x, y), line, font=font, stroke_width=2)
        assert 20 < bounds[0] < bounds[2] < 232


def test_stale_saved_outline_is_recovered_at_the_object_location():
    base = Image.new("RGB", (440, 560), "white")
    draw = ImageDraw.Draw(base)
    for rectangle in ((30, 25, 412, 240), (25, 302, 417, 535)):
        draw.ellipse(rectangle, outline="black", width=4)
    stale = _contour(base, (86, 60, 358, 183))
    obj = _auto_object("one", (84, 356, 348, 457), "TÔI ĐÃ HIỂU RỒI.", stale,
                       style={"fontSize": "auto"})
    out = base.copy()
    count, verdicts = _render(out, [obj])
    ink = _mask_bbox(_ink_mask(base, out))
    assert count == 1 and verdicts["one"]["stale_outline"]
    assert verdicts["one"]["frame"][1] >= 300
    assert 320 <= ink[1] < ink[3] < 515


def test_auto_composition_reserves_room_around_existing_lettering():
    base = Image.new("RGB", (420, 280), "white")
    region = (42, 27, 377, 253)
    opts = dict(container="free", style={"fontSize": "auto", "strokeWidth": 0})
    first = _auto_object("a", region, "Cứ đi trước đi.", **opts)
    second = _auto_object("b", region, "Tôi sẽ theo sau.", **opts)
    out = base.copy()
    count, verdicts = _render(out, [first, second])
    assert count == 2 and verdicts["b"]["guard"] == "recomposed"
    first_only = base.copy()
    _render(first_only, [first])
    first_ink = _ink_mask(base, first_only)
    second_ink = _ink_mask(first_only, out)
    spacing = cv2.distanceTransform((~first_ink).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    assert second_ink.any() and spacing[second_ink].min() >= 4


def test_reserved_ink_allows_search_beyond_the_normal_centre():
    base = Image.new("RGB", (400, 300), "white")
    plan = compose_text(
        base, "Tôi sẽ theo sau.", (10, 10, 390, 290), str(DEFAULT_FONT),
        preferred=30, floor=12, stroke=0, bold=False, reserved_boxes=((10, 30, 390, 220),),
    )
    assert plan is not None
    draw = ImageDraw.Draw(base)
    font = get_font_object(str(DEFAULT_FONT), plan.size)
    assert all(draw.textbbox((x, y), line, font=font)[1] >= 220 for x, y, line in plan.lines)


def test_render_collision_survives_manifest_and_export_without_erasing_translation(monkeypatch, tmp_path):
    from app import config, manifest_utils

    chapter = "abcd0623"
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    for module, attr, value in ((config, "PROCESSED_DIR", processed),
                                (manifest_utils, "PROCESSED_DIR", processed),
                                (render_commit, "OUTPUT_DIR", output),
                                (image, "OUTPUT_DIR", output)):
        monkeypatch.setattr(module, attr, value)
    folder = processed / chapter
    folder.mkdir(parents=True)
    base = folder / "clean.png"
    Image.new("RGB", (140, 54), "white").save(base)
    region = (15, 10, 125, 43)
    first = _auto_object("a", region, "MMMMMM", container="free", origin="manual", auto_generated=False,
                         style={"fontSize": 23, "strokeWidth": 0})
    second = _auto_object("b", region, "WWWWWWWW", container="free", origin="manual", auto_generated=False,
                          style={"fontSize": "auto", "strokeWidth": 0})
    page = {"clean": str(base), "text_objects": [first, second], "boxes": []}
    manifest_utils.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [page]})
    result = render_commit.render_page(export._render_request_from_page(chapter, 0, page))
    assert result["layout_conflicts"] == ["b"] and result["skipped_objects"] == []
    summary = export.render_chapter(chapter, force=True)["chapter_render"]
    assert summary["reused"] == 1 and summary["layout_conflicts"] == 1
    assert summary["review_list"][0]["id"] == "b"
    saved = manifest_utils.load_manifest_raw(chapter)["pages"][0]
    assert saved["text_objects"][1]["translation"] == "WWWWWWWW"
