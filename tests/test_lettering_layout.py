import numpy as np
import cv2
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import clip_bubble_mask, derive_bubble_mask, mask_row_spans, shift_bubble_mask
from app.image_io import encode_mask
from app.render.composition import _word_spans, compose_text
from app.render.font_metrics import _contrast, get_font_object
from app.render.page_renderer import _ink_mask, _mask_bbox, _render_diff, render_text_objects
from app.render.text_renderer import _art_ink, _local_contrast, render_text_in_box
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
    clean = raw.copy()
    ImageDraw.Draw(raw).text((128, 216), "DON'T COME BACK", fill="black", font_size=25)
    original = chapter / "0.png"
    raw.save(original)
    ImageDraw.Draw(clean).rectangle((205, 279, 235, 290), fill="white")
    result = clean.copy()
    box = (76, 199, 384, 266)
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


def test_diagonal_lobes_own_their_source_dialogue_before_placing_ink():
    base = Image.new("RGB", (700, 500), "black")
    draw = ImageDraw.Draw(base)
    draw.ellipse((24, 40, 396, 325), fill="white")
    draw.ellipse((300, 181, 676, 465), fill="white")
    whole = _contour(base, (85, 112, 605, 410))
    first = _auto_object("a", (85, 112, 318, 226), "Chúng ta phải lên đường sớm.", whole,
                         style={"fontSize": "auto"})
    second = _auto_object("b", (384, 244, 605, 410), "Tôi sẽ chuẩn bị ngay.", whole,
                          style={"fontSize": "auto"})
    out = base.copy()
    count, verdicts = _render(out, [first, second])
    assert count == 2
    assert all(v["owned_chamber"] and v["guard"] == "clear" for v in verdicts.values())
    ax1, ay1, ax2, ay2 = verdicts["a"]["frame"]
    bx1, by1, bx2, by2 = verdicts["b"]["frame"]
    assert (ax1 + ax2) / 2 < (bx1 + bx2) / 2 - 210
    assert (ay1 + ay2) / 2 < (by1 + by2) / 2 - 100
    assert not np.any(np.asarray(base)[_ink_mask(base, out)] == 0)


def test_multiple_fragments_follow_one_bubble_with_asymmetric_clipping():
    base = _oval()
    whole = _contour(base, (90, 105, 350, 185))
    boxes = [(86, 131, 173, 180), (176, 130, 263, 179), (267, 129, 354, 181)]
    clips = [(25, 37, 174, 285), (175, 25, 265, 285), (266, 29, 415, 267)]
    objects = [
        _auto_object(str(i), box, word, clip_bubble_mask(whole, clip), style={"fontSize": "auto"})
        for i, (box, clip, word) in enumerate(zip(boxes, clips, ("Vậy.", "Xong.", "Đi thôi!")))
    ]
    out = base.copy()
    count, verdicts = _render(out, objects)
    assert count == 3 and sum(v["guard"] == "joined_in_bubble" for v in verdicts.values()) == 2
    assert [obj["translation"] for obj in objects] == ["Vậy.", "Xong.", "Đi thôi!"]


def test_blank_source_flags_machine_text_without_erasing_its_translation(monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    raw_path = raw_root / "sample" / "0.png"
    raw_path.parent.mkdir(parents=True)
    base = _oval()
    region = (78, 79, 360, 163)
    mask = _contour(base, region)
    label = _auto_object("label", region, "SHINGEN YAMAZAKI", mask, container="box")
    glyph = _auto_object("glyph", (101, 204, 285, 249), "TÔI ĐANG Ở ĐÂY.", mask,
                         verified_balloon=True)
    stale_ink = np.zeros((45, 184), np.uint8)
    stale_ink[9:25, 9:102] = 255
    glyph["letters_mask"] = encode_mask(stale_ink)
    page = {"original": str(raw_path), "clean": "clean.png", "text_objects": [label, glyph]}
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", raw_root)
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})
    with_source_ink = base.copy()
    ImageDraw.Draw(with_source_ink).text((104, 98), "SHINGEN YAMAZAKI", fill="black", font_size=24)
    with_source_ink.save(raw_path)
    issues = {}
    conflict_ids = []
    render_commit._render_snapshot(base.copy(), req, page, {}, render_commit._request_style_maps(req),
                                   layout_conflicts=conflict_ids, layout_issue_types=issues)
    assert issues == {"glyph": "placement_unverified"} and conflict_ids == ["glyph"]
    assert glyph["translation"] == "TÔI ĐANG Ở ĐÂY."

    ImageDraw.Draw(with_source_ink).text((109, 209), "I AM HERE", fill="black", font_size=18)
    with_source_ink.save(raw_path)
    issues.clear()
    render_commit._render_snapshot(base.copy(), req, page, {}, render_commit._request_style_maps(req),
                                   layout_issue_types=issues)
    assert "placement_unverified" not in issues.values()


def test_auto_text_cannot_borrow_a_nearby_balloon(monkeypatch, tmp_path):
    clean = Image.new("RGB", (420, 340), "#dddddd")
    ImageDraw.Draw(clean).ellipse((56, 30, 364, 175), fill="white", outline="black", width=4)
    next_balloon = _contour(clean, (103, 61, 309, 140))
    region = (104, 175, 315, 239)
    obj = _auto_object("one", region, "ĐI MAU!", next_balloon, style={"fontSize": "auto"})
    raw = clean.copy()
    ImageDraw.Draw(raw).text((125, 189), "GO NOW!", fill="black", font_size=32)
    source = tmp_path / "sample" / "0.png"
    source.parent.mkdir()
    raw.save(source)
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", tmp_path)
    page = {"original": str(source), "clean": "clean.png", "text_objects": [obj]}
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})

    output = clean.copy()
    count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req))
    ink = _mask_bbox(_ink_mask(clean, output))
    assert count == 1 and ink is not None
    assert region[1] <= ink[1] < ink[3] <= region[3]


def test_two_automatic_translations_cannot_both_claim_the_same_source(monkeypatch, tmp_path):
    box = (82, 80, 358, 148)
    base = _oval()
    outline = _contour(base, box)
    one = _auto_object("a", box, "ĐI THÔI!", outline, source_boxes=["same"])
    two = _auto_object("b", box, "Ở LẠI ĐÂY.", outline, source_boxes=["same"])
    raw = base.copy()
    ImageDraw.Draw(raw).text((130, 105), "COME ON", fill="black", font_size=27)
    source = tmp_path / "sample" / "0.png"
    source.parent.mkdir()
    raw.save(source)
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", tmp_path)
    page = {"original": str(source), "clean": "clean.png", "text_objects": [one, two],
            "boxes": [{"id": "same", **_region(box)}]}
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})

    output = base.copy()
    issues = {}
    count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req),
                                               layout_issue_types=issues)
    assert count == 0 and np.array_equal(np.asarray(output), np.asarray(base))
    assert issues == {"a": "placement_ambiguous", "b": "placement_ambiguous"}


def test_stale_source_link_cannot_move_lettering_to_another_balloon(monkeypatch, tmp_path):
    base = Image.new("RGB", (420, 460), "white")
    draw = ImageDraw.Draw(base)
    draw.ellipse((40, 15, 380, 208), outline="black", width=4)
    draw.ellipse((45, 253, 375, 436), outline="black", width=4)
    upper, lower = (85, 82, 332, 149), (85, 310, 332, 384)
    mask = _contour(base, lower)
    obj = _auto_object("one", upper, "ĐI ĐI!", mask, source_boxes=["lower"], verified_balloon=True)
    raw = base.copy()
    ImageDraw.Draw(raw).text((137, 338), "GO!", fill="black", font_size=30)
    source = tmp_path / "sample" / "0.png"
    source.parent.mkdir()
    raw.save(source)
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", tmp_path)
    page = {"original": str(source), "clean": "clean.png", "text_objects": [obj],
            "boxes": [{"id": "lower", **_region(lower)}]}
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})
    output = base.copy()
    issues = {}
    count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req),
                                               layout_issue_types=issues)
    assert count == 0 and issues == {"one": "placement_unverified"}
    assert np.array_equal(np.asarray(output), np.asarray(base))


def test_manually_moved_region_can_be_rendered_without_source_ink():
    base = Image.new("RGB", (440, 340), "white")
    region = (165, 170, 392, 274)
    obj = _auto_object("one", (60, 40, 295, 110), "ĐÃ XÁC NHẬN.", container="free",
                       style={"fontSize": "auto"})
    obj["region"] = _region(region)
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})
    output = base.copy()
    count, _ = render_commit._render_snapshot(output, req, {"text_objects": [obj]}, {},
                                               render_commit._request_style_maps(req))
    ink = _mask_bbox(_ink_mask(base, output))
    assert count == 1 and ink
    assert region[0] <= ink[0] < ink[2] <= region[2] and region[1] <= ink[1] < ink[3] <= region[3]


def test_projected_balloon_keeps_its_position_across_two_slices(monkeypatch, tmp_path):
    base = Image.new("RGB", (360, 330), "white")
    ImageDraw.Draw(base).ellipse((32, 68, 322, 260), outline="black", width=4)
    raw = base.copy()
    ImageDraw.Draw(raw).text((130, 145), "WAIT HERE", fill="black", font_size=23)
    record = (83, 118, 278, 200)
    mask = _contour(base, record)
    first, second = 210, 95
    source_directory = tmp_path / "sample"
    source_directory.mkdir()
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", tmp_path)

    def render(start, end, core):
        source = source_directory / f"{start}.png"
        raw.crop((0, start, 360, end)).save(source)
        x1, y1, x2, y2 = mask["bbox"]
        region = (x1, y1 - start, x2, y2 - start)
        x1, y1, x2, y2 = record
        evidence = {"id": "box", **_region((x1, y1 - start, x2, y2 - start))}
        obj = _auto_object("one", region, "ĐỢI Ở ĐÂY.", shift_bubble_mask(mask, dy=-start),
                           verified_balloon=True, source_boxes=["box"])
        if start:
            obj["seam_owner"] = {"page": 0, "id": "one"}
        page = {"original": str(source), "clean": "clean.png", "text_objects": [obj], "boxes": [evidence],
                "stitch_core": {"source_y1": start, "core_source_y1": core[0], "core_source_y2": core[1]}}
        clean = base.crop((0, start, 360, end))
        output = clean.copy()
        req = RenderRequest(chapter_id="sample", page_index=0, translations={})
        count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req))
        assert count == 1 and _ink_mask(clean, output).any()
        return np.asarray(output)

    top = render(0, first, (0, 163))
    bottom = render(second, 330, (163, 330))
    assert np.array_equal(top[second:first], bottom[:first-second])


def test_legacy_detector_requires_source_ink_and_manual_boxes_render():
    base = Image.new("RGB", (440, 340), "white")
    auto = (28, 24, 240, 124)
    manual = (125, 173, 413, 262)
    page = {"boxes": [{"origin": "detector", **_region(auto)}, {"manual": True, **_region(manual)}]}
    req = RenderRequest(chapter_id="sample", page_index=0, translations={"0": "ĐỪNG ĐI.", "1": "CỨ ĐI."})
    output = base.copy()
    issues = {}
    count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req),
                                               layout_issue_types=issues)
    assert count == 1 and issues == {"0": "placement_unverified"}
    assert not _ink_mask(base, output)[:auto[3]].any()


def test_auto_art_ink_pairs_fill_and_outline_over_mixed_tones():
    backgrounds = np.array([.02, .15, .55, .98])
    fill, outline = _art_ink([(105, 92, 87)], backgrounds, (93, 58, 50))
    assert outline is not None and _contrast(fill[0], outline) >= 4.5
    assert np.maximum(_local_contrast(fill[0], backgrounds),
                      _local_contrast(outline, backgrounds)).min() >= 4.5


def test_local_ink_diff_keeps_stroke_glow_and_caption_pixels():
    base = Image.new("RGB", (430, 350), "#7898b2")
    out = base.copy()
    stats = {}
    render_text_in_box(out, "ĐỪNG QUAY LẠI!", (65, 75, 360, 270), fill="#ffffff",
                       font_size=42, stroke_width=4, stroke_color="#000000", is_bold=True,
                       bg_color="#534251", corner_radius=20, stats=stats, adapt_contrast=False,
                       color_profile={"glow_color": "#ffff66", "glow_radius": 7})
    localized, _ = _render_diff(base, out, stats)
    assert np.array_equal(localized, _ink_mask(base, out))


def test_auto_title_uses_space_in_a_verified_card_for_legibility():
    base = Image.new("RGB", (460, 430), "white")
    ImageDraw.Draw(base).rectangle((30, 30, 430, 399), outline="black", width=3)
    rect = (95, 150, 365, 245)
    obj = _auto_object("title", rect, "Shingen Yamazaki\nThủ lĩnh băng Yamazaki", _contour(base, rect),
                       container="box", typography_role="title", source_cap_px=14,
                       style={"fontSize": "auto"})
    out = base.copy()
    count, verdicts = _render(out, [obj])
    assert count == 1 and verdicts["title"]["size"] >= 28
    assert not np.any(np.asarray(base)[_ink_mask(base, out)] == 0)


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


def test_verified_balloon_uses_clearance_for_each_row_of_accented_ink():
    text = "AI ĐÓ ĐẾN!"
    width, height = 246, 48
    bitmap = Image.new("L", (width, height))
    draw = ImageDraw.Draw(bitmap)
    font = get_font_object(str(DEFAULT_FONT), 37)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - right + left) // 2 - left, (height - bottom + top) // 2 - top),
              text, font=font, fill=255)
    contour = np.zeros((height, width), np.uint8)
    for row, values in enumerate(np.asarray(bitmap)):
        ink = np.flatnonzero(values)
        if ink.size:
            contour[row, ink[0]:ink[-1] + 1] = 1
    contour = cv2.dilate(contour, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    rows = []
    for band in np.split(contour, 24):
        columns = np.nonzero(band)[1]
        rows.append([[int(columns.min()), int(columns.max()) + 1]] if columns.size else [])
    bounds = (0, 0, width, height)
    mask = dict(kind="bubble", bbox=bounds, rows=rows, cx=width // 2, cy=height // 2)
    canvas = Image.new("RGB", (width, height), "white")

    plan = compose_text(canvas, text, bounds, str(DEFAULT_FONT), preferred=37, floor=37,
                        stroke=0, bold=False, mask=mask, verified_balloon=True)
    assert plan is not None and plan.size == 37 and len(plan.lines) == 1
    x, y, words = plan.lines[0]
    ImageDraw.Draw(canvas).text((x, y), words, font=font, fill="black")
    assert np.all(contour[np.asarray(canvas)[..., 0] < 255])


def test_render_does_not_touch_preserved_art_partly_inside_a_text_box(monkeypatch, tmp_path):
    base = _oval()
    region = (80, 100, 355, 205)
    obj = _auto_object("one", region, "ĐỪNG QUAY LẠI.", _contour(base, region),
                       verified_balloon=True, style={"fontSize": "auto"})
    protected = _region((120, 110, 195, 198))
    raw_path = tmp_path / "sample" / "0.png"
    raw_path.parent.mkdir()
    raw = base.copy()
    ImageDraw.Draw(raw).text((205, 125), "DON'T TURN", fill="black", font_size=24)
    raw.save(raw_path)
    monkeypatch.setattr(render_commit._app_config, "RAW_DIR", tmp_path)
    page = {"original": str(raw_path), "text_objects": [obj], "preserve_regions": [protected]}
    req = RenderRequest(chapter_id="sample", page_index=0, translations={})
    output = base.copy()
    count, _ = render_commit._render_snapshot(output, req, page, {}, render_commit._request_style_maps(req))
    assert count == 1
    assert not _ink_mask(base, output)[110:198, 120:195].any()


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
