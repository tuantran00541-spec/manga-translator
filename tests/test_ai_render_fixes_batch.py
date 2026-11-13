from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

import app.ai_mode.job as ai_job
from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings, RENDER_RESTORE_ATTEMPTS
from app.ai_mode.seams import (
    drop_overlapping_letters,
    join_stacked_lines,
    set_letter_bounds,
    sync_seam_mirrors,
)
from app.ai_providers import PROVIDERS
from app.detector.bubble_mask import MASK_ROWS, derive_bubble_mask
from app.image_io import encode_mask
from app.mask_store import decode_mask_value
from app.render.page_renderer import RenderRequest, _ink_mask, render_text_objects
from app.render.source_size import source_ink_hex
from app.render.text_renderer import render_text_in_box
from app.text_objects import ensure_page_text_objects
from app.translation.vision import (
    _line_is_bare_scream,
    _line_is_english,
    _salvage_translation_entries,
    _salvage_translations,
)

FONT = "dialogue.mac-dinh-3"


def _mask_from_widths(widths: list[int], bbox: tuple[int, int, int, int], kind: str = "bubble") -> dict:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rows = []
    half = (x2 - x1) // 2
    for w in widths:
        lo = max(0, half - w // 2)
        hi = min(x2 - x1, lo + w)
        rows.append([[lo, hi]] if hi > lo else [])
    return {"bbox": [x1, y1, x2, y2], "cx": cx, "cy": cy, "rows": rows, "kind": kind}


def test_render_one_restores_on_final_attempt_and_clears_translation(monkeypatch):
    from app.dependencies import pipeline
    import app.routers.export as export_router
    import app.routers.render_commit as render_commit

    region = {"x1": 10, "y1": 20, "x2": 60, "y2": 40}
    manifest = {
        "pages": [
            {
                "text_objects": [
                    {
                        "id": "text_ok",
                        "region": region,
                        "translation": "Được",
                        "auto_translation": "Được",
                    },
                    {
                        "id": "text_long",
                        "region": region,
                        "translation": "Rất dài",
                        "auto_translation": "Rất dài",
                    },
                ]
            }
        ]
    }
    calls, preserved = [], []

    def render(req):
        calls.append(len(calls) + 1)
        if len(calls) <= RENDER_RESTORE_ATTEMPTS + 1:
            return {"skipped_objects": ["text_long"]}
        return {}

    monkeypatch.setattr(render_commit, "render_page", render)
    monkeypatch.setattr(export_router, "_render_request_from_page", lambda chapter_id, index, page: (index, page))
    monkeypatch.setattr(
        pipeline,
        "preserve_and_reinpaint",
        lambda chapter_id, index, regions: preserved.append((index, regions)),
        raising=False,
    )

    job = AIModeJob(
        job_id="j1",
        settings=AIModeSettings(url="https://example.com", provider="openai", model="m"),
        chapter_id="a1b2c3d6",
        stage="render",
        stages={"render": {"done": 0, "total": 0, "detail": ""}},
    )
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    monkeypatch.setattr(runner, "_ensure_objects", lambda index: None)

    asyncio.run(runner._render_one(0))

    assert len(preserved) == RENDER_RESTORE_ATTEMPTS + 1
    assert len(calls) == RENDER_RESTORE_ATTEMPTS + 2
    restored_obj = manifest["pages"][0]["text_objects"][1]
    assert restored_obj["translation"] == ""
    assert "auto_translation" not in restored_obj
    assert restored_obj["translation_attempted"] is True


def test_export_restore_skipped_and_rerender_loops_and_clears_translation(monkeypatch, tmp_path):
    import app.config as config
    import app.dependencies as deps
    import app.manifest_utils as manifests
    import app.routers.export as export_router

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(manifests, "PROCESSED_DIR", tmp_path)
    chapter_id = "e1f2a3b4"
    chap_dir = tmp_path / chapter_id
    chap_dir.mkdir(parents=True)
    manifest = {
        "pages": [
            {
                "text_objects": [
                    {
                        "id": "text_long",
                        "region": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                        "translation": "Dài",
                        "auto_translation": "Dài",
                    }
                ]
            }
        ]
    }
    (chap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    preserved, rerender_calls = [], []

    monkeypatch.setattr(
        deps.pipeline,
        "preserve_and_reinpaint",
        lambda cid, idx, regs: preserved.append((idx, regs)),
        raising=False,
    )

    def fake_render(req):
        rerender_calls.append(req.translations)
        if len(rerender_calls) == 1:
            return {"skipped_objects": ["text_long"]}
        return {"skipped_objects": []}

    monkeypatch.setattr(export_router, "render_page", fake_render)
    restored, rem, err = export_router._restore_skipped_and_rerender(chapter_id, 0, ["text_long"])
    assert err is None
    assert rem == []
    assert restored == 2
    saved = json.loads((chap_dir / "manifest.json").read_text(encoding="utf-8"))
    obj = saved["pages"][0]["text_objects"][0]
    assert obj["translation"] == ""
    assert "auto_translation" not in obj
    assert rerender_calls[0].get("text_long", "") == ""


def test_set_letter_bounds_preserves_multigroup_rect_and_cuts_double_lobe_only():
    letters = np.zeros((160, 100), dtype=np.uint8)
    letters[10:30, 20:80] = 255
    letters[120:145, 20:80] = 255
    rect_page = {
        "boxes": [
            {
                "id": "b_rect",
                "x1": 0,
                "y1": 0,
                "x2": 100,
                "y2": 160,
                "letters_mask": encode_mask(letters),
                "bubble_mask": None,
            }
        ],
        "text_objects": [
            {
                "id": "text_rect",
                "source_boxes": ["b_rect"],
                "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 160},
            }
        ],
    }
    set_letter_bounds(rect_page)
    assert rect_page["text_objects"][0]["letter_bounds"] == {"x1": 20, "y1": 10, "x2": 80, "y2": 145}

    widths = [80] * 8 + [20] * 4 + [80] * 12
    pinched = _mask_from_widths(widths, (0, 0, 100, 160), kind="bubble")
    lobe_page = {
        "boxes": [
            {
                "id": "b_lobe",
                "x1": 0,
                "y1": 0,
                "x2": 100,
                "y2": 160,
                "letters_mask": encode_mask(letters),
                "bubble_mask": pinched,
            }
        ],
        "text_objects": [
            {
                "id": "text_lobe",
                "source_boxes": ["b_lobe"],
                "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 160},
            }
        ],
    }
    set_letter_bounds(lobe_page)
    assert lobe_page["text_objects"][0]["letter_bounds"] == {"x1": 20, "y1": 10, "x2": 80, "y2": 30}


def test_join_and_overlap_merge_masks_and_sync_preserves_merged_metadata():
    m1 = np.zeros((30, 80), dtype=np.uint8)
    m1[5:25, 10:70] = 255
    m2 = np.zeros((30, 80), dtype=np.uint8)
    m2[5:25, 10:70] = 255
    small_bmask = _mask_from_widths([60] * MASK_ROWS, (10, 10, 90, 40), kind="bubble")

    page_join = {
        "boxes": [
            {
                "id": "b1",
                "x1": 10,
                "y1": 10,
                "x2": 90,
                "y2": 40,
                "text": "Line 1",
                "letters_mask": encode_mask(m1),
                "bubble_mask": small_bmask,
            },
            {
                "id": "b2",
                "x1": 10,
                "y1": 42,
                "x2": 90,
                "y2": 72,
                "text": "Line 2",
                "letters_mask": encode_mask(m2),
            },
        ],
        "text_objects": [
            {
                "id": "text_1",
                "box_id": "b1",
                "source_box_ids": ["b1"],
                "auto_generated": True,
                "region": {"x1": 10, "y1": 10, "x2": 90, "y2": 40},
                "letter_bounds": {"x1": 20, "y1": 15, "x2": 80, "y2": 35},
                "letters_mask": encode_mask(m1),
                "bubble_mask": small_bmask,
            },
            {
                "id": "text_2",
                "box_id": "b2",
                "source_box_ids": ["b2"],
                "auto_generated": True,
                "region": {"x1": 10, "y1": 42, "x2": 90, "y2": 72},
                "letter_bounds": {"x1": 20, "y1": 47, "x2": 80, "y2": 67},
                "letters_mask": encode_mask(m2),
            },
        ],
    }
    assert join_stacked_lines({"pages": [page_join]}, [0]) == 1
    top = page_join["text_objects"][0]
    assert top["region"] == {"x1": 10, "y1": 10, "x2": 90, "y2": 72}
    assert "bubble_mask" not in top, "undersized 1-line bubble mask must be cleared"
    merged_arr = decode_mask_value(top["letters_mask"])
    assert merged_arr is not None and merged_arr.shape == (62, 80)
    assert merged_arr[5:25, 10:70].any() and merged_arr[37:57, 10:70].any()

    # Overlap merge sets overlap_merged and ensure_page_text_objects must not clobber merged masks.
    ov_page = {
        "boxes": [
            {
                "id": "o1",
                "x1": 10,
                "y1": 10,
                "x2": 90,
                "y2": 40,
                "text": "Hello",
                "letters_mask": encode_mask(m1),
                "bubble_mask": small_bmask,
            },
            {"id": "o2", "x1": 10, "y1": 20, "x2": 90, "y2": 50, "text": "Hello"},
        ],
        "text_objects": [
            {
                "id": "t_o1",
                "box_id": "o1",
                "source_box_ids": ["o1"],
                "auto_generated": True,
                "translation": "Xin chào bạn",
                "region": {"x1": 10, "y1": 10, "x2": 90, "y2": 40},
                "letter_bounds": {"x1": 10, "y1": 10, "x2": 90, "y2": 40},
                "letters_mask": encode_mask(m1),
                "bubble_mask": small_bmask,
            },
            {
                "id": "t_o2",
                "box_id": "o2",
                "source_box_ids": ["o2"],
                "auto_generated": True,
                "translation": "Xin chào bạn",
                "region": {"x1": 10, "y1": 20, "x2": 90, "y2": 50},
                "letter_bounds": {"x1": 10, "y1": 20, "x2": 90, "y2": 50},
                "letters_mask": encode_mask(m2),
            },
        ],
    }
    dropped = drop_overlapping_letters({"pages": [ov_page]})
    assert dropped == [(0, "t_o2")]
    merged_obj = ov_page["text_objects"][0]
    assert merged_obj["overlap_merged"] is True
    assert merged_obj["region"] == {"x1": 10, "y1": 10, "x2": 90, "y2": 50}
    ensure_page_text_objects(ov_page)
    assert merged_obj["region"] == {"x1": 10, "y1": 10, "x2": 90, "y2": 50}
    assert decode_mask_value(merged_obj["letters_mask"]).shape == (40, 80)


def test_sync_seam_mirrors_keeps_local_letters_mask_on_large_page_shift():
    local = np.zeros((40, 80), dtype=np.uint8)
    local[10:30, 15:65] = 255
    owner_page = {
        "source_page": "p1",
        "stitch_core": {"source_y1": 0, "core_source_y1": 0, "core_source_y2": 1000},
        "text_objects": [
            {
                "id": "text_seam",
                "translation": "Câu thoại",
                "region": {"x1": 10, "y1": 900, "x2": 90, "y2": 940},
                "letters_mask": encode_mask(local),
            }
        ],
    }
    mirror_page = {
        "source_page": "p1",
        "stitch_core": {"source_y1": 800, "core_source_y1": 800, "core_source_y2": 1600},
        "text_objects": [
            {"id": "text_seam_m", "region": {"x1": 10, "y1": 100, "x2": 90, "y2": 140}}
        ],
    }
    sync_seam_mirrors({"pages": [owner_page, mirror_page]})
    mirror_letters = decode_mask_value(mirror_page["text_objects"][0].get("letters_mask"))
    assert mirror_letters is not None and mirror_letters.any()
    assert np.array_equal(mirror_letters > 0, local > 0)


def test_page_renderer_shifted_rerenders_cleanly_over_bgcolor_and_skips_unresolvable_collision():
    # Background with diagonal texture; two identical short boxes stack cleanly via shifted/stacked.
    img = Image.new("RGB", (400, 300), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for x in range(0, 400, 12):
        draw.line([(x, 0), (x, 300)], fill=(180, 210, 240), width=2)
    base = img.copy()

    # Use shifted on a vertically tall box where middle overlaps previous placed box.
    # Container="free" with large region leaves room for shifted without hitting cand_frame top/bottom.
    a = {
        "id": "a",
        "translation": "Một",
        "region": {"x1": 50, "y1": 60, "x2": 220, "y2": 100},
        "container": "free",
        "style": {"font": FONT, "bgColor": "#ffffff", "color": "#000000"},
    }
    b = {
        "id": "b",
        "translation": "Hai",
        "region": {"x1": 50, "y1": 75, "x2": 220, "y2": 200},
        "container": "free",
        "style": {"font": FONT, "bgColor": "#ffffff", "color": "#000000"},
    }
    req = RenderRequest(chapter_id="c1", page_index=0, translations={"a": "Một", "b": "Hai"})
    verdicts: list[dict] = []
    out = img.copy()
    render_text_objects(out, req, [a, b], *({} for _ in range(10)), verdicts=verdicts)
    assert verdicts[1]["guard"] in ("shifted", "stacked")
    # Every changed pixel from b below y=110 must sit within a single connected box (no split bg_color tear).
    # Verify background stripes between y=102..112 stay uncorrupted (or clean gap before shifted b).
    band = np.asarray(out)[102:105, 50:220]
    band_base = np.asarray(base)[102:105, 50:220]
    if verdicts[1]["guard"] == "shifted":
        assert verdicts[1]["shift_dy"] != 0
        # Above shifted b's bg_color backdrop (y=102..105 is the gap), original stripe texture remains untouched.
        assert np.array_equal(band, band_base)

    # Lateral stacked fallback: blocker spans vertical middle inside cand_frame, forcing left_of/right_of.
    wide_mask = _mask_from_widths([280] * MASK_ROWS, (20, 170, 300, 220), kind="rect")
    blk = {
        "id": "blk",
        "translation": "GIỮA\nGIỮA\nGIỮA\nGIỮA",
        "container": "box",
        "region": {"x1": 120, "y1": 120, "x2": 200, "y2": 225},
        "style": {"font": FONT, "fontSize": 18, "bgColor": "#ffffff", "color": "#000000"},
    }
    lat = {
        "id": "lat",
        "translation": "Trái",
        "region": {"x1": 20, "y1": 170, "x2": 300, "y2": 220},
        "container": "bubble",
        "bubble_mask": wide_mask,
        "style": {"font": FONT, "color": "#000000"},
    }
    out_lat = base.copy()
    v_lat: list[dict] = []
    render_text_objects(out_lat, RenderRequest(chapter_id="cl", page_index=0, translations={"blk": "GIỮA\nGIỮA\nGIỮA\nGIỮA", "lat": "Trái"}),
                        [blk, lat], *({} for _ in range(10)), verdicts=v_lat)
    lat_verdict = next(item for item in v_lat if item["id"] == "lat")
    assert lat_verdict["guard"] == "stacked"
    # For unresolvable tight box collision inside a constrained bubble frame, second object
    # must be skipped with review=True (never pasting colliding ink).
    small_mask = _mask_from_widths([60] * MASK_ROWS, (250, 50, 320, 86), kind="bubble")
    tiny_a = {
        "id": "ta",
        "translation": "ABCD EFGH",
        "region": {"x1": 250, "y1": 50, "x2": 320, "y2": 86},
        "container": "bubble",
        "bubble_mask": small_mask,
        "style": {"font": FONT, "color": "#000000"},
    }
    tiny_b = {
        "id": "tb",
        "translation": "WXYZ MNOP",
        "region": {"x1": 250, "y1": 50, "x2": 320, "y2": 86},
        "container": "bubble",
        "bubble_mask": small_mask,
        "style": {"font": FONT, "color": "#000000"},
    }
    only_a = base.copy()
    render_text_objects(only_a, RenderRequest(chapter_id="c2", page_index=0, translations={"ta": "ABCD EFGH"}),
                        [tiny_a], *({} for _ in range(10)))
    out2 = base.copy()
    v2: list[dict] = []
    req2 = RenderRequest(chapter_id="c2", page_index=0, translations={"ta": "ABCD EFGH", "tb": "WXYZ MNOP"})
    render_text_objects(out2, req2, [tiny_a, tiny_b], *({} for _ in range(10)), verdicts=v2)
    assert len(v2) == 2
    assert v2[1]["guard"] == "skipped_collision"
    assert v2[1]["review"] is True
    assert np.array_equal(np.asarray(out2), np.asarray(only_a)), "colliding text must not be pasted"


def test_page_renderer_hex_lettering_color_precedence_and_spills_false_in_mask():
    img = Image.new("RGB", (320, 260), "black")
    ImageDraw.Draw(img).ellipse([20, 20, 300, 240], fill="white")
    bmask = derive_bubble_mask(np.asarray(img)[:, :, ::-1].copy(), (20, 20, 300, 240))
    assert bmask is not None
    obj = {
        "id": "h1",
        "translation": "Chào các bạn nhỏ",
        "region": {"x1": 20, "y1": 20, "x2": 300, "y2": 240},
        "letter_bounds": {"x1": 140, "y1": 80, "x2": 175, "y2": 180},
        "bubble_mask": bmask,
        "container": "bubble",
        "lettering_color": "#cc0000",
        "ocr_text_color": "#0000cc",
        "style": {"font": FONT},
    }
    req = RenderRequest(chapter_id="hc", page_index=0, translations={"h1": obj["translation"]})
    verdicts: list[dict] = []
    out = img.copy()
    render_text_objects(out, req, [obj], *({} for _ in range(10)), verdicts=verdicts)
    assert verdicts[0]["frame_source"] == "mask"
    assert verdicts[0]["spills_letter_box"] is False
    arr = np.asarray(out)
    ink = _ink_mask(img, out)
    assert ink.any()
    med_rgb = np.median(arr[ink], axis=0)
    assert med_rgb[0] > med_rgb[2] + 60, f"expected red hex lettering_color over blue ocr_text_color, got {med_rgb}"


def test_text_renderer_auto_fill_on_dark_bgcolor_and_enlarge_inscribed_fallback():
    white_img = Image.new("RGB", (260, 120), "white")
    rendered = render_text_in_box(
        white_img.copy(),
        "Sáng",
        (20, 20, 220, 90),
        font_name=FONT,
        fill="auto",
        bg_color="#101010",
    )
    arr = np.asarray(rendered)
    # Interior text pixels inside the dark caption box must be white (high luminance), not dark.
    crop = arr[25:85, 25:215]
    bright_text_pixels = np.count_nonzero((crop[..., 0] > 200) & (crop[..., 1] > 200) & (crop[..., 2] > 200))
    assert bright_text_pixels > 30

    # Enlarge=True with dense text inside an oval should fall back to readable size in inscribed rect
    # instead of raising below_readable_floor when floor alone doesn't fit.
    bubble_img = Image.new("RGB", (300, 220), "black")
    ImageDraw.Draw(bubble_img).ellipse([20, 20, 280, 200], fill="white")
    bmask = derive_bubble_mask(np.asarray(bubble_img)[:, :, ::-1].copy(), (20, 20, 280, 200))
    stats: dict = {}
    out = render_text_in_box(
        bubble_img.copy(),
        "Đây là một đoạn thoại khá dài cần tự động thu xuống cỡ đọc được thay vì lỗi",
        (20, 20, 280, 200),
        font_name=FONT,
        container="bubble",
        enlarge=True,
        source_cap_px=24,
        bubble_mask=bmask,
        stats=stats,
    )
    assert stats["size"] >= 11
    assert _ink_mask(bubble_img, out).any()

    # Fixed oversized font_size that overflows horizontally must fall back instead of clipping.
    ov_stats: dict = {}
    render_text_in_box(
        white_img.copy(),
        "SIÊUDÀI",
        (10, 10, 160, 110),
        font_name=FONT,
        font_size=60,
        stroke_width=3,
        stats=ov_stats,
    )
    assert ov_stats["size"] < 60
    assert ov_stats["ink_box"][2] <= 160


def test_source_ink_hex_handles_light_text_on_dark_and_mixed_panel_border():
    dark_img = np.zeros((100, 160, 3), dtype=np.uint8)
    # Bright cyan-ish text strokes (>5px thick) on black panel -> BGR (240, 200, 20)
    cv2.putText(dark_img, "HEY!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (240, 200, 20), 6)
    hx = source_ink_hex(dark_img, (0, 0, 160, 100))
    assert hx is not None
    r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    assert b > 180 and g > 150 and r < 80, f"expected light cyan-blue BGR->RGB hex, got {hx}"

    # Mixed border (left 50% black panel, right 50% white bubble with dark red text).
    mixed = np.full((120, 200, 3), 255, dtype=np.uint8)
    mixed[:, :95] = 0
    cv2.putText(mixed, "AB", (105, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 220), 6)
    hx_mix = source_ink_hex(mixed, (0, 0, 200, 120))
    assert hx_mix is not None
    mr, mg, mb = int(hx_mix[1:3], 16), int(hx_mix[3:5], 16), int(hx_mix[5:7], 16)
    assert mr > 160 and mb < 80, f"expected red dark-fg text hex on mixed border, got {hx_mix}"


def test_vision_salvage_entries_preserves_metadata_and_language_filters_protect_vietnamese():
    truncated = (
        '{"translations":[{"id":"b1","translated_text":"Khoan đã!","role":"shout",'
        '"container":"spiky","color":"red","enlarge":true,"review":false,"source":"待"},'
        '{"id":"b2","translated_text":"cắt ngang chưa đóng'
    )
    entries = _salvage_translation_entries({"message": {"content": truncated}}, {"b1", "b2"})
    assert len(entries) == 1
    assert entries[0]["role"] == "shout"
    assert entries[0]["container"] == "spiky"
    assert entries[0]["color"] == "red"
    assert entries[0]["enlarge"] is True
    assert _salvage_translations({"message": {"content": truncated}}, {"b1", "b2"}) == {"b1": "Khoan đã!"}

    assert _line_is_english("CON TO") is False
    assert _line_is_english("BA ME") is False
    assert _line_is_english("IN AN") is False
    assert _line_is_english("Were you talking to me?") is True
    assert _line_is_english("GO TO ME") is True

    assert _line_is_bare_scream("ÁÁÁÁ!!!") is True
    assert _line_is_bare_scream("ừừừ") is False
