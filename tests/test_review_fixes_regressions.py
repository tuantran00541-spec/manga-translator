from __future__ import annotations

import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.ai_mode.seams import join_stacked_lines
from app.detector.kiuyha_detector import _restore_outline_glyphs
from app.inpaint import lama_inpainter
from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.linework import continue_linework
from app.ocr.service import OCRService
from app.render.composition import _break_costs, _word_spans, compose_text
from app.render.font_metrics import get_font_path
from app.render.font_selection import _title_width_ratio, resolve_object_font
from app.render.page_renderer import render_text_objects
from app.render.placement import source_ink_bounds, user_position
from app.routers.translation import _apply_ai_font_choice
from app.schemas import RenderRequest
from app.text_objects import _can_absorb, _sync_balloon_ocr


def test_user_font_wins_when_ai_font_id_is_auto():
    obj = {"style": {"font": "dialogue.roboto"}, "font_selection_mode": "user"}
    font_id, mode, metadata = resolve_object_font(
        obj,
        ai_font_id="auto",
        ai_font_mode="ai",
    )
    assert (font_id, mode) == ("dialogue.roboto", "user")
    assert metadata["reason"] == "explicit_user_selection"

    _apply_ai_font_choice(obj, {"font_id": "auto", "font_mode": "ai"})
    assert obj["font_selection_mode"] == "user"
    assert obj["style"]["font"] == "dialogue.roboto"


def test_title_width_ratio_handles_empty_letters_mask(monkeypatch):
    from app.render import font_selection

    monkeypatch.setattr(font_selection, "decode_mask_value", lambda _: np.zeros((0, 0), dtype=np.uint8))
    obj = {
        "verified_balloon": True,
        "container": "box",
        "typography_role": "title",
        "source_cap_px": 20,
        "letters_mask": "dummy",
    }
    assert _title_width_ratio(obj) is None


def test_restore_outline_glyphs_handles_empty_letters_without_nan():
    gray = np.full((64, 64), 240, dtype=np.uint8)
    letters = np.zeros((64, 64), dtype=bool)
    protected = np.zeros((64, 64), dtype=bool)
    labels = np.zeros((64, 64), dtype=np.int32)
    stats = np.zeros((1, 5), dtype=np.int32)
    _restore_outline_glyphs(gray, letters, protected, labels, stats, [], (0, 0, 64, 64))
    assert not np.any(letters)


def test_dynamic_lama_max_dim_respects_smaller_max_dim(monkeypatch):
    crop = np.full((420, 420, 3), 240, dtype=np.uint8)
    mask = np.zeros((420, 420), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    monkeypatch.setattr(lama_inpainter, "DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 256)
    assert Inpainter._dynamic_lama_max_dim(crop, mask) == 256
    monkeypatch.setattr(lama_inpainter, "DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 512)
    assert Inpainter._dynamic_lama_max_dim(crop, mask) == 384


def test_continue_linework_supports_grayscale_and_mismatched_shapes():
    truth = np.full((128, 160), 220, dtype=np.uint8)
    cv2.line(truth, (14, 27), (132, 96), 35, 2, cv2.LINE_AA)
    mask = np.zeros(truth.shape[:2], np.uint8)
    mask[43:77, 68:91] = 255
    source = truth.copy()
    source[mask > 0] = 245
    painted = truth.copy()
    painted[mask > 0] = 220

    repaired = continue_linework(source, painted, mask)
    assert repaired.shape == painted.shape
    assert np.abs(truth.astype(np.int16) - repaired).sum() < np.abs(truth.astype(np.int16) - painted).sum()
    assert np.array_equal(repaired[mask == 0], source[mask == 0])

    mismatched = np.zeros((64, 64), dtype=np.uint8)
    assert np.array_equal(continue_linework(mismatched, painted, mask), painted)


def test_source_ink_bounds_supports_grayscale_and_tight_edge_glyphs():
    clean = Image.new("L", (120, 40), 240)
    original = clean.copy()
    draw = ImageDraw.Draw(original)
    # Glyph touches left border (x=0) and spans full height (>90% of box height)
    draw.rectangle((10, 5, 16, 34), fill=20)
    draw.rectangle((10, 18, 95, 24), fill=20)
    record = {"x1": 10, "y1": 5, "x2": 100, "y2": 35}

    anchor = source_ink_bounds(record, original, clean)
    assert anchor is not None
    assert anchor[0] == 10 and anchor[2] >= 90


def test_manual_object_not_locked_out_when_source_boxes_present():
    clean = Image.new("RGB", (300, 200), (255, 255, 255))
    original = clean.copy()
    obj = {
        "id": "manual-1",
        "origin": "manual",
        "manual": True,
        "region": {"x1": 40, "y1": 40, "x2": 110, "y2": 72},
        "translation": "Xin chào!",
        "style": {"fontSize": "24", "strokeWidth": "4", "strokeColor": "#ffffff"},
    }
    req = RenderRequest(chapter_id="ch1", page_index=0, translations={"manual-1": "Xin chào!"})
    verdicts: list[dict] = []
    count = render_text_objects(
        clean,
        req,
        [obj],
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        verdicts=verdicts,
        source_image_loader=lambda: original,
        source_boxes={},
    )
    assert count == 1
    assert not any(v.get("guard") == "placement_outside_bounds" for v in verdicts)


def test_join_stacked_lines_merges_source_boxes_and_skips_only_verified_balloons():
    manifest = {
        "pages": [
            {
                "balloons": [{"id": "bal-1"}],
                "text_objects": [
                    {
                        "id": "obj-balloon",
                        "verified_balloon": True,
                        "region": {"x1": 20, "y1": 20, "x2": 120, "y2": 50},
                        "source_boxes": ["b0"],
                    },
                    {
                        "id": "obj-1",
                        "region": {"x1": 20, "y1": 100, "x2": 140, "y2": 130},
                        "source_boxes": ["b1"],
                        "source_cap_px": 20,
                    },
                    {
                        "id": "obj-2",
                        "region": {"x1": 22, "y1": 136, "x2": 138, "y2": 166},
                        "source_boxes": ["b2"],
                        "source_cap_px": 20,
                    },
                ],
            }
        ]
    }
    joined = join_stacked_lines(manifest, [0])
    assert joined == 1
    objs = {o["id"]: o for o in manifest["pages"][0]["text_objects"]}
    assert "joined_lines" not in objs["obj-balloon"]
    assert objs["obj-1"]["joined_lines"] == ["obj-2"]
    assert objs["obj-1"]["source_boxes"] == ["b1", "b2"]
    assert objs["obj-2"]["joined_into"] == "obj-1"


def test_compose_text_normalizes_nfd_vietnamese_and_leading_quote_forward_word():
    nfd = unicodedata.normalize("NFD", "Tiếng Việt đẹp lắm")
    canvas = Image.new("RGB", (320, 160), (255, 255, 255))
    comp = compose_text(
        canvas,
        nfd,
        (20, 20, 300, 140),
        get_font_path("default"),
        preferred=26,
        floor=14,
        stroke=0,
        bold=False,
    )
    assert comp is not None
    joined = " ".join(line for _, _, line in comp.lines)
    assert joined == unicodedata.normalize("NFC", nfd)

    text, spans, ends = _word_spans('"của tôi')
    costs = _break_costs(text, spans, ends, balloon=True)
    assert costs.get(1) == 0.85


def test_compose_text_avoids_middle_foreign_art_with_blocker_search():
    canvas = Image.new("RGB", (260, 260), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # Dark horizontal foreign art strip (<20% of interior area) across the middle of the bubble
    draw.rectangle((35, 112, 225, 148), fill=(15, 15, 15))
    mask = {
        "kind": "bubble",
        "bbox": [30, 20, 230, 240],
        "cx": 130,
        "cy": 130,
        "rows": [[[0, 200]] for _ in range(24)],
    }
    comp = compose_text(
        canvas,
        "Xin chào!",
        (30, 20, 230, 240),
        get_font_path("default"),
        preferred=24,
        floor=14,
        stroke=0,
        bold=False,
        mask=mask,
        verified_balloon=True,
    )
    assert comp is not None
    # Placed line should avoid the middle dark strip (y in [112, 148])
    for _, line_y, _ in comp.lines:
        assert line_y < 112 or line_y > 148


def test_user_position_respects_geometry_overridden_on_seam_owner():
    seam_obj = {
        "seam_owner": True,
        "region": {"x1": 10, "y1": 10, "x2": 100, "y2": 50},
        "auto_geometry": {"x1": 0, "y1": 0, "x2": 90, "y2": 40},
    }
    assert not user_position(seam_obj)
    seam_obj["geometry_overridden"] = True
    assert user_position(seam_obj)


def test_ocr_group_stale_allows_balloon_fallback_when_fragments_empty(tmp_path):
    orig = tmp_path / "p0.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(orig)
    from app.ocr.service import file_revision

    rev = file_revision(orig)
    page = {
        "original": str(orig),
        "source_revision": 1,
        "boxes": [{"id": "b1", "x1": 10, "y1": 10, "x2": 50, "y2": 40}],
    }
    obj = {
        "id": "obj-1",
        "verified_balloon": True,
        "source_boxes": ["b1"],
        "region": {"x1": 10, "y1": 10, "x2": 50, "y2": 40},
        "ocr_text": "",
        "auto_ocr_text": "",
    }
    from app.ocr.service import active_overlap_signatures

    snapshot = {
        "original_value": str(orig),
        "source_revision": 1,
        "region": dict(obj["region"]),
        "text": "",
        "balloon_sources": ("b1",),
        "follows_ocr": True,
        "signatures": active_overlap_signatures(page, obj["region"], ("b1",)),
    }
    # Even if fallback OCR read non-empty text, expected_synced is "" (from fragments), so NOT stale
    assert not OCRService._group_result_stale(page, obj, snapshot, rev, combined="")


def test_sync_balloon_ocr_empty_members_and_can_absorb_corner_radius():
    obj = {"ocr_text": "", "auto_ocr_text": ""}
    assert _sync_balloon_ocr(obj, []) is False

    reg = {"x1": 0, "y1": 0, "x2": 50, "y2": 30}
    absorb_obj = {
        "region": reg,
        "auto_geometry": dict(reg),
        "shape": "rectangle",
        "ocr_text": "hi",
        "auto_ocr_text": "hi",
        "font_selection_mode": "auto",
        "style": {"cornerRadius": "0", "font": "auto"},
    }
    assert _can_absorb(absorb_obj) is True
