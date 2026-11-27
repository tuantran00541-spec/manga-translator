from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

import app.config as config
import app.manifest_utils as manifest_utils
from app.ai_mode.seams import drop_overlapping_letters
from app.detector.bubble_mask import derive_bubble_mask
from app.pipeline import ChapterPipeline
from app.render.composition import compose_text
from app.render.font_metrics import DEFAULT_FONT
from app.render.page_renderer import render_text_objects
from app.render.placement import user_position
from app.render.text_renderer import render_text_in_box
from app.schemas import RenderRequest


def _ink_mask(base: Image.Image, rendered: Image.Image) -> np.ndarray:
    diff = np.abs(np.asarray(rendered, dtype=np.int16) - np.asarray(base, dtype=np.int16))
    return diff.max(axis=2) > 8


@pytest.mark.parametrize("h_align", ["left", "center", "right"])
@pytest.mark.parametrize("v_align", ["top", "middle", "bottom"])
@pytest.mark.parametrize("is_bold", [False, True])
def test_vietnamese_stacked_diacritics_stay_inside_box_all_alignments(h_align, v_align, is_bold):
    base = Image.new("RGB", (400, 260), "white")
    out = base.copy()
    box = (60, 50, 340, 210)
    text = "Ệ Ộ Ữ Ấ Ể Ợ ĐƯỜNG ĐI KHÔNG DỄ DÀNG!"
    stats: dict = {}
    render_text_in_box(
        out,
        text,
        box,
        font_name="dialogue.mac-dinh-3",
        horizontal_align=h_align,
        vertical_align=v_align,
        is_bold=is_bold,
        stroke_width=3,
        stats=stats,
    )
    ink = _ink_mask(base, out)
    ys, xs = np.nonzero(ink)
    assert xs.size > 0
    assert int(xs.min()) >= box[0] and int(xs.max()) < box[2]
    assert int(ys.min()) >= box[1] and int(ys.max()) < box[3]
    if h_align == "center" and v_align == "middle":
        cx = (int(xs.min()) + int(xs.max())) / 2.0
        cy = (int(ys.min()) + int(ys.max())) / 2.0
        assert abs(cx - (box[0] + box[2]) / 2.0) <= 5.0
        assert abs(cy - (box[1] + box[3]) / 2.0) <= 5.0


def test_fixed_font_size_with_diacritics_and_stroke_never_spills():
    base = Image.new("RGB", (360, 200), "white")
    out = base.copy()
    box = (50, 40, 310, 160)
    render_text_in_box(
        out,
        "Ệ Ộ Ữ Ấ Ể Ợ",
        box,
        font_name="dialogue.mac-dinh-3",
        font_size=28,
        stroke_width=4,
        is_bold=True,
    )
    ink = _ink_mask(base, out)
    ys, xs = np.nonzero(ink)
    assert xs.size > 0
    assert int(xs.min()) >= box[0] and int(xs.max()) < box[2]
    assert int(ys.min()) >= box[1] and int(ys.max()) < box[3]


def test_manual_create_and_move_strip_stale_masks_and_render_in_new_box():
    chapter_id = "deadbeef"
    with TemporaryDirectory() as tmp:
        root = manifest_utils.Path(tmp) if hasattr(manifest_utils, "Path") else __import__("pathlib").Path(tmp)
        proc = root / chapter_id
        proc.mkdir()
        raw_root = root / "raw"
        orig = raw_root / chapter_id / "p.png"
        orig.parent.mkdir(parents=True)
        img = Image.new("RGB", (500, 400), "white")
        img.save(orig)
        manifest_utils.save_manifest_raw(
            proc,
            {
                "schema_version": 3,
                "chapter_id": chapter_id,
                "pages": [
                    {
                        "original": orig.as_posix(),
                        "clean": None,
                        "boxes": [],
                        "text_objects": [],
                        "source_revision": 1,
                        "process_revision": 0,
                        "clean_revision": 0,
                        "render_revision": 0,
                        "artifact_generation": 0,
                    }
                ],
            },
        )
        pipe = ChapterPipeline.__new__(ChapterPipeline)
        with patch.object(manifest_utils, "PROCESSED_DIR", root), patch(
            "app.pipeline_editing.PROCESSED_DIR", root
        ), patch.object(config, "RAW_DIR", raw_root):
            created = pipe.create_text_object(
                chapter_id, 0, "rectangle", {"x1": 40, "y1": 40, "x2": 180, "y2": 120}
            )
            obj = created["pages"][0]["text_objects"][0]
            assert obj.get("origin") == "manual" and obj.get("manual") is True

            # Inject stale bubble mask pointing far away at (300..460, 220..360), then move region.
            manifest = manifest_utils.load_manifest_raw(chapter_id)
            manifest["pages"][0]["text_objects"][0]["bubble_mask"] = {
                "bbox": [300, 220, 460, 360],
                "cx": 380,
                "cy": 290,
                "rows": [[[10, 150]] for _ in range(24)],
                "kind": "bubble",
            }
            manifest_utils.save_manifest_raw(chapter_id, manifest)

            moved = pipe.update_text_object(
                chapter_id,
                0,
                obj["id"],
                {"region": {"x1": 50, "y1": 50, "x2": 220, "y2": 150}, "translation": "XIN CHÀO!"},
            )
            moved_obj = moved["pages"][0]["text_objects"][0]
            assert "bubble_mask" not in moved_obj

        out = img.copy()
        req = RenderRequest(chapter_id=chapter_id, page_index=0, translations={"0": "XIN CHÀO!"})
        count = render_text_objects(out, req, [moved_obj], *({} for _ in range(10)))
        assert count == 1
        ink = _ink_mask(img, out)
        ys, xs = np.nonzero(ink)
        assert xs.size > 0
        assert int(xs.min()) >= 50 and int(xs.max()) < 220
        assert int(ys.min()) >= 50 and int(ys.max()) < 150


def test_compose_text_resilient_to_small_speck_and_elongated_tail():
    base = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(base)
    draw.ellipse((80, 60, 320, 260), fill="white", outline="black", width=3)
    draw.polygon([(220, 240), (260, 240), (310, 440)], fill="white")
    # Small residual 3x3 black speck inside bubble center must not carve a crater
    draw.rectangle((198, 158, 201, 161), fill="black")

    mask = derive_bubble_mask(np.asarray(base)[:, :, ::-1].copy(), (90, 70, 310, 250), seed=(200, 160))
    assert mask["kind"] == "bubble"
    plan = compose_text(
        base,
        "Ta nhất định phải đi ngay bây giờ!",
        tuple(mask["bbox"]),
        str(DEFAULT_FONT),
        preferred=26,
        floor=12,
        stroke=2,
        bold=False,
        mask=mask,
        anchor=(120, 100, 280, 220),
    )
    assert plan is not None and len(plan.lines) >= 1
    avg_y = sum(y for _, y, _ in plan.lines) / len(plan.lines)
    assert avg_y < 270, f"text drifted into tail at y={avg_y}"


def test_short_cjk_unmasked_wide_box_does_not_get_crushed():
    base = Image.new("RGB", (300, 200), "white")
    # Small 25x30 source ink in top-left inside a generous 200x120 unmasked box
    orig = base.copy()
    ImageDraw.Draw(orig).rectangle((60, 45, 85, 75), fill="black")
    obj = {
        "id": "u1",
        "auto_generated": True,
        "region": {"x1": 50, "y1": 40, "x2": 250, "y2": 160},
        "letter_bounds": {"x1": 60, "y1": 45, "x2": 85, "y2": 75},
        "translation": "KHÔNG THỂ TIN ĐƯỢC CHUYỆN NÀY!",
        "container": "box",
        "source_cap_px": 22,
        "style": {"font": "dialogue.mac-dinh-3"},
    }
    req = RenderRequest(chapter_id="c", page_index=0, translations={"u1": obj["translation"]})
    out = base.copy()
    verdicts: list = []
    count = render_text_objects(out, req, [obj], *({} for _ in range(10)), verdicts=verdicts)
    assert count == 1
    assert verdicts[0].get("guard") != "placement_overset"
    ink = _ink_mask(base, out)
    ys, xs = np.nonzero(ink)
    assert xs.size > 0
    assert int(xs.min()) >= 50 and int(xs.max()) < 250


def test_drop_overlapping_letters_syncs_auto_geometry():
    page = {
        "text_objects": [
            {
                "id": "a",
                "auto_generated": True,
                "region": {"x1": 50, "y1": 50, "x2": 150, "y2": 120},
                "letter_bounds": {"x1": 50, "y1": 50, "x2": 150, "y2": 120},
                "auto_geometry": {"x1": 50, "y1": 50, "x2": 150, "y2": 120},
                "translation": "HELLO WORLD TODAY",
            },
            {
                "id": "b",
                "auto_generated": True,
                "region": {"x1": 55, "y1": 55, "x2": 180, "y2": 125},
                "letter_bounds": {"x1": 55, "y1": 55, "x2": 180, "y2": 125},
                "auto_geometry": {"x1": 55, "y1": 55, "x2": 180, "y2": 125},
                "translation": "HELLO WORLD TODAY",
            },
        ]
    }
    dropped = drop_overlapping_letters({"pages": [page]})
    assert len(dropped) == 1
    kept = page["text_objects"][0]
    assert kept["region"] == kept["auto_geometry"]
    assert not user_position(kept)
