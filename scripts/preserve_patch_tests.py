from pathlib import Path


def rep(path, old, new, count=1):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"{path}: expected {count} x {old[:90]!r}, found {found}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")


def splice(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{path}: missing start marker {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{path}: missing end marker {end_marker!r}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
# Focused synthetic contract tests.
Path("tests/test_ai_repair_controls.py").write_text(r'''from contextlib import nullcontext

import numpy as np
import pytest
from PIL import Image

from app.ocr.service import OCRService
from app.pipeline import ChapterPipeline
from app.region_policy import geometry_center_in_regions, subtract_regions_from_mask
from app.render.identity import render_input_signature
from app.routers.editor import _regions_to_repaint_mask
from app.routers.render_commit import _render_snapshot, _request_style_maps
from app.schemas import RegionModel, RenderRequest
from app.text_objects import ensure_page_text_objects


def test_preserve_center_ownership_not_any_overlap():
    regions = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200}]
    assert geometry_center_in_regions({"x1": 120, "y1": 120, "x2": 180, "y2": 180}, regions)
    assert not geometry_center_in_regions({"x1": 80, "y1": 120, "x2": 110, "y2": 180}, regions)


def test_preserve_suppresses_auto_text_object():
    page = {
        "preserve_regions": [{"x1": 0, "y1": 0, "x2": 60, "y2": 60}],
        "boxes": [
            {"id": "box_keep", "x1": 10, "y1": 10, "x2": 40, "y2": 40, "ocr_eligible": True},
            {"id": "box_story", "x1": 100, "y1": 100, "x2": 160, "y2": 150, "ocr_eligible": True},
        ],
        "text_objects": [],
    }
    created, changed = ensure_page_text_objects(page)
    assert changed and created == 1
    assert page["text_objects"][0]["source_boxes"] == ["box_story"]


def test_preserve_clips_repaint_mask():
    mask = np.full((80, 100), 255, np.uint8)
    result = subtract_regions_from_mask(mask, [{"x1": 10, "y1": 20, "x2": 30, "y2": 50}])
    assert np.count_nonzero(result[20:50, 10:30]) == 0
    assert result[0, 0] == 255


def test_ai_rectangle_repaint_mask_is_bounded():
    mask = _regions_to_repaint_mask((100, 120, 3), [RegionModel(x1=10, y1=20, x2=30, y2=40)])
    assert mask.shape == (100, 120)
    assert int(np.count_nonzero(mask)) == 400
    with pytest.raises(ValueError, match="exceeds page dimensions"):
        _regions_to_repaint_mask((100, 120, 3), [{"x1": 0, "y1": 0, "x2": 121, "y2": 10}])


def test_ocr_plan_skips_preserve_and_process_required(monkeypatch):
    import app.ocr.service as module
    manifest = {"pages": [
        {"skipped": False, "process_required": False,
         "preserve_regions": [{"x1": 0, "y1": 0, "x2": 60, "y2": 60}],
         "boxes": [
             {"id": "box_keep", "x1": 10, "y1": 10, "x2": 40, "y2": 40, "ocr_eligible": True},
             {"id": "box_story", "x1": 100, "y1": 100, "x2": 160, "y2": 150, "ocr_eligible": True},
         ]},
        {"skipped": False, "process_required": True, "preserve_regions": [],
         "boxes": [{"id": "box_stale", "x1": 0, "y1": 0, "x2": 20, "y2": 20, "ocr_eligible": True}]},
    ]}
    monkeypatch.setattr(module, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "load_manifest_raw", lambda _: manifest)
    assert OCRService(object(), object()).plan_chapter("abcd1234") == [(0, "box_story")]


def test_render_snapshot_suppresses_preserve_text(monkeypatch):
    import app.routers.render_commit as module
    seen = []
    monkeypatch.setattr(module, "render_text_objects", lambda _i, _r, objs, *_a: seen.extend(o["id"] for o in objs) or len(objs))
    page = {"preserve_regions": [{"x1": 0, "y1": 0, "x2": 70, "y2": 70}], "text_objects": [
        {"id": "protected", "region": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "translation": "NO"},
        {"id": "story", "region": {"x1": 100, "y1": 100, "x2": 160, "y2": 150}, "translation": "OK"},
    ]}
    req = RenderRequest(chapter_id="abcd1234", page_index=0, translations={"protected": "NO", "story": "OK"})
    assert _render_snapshot(Image.new("RGB", (200, 200), "white"), req, page, {}, _request_style_maps(req)) == 1
    assert seen == ["story"]


def test_render_identity_changes_with_preserve_policy(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(path)
    manifest = {"pages": [{"original": str(path), "clean": None, "skipped": False,
                            "process_required": False, "preserve_regions": [], "text_objects": []}]}
    before = render_input_signature(manifest, 0)
    manifest["pages"][0]["preserve_regions"] = [{"x1": 1, "y1": 1, "x2": 10, "y2": 10}]
    assert before != render_input_signature(manifest, 0)


def test_repaint_refuses_skipped_page(monkeypatch):
    import app.pipeline_editing as module
    manifest = {"pages": [{"skipped": True, "process_required": False, "original": "unused.png", "boxes": []}]}
    monkeypatch.setattr(module, "get_page_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "load_manifest_raw", lambda _: manifest)
    with pytest.raises(ValueError, match="Cannot repaint a skipped page"):
        ChapterPipeline.__new__(ChapterPipeline).repaint_mask("abcd1234", 0, np.ones((10, 10), np.uint8) * 255)


def test_skip_unskip_requires_reprocessing(monkeypatch):
    import app.pipeline as module
    page = {"skipped": False, "process_required": False, "clean": "clean.png",
            "boxes": [{"id": "box_1"}], "preserve_regions": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}],
            "clean_revision": 2}
    manifest = {"pages": [page]}
    monkeypatch.setattr(module, "get_page_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "load_manifest_raw", lambda _: manifest)
    monkeypatch.setattr(module, "save_manifest_raw", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "invalidate_page_render", lambda *_a, **_k: None)
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._sync_output_dir = lambda *_a, **_k: None
    assert pipeline.mark_skipped("abcd1234", [0], True)["pages"][0]["process_required"] is False
    current = pipeline.mark_skipped("abcd1234", [0], False)["pages"][0]
    assert current["process_required"] is True and current["preserve_regions"] == page["preserve_regions"]


def test_preserve_change_requires_reprocessing(monkeypatch):
    import app.routers.chapters as module
    page = {"original": "orig.png", "skipped": False, "process_required": False,
            "clean": "clean.png", "boxes": [], "text_objects": [], "preserve_regions": [], "clean_revision": 3}
    manifest = {"chapter_id": "abcd1234", "pages": [page]}
    monkeypatch.setattr(module, "get_page_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "load_manifest_raw", lambda _: manifest)
    monkeypatch.setattr(module, "save_manifest_raw", lambda *_a, **_k: None)
    monkeypatch.setattr(module.pipeline, "_sync_output_dir", lambda *_a, **_k: None)
    current = module._set_page_preserve_regions("abcd1234", 0, [RegionModel(x1=1, y1=1, x2=10, y2=10)])["pages"][0]
    assert current["process_required"] is True and current["clean"] is None
''', encoding="utf-8")

print("preserve-region design patch applied")
