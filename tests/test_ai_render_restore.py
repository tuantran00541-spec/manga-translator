from pathlib import Path

import asyncio
import numpy as np

from app.image_io import write_image
from app.optimized_pipeline import OptimizedChapterPipeline


class _FakeInpainter:
    def inpaint(self, image, boxes, *, protected_regions=None):
        return np.full_like(image, 240)

    def inpaint_mask(self, image, mask, *, force_lama=False):
        return image


def test_render_chapter_restores_an_unletterable_object(tmp_path: Path, monkeypatch):
    """Render N6: render_chapter called render_page and ignored the returned
    skipped_objects, so a floor-skipped object kept a cleaned-white hole.
    It now restores the skipped regions and re-renders, as the job flow
    does — the output shows the original pixels, not the clean image."""
    import app.config as config
    import app.dependencies as deps
    import app.manifest_utils as manifests
    import app.pipeline_editing as editing
    import app.routers.export as export_router
    import app.routers.image as image_router
    import app.routers.render_commit as render_commit
    from PIL import Image

    chapter = "e7c3b2a0"
    processed, output, raw = tmp_path / "processed", tmp_path / "output", tmp_path / "raw"
    (processed / chapter).mkdir(parents=True)
    (raw / chapter).mkdir(parents=True)
    for module, name, value in (
        (editing, "PROCESSED_DIR", processed), (manifests, "PROCESSED_DIR", processed),
        (config, "PROCESSED_DIR", processed), (config, "OUTPUT_DIR", output), (config, "RAW_DIR", raw),
        (render_commit, "OUTPUT_DIR", output), (export_router, "OUTPUT_DIR", output),
        (image_router, "OUTPUT_DIR", output), (image_router, "PROCESSED_DIR", processed), (image_router, "RAW_DIR", raw),
    ):
        monkeypatch.setattr(module, name, value)
    original = np.full((200, 300, 3), 30, dtype=np.uint8)
    img_path = raw / chapter / "page.png"
    write_image(img_path, original)
    boxes = [{"id": "b1", "x1": 10, "y1": 10, "x2": 30, "y2": 20, "confidence": 1.0, "manual": True,
              "origin": "manual", "ocr_eligible": True}]
    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._inpainter = _FakeInpainter()
    clean = pipeline._do_reinpaint(processed / chapter, img_path, original, boxes,
                                   manual_mask_posix=None, manual_lama_mask_posix=None, preserve_regions=[])
    long_line = "Một câu dịch rất dài không thể nào vừa trong một ô nhỏ như thế này được " * 6
    manifests.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [{
        "original": img_path.as_posix(), "clean": clean, "boxes": boxes,
        "text_objects": [{"id": "tiny", "region": {"x1": 10, "y1": 10, "x2": 30, "y2": 20},
                          "translation": long_line, "source_boxes": ["b1"], "style": {},
                          "auto_generated": True, "auto_geometry": {"x1": 10, "y1": 10, "x2": 30, "y2": 20}}],
        "preserve_regions": [], "skipped": False, "process_required": False, "source_page": 0, "slice_index": 0,
        "source_revision": 1, "clean_revision": 1, "render_revision": 0, "rendered": False,
    }]})
    monkeypatch.setattr(deps, "pipeline", pipeline)

    result = export_router.render_chapter(chapter, force=True)

    assert result["chapter_render"]["restored_regions"] == 1, result["chapter_render"]
    assert result["chapter_render"]["restore_failures"] == [], result["chapter_render"]
    manifest = manifests.load_manifest_raw(chapter)
    assert image_router._current_rendered_path(chapter, 0, manifest) is not None, "the slice must be rendered"
    with Image.open(output / chapter / "page_000.png") as image:
        crop = np.asarray(image.convert("L").crop((10, 10, 30, 20)))
    assert float(crop.mean()) < 100, (
        f"the skipped region must show the dark original, not the clean image (mean {crop.mean():.1f})")


def test_render_stage_leaves_a_slice_rendered_before_the_chapter_finishes_alone(tmp_path: Path, monkeypatch):
    """A slice rendered while the chapter was unfinished stays current once it finishes.

    The render signature covers only the page's own objects -- the
    chapter band is dead -- so finishing the chapter does not stale
    an early slice's render, and the render stage does not letter it
    again.
    """
    import app.config as config
    import app.dependencies as deps
    import app.manifest_utils as manifests
    import app.pipeline_editing as editing
    import app.routers.export as export_router
    import app.routers.image as image_router
    import app.routers.render_commit as render_commit
    from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings
    from app.ai_providers import PROVIDERS
    from app.schemas import RenderRequest
    from PIL import Image

    chapter = "e7c3b2a2"
    processed, output, raw = tmp_path / "processed", tmp_path / "output", tmp_path / "raw"
    (processed / chapter).mkdir(parents=True)
    (raw / chapter).mkdir(parents=True)
    for module, name, value in (
        (editing, "PROCESSED_DIR", processed), (manifests, "PROCESSED_DIR", processed),
        (config, "PROCESSED_DIR", processed), (config, "OUTPUT_DIR", output), (config, "RAW_DIR", raw),
        (render_commit, "OUTPUT_DIR", output), (export_router, "OUTPUT_DIR", output),
        (image_router, "OUTPUT_DIR", output), (image_router, "PROCESSED_DIR", processed), (image_router, "RAW_DIR", raw),
    ):
        monkeypatch.setattr(module, name, value)

    clean = raw / chapter / "clean.png"
    Image.new("RGB", (900, 1000), "white").save(clean)
    page0 = {
        "original": clean.as_posix(), "clean": clean.as_posix(), "boxes": [],
        "text_objects": [{
            "id": f"d{i}", "region": {"x1": 100, "y1": 100 + i * 100, "x2": 500, "y2": 180 + i * 100},
            "translation": "", "typography_role": "dialogue", "container": "bubble",
            "style": {"font": "dialogue.mac-dinh-3"},
        } for i in range(8)],
        "preserve_regions": [], "skipped": False, "process_required": False, "source_page": 0, "slice_index": 0,
        "source_revision": 1, "clean_revision": 1, "render_revision": 0, "rendered": False,
    }
    dots = {"id": "dots", "region": {"x1": 100, "y1": 100, "x2": 282, "y2": 205},
            "translation": ".......", "source_cap_px": 8, "typography_role": "dialogue",
            "container": "bubble", "style": {"font": "dialogue.mac-dinh-3"}}
    page1 = {
        "original": clean.as_posix(), "clean": clean.as_posix(), "boxes": [],
        "text_objects": [dots],
        "preserve_regions": [], "skipped": False, "process_required": False, "source_page": 1, "slice_index": 1,
        "source_revision": 1, "clean_revision": 1, "render_revision": 0, "rendered": False,
    }
    manifests.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [page0, page1]})
    render_commit.render_page(RenderRequest(chapter_id=chapter, page_index=1, translations={}))

    def dots_ink() -> int:
        with Image.open(output / chapter / "page_001.png") as image:
            grey = image.convert("L").crop((100, 100, 282, 205))
        return sum(1 for px in grey.getdata() if px < 128)

    small = dots_ink()
    manifest = manifests.load_manifest_raw(chapter)
    assert image_router._current_rendered_path(chapter, 1, manifest) is not None, "setup: slice 1 must be current"
    with manifests.get_manifest_lock(chapter):
        manifest = manifests.load_manifest_raw(chapter)
        for obj in manifest["pages"][0]["text_objects"]:
            obj["translation"] = "Lời thoại."
            obj["source_cap_px"] = 38
        manifests.save_manifest_raw(chapter, manifest)
    manifest = manifests.load_manifest_raw(chapter)
    assert image_router._current_rendered_path(chapter, 1, manifest) is not None, (
        "finishing the chapter must not stale slice 1: its own objects did not change")

    rendered_pages: list[int] = []
    real_render_page = render_commit.render_page

    def counting_render_page(req):
        rendered_pages.append(req.page_index)
        return real_render_page(req)

    monkeypatch.setattr(render_commit, "render_page", counting_render_page)
    job = AIModeJob(job_id="j", settings=AIModeSettings(url="https://x", provider="openai", model="m"),
                    chapter_id=chapter, stage="render", stages={"render": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    asyncio.run(runner.render())

    assert 1 not in rendered_pages, "the render stage re-lettered a slice that was still current"
    assert dots_ink() == small, "the dots must render identically"


def test_render_stage_restores_an_unletterable_object_and_the_slice_exports(tmp_path: Path, monkeypatch):
    import app.config as config
    import app.dependencies as deps
    import app.manifest_utils as manifests
    import app.pipeline_editing as editing
    import app.routers.export as export_router
    import app.routers.image as image_router
    import app.routers.render_commit as render_commit
    from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings
    from app.ai_providers import PROVIDERS

    chapter = "e7c3b2a1"
    processed, output, raw = tmp_path / "processed", tmp_path / "output", tmp_path / "raw"
    (processed / chapter).mkdir(parents=True)
    (raw / chapter).mkdir(parents=True)
    for module, name, value in (
        (editing, "PROCESSED_DIR", processed), (manifests, "PROCESSED_DIR", processed),
        (config, "PROCESSED_DIR", processed), (config, "OUTPUT_DIR", output), (config, "RAW_DIR", raw),
        (render_commit, "OUTPUT_DIR", output), (export_router, "OUTPUT_DIR", output),
        (image_router, "OUTPUT_DIR", output), (image_router, "PROCESSED_DIR", processed), (image_router, "RAW_DIR", raw),
    ):
        monkeypatch.setattr(module, name, value)
    original = np.full((200, 300, 3), 30, dtype=np.uint8)
    img_path = raw / chapter / "page.png"
    write_image(img_path, original)
    boxes = [{"id": "b1", "x1": 10, "y1": 10, "x2": 30, "y2": 20, "confidence": 1.0, "manual": True,
              "origin": "manual", "ocr_eligible": True}]
    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._inpainter = _FakeInpainter()
    clean = pipeline._do_reinpaint(processed / chapter, img_path, original, boxes,
                                   manual_mask_posix=None, manual_lama_mask_posix=None, preserve_regions=[])
    long_line = "Một câu dịch rất dài không thể nào vừa trong một ô nhỏ như thế này được " * 6
    manifests.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [{
        "original": img_path.as_posix(), "clean": clean, "boxes": boxes,
        "text_objects": [{"id": "tiny", "region": {"x1": 10, "y1": 10, "x2": 30, "y2": 20},
                          "translation": long_line, "source_boxes": ["b1"], "style": {},
                          "auto_generated": True, "auto_geometry": {"x1": 10, "y1": 10, "x2": 30, "y2": 20}}],
        "preserve_regions": [], "skipped": False, "process_required": False, "source_page": 0, "slice_index": 0,
        "source_revision": 1, "clean_revision": 1, "render_revision": 0, "rendered": False,
    }]})
    monkeypatch.setattr(deps, "pipeline", pipeline)

    job = AIModeJob(job_id="j", settings=AIModeSettings(url="https://x", provider="openai", model="m"),
                    chapter_id=chapter, stage="render", stages={"render": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    asyncio.run(runner.render())

    manifest = manifests.load_manifest_raw(chapter)
    assert runner.report["restored_regions"] == 1, runner.report["render_errors"]
    assert image_router._current_rendered_path(chapter, 0, manifest) is not None, "the slice must be rendered"
    snapshot = export_router._snapshot_export_inputs(chapter)
    assert [item["page_index"] for item in snapshot] == [0]


def test_export_snapshot_does_not_save_the_manifest(tmp_path, monkeypatch):
    """M6: export is a read path — the ensured objects live in the
    in-memory manifest the gate and snapshot read, and saving here
    would mutate the chapter on every export."""
    import app.config as config
    import app.manifest_utils as manifests
    import app.routers.export as export_router

    chapter = "c3d4e5f6"
    processed = tmp_path / "processed"
    (processed / chapter).mkdir(parents=True)
    monkeypatch.setattr(manifests, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    manifests.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [{
        "boxes": [{"id": "b1", "x1": 10, "y1": 10, "x2": 30, "y2": 20}],
        "text_objects": [], "skipped": False, "source_page": 0, "slice_index": 0,
    }]})
    saved = []
    monkeypatch.setattr(export_router, "save_manifest_raw", lambda chapter_id, data: saved.append(chapter_id))
    monkeypatch.setattr(export_router, "_editorial_gate_or_409", lambda manifest, **kw: None)
    monkeypatch.setattr(export_router, "_export_path_for_page",
                        lambda chapter_id, page_index, page, manifest: tmp_path / "page.png")
    monkeypatch.setattr(export_router, "_file_revision", lambda path: 1)
    snapshot = export_router._snapshot_export_inputs(chapter)
    assert saved == []
    assert [item["page_index"] for item in snapshot] == [0]
    assert manifests.load_manifest_raw(chapter)["pages"][0]["text_objects"] == []
