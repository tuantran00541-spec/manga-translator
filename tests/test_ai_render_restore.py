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


def test_render_stage_restores_an_unletterable_object_and_the_slice_exports(tmp_path: Path, monkeypatch):
    import app.ai_mode.job as ai_job
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
        # Auto object of a box the repair stage added, as in the failing run.
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
    # Export re-syncs text objects first; that must not invalidate the render.
    snapshot = export_router._snapshot_export_inputs(chapter, enforce_editorial_gate=False)
    assert [item["page_index"] for item in snapshot] == [0]
