"""Audit §8.14 (M1+M2+M3): the render-output dir must agree with the manifest.

M1: a skipped page ships in export as its clean-or-original fallback, so
    _sync_output_dir must materialize that same image in the output dir.
M2: invalidating a render (AI-mode _invalidate_renders, render-commit
    failure) must drop the stale file, not just flip the manifest flag.
M3: GET /api/image/.../rendered must 409 on a stale/missing render like
    download_page does, never silently serve the base image.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

import app.ai_mode.job as ai_job
import app.config as config
import app.manifest_utils as manifests
import app.routers.image as image_router
import app.routers.render_commit as render_commit
from app.ai_mode.job import AIModeRunner, AIModeSettings
from app.ai_providers import PROVIDERS
from app.pipeline import ChapterPipeline
from app.schemas import RenderRequest

CHAPTER = "a1b2c3d5"
SETTINGS = AIModeSettings(url="https://example.com/c/1", provider="openai", model="vision-test")


def _write_png(path: Path, color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 30), color).save(path)
    return path


def _manifest_with_pages(pages):
    return {"pages": pages}


# -- M1 -----------------------------------------------------------------------


def test_sync_output_dir_materializes_skipped_page_fallback(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    out_dir = tmp_path / "output"
    original = _write_png(raw_dir / CHAPTER / "000.png", "red")
    stale = _write_png(out_dir / CHAPTER / "page_001.png", "blue")
    rendered = _write_png(out_dir / CHAPTER / "page_002.png", "green")
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(image_router, "RAW_DIR", raw_dir)
    monkeypatch.setattr(image_router, "PROCESSED_DIR", proc_dir)

    manifest = _manifest_with_pages([
        {"skipped": True, "rendered": False, "clean": None, "original": str(original)},
        {"skipped": False, "rendered": False, "clean": None, "original": str(original)},
        {"skipped": False, "rendered": True, "clean": None, "original": str(original)},
    ])
    ChapterPipeline._sync_output_dir(CHAPTER, manifest)

    fallback = out_dir / CHAPTER / "page_000.png"
    assert fallback.exists(), "a skipped page must have a file in the output dir (M1)"
    assert fallback.read_bytes() == original.read_bytes()
    assert not stale.exists(), "a stale file for an unrendered active page must go (M2)"
    assert rendered.exists(), "a current render must be left alone"


def test_mark_skipped_writes_fallback_into_output_dir(monkeypatch, tmp_path):
    import app.pipeline as module

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "output"
    original = _write_png(raw_dir / CHAPTER / "000.png", "red")
    manifest = _manifest_with_pages([
        {"skipped": False, "rendered": False, "clean": None, "original": str(original),
         "boxes": [{"id": "box_1"}], "clean_revision": 2},
    ])
    monkeypatch.setattr(module, "get_page_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(module, "load_manifest_raw", lambda _: manifest)
    monkeypatch.setattr(module, "save_manifest_raw", lambda *_a, **_k: None)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(image_router, "RAW_DIR", raw_dir)
    monkeypatch.setattr(image_router, "PROCESSED_DIR", tmp_path / "processed")

    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline.mark_skipped(CHAPTER, [0], True)

    fallback = out_dir / CHAPTER / "page_000.png"
    assert fallback.exists()
    assert fallback.read_bytes() == original.read_bytes()


# -- M2 -----------------------------------------------------------------------


def test_invalidate_renders_drops_stale_output_files(monkeypatch):
    manifest = _manifest_with_pages([
        {"rendered": True, "skipped": False},
        {"rendered": True, "skipped": False},
    ])
    saved = []
    synced = []
    monkeypatch.setattr(ai_job, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(ai_job, "load_manifest_raw", lambda _: manifest)
    monkeypatch.setattr(manifests, "save_manifest_raw", lambda *_a, **_k: saved.append(True))

    job = ai_job.AIModeJob(
        job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="render",
        cost_usd=None, stages={"render": {"done": 0, "total": 0, "detail": ""}},
    )
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    # Patch the instance _invalidate_renders actually calls (from
    # app.dependencies import pipeline), not the class: an earlier test
    # leaves a shadowing instance attribute on that shared object, so a
    # class-level patch would never be reached.
    from app.dependencies import pipeline as pipeline_instance

    monkeypatch.setattr(
        pipeline_instance, "_sync_output_dir",
        lambda chapter_id, m, indices: synced.append((chapter_id, indices)),
    )
    runner._invalidate_renders([0, 1])

    assert all(not page.get("rendered") for page in manifest["pages"])
    assert saved, "the manifest must still be saved"
    assert synced == [(CHAPTER, [0, 1])], "invalidating must sync (delete) the stale files (M2)"


def test_failed_render_commit_drops_stale_output(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    base = _write_png(tmp_path / "base.png")
    stale = _write_png(out_dir / CHAPTER / "page_000.png", "blue")
    monkeypatch.setattr(render_commit, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(render_commit, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(
        render_commit, "_load_render_snapshot",
        lambda req: ({"clean": str(base), "original": str(base)}, {}, "sig", []),
    )
    monkeypatch.setattr(render_commit, "_render_snapshot", lambda *_a, **_k: (0, []))
    monkeypatch.setattr(render_commit, "_commit_render", lambda *_a, **_k: (False, None))

    manifest = _manifest_with_pages([{"rendered": False}])
    monkeypatch.setattr(render_commit, "load_manifest_raw", lambda _: manifest)
    req = RenderRequest(chapter_id=CHAPTER, page_index=0, translations={})
    with pytest.raises(HTTPException) as excinfo:
        render_commit.render_page(req)
    assert excinfo.value.status_code == 409
    assert not stale.exists(), "a failed commit must not leave the stale render on disk (M2)"


def test_failed_render_commit_keeps_file_when_manifest_still_says_rendered(monkeypatch, tmp_path):
    # A concurrent render may have committed a fresh file after this
    # snapshot went stale; that file must survive the 409.
    out_dir = tmp_path / "output"
    base = _write_png(tmp_path / "base.png")
    fresh = _write_png(out_dir / CHAPTER / "page_000.png", "blue")
    monkeypatch.setattr(render_commit, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(render_commit, "get_manifest_lock", lambda *_: nullcontext())
    monkeypatch.setattr(
        render_commit, "_load_render_snapshot",
        lambda req: ({"clean": str(base), "original": str(base)}, {}, "sig", []),
    )
    monkeypatch.setattr(render_commit, "_render_snapshot", lambda *_a, **_k: (0, []))
    monkeypatch.setattr(render_commit, "_commit_render", lambda *_a, **_k: (False, None))

    manifest = _manifest_with_pages([{"rendered": True}])
    monkeypatch.setattr(render_commit, "load_manifest_raw", lambda _: manifest)
    req = RenderRequest(chapter_id=CHAPTER, page_index=0, translations={})
    with pytest.raises(HTTPException) as excinfo:
        render_commit.render_page(req)
    assert excinfo.value.status_code == 409
    assert fresh.exists(), "a file the manifest still claims must not be deleted"


# -- M3 -----------------------------------------------------------------------


def test_rendered_image_endpoint_409s_when_no_current_render(monkeypatch, tmp_path):
    manifest = _manifest_with_pages([{"rendered": False, "original": "000.png"}])
    monkeypatch.setattr(image_router, "load_manifest_raw", lambda _: manifest)
    with pytest.raises(HTTPException) as excinfo:
        image_router.get_image(CHAPTER, 0, "rendered")
    assert excinfo.value.status_code == 409


def test_rendered_image_endpoint_serves_current_render(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    rendered = _write_png(out_dir / CHAPTER / "page_000.png")
    manifest = _manifest_with_pages([{"rendered": True, "original": "000.png"}])
    monkeypatch.setattr(image_router, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(image_router, "load_manifest_raw", lambda _: manifest)
    monkeypatch.setattr(image_router, "render_artifact_is_current", lambda *_a, **_k: True)
    response = image_router.get_image(CHAPTER, 0, "rendered")
    assert response.status_code == 200
    assert Path(str(response.path)).resolve() == rendered.resolve()


def test_original_image_endpoint_still_serves(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    original = _write_png(raw_dir / CHAPTER / "000.png")
    manifest = _manifest_with_pages([{"rendered": False, "original": str(original)}])
    monkeypatch.setattr(image_router, "RAW_DIR", raw_dir)
    monkeypatch.setattr(image_router, "load_manifest_raw", lambda _: manifest)
    response = image_router.get_image(CHAPTER, 0, "original")
    assert response.status_code == 200
