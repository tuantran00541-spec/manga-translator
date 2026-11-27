from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.detector.boxes import BubbleBox
from app.downloader import generic_js, slicer
from app.ocr import identity as ocr_identity
from app.routers import editor as editor_router


def test_update_text_objects_bulk_reconciles_ocr_edit_without_500(monkeypatch, tmp_path):
    """Editing ocr_text via /api/text_object/update_bulk with translation=None must not raise AttributeError."""
    from app import main as app_main

    chapter_id = "a1b2c3d4"
    saved: list[dict] = []
    manifest = {
        "chapter_id": chapter_id,
        "pages": [
            {
                "original": str(tmp_path / "000.png"),
                "clean": str(tmp_path / "clean_000.png"),
                "text_objects": [
                    {
                        "id": "obj1",
                        "region": {"x1": 10, "y1": 10, "x2": 100, "y2": 60},
                        "ocr_text": "old text",
                        "translation": "bản dịch cũ",
                        "auto_translation": "bản dịch cũ",
                        "translation_source": "gemini",
                        "translation_input_text": "old text",
                    }
                ],
            }
        ],
    }

    class _NoopLock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_update(cid, page_index, oid, changes):
        assert cid == chapter_id
        obj = manifest["pages"][page_index]["text_objects"][0]
        obj.update(changes)
        return manifest

    monkeypatch.setattr(editor_router.pipeline, "update_text_object", fake_update)
    monkeypatch.setattr(editor_router.pipeline, "_sync_output_dir", lambda *_a, **_k: None)
    monkeypatch.setattr(editor_router, "get_manifest_lock", lambda _cid: _NoopLock())
    monkeypatch.setattr(editor_router, "load_manifest_raw", lambda _cid: manifest)
    monkeypatch.setattr(editor_router, "save_manifest_raw", lambda _cid, m: saved.append(m))
    monkeypatch.setattr(editor_router, "urlify_manifest", lambda m: m)

    client = TestClient(app_main.app, base_url="http://127.0.0.1")
    resp = client.post(
        "/api/text_object/update_bulk",
        json={
            "chapter_id": chapter_id,
            "updates": [
                {"page_index": 0, "id": "obj1", "ocr_text": "new source text"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    obj = resp.json()["pages"][0]["text_objects"][0]
    assert obj["ocr_text"] == "new source text"
    assert obj.get("translation", "") == ""
    assert saved, "stale machine translation must be persisted when cleared"


def test_generic_js_wait_predicate_requires_positive_natural_width():
    """Even when AsuraScansJsAdapter sets min_width=0, wait_for_function must require naturalWidth > 0."""
    source = Path(generic_js.__file__).read_text(encoding="utf-8")
    assert "img.complete && w > 0 && w >= minWidth" in source


def test_ocr_crop_signature_skips_file_stat_on_mask_pointers_and_base64(monkeypatch):
    def fail_stat(_path):
        raise AssertionError("file_revision must not be called on @mask: pointers or base64 blobs")

    monkeypatch.setattr(ocr_identity, "file_revision", fail_stat)

    sig_ptr = ocr_identity.ocr_crop_signature({"x1": 1, "y1": 2, "x2": 30, "y2": 40, "mask": "@mask:masks/p0_b0.png"})
    sig_b64 = ocr_identity.ocr_crop_signature({"x1": 1, "y1": 2, "x2": 30, "y2": 40, "mask": "A" * 600})
    sig_data = ocr_identity.ocr_crop_signature({"x1": 1, "y1": 2, "x2": 30, "y2": 40, "mask": "data:image/png;base64,abc"})
    assert len({sig_ptr, sig_b64, sig_data}) == 3


def test_slicer_chunked_color_mask_matches_full_array():
    rng = np.random.default_rng(42)
    h, w = 3500, 120
    color = np.full((h, w, 3), 240, dtype=np.uint8)
    # Paint varied colored bands across chunk boundaries (around y=1024 and y=2048)
    color[1010:1040, 20:90] = (40, 120, 220)
    color[2035:2065, 30:100] = (200, 50, 70)
    color[2800:2820, 40:80] = rng.integers(0, 180, size=(20, 40, 3), dtype=np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    chunked_mask = slicer._get_content_row_mask(gray, h, w, color=color)

    # Reference full-image color deviation calculation
    base_mask = slicer._get_content_row_mask(gray, h, w, color=None)
    kernel = slicer._horizontal_kernel_width(w)
    ref_bg = cv2.blur(color, (kernel, 1))
    ref_dev = cv2.absdiff(color, ref_bg)
    ref_mask = base_mask | np.any(ref_dev >= slicer.SLICE_BACKGROUND_DISTANCE, axis=(1, 2))
    assert np.array_equal(chunked_mask, ref_mask)


def test_leftover_passes_only_rescan_prior_leftovers(monkeypatch, tmp_path):
    """After pass 1 finds leftovers, pass 2 must only re-check those leftover boxes, not all page boxes."""
    from app.page_processing import PageProcessingMixin

    img_path = tmp_path / "page.png"
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(img_path), img)
    processed_dir = tmp_path / "proc"
    processed_dir.mkdir()

    b1 = BubbleBox(10, 10, 60, 40, 0.9, np.full((30, 50), 255, np.uint8), "kiuyha", safe_to_inpaint=True)
    b2 = BubbleBox(100, 100, 160, 140, 0.9, np.full((40, 60), 255, np.uint8), "kiuyha", safe_to_inpaint=True)
    l1 = BubbleBox(12, 12, 58, 38, 0.9, np.full((26, 46), 255, np.uint8), "kiuyha", safe_to_inpaint=True)

    seen_target_counts: list[int] = []

    class _FakeDetector:
        def detect(self, _im):
            return [b1, b2]

        def last_metrics(self):
            return {}

        def leftover_boxes(self, _clean, targets):
            seen_target_counts.append(len(targets))
            if len(seen_target_counts) == 1:
                return [l1]
            return []

    class _FakeInpainter:
        def inpaint(self, im, _boxes, **_kw):
            return im.copy()

        def last_metrics(self):
            return {}

    class _Dummy(PageProcessingMixin):
        detector = _FakeDetector()
        inpainter = _FakeInpainter()

        def _write_page_cleanup(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("app.render.balloon_layout.plan_balloon_regions", lambda *_a, **_k: [])
    _Dummy()._process_page(img_path, processed_dir)
    assert seen_target_counts == [2, 1], f"expected pass 1 to scan 2 boxes and pass 2 to scan 1 leftover, got {seen_target_counts}"


def test_lifespan_preload_runs_in_background_thread(monkeypatch):
    """Startup lifespan must not block waiting on a slow LaMa preload."""
    from app import main as app_main

    preload_started = threading.Event()
    release_preload = threading.Event()

    def slow_preload():
        preload_started.set()
        release_preload.wait(timeout=2.0)

    monkeypatch.setattr(app_main, "ensure_directories", lambda: None)
    monkeypatch.setattr(app_main, "check_models", lambda: [])
    monkeypatch.setattr(app_main, "_cleanup_extra_stale_artifacts", lambda: 0)
    monkeypatch.setattr(app_main, "INPAINT_PRELOAD_ENABLED", True)
    monkeypatch.setattr(app_main, "PROCESSED_DIR", Path("__nonexistent_proc__"))
    monkeypatch.setattr(app_main, "OUTPUT_DIR", Path("__nonexistent_out__"))
    monkeypatch.setattr(app_main, "pipeline", SimpleNamespace(inpainter=SimpleNamespace(preload=slow_preload)))

    async def _run():
        t0 = time.monotonic()
        async with app_main.lifespan(app_main.app):
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5, f"lifespan blocked {elapsed:.2f}s waiting for preload()"
            assert preload_started.wait(timeout=1.0), "background preload thread did not start"
        release_preload.set()

    asyncio.run(_run())
