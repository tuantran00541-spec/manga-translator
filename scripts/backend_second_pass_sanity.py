from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def manifest_bytes(clean_revision: int, artifact_generation: int = 0, transaction_id=None) -> bytes:
    page = {"clean_revision": clean_revision, "artifact_generation": artifact_generation}
    if transaction_id:
        page["artifact_transaction_id"] = transaction_id
    return json.dumps(
        {"schema_version": 3, "chapter_id": "a1b2c3d4", "pages": [page]},
        separators=(",", ":"),
    ).encode("utf-8")


def transaction_identity_checks() -> None:
    import app.manifest_utils as mu

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / "clean.png"
        live.write_bytes(b"OLD")
        (root / "manifest.json").write_bytes(manifest_bytes(1))
        tx = mu.PageArtifactTransaction(root, 0, [live], 2)
        tx.__enter__()
        live.write_bytes(b"NEW")
        (root / "manifest.json").write_bytes(manifest_bytes(2))
        tx.__exit__(RuntimeError, RuntimeError("commit rejected"), None)
        check(live.read_bytes() == b"OLD", "unrelated revision accepted as artifact commit")
        check(not tx.journal_path.exists(), "rolled-back transaction journal survived")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / "clean.png"
        live.write_bytes(b"OLD")
        (root / "manifest.json").write_bytes(manifest_bytes(1))
        first = mu.PageArtifactTransaction(root, 0, [live], 2)
        first.__enter__()
        live.write_bytes(b"FIRST")
        manifest = json.loads((root / "manifest.json").read_text())
        page = manifest["pages"][0]
        page["clean_revision"] = 2
        first.mark_manifest_commit(page)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        second = mu.PageArtifactTransaction(root, 0, [live], 3)
        second.__enter__()
        live.write_bytes(b"SECOND")
        manifest = json.loads((root / "manifest.json").read_text())
        page = manifest["pages"][0]
        page["clean_revision"] = 3
        second.mark_manifest_commit(page)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        second.commit()

        check(mu.recover_page_artifact_transactions(root) == 1, "superseded journal not resolved")
        check(live.read_bytes() == b"SECOND", "older recovery overwrote newer artifact")
        check(not first.journal_path.exists(), "superseded journal survived")


def stale_outcome_checks() -> None:
    import types
    registry = types.ModuleType("app.downloader.registry")
    registry.download_chapter = lambda *args, **kwargs: []
    sys.modules.setdefault("app.downloader.registry", registry)
    sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

    import app.manifest_utils as mu
    import app.pipeline as pipeline_module
    from app.pipeline import ChapterPipeline, StaleProcessingStateError

    class DummyInpainter:
        session_loaded = False
        _prefer_dynamic = False
        serialized_inference = True
        lama_model_path = ""

    class FakePipeline(ChapterPipeline):
        def __init__(self):
            super().__init__()
            self._detector = object()
            self._inpainter = DummyInpainter()

        def _shared_seam_detections(self, chapter_id, work_items):
            return {}, set(), {
                "attempts": 0,
                "failures": 0,
                "wall_ms": 0.0,
                "bubble_model_ms": 0.0,
                "text_model_ms": 0.0,
                "mser_ms": 0.0,
                "text_grayscale_fallback_runs": 0,
                "text_grayscale_fallback_ms": 0.0,
                "detector_total_ms": 0.0,
            }

        def _process_page(self, *args, **kwargs):
            return {"boxes": []}

        def _commit_processed_page(self, *args, **kwargs):
            return False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chapter = root / "a1b2c3d4"
        chapter.mkdir()
        original = chapter / "original.png"
        original.write_bytes(b"fixture")
        manifest = {
            "schema_version": 3,
            "chapter_id": "a1b2c3d4",
            "pages": [{
                "original": original.as_posix(),
                "clean": None,
                "boxes": [],
                "skipped": False,
                "excluded_regions": [],
                "clean_revision": 0,
                "source_revision": 1,
                "process_revision": 0,
                "render_revision": 0,
                "artifact_generation": 0,
            }],
        }
        (chapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        old_mu_dir, old_pipeline_dir = mu.PROCESSED_DIR, pipeline_module.PROCESSED_DIR
        mu.PROCESSED_DIR = root
        pipeline_module.PROCESSED_DIR = root
        try:
            try:
                FakePipeline().process_pages("a1b2c3d4", [0], workers=1)
            except StaleProcessingStateError:
                pass
            else:
                raise AssertionError("stale-only process returned success")
            saved = json.loads((chapter / "manifest.json").read_text())
            run = saved["last_processing_run"]
            check(run["outcome"] == "stale_only", "stale-only outcome missing")
            check(run["discarded_stale_page_indices"] == [0], "stale page index missing")
            check(
                run["discarded_stale_pages"] == [{"page_index": 0, "reason": "processing_state_changed"}],
                "stale reason missing",
            )
        finally:
            mu.PROCESSED_DIR = old_mu_dir
            pipeline_module.PROCESSED_DIR = old_pipeline_dir


def protected_region_checks() -> None:
    from app.inpaint.lama_inpainter import Inpainter

    mask = np.full((100, 100), 255, np.uint8)
    clipped = Inpainter._subtract_protected_regions(
        mask,
        (0, 0, 100, 100),
        [{"x1": 0, "y1": 0, "x2": 20, "y2": 100}],
    )
    check(not np.any(clipped[:, :20] > 0), "protected pixels retained destructive authority")
    check(np.all(clipped[:, 20:] > 0), "protection removed unrelated mask pixels")
    source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check("_box_in_excluded" not in source, "center-based exclusion predicate remains")
    check("protected_regions=excluded_regions" in source, "auto inpaint does not receive protection")


def recovery_cache_checks() -> None:
    from app.detector.recovery import SecondaryTextRecovery

    class FakeMser:
        def __init__(self):
            self.calls = 0

        def detectRegions(self, gray):
            self.calls += 1
            return [], np.array([[2, 2, 4, 4]], dtype=np.int32)

    recovery = SecondaryTextRecovery()
    fake = FakeMser()
    recovery._mser = fake
    image = np.zeros((32, 32, 3), np.uint8)
    gray = np.zeros((32, 32), np.uint8)
    first = recovery._extract_primitives(image, gray)
    second = recovery._extract_primitives(image, gray)
    check(fake.calls == 1, "same-image MSER extraction repeated")
    check(first is second, "same-image primitive cache was not reused")


def render_identity_checks() -> None:
    from app.render.identity import render_input_signature

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.png"
        base.write_bytes(b"base")
        manifest = {
            "pages": [{
                "original": base.as_posix(),
                "clean": None,
                "source_revision": 1,
                "process_revision": 1,
                "clean_revision": 1,
                "render_revision": 0,
                "skipped": False,
                "text_objects": [{
                    "id": "obj",
                    "shape": "rectangle",
                    "region": {"x1": 1, "y1": 1, "x2": 5, "y2": 5},
                    "translation": "hello",
                    "style": {},
                    "source_missing": False,
                }],
            }],
        }
        before = render_input_signature(manifest, 0)
        manifest["pages"][0]["text_objects"][0]["source_missing"] = True
        after = render_input_signature(manifest, 0)
        check(before != after, "source_missing does not invalidate render identity")


def repaint_input_checks() -> None:
    from app.routers.editor import _decode_repaint_mask_payload

    mask = np.zeros((16, 20), np.uint8)
    mask[3:7, 4:9] = 255
    ok, encoded = cv2.imencode(".png", mask)
    check(ok, "could not encode repaint fixture")
    decoded = _decode_repaint_mask_payload(encoded.tobytes())
    check(decoded.shape == mask.shape and np.array_equal(decoded, mask), "repaint decode changed pixels")
    editor_source = (ROOT / "app/routers/editor.py").read_text(encoding="utf-8")
    pipeline_source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check("await mask.read()" not in editor_source, "repaint route still performs unbounded read")
    check("read_upload_limited(mask, MAX_REQUEST_BYTES)" in editor_source, "repaint upload is not bounded")
    check("_decode_repaint_mask_payload, mask_bytes" in editor_source, "repaint decode is not offloaded")
    check("must exactly match page dimensions" in pipeline_source, "mismatched repaint masks are still stretched")


def main() -> None:
    transaction_identity_checks()
    stale_outcome_checks()
    protected_region_checks()
    recovery_cache_checks()
    render_identity_checks()
    repaint_input_checks()
    print("backend second-pass sanity: PASS")


if __name__ == "__main__":
    main()
