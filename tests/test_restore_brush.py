from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from app.image_io import read_image, write_image
from app.optimized_pipeline import OptimizedChapterPipeline


class _Inpainter:
    """Auto clean paints 100, a manual repaint paints 220."""

    def inpaint(self, image, boxes, *, protected_regions=None):
        return np.full_like(image, 100)

    def _smart_paint_region(self, image, local_mask, crop_box, feather=False, force_lama=False):
        out = image.copy()
        cx1, cy1, cx2, cy2 = crop_box
        out[cy1:cy2, cx1:cx2][local_mask > 0] = 220
        return out


class _Transaction:
    """The artifact journal is tested elsewhere; here it only has to stay out of the way."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def mark_manifest_commit(self, _page):
        pass

    def commit(self):
        pass


@pytest.fixture
def chapter(tmp_path, monkeypatch):
    import app.pipeline_editing as editing

    processed = tmp_path / "processed"
    (processed / "abcd1234").mkdir(parents=True)
    img_path = tmp_path / "page.png"
    write_image(img_path, np.full((40, 60, 3), 10, np.uint8))
    manifest = {"pages": [{"original": img_path.as_posix(), "boxes": [], "skipped": False,
                           "process_required": False, "clean_revision": 1}]}
    monkeypatch.setattr(editing, "PROCESSED_DIR", processed)
    for name in ("get_page_lock", "get_manifest_lock"):
        monkeypatch.setattr(editing, name, lambda *_a: nullcontext())
    monkeypatch.setattr(editing, "load_manifest_raw", lambda _cid: manifest)
    monkeypatch.setattr(editing, "save_manifest_raw", lambda *_a, **_k: None)
    monkeypatch.setattr(editing, "invalidate_page_render", lambda *_a, **_k: None)
    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._inpainter = _Inpainter()
    pipeline._sync_output_dir = lambda *_a, **_k: None
    pipeline._page_artifact_transaction = lambda *_a, **_k: _Transaction()
    return pipeline, manifest


def _mask(y1, y2, x1, x2):
    mask = np.zeros((40, 60), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _clean(manifest):
    return read_image(Path(manifest["pages"][0]["clean"]))


def test_the_restore_brush_puts_the_original_back_only_where_painted(chapter):
    pipeline, manifest = chapter
    pipeline.restore_mask("abcd1234", 0, _mask(10, 20, 10, 30))
    clean = _clean(manifest)
    assert (clean[10:20, 10:30] == 10).all(), "painted pixels are the original again"
    assert (clean[25:35, 40:55] == 100).all(), "the rest keeps the automatic clean"
    assert manifest["pages"][0]["restore_mask"] and manifest["pages"][0]["clean_revision"] == 2


def test_a_later_repaint_over_a_restored_area_wins(chapter):
    pipeline, manifest = chapter
    pipeline.restore_mask("abcd1234", 0, _mask(10, 20, 10, 30))
    pipeline.repaint_mask("abcd1234", 0, _mask(10, 20, 10, 20))
    clean = _clean(manifest)
    assert (clean[10:20, 10:20] == 220).all(), "the newer repaint stroke wins"
    assert (clean[10:20, 22:30] == 10).all(), "the rest of the restore stays"


def test_restored_pixels_survive_a_repaint_elsewhere(chapter):
    pipeline, manifest = chapter
    pipeline.restore_mask("abcd1234", 0, _mask(10, 20, 10, 30))
    pipeline.repaint_mask("abcd1234", 0, _mask(25, 35, 40, 55))
    clean = _clean(manifest)
    assert (clean[10:20, 10:30] == 10).all()
    assert (clean[25:35, 40:55] == 220).all()


def test_undoing_repaints_forgets_the_restore_brush_too(chapter):
    pipeline, manifest = chapter
    pipeline.restore_mask("abcd1234", 0, _mask(10, 20, 10, 30))
    pipeline.reset_manual_mask("abcd1234", 0)
    assert (_clean(manifest) == 100).all()
    assert "restore_mask" not in manifest["pages"][0]


def test_an_empty_restore_stroke_is_refused(chapter):
    pipeline, _ = chapter
    with pytest.raises(ValueError, match="empty"):
        pipeline.restore_mask("abcd1234", 0, np.zeros((40, 60), np.uint8))
