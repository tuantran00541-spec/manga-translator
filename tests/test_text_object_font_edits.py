"""Font picks sent by the editor must persist through update_text_object, and an explicit null clears them."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

import app.manifest_utils as manifest_utils
import app.routers.editor as editor
from app.pipeline import ChapterPipeline
from app.schemas import UpdateTextObjectRequest

CHAPTER = "abcd1234"


def test_font_picks_persist_and_explicit_null_clears_them():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        processed = root / CHAPTER
        processed.mkdir()
        original = processed / "page.png"
        assert cv2.imwrite(str(original), np.zeros((80, 120, 3), np.uint8))
        manifest_utils.save_manifest_raw(processed, {
            "schema_version": 3, "chapter_id": CHAPTER,
            "pages": [{"original": original.as_posix(), "clean": None, "boxes": [], "text_objects": [],
                       "source_revision": 1, "process_revision": 0, "clean_revision": 0,
                       "render_revision": 0, "artifact_generation": 0}],
        })
        pipeline = ChapterPipeline.__new__(ChapterPipeline)
        with patch.object(manifest_utils, "PROCESSED_DIR", root), patch("app.pipeline_editing.PROCESSED_DIR", root):
            created = pipeline.create_text_object(CHAPTER, 0, "rectangle", {"x1": 10, "y1": 15, "x2": 70, "y2": 55})
            oid = created["pages"][0]["text_objects"][0]["id"]
            updated = pipeline.update_text_object(CHAPTER, 0, oid, {
                "font_selection_mode": "user", "font_ai_id": "dialogue.roboto",
                "font_match": {"reason": "explicit_user_selection"},
            })
            obj = updated["pages"][0]["text_objects"][0]
            assert obj["font_selection_mode"] == "user"
            assert obj["font_ai_id"] == "dialogue.roboto"
            assert obj["font_match"] == {"reason": "explicit_user_selection"}

            cleared = pipeline.update_text_object(CHAPTER, 0, oid, {
                "font_selection_mode": None, "font_ai_id": None, "font_match": None,
            })
            obj = cleared["pages"][0]["text_objects"][0]
            assert obj["font_selection_mode"] is None
            assert obj["font_ai_id"] is None
            assert obj["font_match"] is None


def test_router_forwards_explicit_null_font_clears_but_not_omitted_fields(monkeypatch):
    captured = {}

    def fake_update(chapter_id, page_index, oid, changes):
        captured.clear()
        captured.update(changes)
        return {"pages": []}

    monkeypatch.setattr(editor.pipeline, "update_text_object", fake_update)
    monkeypatch.setattr(editor, "urlify_manifest", lambda manifest: manifest)

    editor.update_text_object(UpdateTextObjectRequest(
        chapter_id=CHAPTER, page_index=0, id="obj_1",
        font_selection_mode=None, font_ai_id=None, font_match=None))
    assert captured == {"font_selection_mode": None, "font_ai_id": None, "font_match": None}, \
        "an explicit null must reach the pipeline so the cleared pick persists"

    editor.update_text_object(UpdateTextObjectRequest(
        chapter_id=CHAPTER, page_index=0, id="obj_1", translation="Hi"))
    assert captured == {"translation": "Hi"}, "omitted font fields stay out of a partial edit"