"""Focused regression checks for issues found during the project audit.

These intentionally use normal assertions: existing defects should be visible as
failures in the audit report until the production code is corrected.
"""
import copy

import numpy as np
from PIL import Image

from app.render.page_renderer import render_text_objects
from app.schemas import RenderRequest
from app.text_objects import ensure_page_text_objects


def test_render_keeps_text_crossing_slice_edge_and_later_objects():
    page = Image.new("RGB", (400, 200), "white")
    objects = [
        {"id": "cross", "translation": "First line", "region": {"x1": 30, "y1": 110, "x2": 370, "y2": 240},
         "style": {"fontSize": "24", "color": "#000000"}},
        {"id": "after", "translation": "Second line", "region": {"x1": 30, "y1": 10, "x2": 370, "y2": 80},
         "style": {"fontSize": "24", "color": "#000000"}},
    ]
    req = RenderRequest(chapter_id="audit-test", page_index=0, translations={})
    assert render_text_objects(page, req, objects, *({} for _ in range(10))) == 2
    pixels = np.asarray(page)
    assert (pixels[110:] < 240).any(), "slice-edge text was not drawn to the saved image"
    assert (pixels[:80] < 240).any(), "object after slice-edge text was not drawn to the saved image"


def test_deleted_auto_object_does_not_reappear_on_sync():
    from app.pipeline_editing import PipelineEditingMixin
    # Use the actual deletion path with an isolated in-memory manifest.
    from unittest.mock import patch
    from contextlib import nullcontext
    chapter = {"pages": [{"boxes": [{"id": "box_1", "x1": 0, "y1": 0, "x2": 100, "y2": 100,
                                      "ocr_eligible": True}], "text_objects": []}]}
    ensure_page_text_objects(chapter["pages"][0])
    obj_id = chapter["pages"][0]["text_objects"][0]["id"]
    with patch("app.pipeline_editing.get_manifest_lock", lambda *_: nullcontext()), \
         patch("app.pipeline_editing.load_manifest_raw", lambda *_: chapter), \
         patch("app.pipeline_editing.save_manifest_raw", lambda *_: None):
        PipelineEditingMixin().delete_text_object("audit-test", 0, obj_id)
    ensure_page_text_objects(chapter["pages"][0])
    assert not chapter["pages"][0]["text_objects"], "deleted detector object was recreated"


def test_saving_font_selection_metadata_persists_it():
    from app.pipeline_editing import PipelineEditingMixin
    from unittest.mock import patch
    from contextlib import nullcontext
    manifest = {"pages": [{"text_objects": [{"id": "obj", "style": {}}]}]}
    changes = {"font_selection_mode": "user", "font_match": {"reason": "user"}, "font_ai_id": "comic"}
    with patch("app.pipeline_editing.get_manifest_lock", lambda *_: nullcontext()), \
         patch("app.pipeline_editing.load_manifest_raw", lambda *_: manifest), \
         patch("app.pipeline_editing.save_manifest_raw", lambda *_: None):
        PipelineEditingMixin().update_text_object("audit-test", 0, "obj", copy.deepcopy(changes))
    obj = manifest["pages"][0]["text_objects"][0]
    assert all(obj.get(key) == value for key, value in changes.items()), "font metadata was dropped"


def test_empty_render_translation_does_not_reuse_stored_text():
    page = Image.new("RGB", (300, 120), "white")
    obj = {"id": "obj", "translation": "old text", "region": {"x1": 10, "y1": 10, "x2": 290, "y2": 110},
           "style": {"fontSize": "24", "color": "#000000"}}
    req = RenderRequest(chapter_id="audit-test", page_index=0, translations={"obj": ""})
    assert render_text_objects(page, req, [obj], *({} for _ in range(10))) == 0
    assert np.asarray(page).min() == 255


def test_ai_polish_does_not_overwrite_a_newer_user_edit():
    from contextlib import nullcontext
    from unittest.mock import patch
    from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings
    from app.ai_mode.polish import Line
    from app.ai_providers import PROVIDERS
    manifest = {"pages": [{"text_objects": [{"id": "obj", "translation": "Edited by user"}]}]}
    job = AIModeJob(job_id="audit", settings=AIModeSettings(url="https://example.com", provider="openai", model="test"),
                    chapter_id="audit-test")
    runner = AIModeRunner(job, PROVIDERS["openai"], "unused")
    # The model started with "Machine draft" but returned after the user edited it.
    line = Line(page_index=0, id="obj", source="Hello", text="AI rewrite")
    with patch("app.ai_mode.job.get_manifest_lock", lambda *_: nullcontext()), \
         patch("app.ai_mode.job.load_manifest_raw", lambda *_: manifest), \
         patch("app.ai_mode.job.save_manifest_raw", lambda *_: None):
        runner._save_polished([line])
    assert manifest["pages"][0]["text_objects"][0]["translation"] == "Edited by user"
