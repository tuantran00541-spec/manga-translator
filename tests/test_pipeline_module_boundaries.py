from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

import app.manifest_utils as manifest_utils
from app.pipeline import ChapterPipeline


def test_editor_state_operations_survive_pipeline_module_split():
    chapter_id = "facefeed"
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        processed = root / chapter_id
        processed.mkdir()
        original = processed / "page.png"
        assert cv2.imwrite(str(original), np.zeros((80, 120, 3), np.uint8))
        manifest_utils.save_manifest_raw(
            processed,
            {
                "schema_version": 3,
                "chapter_id": chapter_id,
                "pages": [{
                    "original": original.as_posix(),
                    "clean": None,
                    "boxes": [],
                    "text_objects": [],
                    "source_revision": 1,
                    "process_revision": 0,
                    "clean_revision": 0,
                    "render_revision": 0,
                    "artifact_generation": 0,
                }],
            },
        )
        pipeline = ChapterPipeline.__new__(ChapterPipeline)
        with patch.object(manifest_utils, "PROCESSED_DIR", root), patch(
            "app.pipeline_editing.PROCESSED_DIR", root
        ):
            created = pipeline.create_text_object(
                chapter_id,
                0,
                "rectangle",
                {"x1": 10, "y1": 15, "x2": 70, "y2": 55},
            )
            text_object = created["pages"][0]["text_objects"][0]
            updated = pipeline.update_text_object(
                chapter_id,
                0,
                text_object["id"],
                {
                    "translation": "Hello",
                    "style": {"bold": True, "fontSize": "22"},
                },
            )

        saved = updated["pages"][0]["text_objects"][0]
        assert saved["translation"] == "Hello"
        assert saved["style"]["bold"] is True
        assert saved["style"]["fontSize"] == "22"
        assert saved["style"]["horizontalAlign"] == "center"
        assert ChapterPipeline.create_text_object.__module__ == "app.pipeline_editing"
        assert ChapterPipeline._process_page.__module__ == "app.page_processing"
