import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app.config as config
import app.manifest_utils as manifest_utils
import app.pipeline as pipeline_mod
import app.routers.editor as editor_mod
from app.pipeline import ChapterPipeline
from app.schemas import OcrBoxRequest


CHAPTER_ID = "d4e5f6a7"


class Phase44OCRHarness(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.processed = self.root / "processed"
        self.output = self.root / "output"
        self.processed.mkdir()
        self.output.mkdir()
        (self.processed / CHAPTER_ID).mkdir()
        self.original = self.root / "original.png"
        self.replacement = self.root / "replacement.png"
        self.clean = self.root / "clean.png"
        image = np.full((96, 128, 3), 245, dtype=np.uint8)
        cv2.putText(image, "OCR", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.assertTrue(cv2.imwrite(str(self.original), image))
        self.assertTrue(cv2.imwrite(str(self.clean), image))
        replacement = np.full((96, 128, 3), 200, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.replacement), replacement))
        self.patchers = [
            patch.object(config, "PROCESSED_DIR", self.processed),
            patch.object(config, "OUTPUT_DIR", self.output),
            patch.object(manifest_utils, "PROCESSED_DIR", self.processed),
            patch.object(pipeline_mod, "PROCESSED_DIR", self.processed),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.td.cleanup()

    def box(self, box_id: str, **updates):
        box = {
            "id": box_id,
            "origin": "detector",
            "x1": 10,
            "y1": 10,
            "x2": 90,
            "y2": 60,
            "confidence": 0.9,
            "mask": None,
        }
        box.update(updates)
        return box

    def text_object(self, **updates):
        obj = {
            "id": "obj1",
            "shape": "rectangle",
            "region": {"x1": 5, "y1": 5, "x2": 100, "y2": 70},
            "source_boxes": [],
            "ocr_text": "",
            "translation": "",
            "style": {},
        }
        obj.update(updates)
        return obj

    def save_manifest(self, *, boxes=None, text_objects=None):
        manifest = {
            "chapter_id": CHAPTER_ID,
            "source_url": None,
            "pages": [{
                "original": self.original.as_posix(),
                "clean": self.clean.as_posix(),
                "boxes": list(boxes or []),
                "text_objects": list(text_objects or []),
                "skipped": False,
                "excluded_regions": [],
                "source_page": 0,
                "slice_index": 0,
                "source_revision": 1,
                "process_revision": 1,
                "clean_revision": 1,
                "render_revision": 0,
            }],
            "workflow": {"stage": "editor", "page_index": 0},
        }
        manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)

    def load_page(self):
        return manifest_utils.load_manifest_raw(CHAPTER_ID)["pages"][0]


class OCRIdentityRegressionTests(Phase44OCRHarness):
    def test_box_ocr_commits_by_stable_id_after_box_reorder(self):
        self.save_manifest(boxes=[self.box("box_a"), self.box("box_b")])

        def read_then_reorder(_rgb, _lang):
            with manifest_utils.get_manifest_lock(CHAPTER_ID):
                manifest = manifest_utils.load_manifest_raw(CHAPTER_ID)
                manifest["pages"][0]["boxes"].reverse()
                manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)
            return "ALPHA"

        req = OcrBoxRequest(chapter_id=CHAPTER_ID, page_index=0, box_index=0, lang="en")
        with patch.object(editor_mod.ocr, "read", side_effect=read_then_reorder):
            editor_mod.ocr_box(req)

        boxes = {box["id"]: box for box in self.load_page()["boxes"]}
        self.assertEqual(boxes["box_a"].get("ocr_text"), "ALPHA")
        self.assertFalse(boxes["box_b"].get("ocr_text"))

    def test_box_ocr_discards_result_when_original_source_changes(self):
        self.save_manifest(boxes=[self.box("box_a")])

        def read_then_replace_source(_rgb, _lang):
            with manifest_utils.get_manifest_lock(CHAPTER_ID):
                manifest = manifest_utils.load_manifest_raw(CHAPTER_ID)
                page = manifest["pages"][0]
                page["original"] = self.replacement.as_posix()
                manifest_utils.bump_page_revision(page, "source_revision")
                manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)
            return "STALE"

        req = OcrBoxRequest(chapter_id=CHAPTER_ID, page_index=0, box_index=0, lang="en")
        with patch.object(editor_mod.ocr, "read", side_effect=read_then_replace_source):
            editor_mod.ocr_box(req)

        box = self.load_page()["boxes"][0]
        self.assertNotIn("ocr_text", box)

    def test_geometry_edit_invalidates_machine_ocr_cache(self):
        box = self.box(
            "box_a",
            ocr_text="OLD",
            ocr_lang="en",
            ocr_source="machine",
            ocr_engine="legacy-engine",
            ocr_source_revision=1,
            ocr_geometry=[10, 10, 90, 60],
        )
        self.save_manifest(boxes=[box])
        pipe = ChapterPipeline()
        with patch.object(pipe, "_do_reinpaint", return_value=self.clean.as_posix()), patch.object(
            pipe, "_sync_output_dir", return_value=None
        ):
            pipe.update_box(CHAPTER_ID, 0, 0, 20, 18, 100, 68)

        updated = self.load_page()["boxes"][0]
        for key in (
            "ocr_text",
            "ocr_lang",
            "ocr_source",
            "ocr_engine",
            "ocr_source_revision",
            "ocr_geometry",
            "ocr_file_revision",
        ):
            self.assertNotIn(key, updated)


class OCRManualEditRegressionTests(Phase44OCRHarness):
    def test_user_ocr_edit_is_marked_manual_and_drops_machine_identity(self):
        obj = self.text_object(
            ocr_text="MACHINE",
            ocr_source="machine",
            ocr_engine="legacy-engine",
            ocr_lang="en",
        )
        self.save_manifest(boxes=[self.box("box_a")], text_objects=[obj])
        pipe = ChapterPipeline()
        pipe.update_text_object(CHAPTER_ID, 0, "obj1", {"ocr_text": "USER EDIT"})

        updated = self.load_page()["text_objects"][0]
        self.assertEqual(updated["ocr_text"], "USER EDIT")
        self.assertEqual(updated.get("ocr_source"), "manual")
        self.assertNotIn("ocr_engine", updated)
        self.assertNotIn("ocr_lang", updated)

    def test_group_ocr_does_not_overwrite_manual_text_without_force(self):
        obj = self.text_object(ocr_text="USER EDIT", ocr_source="manual")
        self.save_manifest(boxes=[self.box("box_a")], text_objects=[obj])
        with patch.object(editor_mod.ocr, "read", return_value="MACHINE"):
            editor_mod._group_text_object_ocr(CHAPTER_ID, 0, "obj1", "en")

        updated = self.load_page()["text_objects"][0]
        self.assertEqual(updated["ocr_text"], "USER EDIT")
        self.assertEqual(updated.get("ocr_source"), "manual")


if __name__ == "__main__":
    unittest.main()
