import copy
import json
import os
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
import app.routers.render as render_mod
from app.pipeline import ChapterPipeline
from app.schemas import RenderRequest


CHAPTER_ID = "a1b2c3d4"


class Phase40Harness(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.processed = self.root / "processed"
        self.output = self.root / "output"
        self.processed.mkdir()
        self.output.mkdir()
        self.chapter_dir = self.processed / CHAPTER_ID
        self.chapter_dir.mkdir()
        self.original = self.root / "page.png"
        self.clean = self.root / "clean.png"
        img = np.full((96, 128, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 60), (0, 0, 0), 2)
        self.assertTrue(cv2.imwrite(str(self.original), img))
        self.assertTrue(cv2.imwrite(str(self.clean), img))

        self.patchers = [
            patch.object(config, "PROCESSED_DIR", self.processed),
            patch.object(config, "OUTPUT_DIR", self.output),
            patch.object(manifest_utils, "PROCESSED_DIR", self.processed),
            patch.object(pipeline_mod, "PROCESSED_DIR", self.processed),
            patch.object(render_mod, "OUTPUT_DIR", self.output),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.td.cleanup()

    def save_manifest(self, page):
        manifest = {
            "chapter_id": CHAPTER_ID,
            "source_url": None,
            "pages": [page],
            "workflow": {"stage": "review", "page_index": 0},
        }
        manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)
        return manifest

    def load_manifest(self):
        return manifest_utils.load_manifest_raw(CHAPTER_ID)

    def base_page(self, **updates):
        page = {
            "original": self.original.as_posix(),
            "clean": self.clean.as_posix(),
            "boxes": [],
            "skipped": False,
            "excluded_regions": [],
            "source_page": 0,
            "slice_index": 0,
        }
        page.update(updates)
        return page


class TransactionRegressionTests(Phase40Harness):
    def test_add_manual_box_persists_metadata_used_for_repaint(self):
        self.save_manifest(self.base_page())
        pipe = ChapterPipeline()
        with patch.object(pipe, "_do_reinpaint", return_value=self.clean.as_posix()), \
             patch.object(pipe, "_sync_output_dir", return_value=None):
            result = pipe.add_manual_box(CHAPTER_ID, 0, 10, 12, 70, 60)

        expected = {"x1": 10, "y1": 12, "x2": 70, "y2": 60}
        for manifest in (result, self.load_manifest()):
            self.assertEqual(len(manifest["pages"][0]["boxes"]), 1)
            box = manifest["pages"][0]["boxes"][0]
            self.assertTrue(box.get("manual"))
            self.assertEqual({k: box[k] for k in expected}, expected)

    def test_update_box_persists_geometry_used_for_repaint(self):
        old_box = {
            "x1": 8, "y1": 8, "x2": 40, "y2": 40,
            "confidence": 1.0, "mask": None, "manual": True,
        }
        self.save_manifest(self.base_page(boxes=[old_box]))
        pipe = ChapterPipeline()
        with patch.object(pipe, "_do_reinpaint", return_value=self.clean.as_posix()), \
             patch.object(pipe, "_sync_output_dir", return_value=None):
            result = pipe.update_box(CHAPTER_ID, 0, 0, 20, 18, 82, 66)

        expected = {"x1": 20, "y1": 18, "x2": 82, "y2": 66}
        for manifest in (result, self.load_manifest()):
            box = manifest["pages"][0]["boxes"][0]
            self.assertEqual({k: box[k] for k in expected}, expected)
            self.assertIsNone(box.get("mask"))

    def test_reset_manual_mask_failure_preserves_file_and_manifest_pointer(self):
        mask_path = self.chapter_dir / f"manual_mask_{self.original.name}"
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:40, 30:70] = 255
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        self.save_manifest(self.base_page(manual_mask=mask_path.as_posix()))
        pipe = ChapterPipeline()

        with patch.object(pipe, "_do_reinpaint", side_effect=RuntimeError("synthetic LaMa failure")), \
             patch.object(pipe, "_sync_output_dir", return_value=None):
            with self.assertRaises(RuntimeError):
                pipe.reset_manual_mask(CHAPTER_ID, 0)

        self.assertTrue(mask_path.is_file())
        self.assertEqual(self.load_manifest()["pages"][0].get("manual_mask"), mask_path.as_posix())


class StaleWorkRegressionTests(Phase40Harness):
    def test_stale_process_does_not_delete_canonical_auto_clean_cache(self):
        self.save_manifest(self.base_page(clean=None))
        auto_clean = self.chapter_dir / f"auto_clean_{self.original.name}"
        auto_clean.write_bytes(b"existing-cache")

        class FakeDetector:
            def detect(self, image, *, parallel=False):
                return []

        class FakeInpainter:
            def inpaint(inner_self, image, boxes):
                with manifest_utils.get_manifest_lock(CHAPTER_ID):
                    m = manifest_utils.load_manifest_raw(CHAPTER_ID)
                    m["pages"][0]["excluded_regions"] = [{"x1": 1, "y1": 1, "x2": 2, "y2": 2}]
                    manifest_utils.save_manifest_raw(CHAPTER_ID, m)
                return image.copy()

            def inpaint_mask(inner_self, image, mask):
                return image.copy()

        pipe = ChapterPipeline()
        pipe._detector = FakeDetector()
        pipe._inpainter = FakeInpainter()
        with patch.object(pipe, "_sync_output_dir", return_value=None):
            pipe.process_pages(CHAPTER_ID, [0], workers=1)

        self.assertTrue(auto_clean.exists(), "stale worker must not mutate canonical auto-clean cache")
        self.assertEqual(auto_clean.read_bytes(), b"existing-cache")

    def test_render_rejects_same_path_clean_file_replacement(self):
        self.save_manifest(self.base_page(text_objects=[]))
        original_save = render_mod.Image.Image.save

        def save_then_replace_clean(image_obj, fp, *args, **kwargs):
            result = original_save(image_obj, fp, *args, **kwargs)
            replacement = np.full((96, 128, 3), 10, dtype=np.uint8)
            cv2.imwrite(str(self.clean), replacement)
            os.utime(self.clean, None)
            return result

        req = RenderRequest(chapter_id=CHAPTER_ID, page_index=0, translations={})
        with patch.object(render_mod.Image.Image, "save", new=save_then_replace_clean):
            result = render_mod.render_page(req)

        self.assertIn("warning", result)
        self.assertFalse(self.load_manifest()["pages"][0].get("rendered", False))

    def test_text_object_ocr_does_not_overwrite_newer_user_edit(self):
        box = {
            "x1": 10, "y1": 10, "x2": 80, "y2": 55,
            "confidence": 1.0, "mask": None,
        }
        obj = {
            "id": "obj1",
            "shape": "rectangle",
            "region": {"x1": 5, "y1": 5, "x2": 90, "y2": 65},
            "source_boxes": [],
            "ocr_text": "",
            "translation": "",
            "style": {},
        }
        self.save_manifest(self.base_page(boxes=[box], text_objects=[obj]))

        def ocr_then_user_edit(rgb, lang):
            with manifest_utils.get_manifest_lock(CHAPTER_ID):
                m = manifest_utils.load_manifest_raw(CHAPTER_ID)
                m["pages"][0]["text_objects"][0]["ocr_text"] = "USER EDIT"
                manifest_utils.save_manifest_raw(CHAPTER_ID, m)
            return "STALE OCR"

        with patch.object(editor_mod.ocr, "read", side_effect=ocr_then_user_edit):
            editor_mod._group_text_object_ocr(CHAPTER_ID, 0, "obj1", "en")

        self.assertEqual(self.load_manifest()["pages"][0]["text_objects"][0]["ocr_text"], "USER EDIT")


class RuntimeContractTests(unittest.TestCase):
    def test_requirements_pin_benchmarked_onnxruntime(self):
        req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("onnxruntime==1.21.0", req)


if __name__ == "__main__":
    unittest.main()
