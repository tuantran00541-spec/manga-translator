import json
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
from app.detector.bubble_detector import BubbleBox
from app.pipeline import ChapterPipeline
from app.schemas import RenderRequest


CHAPTER_ID = "b1c2d3e4"


class Phase41Harness(unittest.TestCase):
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

    def write_legacy_manifest(self, page):
        manifest = {
            "chapter_id": CHAPTER_ID,
            "source_url": None,
            "pages": [page],
            "workflow": {"stage": "review", "page_index": 0},
        }
        (self.chapter_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    def save_manifest(self, page):
        manifest = {
            "chapter_id": CHAPTER_ID,
            "source_url": None,
            "pages": [page],
            "workflow": {"stage": "review", "page_index": 0},
        }
        manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)
        return manifest

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


class ManifestMigrationTests(Phase41Harness):
    def test_legacy_manifest_load_adds_dimensions_revisions_and_stable_box_ids(self):
        box = {
            "x1": 10, "y1": 12, "x2": 70, "y2": 60,
            "confidence": 0.9, "mask": None,
        }
        obj = {
            "id": "obj1", "shape": "rectangle",
            "region": {"x1": 5, "y1": 5, "x2": 90, "y2": 70},
            "source_boxes": [0], "ocr_text": "", "translation": "", "style": {},
        }
        self.write_legacy_manifest(self.base_page(boxes=[box], text_objects=[obj]))

        first = manifest_utils.load_manifest_raw(CHAPTER_ID)
        second = manifest_utils.load_manifest_raw(CHAPTER_ID)
        p1 = first["pages"][0]
        p2 = second["pages"][0]

        self.assertGreaterEqual(first.get("schema_version", 0), 2)
        self.assertEqual((p1["width"], p1["height"]), (128, 96))
        self.assertEqual(p1["source_revision"], 1)
        self.assertEqual(p1["process_revision"], 1)
        self.assertEqual(p1["clean_revision"], 1)
        self.assertEqual(p1["render_revision"], 0)
        self.assertTrue(p1["boxes"][0]["id"].startswith("box_"))
        self.assertEqual(p1["boxes"][0]["origin"], "detector")
        self.assertEqual(p1["text_objects"][0]["source_boxes"], [p1["boxes"][0]["id"]])
        self.assertEqual(p1["boxes"][0]["id"], p2["boxes"][0]["id"])


class StableIdentityAndRevisionTests(Phase41Harness):
    def test_add_manual_box_assigns_id_origin_and_bumps_clean_revision(self):
        self.save_manifest(self.base_page(clean_revision=4, process_revision=2))
        pipe = ChapterPipeline()
        with patch.object(pipe, "_do_reinpaint", return_value=self.clean.as_posix()), \
             patch.object(pipe, "_sync_output_dir", return_value=None):
            result = pipe.add_manual_box(CHAPTER_ID, 0, 10, 12, 70, 60)

        page = result["pages"][0]
        box = page["boxes"][0]
        self.assertTrue(box["id"].startswith("box_"))
        self.assertEqual(box["origin"], "manual")
        self.assertEqual(page["clean_revision"], 5)
        self.assertEqual(page["process_revision"], 2)

    def test_process_reuses_detector_box_id_and_bumps_process_and_clean_revisions(self):
        old_box = {
            "id": "box_keep_me", "origin": "detector",
            "x1": 10, "y1": 10, "x2": 50, "y2": 40,
            "confidence": 0.9, "mask": None,
        }
        self.save_manifest(self.base_page(
            clean=None, boxes=[old_box], process_revision=2, clean_revision=3,
        ))

        class Detector:
            def detect(self, image, *, parallel=False):
                return [BubbleBox(11, 10, 51, 41, 0.95, None)]

        class Inpainter:
            def inpaint(self, image, boxes):
                return image.copy()
            def inpaint_mask(self, image, mask):
                return image.copy()

        pipe = ChapterPipeline()
        pipe._detector = Detector()
        pipe._inpainter = Inpainter()
        with patch.object(pipe, "_sync_output_dir", return_value=None):
            result = pipe.process_pages(CHAPTER_ID, [0], workers=1)

        page = result["pages"][0]
        self.assertEqual(page["boxes"][0]["id"], "box_keep_me")
        self.assertEqual(page["boxes"][0]["origin"], "detector")
        self.assertEqual(page["process_revision"], 3)
        self.assertEqual(page["clean_revision"], 4)

    def test_text_object_ocr_stores_box_ids_not_array_indices(self):
        box = {
            "id": "box_alpha", "origin": "detector",
            "x1": 10, "y1": 10, "x2": 80, "y2": 55,
            "confidence": 1.0, "mask": None,
        }
        obj = {
            "id": "obj1", "shape": "rectangle",
            "region": {"x1": 5, "y1": 5, "x2": 90, "y2": 65},
            "source_boxes": [], "ocr_text": "", "translation": "", "style": {},
        }
        self.save_manifest(self.base_page(boxes=[box], text_objects=[obj]))
        with patch.object(editor_mod.ocr, "read", return_value="hello"):
            result = editor_mod._group_text_object_ocr(CHAPTER_ID, 0, "obj1", "en")
        self.assertEqual(result["pages"][0]["text_objects"][0]["source_boxes"], ["box_alpha"])

    def test_successful_render_bumps_render_revision(self):
        self.save_manifest(self.base_page(render_revision=3, text_objects=[]))
        result = render_mod.render_page(RenderRequest(
            chapter_id=CHAPTER_ID, page_index=0, translations={}
        ))
        self.assertNotIn("warning", result)
        page = manifest_utils.load_manifest_raw(CHAPTER_ID)["pages"][0]
        self.assertTrue(page["rendered"])
        self.assertEqual(page["render_revision"], 4)


class PersistentBoxDecisionTests(Phase41Harness):
    class RecordingInpainter:
        def __init__(self):
            self.boxes = None
        def inpaint(self, image, boxes):
            self.boxes = list(boxes)
            return image.copy()
        def inpaint_mask(self, image, mask):
            return image.copy()

    def _run_process(self, existing_boxes, detected):
        self.save_manifest(self.base_page(clean=None, boxes=existing_boxes))

        class Detector:
            def detect(inner_self, image, *, parallel=False):
                return detected

        inpainter = self.RecordingInpainter()
        pipe = ChapterPipeline()
        pipe._detector = Detector()
        pipe._inpainter = inpainter
        with patch.object(pipe, "_sync_output_dir", return_value=None):
            result = pipe.process_pages(CHAPTER_ID, [0], workers=1)
        return result, inpainter

    def test_removed_detector_box_stays_suppressed_after_reprocess(self):
        existing = [{
            "id": "box_removed", "origin": "detector",
            "x1": 10, "y1": 10, "x2": 50, "y2": 40,
            "confidence": 0.9, "mask": None, "removed": True,
        }]
        result, inpainter = self._run_process(
            existing, [BubbleBox(10, 10, 50, 40, 0.95, None)]
        )
        box = result["pages"][0]["boxes"][0]
        self.assertEqual(box["id"], "box_removed")
        self.assertTrue(box.get("removed"))
        self.assertEqual(inpainter.boxes, [])

    def test_detector_geometry_override_survives_reprocess(self):
        existing = [{
            "id": "box_override", "origin": "detector",
            "x1": 20, "y1": 18, "x2": 82, "y2": 66,
            "confidence": 0.9, "mask": None,
            "geometry_overridden": True,
            "detector_anchor": {"x1": 10, "y1": 10, "x2": 50, "y2": 40},
        }]
        result, inpainter = self._run_process(
            existing, [BubbleBox(11, 10, 51, 41, 0.95, None)]
        )
        box = result["pages"][0]["boxes"][0]
        self.assertEqual(box["id"], "box_override")
        self.assertTrue(box.get("geometry_overridden"))
        self.assertEqual(
            {k: box[k] for k in ("x1", "y1", "x2", "y2")},
            {"x1": 20, "y1": 18, "x2": 82, "y2": 66},
        )
        self.assertEqual(len(inpainter.boxes), 1)
        b = inpainter.boxes[0]
        self.assertEqual((b.x1, b.y1, b.x2, b.y2), (20, 18, 82, 66))

    def test_manual_box_is_still_applied_during_full_reprocess(self):
        existing = [{
            "id": "box_manual", "origin": "manual", "manual": True,
            "x1": 22, "y1": 20, "x2": 78, "y2": 58,
            "confidence": 1.0, "mask": None,
        }]
        result, inpainter = self._run_process(existing, [])
        self.assertEqual(len(result["pages"][0]["boxes"]), 1)
        self.assertEqual(result["pages"][0]["boxes"][0]["id"], "box_manual")
        self.assertEqual(len(inpainter.boxes), 1)
        b = inpainter.boxes[0]
        self.assertEqual((b.x1, b.y1, b.x2, b.y2), (22, 20, 78, 58))


if __name__ == "__main__":
    unittest.main()
