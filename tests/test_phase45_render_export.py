import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException

import app.config as config
import app.manifest_utils as manifest_utils
import app.routers.image as image_mod
import app.routers.render45 as render45
from app.main import app
from app.render.identity import render_artifact_is_current
from app.schemas import RenderRequest


CHAPTER_ID = "45abc123"


class Phase45RenderHarness(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.output = self.root / "output"
        self.raw_page_dir = self.raw / CHAPTER_ID
        self.processed_page_dir = self.processed / CHAPTER_ID
        self.output_page_dir = self.output / CHAPTER_ID
        for path in (
            self.raw_page_dir,
            self.processed_page_dir,
            self.output_page_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.original = self.raw_page_dir / "000.png"
        self.clean = self.processed_page_dir / "clean_000.png"
        image = np.full((96, 128, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (110, 80), (220, 220, 220), -1)
        self.assertTrue(cv2.imwrite(str(self.original), image))
        self.assertTrue(cv2.imwrite(str(self.clean), image))

        self.patchers = [
            patch.object(config, "RAW_DIR", self.raw),
            patch.object(config, "PROCESSED_DIR", self.processed),
            patch.object(config, "OUTPUT_DIR", self.output),
            patch.object(manifest_utils, "PROCESSED_DIR", self.processed),
            patch.object(render45, "OUTPUT_DIR", self.output),
            patch.object(image_mod, "RAW_DIR", self.raw),
            patch.object(image_mod, "PROCESSED_DIR", self.processed),
            patch.object(image_mod, "OUTPUT_DIR", self.output),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.td.cleanup()

    def text_object(self):
        return {
            "id": "obj1",
            "shape": "rectangle",
            "region": {"x1": 16, "y1": 16, "x2": 104, "y2": 72},
            "source_boxes": [],
            "ocr_text": "SOURCE",
            "translation": "OLD",
            "style": {
                "color": "auto",
                "font": "default",
                "fontSize": "auto",
                "bold": False,
                "strokeWidth": "auto",
                "strokeColor": "auto",
                "bgColor": "transparent",
                "cornerRadius": "0",
                "horizontalAlign": "center",
                "verticalAlign": "middle",
            },
        }

    def save_manifest(self, **page_updates):
        page = {
            "original": self.original.as_posix(),
            "clean": self.clean.as_posix(),
            "boxes": [],
            "text_objects": [self.text_object()],
            "skipped": False,
            "excluded_regions": [],
            "source_page": 0,
            "slice_index": 0,
            "source_revision": 1,
            "process_revision": 1,
            "clean_revision": 1,
            "render_revision": 0,
            "rendered": False,
        }
        page.update(page_updates)
        manifest_utils.save_manifest_raw(
            CHAPTER_ID,
            {
                "chapter_id": CHAPTER_ID,
                "source_url": None,
                "pages": [page],
                "workflow": {"stage": "editor", "page_index": 0},
            },
        )

    def load_manifest(self):
        return manifest_utils.load_manifest_raw(CHAPTER_ID)

    def request(self, **overrides):
        payload = {
            "chapter_id": CHAPTER_ID,
            "page_index": 0,
            "translations": {"obj1": "HELLO"},
            "colors": {"obj1": "#112233"},
            "font_sizes": {"obj1": 24},
            "bolds": {"obj1": True},
            "horizontal_aligns": {"obj1": "left"},
            "vertical_aligns": {"obj1": "top"},
        }
        payload.update(overrides)
        return RenderRequest(**payload)

    def render_successfully(self):
        with patch.object(render45, "_render_snapshot", return_value=1):
            return render45.render_page(self.request())


class RenderIdentityTests(Phase45RenderHarness):
    def test_successful_render_stamps_identity_and_persists_render_state(self):
        self.save_manifest()
        result = self.render_successfully()

        self.assertTrue(result["committed"])
        manifest = self.load_manifest()
        page = manifest["pages"][0]
        output = self.output_page_dir / "page_000.png"
        self.assertTrue(output.is_file())
        self.assertTrue(page["rendered"])
        self.assertTrue(page.get("render_input_signature"))
        self.assertEqual(page.get("render_identity_version"), "phase45-v1")
        self.assertEqual(len(page.get("render_output_revision", [])), 3)
        self.assertTrue(render_artifact_is_current(manifest, 0, output))

        obj = page["text_objects"][0]
        self.assertEqual(obj["translation"], "HELLO")
        self.assertEqual(obj["style"]["color"], "#112233")
        self.assertEqual(obj["style"]["fontSize"], "24")
        self.assertTrue(obj["style"]["bold"])
        self.assertEqual(obj["style"]["horizontalAlign"], "left")
        self.assertEqual(obj["style"]["verticalAlign"], "top")

    def test_concurrent_editor_change_discards_render_and_reports_not_committed(self):
        self.save_manifest()

        def mutate_during_render(*_args, **_kwargs):
            with manifest_utils.get_manifest_lock(CHAPTER_ID):
                manifest = manifest_utils.load_manifest_raw(CHAPTER_ID)
                manifest["pages"][0]["text_objects"][0]["translation"] = "NEWER USER EDIT"
                manifest_utils.save_manifest_raw(CHAPTER_ID, manifest)
            return 1

        with patch.object(render45, "_render_snapshot", side_effect=mutate_during_render):
            result = render45.render_page(self.request())

        self.assertFalse(result["committed"])
        self.assertIn("warning", result)
        page = self.load_manifest()["pages"][0]
        self.assertFalse(page.get("rendered", False))
        self.assertEqual(page["text_objects"][0]["translation"], "NEWER USER EDIT")
        self.assertFalse((self.output_page_dir / "page_000.png").exists())

    def test_same_path_base_replacement_invalidates_rendered_artifact(self):
        self.save_manifest()
        self.render_successfully()
        output = self.output_page_dir / "page_000.png"

        replacement = np.full((96, 128, 3), 7, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.clean), replacement))

        manifest = self.load_manifest()
        self.assertFalse(render_artifact_is_current(manifest, 0, output))
        with self.assertRaises(HTTPException) as caught:
            image_mod.download_page(CHAPTER_ID, 0)
        self.assertEqual(caught.exception.status_code, 409)

    def test_output_tamper_invalidates_download(self):
        self.save_manifest()
        self.render_successfully()
        output = self.output_page_dir / "page_000.png"
        output.write_bytes(output.read_bytes() + b"tamper")

        self.assertFalse(render_artifact_is_current(self.load_manifest(), 0, output))
        with self.assertRaises(HTTPException) as caught:
            image_mod.download_page(CHAPTER_ID, 0)
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_rendered_boolean_without_identity_is_not_exportable(self):
        self.save_manifest(rendered=True, render_revision=4)
        output = self.output_page_dir / "page_000.png"
        output.write_bytes(self.clean.read_bytes())

        self.assertFalse(render_artifact_is_current(self.load_manifest(), 0, output))
        with self.assertRaises(HTTPException) as caught:
            image_mod.download_page(CHAPTER_ID, 0)
        self.assertEqual(caught.exception.status_code, 409)

    def test_rendered_preview_falls_back_to_current_base_when_artifact_is_stale(self):
        self.save_manifest()
        self.render_successfully()
        output = self.output_page_dir / "page_000.png"
        output.write_bytes(output.read_bytes() + b"tamper")

        response = image_mod.get_image(CHAPTER_ID, 0, "rendered")
        self.assertEqual(Path(response.path), self.clean)


class RenderRouteAndUiContractTests(unittest.TestCase):
    def test_phase45_render_route_precedes_legacy_render_route(self):
        modules = [
            route.endpoint.__module__
            for route in app.routes
            if getattr(route, "path", None) == "/api/render"
            and "POST" in (getattr(route, "methods", None) or set())
        ]
        self.assertGreaterEqual(len(modules), 2)
        self.assertEqual(modules[0], "app.routers.render45")

    def test_phase45_script_loads_after_api_and_before_editor(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "app/templates/index.html").read_text(encoding="utf-8")
        self.assertLess(html.index("/static/js/api.js"), html.index("/static/js/render-export45.js"))
        self.assertLess(html.index("/static/js/render-export45.js"), html.index("/static/js/editor.js"))

    def test_frontend_only_marks_rendered_after_committed_response(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "app/static/js/render-export45.js").read_text(encoding="utf-8")
        stale_guard = source.index("data.committed !== true")
        rendered_assignment = source.index("currentPage.rendered = true")
        result_display = source.index("showRenderResult(pageIndex, data.output)")
        self.assertLess(stale_guard, rendered_assignment)
        self.assertLess(stale_guard, result_display)


if __name__ == "__main__":
    unittest.main()
