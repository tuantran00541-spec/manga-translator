import unittest

from app.main import app
from app.ocr.schemas import ChapterOCRRequest


class OCRRouteContractTests(unittest.TestCase):
    def _first_post_endpoint_module(self, path: str) -> str:
        for route in app.routes:
            if getattr(route, "path", None) != path:
                continue
            if "POST" not in (getattr(route, "methods", None) or set()):
                continue
            return route.endpoint.__module__
        self.fail(f"POST route not found: {path}")

    def test_revision_safe_box_ocr_route_precedes_legacy_editor_route(self):
        self.assertEqual(self._first_post_endpoint_module("/api/ocr_box"), "app.routers.ocr")

    def test_revision_safe_group_ocr_route_precedes_legacy_editor_route(self):
        self.assertEqual(
            self._first_post_endpoint_module("/api/text_object/ocr"),
            "app.routers.ocr",
        )

    def test_chapter_ocr_concurrency_is_cpu_bounded(self):
        self.assertEqual(ChapterOCRRequest(chapter_id="abc12345").concurrency, 1)
        self.assertEqual(
            ChapterOCRRequest(chapter_id="abc12345", concurrency=2).concurrency,
            2,
        )
        with self.assertRaises(ValueError):
            ChapterOCRRequest(chapter_id="abc12345", concurrency=3)


if __name__ == "__main__":
    unittest.main()
