import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from fastapi import HTTPException

from app.schemas import VisualQCInspectRequest
from app.secret_store import gemini_key_status, get_gemini_api_key
from app.visual_qc.gemini import GeminiVisualQC, _parse_issues


class TestVisualQCParsing(unittest.TestCase):
    def test_bbox_and_relative_polygon_are_descaled_to_page_pixels(self):
        parsed = {
            "issues": [{
                "issue_type": "residual_text",
                "confidence": 0.91,
                "label": "leftover dialogue glyph",
                "box_2d": [100, 200, 300, 400],
                "mask": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            }]
        }
        issues = _parse_issues(parsed, width=2000, height=1000)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.box_2d, (100, 400, 300, 800))
        self.assertEqual(issue.polygon, [(400, 100), (800, 100), (800, 300), (400, 300)])
        self.assertAlmostEqual(issue.confidence, 0.91)

    def test_invalid_or_unknown_issues_are_ignored(self):
        parsed = {
            "issues": [
                {"issue_type": "unknown", "confidence": 1, "box_2d": [0, 0, 10, 10], "mask": [[0, 0], [1, 1], [2, 2]]},
                {"issue_type": "residual_text", "confidence": 1, "box_2d": [10, 10, 10, 20], "mask": [[0, 0], [1000, 0], [0, 1000]]},
                {"issue_type": "partial_text", "confidence": 1, "box_2d": [10, 10, 20, 20], "mask": [[0, 0], [1000, 0]]},
            ]
        }
        self.assertEqual(_parse_issues(parsed, 1000, 1000), [])

    def test_non_finite_values_are_ignored(self):
        parsed = {
            "issues": [
                {"issue_type": "residual_text", "confidence": float("nan"), "label": "bad", "box_2d": [0, 0, 100, 100], "mask": [[0, 0], [1000, 0], [0, 1000]]},
                {"issue_type": "residual_text", "confidence": 0.9, "label": "bad", "box_2d": [0, 0, float("inf"), 100], "mask": [[0, 0], [1000, 0], [0, 1000]]},
                {"issue_type": "residual_text", "confidence": 0.9, "label": "bad", "box_2d": [0, 0, 100, 100], "mask": [[0, 0], [float("nan"), 0], [0, 1000]]},
            ]
        }
        issues = _parse_issues(parsed, 1000, 1000)
        self.assertEqual(issues, [])


class TestGeminiVisualQCClient(unittest.TestCase):
    def test_inspect_sends_two_images_and_parses_structured_output(self):
        img = np.full((120, 200, 3), 255, dtype=np.uint8)
        cv2.putText(img, "TXT", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        response_payload = {
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "issues": [{
                            "issue_type": "partial_text",
                            "confidence": 0.88,
                            "label": "fragment",
                            "box_2d": [100, 100, 300, 300],
                            "mask": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                        }]
                    }),
                }],
            }]
        }
        fake_response = MagicMock()
        fake_response.ok = True
        fake_response.json.return_value = response_payload

        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original.png"
            cleaned = Path(tmp) / "cleaned.png"
            self.assertTrue(cv2.imwrite(str(original), img))
            self.assertTrue(cv2.imwrite(str(cleaned), img))

            with patch("app.visual_qc.gemini.requests.post", return_value=fake_response) as post:
                issues = GeminiVisualQC().inspect(original, cleaned, "test-key")

        self.assertEqual(len(issues), 1)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        body = kwargs["json"]
        image_inputs = [item for item in body["input"] if item.get("type") == "image"]
        self.assertEqual(len(image_inputs), 2)
        self.assertEqual(body["generation_config"]["thinking_level"], "low")
        self.assertFalse(body["store"])


class TestSecretStoreEnvironment(unittest.TestCase):
    def test_environment_key_takes_precedence_without_touching_keyring(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": " env-key ", "GOOGLE_API_KEY": ""}, clear=False):
            self.assertEqual(get_gemini_api_key(), "env-key")
            self.assertEqual(gemini_key_status()["source"], "environment")
            self.assertTrue(gemini_key_status()["configured"])


class TestVisualQCRouterConcurrency(unittest.TestCase):
    def test_changed_clean_image_rejects_stale_qc_result(self):
        from app.routers import visual_qc as router_mod

        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original.png"
            cleaned = Path(tmp) / "cleaned.png"
            original.write_bytes(b"original-v1")
            cleaned.write_bytes(b"clean-v1")
            manifest = {"pages": [{"original": str(original), "clean": str(cleaned)}]}

            def fake_inspect(*_args, **_kwargs):
                cleaned.write_bytes(b"clean-v2-longer")
                return []

            req = VisualQCInspectRequest(chapter_id="deadbeef", page_index=0)
            with patch.object(router_mod, "load_manifest_raw", return_value=manifest), \
                 patch.object(router_mod, "get_gemini_api_key", return_value="test-key"), \
                 patch.object(router_mod.visual_qc, "inspect", side_effect=fake_inspect):
                with self.assertRaises(HTTPException) as cm:
                    asyncio.run(router_mod.inspect_visual_qc(req))

            self.assertEqual(cm.exception.status_code, 409)
            self.assertIn("changed", str(cm.exception.detail).lower())


if __name__ == "__main__":
    unittest.main()
