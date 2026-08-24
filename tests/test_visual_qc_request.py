import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.visual_qc.gemini import GeminiVisualQC


class _FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": '{"issues": []}'},
                    ],
                }
            ]
        }


class VisualQCRequestTests(unittest.TestCase):
    def test_request_contains_two_inline_images_and_supported_thinking_level(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = np.full((120, 160, 3), 255, dtype=np.uint8)
            cleaned = original.copy()
            cv2.putText(original, "QC42", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            original_path = root / "original.png"
            cleaned_path = root / "cleaned.png"
            self.assertTrue(cv2.imwrite(str(original_path), original))
            self.assertTrue(cv2.imwrite(str(cleaned_path), cleaned))

            captured = {}

            def fake_post(url, *, headers, json, timeout):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                captured["timeout"] = timeout
                return _FakeResponse()

            with patch("app.visual_qc.gemini.requests.post", side_effect=fake_post):
                issues = GeminiVisualQC().inspect(original_path, cleaned_path, "test-key")

            self.assertEqual(issues, [])
            payload = captured["json"]
            self.assertEqual(payload["model"], "gemini-3.7-flash")
            self.assertEqual(payload["generation_config"]["thinking_level"], "low")
            self.assertFalse(payload["store"])

            image_parts = [part for part in payload["input"] if part.get("type") == "image"]
            self.assertEqual(len(image_parts), 2)
            for part in image_parts:
                self.assertEqual(part["mime_type"], "image/jpeg")
                encoded = part.get("data")
                self.assertIsInstance(encoded, str)
                raw = base64.b64decode(encoded, validate=True)
                decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(decoded)
                self.assertGreater(decoded.size, 0)

            self.assertEqual(payload["response_format"]["type"], "text")
            self.assertEqual(payload["response_format"]["mime_type"], "application/json")
            self.assertIn("schema", payload["response_format"])
            self.assertEqual(captured["headers"]["x-goog-api-key"], "test-key")
            self.assertEqual(captured["headers"]["Api-Revision"], "2026-05-20")


if __name__ == "__main__":
    unittest.main()
