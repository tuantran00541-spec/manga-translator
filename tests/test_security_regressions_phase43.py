from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from app.schemas import VisualQCInspectRequest
from app.security import browser_request_allowed, validate_managed_path, validate_url


class _FakeResponse:
    def __init__(self, status_code=200, *, headers=None, body=b""):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._body = body
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class TestURLSecurityRegression(unittest.TestCase):
    def test_ipv4_mapped_ipv6_loopback_is_blocked(self):
        with self.assertRaises(HTTPException):
            validate_url("http://[::ffff:127.0.0.1]/private")

    def test_ipv6_unspecified_address_is_blocked(self):
        with self.assertRaises(HTTPException):
            validate_url("http://[::]/private")

    def test_url_credentials_are_rejected(self):
        with self.assertRaises(HTTPException):
            validate_url("https://user:secret@8.8.8.8/image.png")

    def test_browser_guard_blocks_private_and_file_urls(self):
        self.assertFalse(browser_request_allowed("http://127.0.0.1/admin"))
        self.assertFalse(browser_request_allowed("file:///etc/passwd"))
        self.assertTrue(browser_request_allowed("data:image/png;base64,AA=="))


class TestDownloaderSecurityRegression(unittest.TestCase):
    def test_redirect_target_is_revalidated_before_second_request(self):
        from app.downloader.http import safe_get

        first = _FakeResponse(302, headers={"Location": "http://127.0.0.1/admin"})
        with patch("app.downloader.http.requests.get", return_value=first) as request:
            with self.assertRaises(HTTPException):
                safe_get("http://93.184.216.34/start", stream=True)

        self.assertEqual(request.call_count, 1)
        self.assertTrue(first.closed)
        self.assertFalse(request.call_args.kwargs.get("allow_redirects", True))

    def test_remote_image_stream_over_limit_leaves_no_partial_file(self):
        from app.downloader.http import safe_download_file

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
        payload = buf.getvalue()
        response = _FakeResponse(200, body=payload)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.downloader.http.requests.get", return_value=response
        ):
            out = Path(tmp) / "page.png"
            with self.assertRaises(ValueError):
                safe_download_file(
                    "http://93.184.216.34/page.png",
                    out,
                    headers={},
                    limit_bytes=max(1, len(payload) - 1),
                )
            self.assertFalse(out.exists())
            self.assertEqual(list(Path(tmp).glob("*.part")), [])


class TestManagedPathRegression(unittest.TestCase):
    def test_managed_path_rejects_file_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "managed"
            root.mkdir()
            inside = root / "page.png"
            outside = base / "secret.png"
            self.assertEqual(validate_managed_path(inside, root), inside.resolve())
            with self.assertRaises(HTTPException):
                validate_managed_path(outside, root)

    def test_image_router_does_not_serve_manifest_path_outside_project_root(self):
        from app.routers import image as image_router

        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.png"
            Image.new("RGB", (16, 16), "white").save(outside)
            manifest = {"pages": [{"original": str(outside), "clean": None, "rendered": False}]}
            with patch.object(image_router, "load_manifest_raw", return_value=manifest):
                with self.assertRaises(HTTPException) as cm:
                    image_router.get_image("deadbeef", 0, "original")
            self.assertEqual(cm.exception.status_code, 403)


class TestVisualQCSecretRegression(unittest.TestCase):
    def test_gemini_exception_does_not_echo_api_key(self):
        from app.routers import visual_qc as router_mod

        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original.png"
            cleaned = Path(tmp) / "cleaned.png"
            Image.new("RGB", (16, 16), "white").save(original)
            Image.new("RGB", (16, 16), "white").save(cleaned)
            manifest = {"pages": [{"original": str(original), "clean": str(cleaned)}]}
            req = VisualQCInspectRequest(chapter_id="deadbeef", page_index=0)
            secret = "super-secret-gemini-key"

            with patch.object(router_mod, "load_manifest_raw", return_value=manifest), \
                 patch.object(router_mod, "_page_paths", return_value=(original, cleaned)), \
                 patch.object(router_mod, "get_gemini_api_key", return_value=secret), \
                 patch.object(
                     router_mod.visual_qc,
                     "inspect",
                     side_effect=RuntimeError(f"transport error leaked {secret}"),
                 ):
                with self.assertRaises(HTTPException) as cm:
                    asyncio.run(router_mod.inspect_visual_qc(req))

            self.assertEqual(cm.exception.status_code, 502)
            self.assertNotIn(secret, str(cm.exception.detail))


if __name__ == "__main__":
    unittest.main()
