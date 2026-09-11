#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("MT_UI_ROOT", str(DEFAULT_ROOT))).resolve()
STATIC_ROOT = ROOT / "app" / "static"
FIXTURE = ROOT / "scripts" / "fixtures" / "frontend_ui_transition_browser.html"
EXPECTED_BATCHES = [16, 16, 16, 16, 12]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/fixture":
            return self._send(FIXTURE, "text/html; charset=utf-8")
        if parsed.path.startswith("/static/"):
            rel = parsed.path[len("/static/"):]
            path = (STATIC_ROOT / rel).resolve()
            if STATIC_ROOT.resolve() not in path.parents:
                self.send_error(403)
                return
            mime = "text/plain"
            if path.suffix == ".css":
                mime = "text/css"
            elif path.suffix == ".js":
                mime = "text/javascript"
            return self._send(path, mime)
        self.send_error(404)

    def _send(self, path: pathlib.Path, mime: str):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def find_chrome() -> str:
    for candidate in [os.environ.get("CHROME_BIN"), "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if resolved and pathlib.Path(resolved).exists():
            return resolved
    raise SystemExit("Chrome/Chromium not found")


def parse_result(dom: str) -> dict:
    match = re.search(r'<pre id="fixture-result">(.*?)</pre>', dom, flags=re.S)
    if not match:
        raise AssertionError("fixture result element missing")
    raw = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
    if raw == "pending":
        raise AssertionError("UI transition fixture did not finish before browser deadline")
    return json.loads(raw)


def main() -> int:
    chrome = find_chrome()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/fixture"
    try:
        with tempfile.TemporaryDirectory(prefix="mt-ui-transition-") as profile:
            cmd = [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                f"--user-data-dir={profile}",
                "--window-size=1366,900",
                "--virtual-time-budget=10000",
                "--dump-dom",
                url,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
            if proc.returncode != 0:
                raise RuntimeError(f"Chrome failed ({proc.returncode}): {proc.stderr[-2500:]}")
            result = parse_result(proc.stdout)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result.get("ok"):
                raise AssertionError(f"UI transition browser regression failed: {result}")
            if result.get("processBatchSizes") != EXPECTED_BATCHES:
                raise AssertionError(f"processing throughput must stay at 16-page batches: {result}")
            if result.get("manifestBoxes") != 384:
                raise AssertionError(f"browser fixture did not reach supplied chapter box scale: {result}")
            if result.get("reviewNavigatorItems") != 76:
                raise AssertionError(f"review navigator did not represent all non-skipped slices: {result}")
            if not result.get("reviewReached") or not result.get("editorReached"):
                raise AssertionError(f"preview -> review -> editor transition incomplete: {result}")
            if not result.get("auxControlResponsiveDuringProcess"):
                raise AssertionError(f"UI was not interactive while process requests were in flight: {result}")
            if not result.get("settingsResponsiveAfterEditor"):
                raise AssertionError(f"UI did not remain clickable after editor transition: {result}")
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
