#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "app" / "static"
FIXTURE = ROOT / "scripts" / "fixtures" / "frontend_release_browser.html"
WIDTHS = [390, 768, 1024, 1280, 1366, 1920]
HEIGHT = 900


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
            elif path.suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                mime = "image/" + ("jpeg" if path.suffix in {".jpg", ".jpeg"} else path.suffix[1:])
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


def find_chrome(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("CHROME_BIN"), "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if resolved and pathlib.Path(resolved).exists():
            return resolved
    raise SystemExit("Chrome/Chromium not found; frontend browser gate requires a headless Chromium browser")


def run_chrome(chrome: str, url: str, width: int, *, screenshot: pathlib.Path | None = None, dump: bool = False) -> str:
    with tempfile.TemporaryDirectory(prefix="mt-chrome-") as profile:
        cmd = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--enable-precise-memory-info",
            f"--user-data-dir={profile}",
            f"--window-size={width},{HEIGHT}",
            "--virtual-time-budget=10000",
        ]
        if screenshot is not None:
            cmd.append(f"--screenshot={screenshot}")
        if dump:
            cmd.append("--dump-dom")
        cmd.append(url)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
        if proc.returncode != 0:
            raise RuntimeError(f"Chrome failed ({proc.returncode}): {proc.stderr[-2000:]}")
        return proc.stdout


def parse_result(dom: str) -> dict:
    match = re.search(r'<pre id="fixture-result">(.*?)</pre>', dom, flags=re.S)
    if not match:
        raise AssertionError("Fixture result element missing from browser DOM")
    raw = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
    if raw == "pending":
        raise AssertionError("Fixture did not finish before browser deadline")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome")
    parser.add_argument("--artifacts", default="frontend-browser-artifacts")
    args = parser.parse_args()

    chrome = find_chrome(args.chrome)
    artifacts = (ROOT / args.artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/fixture"

    report = {
        "browser": subprocess.run([chrome, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout.strip(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "viewport_height": HEIGHT,
        "cache_state": "cold profile per run",
        "widths": {},
    }

    try:
        for width in WIDTHS:
            before = artifacts / f"before-{width}.png"
            after = artifacts / f"after-{width}.png"
            run_chrome(chrome, f"{base}?baseline=1&capture=1", width, screenshot=before)
            run_chrome(chrome, f"{base}?capture=1", width, screenshot=after)
            dom = run_chrome(chrome, f"{base}?capture=1", width, dump=True)
            result = parse_result(dom)
            if result["overflowX"]:
                raise AssertionError(f"horizontal document overflow at {width}px: {result}")
            if not result["geometryEquivalent"]:
                raise AssertionError(f"geometry round-trip failed at {width}px")
            if width == 1366 and result["canvasWidth"] < 800:
                raise AssertionError(f"1366px canvas target missed: {result['canvasWidth']}px")
            report["widths"][str(width)] = result

        before_dom = run_chrome(chrome, f"{base}?baseline=1&capture=1", 1366, dump=True)
        before_result = parse_result(before_dom)
        benchmark_dom = run_chrome(chrome, base, 1366, dump=True)
        benchmark = parse_result(benchmark_dom)
        if not benchmark.get("mask") or not benchmark["mask"].get("exactPixels"):
            raise AssertionError(f"mask encode/decode equivalence failed: {benchmark.get('mask')}")
        benchmark["canvasWidthBefore"] = before_result.get("canvasWidth")
        benchmark["canvasWidthDelta"] = benchmark.get("canvasWidth", 0) - before_result.get("canvasWidth", 0)
        report["benchmark_1366"] = benchmark
        (artifacts / "frontend_release_browser_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
