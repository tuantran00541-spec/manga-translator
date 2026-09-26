from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "http://127.0.0.1:8100"
APP = "http://127.0.0.1:8000"
ADMIN = "cost-run-admin"
TERMINAL = {"completed", "failed", "cancelled"}


def _wait(url: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"{url} exited with {proc.returncode}")
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise SystemExit(f"{url} did not come up")


def _pages(archive_path: Path, out: Path, max_width: int = 800) -> int:
    """Save every rendered page, so readability can be checked on the whole chapter."""
    pages = out / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if not n.endswith("/"))
        for index, name in enumerate(names, start=1):
            page = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if page is None:
                continue
            if page.shape[1] > max_width:
                scale = max_width / page.shape[1]
                page = cv2.resize(page, (max_width, max(1, round(page.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                (pages / f"{index:03d}.jpg").write_bytes(buf.tobytes())
        return len(names)


def _fit_metrics(obj: dict, page_width: int) -> dict:
    """Reproduce how the renderer sizes one object: the font size it draws at, the lines, and whether it fits."""
    from PIL import Image, ImageDraw

    from app.parameters import RENDER_AUTO_STROKE_WIDTH, RENDER_DEFAULT_PADDING, RENDER_MIN_READABLE_FONT_SIZE, RENDER_PADDING_RATIO_MAX
    from app.render.page_renderer import _render_region_for_text_object, _resolve_ocr_style
    from app.render.text_renderer import _calc_line_height, _fit_text, _wrap_text, font_draws_text, get_font_object, get_font_path

    text = str(obj.get("translation") or "").strip()
    region = _render_region_for_text_object(obj)
    try:
        x1, y1, x2, y2 = (int(region[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return {}
    raw_w, raw_h = abs(x2 - x1), abs(y2 - y1)
    if not text or raw_w <= 0 or raw_h <= 0:
        return {}
    pad = max(2, min(RENDER_DEFAULT_PADDING, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    box_w, box_h = raw_w - 2 * pad, raw_h - 2 * pad
    style = obj.get("style") or {}
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    font_path = get_font_path(style.get("font") or "default")
    missing_glyphs = not font_draws_text(font_path, text)
    size = _resolve_ocr_style(None, style.get("fontSize", "auto"), obj.get("ocr_font_size"), "auto")
    if isinstance(size, (int, float)) or (isinstance(size, str) and size.isdigit()):
        source = "ocr" if style.get("fontSize") in (None, "", "auto") else "style"
        size = int(size)
        font = get_font_object(str(font_path), size)
        lines = _wrap_text(draw, text, font, box_w)
        fits = bool(lines) and _calc_line_height(draw, font, stroke_w=RENDER_AUTO_STROKE_WIDTH) * len(lines) <= box_h
    else:
        source = "fit"
        size, lines, fits = _fit_text(draw, text, box_w, box_h, str(font_path),
                                      stroke_w=RENDER_AUTO_STROKE_WIDTH, minimum_size=RENDER_MIN_READABLE_FONT_SIZE)
    return {"font_px": size, "font_px_at_800": round(size * 800 / max(1, page_width), 1), "size_source": source,
            "lines": len(lines), "wrapped": lines, "box_w": raw_w, "box_h": raw_h, "fits": fits,
            "missing_glyphs": missing_glyphs,
            # The wrap breaks a word apart when its lines no longer hold the text's own words.
            "split_word": bool(lines) and [w for line in lines for w in line.split()] != text.split()}


def _slice_width(chapter_id: str, index: int) -> int:
    for folder in (ROOT / "data" / "processed" / chapter_id, ROOT / "data" / "raw" / chapter_id / "sliced"):
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if index < len(files):
            image = cv2.imread(str(files[index]))
            if image is not None:
                return int(image.shape[1])
    return 800


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", default="pro")
    parser.add_argument("--timeout-min", type=float, default=60)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit(f"{args.out} must be inside {ROOT}")
    out.mkdir(parents=True, exist_ok=True)
    db = Path(tempfile.mkdtemp()) / "gateway.sqlite"
    logs = Path(tempfile.gettempdir())

    gateway_env = {**os.environ, "GATEWAY_DB": str(db), "GATEWAY_ADMIN_KEY": ADMIN}
    gateway = subprocess.Popen([sys.executable, "-m", "gateway"], cwd=ROOT, env=gateway_env,
                               stdout=open(logs / "gateway.log", "w"), stderr=subprocess.STDOUT)
    app = None
    try:
        _wait(f"{GATEWAY}/health", gateway)
        account = requests.post(f"{GATEWAY}/v1/admin/accounts", json={"email": "cost-run@example.com"},
                                headers={"X-Admin-Key": ADMIN}, timeout=10).json()
        requests.post(f"{GATEWAY}/v1/admin/accounts/{account['account_id']}/plan", json={"plan": args.plan},
                      headers={"X-Admin-Key": ADMIN}, timeout=10).raise_for_status()
        app_env = {k: v for k, v in os.environ.items() if not k.startswith("GATEWAY_")}
        app_env.update(MANGA_TIERS="1", MANGA_CLOUD_URL=f"{GATEWAY}/v1", MANGA_CLOUD_TOKEN=account["token"])
        app = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT, env=app_env,
                               stdout=open(logs / "app.log", "w"), stderr=subprocess.STDOUT)
        _wait(f"{APP}/health", app)

        started = time.perf_counter()
        response = requests.post(f"{APP}/api/ai_mode/start", json={"url": args.url, "provider": "manga-cloud"}, timeout=60)
        if not response.ok:
            raise SystemExit(f"start failed: {response.status_code} {response.text[:400]}")
        job = response.json()
        deadline = time.time() + args.timeout_min * 60
        last = None
        while job["status"] not in TERMINAL and time.time() < deadline:
            time.sleep(5)
            job = requests.get(f"{APP}/api/ai_mode/jobs/{job['job_id']}", timeout=30).json()
            line = " | ".join(f"{s['key']}:{s['status']}:{s['done']}/{s['total']}" for s in job["stages"])
            if line != last:
                print(f"[{time.perf_counter() - started:6.0f}s] {line}", flush=True)
                last = line
        wall = time.perf_counter() - started
        time.sleep(3)
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT status, requests, prompt_tokens, completion_tokens, cost_usd FROM jobs")]
    finally:
        for proc in (app, gateway):
            if proc is not None:
                proc.terminate()

    usage = rows[0] if rows else {}
    report = {
        "url": args.url,
        "plan": args.plan,
        "model": os.environ.get("GATEWAY_UPSTREAM_MODEL"),
        "price_usd_per_m": {"input": float(os.environ.get("GATEWAY_PRICE_INPUT_PER_M", "0")),
                            "output": float(os.environ.get("GATEWAY_PRICE_OUTPUT_PER_M", "0"))},
        "status": job["status"],
        "error": job.get("error"),
        "wall_s": round(wall, 1),
        "stages": [{k: s.get(k) for k in ("key", "status", "done", "total", "elapsed_s", "detail")} for s in job["stages"]],
        "report": job.get("report"),
        "gateway_job": usage,
    }
    chapter_id = job.get("chapter_id")
    if job["status"] == "completed" and chapter_id:
        archive = ROOT / "data" / "output" / chapter_id / f"ai_mode_{chapter_id}.zip"
        if archive.is_file():
            report["zip_pages"] = _pages(archive, out)
    manifest_path = ROOT / "data" / "processed" / str(chapter_id) / "manifest.json"
    if chapter_id and manifest_path.is_file():
        pages = json.loads(manifest_path.read_text(encoding="utf-8")).get("pages", [])
        report["slices"] = len(pages)
        report["slices_active"] = sum(1 for p in pages if not p.get("skipped"))
        report["lines"] = []
        for index, page in enumerate(pages):
            if page.get("skipped"):
                continue
            width = int(page.get("width") or 0) or _slice_width(chapter_id, index)
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict):
                    continue
                report["lines"].append({
                    "slice": index + 1, "id": obj.get("id"), "source": obj.get("ocr_text") or obj.get("text") or "",
                    "translation": obj.get("translation") or "",
                    "role": obj.get("typography_role"), "font": (obj.get("style") or {}).get("font"),
                    "review": bool(obj.get("needs_review")), "page_width": width,
                    **_fit_metrics(obj, width),
                })
    measured = [line for line in report.get("lines", []) if line.get("font_px")]
    if measured:
        sizes = sorted(line["font_px_at_800"] for line in measured)
        report["readability"] = {
            "rendered": len(measured),
            "font_px_at_800": {"min": sizes[0], "p10": sizes[len(sizes) // 10], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
            "under_16px": sum(size < 16 for size in sizes),
            "under_20px": sum(size < 20 for size in sizes),
            "split_word": sum(bool(line.get("split_word")) for line in measured),
            "does_not_fit": sum(not line.get("fits") for line in measured),
            "missing_glyphs": sum(bool(line.get("missing_glyphs")) for line in measured),
            "size_from_ocr": sum(line.get("size_source") == "ocr" for line in measured),
            "by_role": {
                role: sorted(line["font_px_at_800"] for line in measured if (line.get("role") or "?") == role)
                for role in sorted({line.get("role") or "?" for line in measured})
            },
        }
    if usage.get("requests"):
        slices = max(1, report.get("slices_active") or 1)
        report["per_chapter"] = {
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "cost_usd": round(usage["cost_usd"], 5),
        }
        report["per_slice"] = {
            "prompt_tokens": round(usage["prompt_tokens"] / slices),
            "completion_tokens": round(usage["completion_tokens"] / slices),
        }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "wall_s", "gateway_job", "per_chapter", "per_slice", "readability")},
                     ensure_ascii=False, indent=1))
    return 0 if job["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
