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


def _pairs(chapter_id: str, pages: list, out: Path, width: int = 560) -> int:
    """Save each active slice as original | final side by side for a close review."""
    from app.routers.image import _rendered_file_path

    folder = out / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index, page in enumerate(pages):
        if page.get("skipped") or not page.get("original"):
            continue
        final_path = _rendered_file_path(chapter_id, index)
        if not final_path.is_file():
            final_path = Path(page.get("clean") or "")
        raw, final = cv2.imread(str(page["original"])), cv2.imread(str(final_path))
        if raw is None or final is None:
            continue
        pair = [cv2.resize(im, (width, max(1, round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
                for im in (raw, final)]
        height = max(im.shape[0] for im in pair)
        pair = [np.vstack([im, np.full((height - im.shape[0], width, 3), 255, np.uint8)]) for im in pair]
        joined = np.hstack([pair[0], np.full((height, 8, 3), 128, np.uint8), pair[1]])
        ok, buf = cv2.imencode(".jpg", joined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            (folder / f"{index + 1:03d}.jpg").write_bytes(buf.tobytes())
            saved += 1
    return saved


def _clean_timing(pages: list) -> dict:
    """Summed cleanup time per step over the chapter, to see which one the stage waits on."""
    totals: dict[str, float] = {}
    for page in pages:
        metrics = page.get("processing_metrics") or {}
        steps = dict(metrics.get("timing_ms") or {})
        for section in ("auto_inpaint", "manual_inpaint"):
            for key in ("lama_model_ms", "session_lock_wait_ms", "ort_global_lock_wait_ms"):
                steps[f"{section}.{key}"] = (metrics.get(section) or {}).get(key, 0)
        for key, value in steps.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {key: round(value) for key, value in sorted(totals.items())}


def _objects(pages: list) -> list[dict]:
    """Every slice's boxes and text objects with coordinates, to trace a defect back to where it began."""
    keys = ("x1", "y1", "x2", "y2")
    return [{
        "slice": index + 1, "skipped": bool(page.get("skipped")), "stitch_core": page.get("stitch_core"),
        "boxes": [{**{k: box.get(k) for k in keys}, "origin": box.get("origin"), "manual": bool(box.get("manual")),
                   "overlap_context_only": bool(box.get("overlap_context_only")), "removed": bool(box.get("removed"))}
                  for box in page.get("boxes") or [] if isinstance(box, dict)],
        "objects": [{"id": obj.get("id"), "region": obj.get("region"), "source_boxes": obj.get("source_boxes"),
                     "translation": obj.get("translation"), "role": obj.get("typography_role"),
                     "container": obj.get("container"), "font": obj.get("font_ai_id"),
                     "seam_owner": obj.get("seam_owner"), "overlap_dropped": obj.get("overlap_dropped"),
                     "source_missing": obj.get("source_missing")}
                    for obj in page.get("text_objects") or [] if isinstance(obj, dict)],
        "preserve_regions": page.get("preserve_regions"),
    } for index, page in enumerate(pages)]


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
        maximum = 48
        if obj.get("source_cap_px"):
            from app.render.source_size import SIZE_SLACK, matching_font_px
            maximum = max(RENDER_MIN_READABLE_FONT_SIZE, int(matching_font_px(font_path, obj["source_cap_px"]) * SIZE_SLACK))
        size, lines, fits = _fit_text(draw, text, box_w, box_h, str(font_path), stroke_w=RENDER_AUTO_STROKE_WIDTH,
                                      minimum_size=RENDER_MIN_READABLE_FONT_SIZE, maximum_size=maximum)
    return {"font_px": size, "font_px_at_800": round(size * 800 / max(1, page_width), 1), "size_source": source,
            "lines": len(lines), "wrapped": lines, "box_w": raw_w, "box_h": raw_h, "fits": fits,
            "missing_glyphs": missing_glyphs, "source_cap_px": obj.get("source_cap_px"),
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


def _transport(path: Path) -> dict:
    """Per stage: requests, latency, payload size, tokens and retries from the gateway trace."""
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stages: dict[str, list[dict]] = {}
    for row in rows:
        stages.setdefault(str(row.get("stage") or "other"), []).append(row)
    summary = {}
    for stage, items in stages.items():
        ms = sorted(int(r.get("ms") or 0) for r in items)
        first = min(float(r["t"]) - int(r.get("ms") or 0) / 1000 for r in items)
        last = max(float(r["t"]) for r in items)
        total = lambda key: sum(int(r.get(key) or 0) for r in items)
        summary[stage] = {
            "requests": len(items),
            "failed": sum(1 for r in items if r.get("status") != 200),
            "retried": sum(1 for r in items if int(r.get("attempts") or 1) > 1),
            "truncated": sum(1 for r in items if r.get("finish_reason") == "length"),
            "latency_s": {"p50": round(ms[len(ms) // 2] / 1000, 1), "p90": round(ms[int(len(ms) * 0.9)] / 1000, 1),
                          "max": round(ms[-1] / 1000, 1), "sum": round(sum(ms) / 1000, 1)},
            "span_s": round(last - first, 1),
            "images": total("images"),
            "request_kb_avg": round(sum(float(r.get("request_kb") or 0) for r in items) / len(items), 1),
            "prompt_tokens": total("prompt_tokens"), "cached_tokens": total("cached_tokens"),
            "completion_tokens": total("completion_tokens"), "reasoning_tokens": total("reasoning_tokens"),
        }
    return summary


def _scorecard(pages: list, report: dict, out: Path) -> dict:
    """Score the chapter and save each place the cleanup left text or a ghost, original beside clean."""
    from app.ai_mode.scorecard import score_chapter
    from app.config import KIUYHA_TEXT_MODEL
    from app.detector.kiuyha_detector import KiuyhaTextDetector

    card = score_chapter(pages, KiuyhaTextDetector(KIUYHA_TEXT_MODEL), report)
    folder = out / "leftovers"
    folder.mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(card["clean"]["items"], start=1):
        page = pages[item["slice"] - 1]
        x1, y1, x2, y2 = item["box"]
        crops = [cv2.imread(str(page[key]))[max(0, y1 - 40):y2 + 40, max(0, x1 - 40):x2 + 40] for key in ("original", "clean")]
        side = np.hstack([crops[0], np.full((crops[0].shape[0], 8, 3), 128, np.uint8), crops[1]])
        cv2.imwrite(str(folder / f"{number:02d}-slice{item['slice']:03d}-{item['kind']}.jpg"), side,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--balance", type=float, default=5.0, help="USD credited to the test account")
    parser.add_argument("--timeout-min", type=float, default=60)
    parser.add_argument("--polish", action="store_true", help="let the Jev judge grade and send back weak lines")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit(f"{args.out} must be inside {ROOT}")
    out.mkdir(parents=True, exist_ok=True)
    db = Path(tempfile.mkdtemp()) / "gateway.sqlite"
    logs = Path(tempfile.gettempdir())

    trace = out / "transport.jsonl"
    trace.unlink(missing_ok=True)
    gateway_env = {**os.environ, "GATEWAY_DB": str(db), "GATEWAY_ADMIN_KEY": ADMIN, "GATEWAY_TRACE_PATH": str(trace)}
    gateway = subprocess.Popen([sys.executable, "-m", "gateway"], cwd=ROOT, env=gateway_env,
                               stdout=open(logs / "gateway.log", "w"), stderr=subprocess.STDOUT)
    app = None
    try:
        _wait(f"{GATEWAY}/health", gateway)
        account = requests.post(f"{GATEWAY}/v1/admin/accounts", json={"email": "cost-run@example.com"},
                                headers={"X-Admin-Key": ADMIN}, timeout=10).json()
        requests.post(f"{GATEWAY}/v1/admin/accounts/{account['account_id']}/credit",
                      json={"amount_usd": args.balance, "note": "cost run"},
                      headers={"X-Admin-Key": ADMIN}, timeout=10).raise_for_status()
        app_env = {k: v for k, v in os.environ.items() if not k.startswith("GATEWAY_")}
        app_env.update(MANGA_TIERS="1", MANGA_CLOUD_URL=f"{GATEWAY}/v1", MANGA_CLOUD_TOKEN=account["token"])
        app = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT, env=app_env,
                               stdout=open(logs / "app.log", "w"), stderr=subprocess.STDOUT)
        _wait(f"{APP}/health", app)

        started = time.perf_counter()
        response = requests.post(f"{APP}/api/ai_mode/start", json={"url": args.url, "provider": "manga-cloud", "polish": args.polish}, timeout=60)
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
                "SELECT status, requests, prompt_tokens, completion_tokens, cost_usd, charged_micros FROM jobs")]
        exported = None
        if job["status"] == "completed" and job.get("chapter_id"):
            # Exported the way a user would, from the editor's export, while the app is still up.
            exported = requests.get(f"{APP}/api/export/{job['chapter_id']}.zip", timeout=600)
    finally:
        for proc in (app, gateway):
            if proc is not None:
                proc.terminate()

    usage = rows[0] if rows else {}
    report = {
        "url": args.url,
        "model": os.environ.get("GATEWAY_UPSTREAM_MODEL"),
        "price_usd_per_m": {"input": float(os.environ.get("GATEWAY_PRICE_INPUT_PER_M", "0")),
                            "output": float(os.environ.get("GATEWAY_PRICE_OUTPUT_PER_M", "0"))},
        "polish": args.polish,
        # Run times are only comparable on the same CPU type.
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").is_file() else "unknown",
        "status": job["status"],
        "error": job.get("error"),
        "wall_s": round(wall, 1),
        "stages": [{k: s.get(k) for k in ("key", "status", "done", "total", "elapsed_s", "detail")} for s in job["stages"]],
        "report": job.get("report"),
        "gateway_job": usage,
        "transport": _transport(trace),
    }
    chapter_id = job.get("chapter_id")
    if exported is not None:
        if exported.ok:
            archive = out / "chapter.zip"
            archive.write_bytes(exported.content)
            report["zip_pages"] = _pages(archive, out)
            archive.unlink()
        else:
            report["export_error"] = f"HTTP {exported.status_code}: {exported.text[:300]}"
    manifest_path = ROOT / "data" / "processed" / str(chapter_id) / "manifest.json"
    if chapter_id and manifest_path.is_file():
        pages = json.loads(manifest_path.read_text(encoding="utf-8")).get("pages", [])
        report["slices"] = len(pages)
        report["slices_active"] = sum(1 for p in pages if not p.get("skipped"))
        report["clean_ms"] = _clean_timing(pages)
        try:
            report["pairs"] = _pairs(str(chapter_id), pages, out)
        except Exception as exc:  # noqa: BLE001 - the pairs are a review aid, never a reason to lose the report
            report["pairs_error"] = repr(exc)[:300]
        (out / "objects.json").write_text(json.dumps(_objects(pages), ensure_ascii=False), encoding="utf-8")
        report["lines"] = []
        for index, page in enumerate(pages):
            if page.get("skipped"):
                continue
            width = int(page.get("width") or 0) or _slice_width(chapter_id, index)
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict):
                    continue
                report["lines"].append({
                    "slice": index + 1, "id": obj.get("id"), "source": obj.get("source_read") or obj.get("ocr_text") or obj.get("text") or "",
                    "translation": obj.get("translation") or "",
                    "role": obj.get("typography_role"), "container": obj.get("container"),
                    "font": obj.get("font_ai_id") or (obj.get("style") or {}).get("font"),
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
    if chapter_id and manifest_path.is_file():
        report["scorecard"] = _scorecard(pages, report, out)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "wall_s", "gateway_job", "transport", "per_chapter", "per_slice", "readability", "scorecard")},
                     ensure_ascii=False, indent=1))
    return 0 if job["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
