from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from urllib.parse import urljoin, urlparse

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPORT_DIR = _REPO_ROOT / "benchmark-results"
BROWSE_URL = "https://asurascans.com/browse"
ASURA_HOSTS = {"asurascans.com", "www.asurascans.com"}
HTML_LIMIT = 8 * 1024 * 1024


class E2EFailure(RuntimeError):
    pass


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _fetch_html(url: str) -> str:
    from app.downloader.http import read_response_limited, safe_get

    response = safe_get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/130.0 Safari/537.36 manga-translator-e2e/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
        stream=True,
    )
    try:
        payload = read_response_limited(response, limit_bytes=HTML_LIMIT)
        encoding = response.encoding or "utf-8"
        return payload.decode(encoding, errors="replace")
    finally:
        response.close()


def _asura_url(base: str, href: str) -> str | None:
    value = urljoin(base, href)
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ASURA_HOSTS:
        return None
    return value.split("#", 1)[0]


def _series_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        value = _asura_url(BROWSE_URL, str(anchor.get("href") or ""))
        if value is None or "/comics/" not in urlparse(value).path:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _chapter_links(series_url: str, html: str) -> list[str]:
    from app.downloader.asura import is_asura_chapter_page

    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        value = _asura_url(series_url, str(anchor.get("href") or ""))
        if value is None or not is_asura_chapter_page(value):
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _playwright_links(url: str, contains: str) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return []
    out: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                page.wait_for_timeout(1500)
                hrefs = page.locator("a[href]").evaluate_all(
                    "els => els.map(e => e.href).filter(Boolean)"
                )
                for href in hrefs:
                    value = _asura_url(url, str(href))
                    if value and contains in urlparse(value).path and value not in out:
                        out.append(value)
            finally:
                browser.close()
    except Exception:
        return []
    return out


def discover_candidates(seed: int, *, limit: int = 30) -> dict:
    rng = random.Random(seed)
    browse_html = _fetch_html(BROWSE_URL)
    series = _series_links(browse_html)
    if not series:
        series = _playwright_links(BROWSE_URL, "/comics/")
    if not series:
        raise E2EFailure("Asura browse page exposed no comic series links")
    rng.shuffle(series)

    chapters: list[dict] = []
    series_errors: list[dict] = []
    for series_url in series[: min(len(series), 18)]:
        try:
            html = _fetch_html(series_url)
            links = _chapter_links(series_url, html)
            if not links:
                links = _playwright_links(series_url, "/chapter/")
            rng.shuffle(links)
            for chapter_url in links[:8]:
                chapters.append(
                    {"series_url": series_url, "chapter_url": chapter_url}
                )
                if len(chapters) >= limit:
                    break
        except Exception as exc:
            series_errors.append(
                {
                    "series_url": series_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if len(chapters) >= limit:
            break
    if not chapters:
        raise E2EFailure(
            "Could not discover an Asura chapter URL from sampled series"
        )
    rng.shuffle(chapters)
    return {
        "seed": seed,
        "browse_url": BROWSE_URL,
        "series_discovered": len(series),
        "chapter_candidates": chapters,
        "series_errors": series_errors,
    }


def _chapter_id(seed: int, chapter_url: str) -> str:
    return hashlib.sha256(
        f"{seed}:{chapter_url}".encode("utf-8")
    ).hexdigest()[:8]


def _remove_chapter_artifacts(chapter_id: str) -> None:
    from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR

    for root in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        shutil.rmtree(root / chapter_id, ignore_errors=True)


def ingest_candidate(
    pipeline,
    seed: int,
    candidate: dict,
    workers: int,
) -> tuple[str, dict, float]:
    chapter_url = str(candidate["chapter_url"])
    chapter_id = _chapter_id(seed, chapter_url)
    _remove_chapter_artifacts(chapter_id)
    started = now_ms()
    try:
        manifest = pipeline.download_chapter(
            chapter_url, chapter_id, workers=workers
        )
    except Exception:
        _remove_chapter_artifacts(chapter_id)
        raise
    elapsed = now_ms() - started
    pages = manifest.get("pages", [])
    if not pages:
        _remove_chapter_artifacts(chapter_id)
        raise E2EFailure(f"{chapter_url} downloaded but produced zero slices")
    return chapter_id, manifest, elapsed


def _process_summary(manifest: dict, process_wall_ms: float) -> dict:
    timings: dict[str, list[float]] = {
        "read": [],
        "detect": [],
        "auto_inpaint": [],
        "manual_inpaint": [],
        "write": [],
        "total": [],
    }
    detector_totals = {
        "bubble_model_ms": 0.0,
        "text_model_ms": 0.0,
        "mser_ms": 0.0,
        "text_grayscale_fallback_ms": 0.0,
        "text_grayscale_fallback_runs": 0,
        "total_ms": 0.0,
        "records": 0,
        "authorized": 0,
        "review_only": 0,
    }
    inpaint_totals = {
        "lama_model_runs": 0,
        "lama_model_ms": 0,
        "smart_fill_regions": 0,
        "lama_regions": 0,
        "clusters": 0,
    }
    for page in manifest.get("pages", []):
        metrics = page.get("processing_metrics") or {}
        timing = metrics.get("timing_ms") or {}
        for name in timings:
            value = timing.get(name)
            if value is not None:
                timings[name].append(float(value))
        detector = metrics.get("detector") or {}
        for name in detector_totals:
            value = detector.get(name)
            if value is not None:
                detector_totals[name] += (
                    float(value) if name.endswith("_ms") else int(value)
                )
        auto = metrics.get("auto_inpaint") or {}
        for name in inpaint_totals:
            value = auto.get(name)
            if value is not None:
                inpaint_totals[name] += int(value)
    timing_report = {}
    for name, values in timings.items():
        timing_report[name] = {
            "sum_ms": round(sum(values), 3),
            "mean_ms": round(statistics.fmean(values), 3) if values else None,
            "p95_ms": round(percentile(values, 95), 3) if values else None,
            "max_ms": round(max(values), 3) if values else None,
        }
    return {
        "wall_ms": round(process_wall_ms, 3),
        "per_page_timing": timing_report,
        "detector": detector_totals,
        "inpaint": inpaint_totals,
        "last_processing_run": manifest.get("last_processing_run"),
    }


def run_ocr(
    chapter_id: str,
    pipeline,
    *,
    source_lang: str,
    concurrency: int,
) -> dict:
    from app.ocr.multi_lang_ocr import MultiLangOCR
    from app.ocr.service import OCRService

    service = OCRService(MultiLangOCR(), pipeline)
    plan = service.plan_chapter(chapter_id)
    started = now_ms()
    results: list[dict] = []
    failures: list[dict] = []
    latencies: list[float] = []

    def work(item: tuple[int, str]):
        page_index, box_id = item
        t0 = now_ms()
        result = service.inspect_box_id(
            chapter_id, page_index, box_id, source_lang
        )
        return result, now_ms() - t0

    with ThreadPoolExecutor(
        max_workers=max(1, min(int(concurrency), 2))
    ) as pool:
        futures = {pool.submit(work, item): item for item in plan}
        for future in as_completed(futures):
            page_index, box_id = futures[future]
            try:
                result, latency = future.result()
                results.append(result)
                latencies.append(float(latency))
            except Exception as exc:
                failures.append(
                    {
                        "page_index": page_index,
                        "box_id": box_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    recognized = sum(
        bool(str(result.get("text") or "").strip()) for result in results
    )
    quality: dict[str, int] = {}
    models: dict[str, int] = {}
    for result in results:
        quality_name = str(result.get("quality") or "unknown")
        quality[quality_name] = quality.get(quality_name, 0) + 1
        model = str(
            result.get("model") or result.get("engine") or "unknown"
        )
        models[model] = models.get(model, 0) + 1
    return {
        "status": (
            "pass" if plan and recognized > 0 and not failures else "fail"
        ),
        "planned": len(plan),
        "completed": len(results),
        "recognized": int(recognized),
        "empty": len(results) - int(recognized),
        "failures": failures,
        "quality": quality,
        "models": models,
        "wall_ms": round(now_ms() - started, 3),
        "mean_box_ms": (
            round(statistics.fmean(latencies), 3) if latencies else None
        ),
        "p95_box_ms": (
            round(percentile(latencies, 95), 3) if latencies else None
        ),
    }


def _ensure_text_objects(chapter_id: str) -> int:
    from app.manifest_utils import (
        get_manifest_lock,
        load_manifest_raw,
        save_manifest_raw,
    )
    from app.text_objects import ensure_page_text_objects

    count = 0
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        changed = False
        for page in manifest.get("pages", []):
            _, page_changed = ensure_page_text_objects(page)
            changed = changed or page_changed
            count += sum(
                1
                for obj in (page.get("text_objects") or [])
                if isinstance(obj, dict) and not obj.get("source_missing")
            )
        if changed:
            save_manifest_raw(chapter_id, manifest)
    return count


def _apply_source_echo_for_render(chapter_id: str) -> int:
    from app.manifest_utils import (
        get_manifest_lock,
        invalidate_page_render,
        load_manifest_raw,
        save_manifest_raw,
    )

    changed = 0
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        changed_pages: set[int] = set()
        for page_index, page in enumerate(manifest.get("pages", [])):
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict) or obj.get("source_missing"):
                    continue
                if str(obj.get("translation") or "").strip():
                    continue
                source = str(obj.get("ocr_text") or "").strip()
                if not source:
                    continue
                obj["translation"] = source
                obj["translation_source"] = "e2e_source_echo"
                obj["translation_model"] = "none"
                changed += 1
                changed_pages.add(page_index)
        for page_index in changed_pages:
            invalidate_page_render(manifest, page_index)
        if changed_pages:
            save_manifest_raw(chapter_id, manifest)
    return changed


def run_translation(
    chapter_id: str,
    *,
    source_lang: str,
    target_lang: str,
    budget_usd: float,
) -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return {"status": "missing_secret", "translated": 0, "wall_ms": 0.0}

    from app.routers.translation import (
        TranslateChapterRequest,
        translate_chapter,
    )

    started = now_ms()
    try:
        payload = asyncio.run(
            translate_chapter(
                TranslateChapterRequest(
                    chapter_id=chapter_id,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    budget_usd=budget_usd,
                    force=True,
                )
            )
        )
        run = payload.get("translation_run") or {}
        return {
            "status": "pass",
            **run,
            "wall_ms": round(now_ms() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "fail",
            "translated": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_ms": round(now_ms() - started, 3),
        }


def _contact_sheet(
    chapter_id: str,
    manifest: dict,
    output: Path,
    *,
    max_rows: int = 8,
) -> None:
    from app.routers.image import _current_rendered_path

    pages = [
        (index, page)
        for index, page in enumerate(manifest.get("pages", []))
        if not page.get("skipped")
    ]
    if not pages:
        return
    if len(pages) > max_rows:
        indices = np.linspace(
            0, len(pages) - 1, max_rows, dtype=int
        ).tolist()
        pages = [pages[i] for i in indices]

    thumb_w = 300
    max_h = 720
    gap = 10
    label_h = 26
    rows: list[Image.Image] = []
    for page_index, page in pages:
        candidates: list[tuple[str, Path | None]] = [
            ("original", Path(str(page.get("original") or ""))),
            (
                "clean",
                Path(str(page.get("clean") or ""))
                if page.get("clean")
                else None,
            ),
            (
                "rendered",
                _current_rendered_path(chapter_id, page_index, manifest),
            ),
        ]
        thumbs: list[Image.Image] = []
        for label, path in candidates:
            canvas = Image.new(
                "RGB", (thumb_w, max_h + label_h), "white"
            )
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), f"{page_index + 1}: {label}", fill="black")
            if path and path.is_file():
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((thumb_w, max_h))
                    x = (thumb_w - image.width) // 2
                    canvas.paste(image, (x, label_h))
            thumbs.append(canvas)
        row = Image.new(
            "RGB",
            (thumb_w * 3 + gap * 2, max_h + label_h),
            "white",
        )
        for column, thumb in enumerate(thumbs):
            row.paste(thumb, (column * (thumb_w + gap), 0))
            thumb.close()
        rows.append(row)

    sheet = Image.new(
        "RGB",
        (
            rows[0].width,
            sum(row.height for row in rows) + gap * (len(rows) - 1),
        ),
        "white",
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
        row.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=86, optimize=True)
    sheet.close()


def run_render_export(chapter_id: str) -> dict:
    from app.manifest_utils import load_manifest_raw
    from app.routers.export import export_chapter, render_chapter

    render_started = now_ms()
    render_result = render_chapter(chapter_id)
    render_ms = now_ms() - render_started

    export_started = now_ms()
    response = export_chapter(chapter_id)
    export_ms = now_ms() - export_started
    archive = Path(str(response.path))
    if not archive.is_file():
        raise E2EFailure("export route returned without a chapter archive")

    manifest = load_manifest_raw(chapter_id)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    copied_archive = REPORT_DIR / archive.name
    shutil.copy2(archive, copied_archive)
    manifest_path = REPORT_DIR / "real-e2e-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _contact_sheet(
        chapter_id,
        manifest,
        REPORT_DIR / "real-e2e-contact-sheet.jpg",
    )

    return {
        "status": "pass",
        "render_ms": round(render_ms, 3),
        "export_ms": round(export_ms, 3),
        "archive": copied_archive.as_posix(),
        "archive_size_bytes": copied_archive.stat().st_size,
        "chapter_render": render_result.get("chapter_render"),
    }


def _model_provenance() -> dict:
    path = _REPO_ROOT / "models" / ".e2e-provenance.json"
    if not path.is_file():
        return {"mode": "unknown", "production_exact": False}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "invalid", "production_exact": False}


def execute(args: argparse.Namespace) -> dict:
    from app.manifest_utils import load_manifest_raw
    from app.pipeline import ChapterPipeline

    seed = int(args.seed)
    selection = {
        "seed": seed,
        "requested_chapter_url": args.chapter_url or None,
        "attempts": [],
    }
    if args.chapter_url:
        candidates = [
            {"series_url": None, "chapter_url": args.chapter_url}
        ]
        discovery = None
    else:
        discovery = discover_candidates(
            seed,
            limit=max(10, args.max_download_attempts * 3),
        )
        candidates = discovery["chapter_candidates"]
        selection["discovery"] = discovery

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chosen = None
    chapter_id = None
    manifest = None
    ingest_ms = 0.0

    for candidate in candidates[: args.max_download_attempts]:
        t0 = now_ms()
        try:
            cid, candidate_manifest, elapsed = ingest_candidate(
                pipeline, seed, candidate, args.workers
            )
            chosen = candidate
            chapter_id = cid
            manifest = candidate_manifest
            ingest_ms = elapsed
            selection["attempts"].append(
                {
                    **candidate,
                    "status": "pass",
                    "chapter_id": cid,
                    "wall_ms": round(now_ms() - t0, 3),
                }
            )
            break
        except Exception as exc:
            selection["attempts"].append(
                {
                    **candidate,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_ms": round(now_ms() - t0, 3),
                }
            )
    if manifest is None or chapter_id is None or chosen is None:
        raise E2EFailure(
            "No sampled Asura chapter completed production ingestion"
        )

    selection["selected"] = {**chosen, "chapter_id": chapter_id}
    (REPORT_DIR / "real-e2e-selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    page_indices = list(range(len(manifest.get("pages", []))))
    process_started = now_ms()
    pipeline.process_pages(
        chapter_id, page_indices, workers=args.workers
    )
    process_wall_ms = now_ms() - process_started
    manifest = load_manifest_raw(chapter_id)
    process = _process_summary(manifest, process_wall_ms)

    ocr = run_ocr(
        chapter_id,
        pipeline,
        source_lang=args.source_lang,
        concurrency=args.ocr_concurrency,
    )
    text_objects = _ensure_text_objects(chapter_id)
    translation = run_translation(
        chapter_id,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        budget_usd=args.translation_budget_usd,
    )
    source_echo = 0
    if translation.get("status") != "pass":
        source_echo = _apply_source_echo_for_render(chapter_id)
    render_export = run_render_export(chapter_id)
    manifest = load_manifest_raw(chapter_id)

    source_pages = max(
        (
            int(page.get("source_page", -1))
            for page in manifest.get("pages", [])
        ),
        default=-1,
    ) + 1
    boxes = [
        box
        for page in manifest.get("pages", [])
        for box in (page.get("boxes") or [])
        if isinstance(box, dict) and not box.get("removed")
    ]
    failures: list[str] = []
    if not manifest.get("pages"):
        failures.append("chapter produced zero slices")
    if not boxes:
        failures.append("detector produced zero boxes across full chapter")
    if ocr.get("status") != "pass":
        failures.append(
            "OCR did not complete with recognized text across the chapter"
        )
    if (
        args.require_translation
        and translation.get("status") != "pass"
    ):
        failures.append(
            "live translation required but status is "
            f"{translation.get('status')!r}"
        )
    if render_export.get("status") != "pass":
        failures.append("render/export did not pass")

    model_provenance = _model_provenance()
    bubble_source = model_provenance.get("bubble_detector") or {}
    reproducible_models = bool(
        model_provenance.get("production_exact")
        or (
            model_provenance.get("mode") == "public_equivalent"
            and bubble_source.get("expected_pt_sha256")
        )
    )
    live_translation_ok = translation.get("status") == "pass"
    promotion_ready = (
        not failures and live_translation_ok and reproducible_models
    )

    return {
        "status": "pass" if not failures else "fail",
        "promotion_ready": bool(promotion_ready),
        "source_revision": os.getenv("GITHUB_SHA") or None,
        "seed": seed,
        "chapter_id": chapter_id,
        "chapter_url": chosen["chapter_url"],
        "series_url": chosen.get("series_url"),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "source_pages": source_pages,
        "slices": len(manifest.get("pages", [])),
        "boxes": {
            "total": len(boxes),
            "safe_to_inpaint": sum(
                bool(box.get("safe_to_inpaint")) for box in boxes
            ),
            "needs_review": sum(
                bool(box.get("needs_review")) for box in boxes
            ),
            "ocr_eligible": sum(
                bool(box.get("ocr_eligible")) for box in boxes
            ),
        },
        "ingest_ms": round(ingest_ms, 3),
        "process": process,
        "ocr": ocr,
        "text_objects": text_objects,
        "translation": translation,
        "render_smoke_source_echo_objects": source_echo,
        "render_export": render_export,
        "model_provenance": model_provenance,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real full-chapter AsuraScan E2E: ingest -> detect/inpaint -> "
            "OCR -> translate -> render/export."
        )
    )
    parser.add_argument("--chapter-url", default="")
    parser.add_argument(
        "--seed", default=os.getenv("GITHUB_RUN_ID") or int(time.time())
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ocr-concurrency", type=int, default=2)
    parser.add_argument(
        "--source-lang",
        default="en",
        choices=["en", "ch", "ja", "korean"],
    )
    parser.add_argument("--target-lang", default="vi")
    parser.add_argument("--translation-budget-usd", type=float, default=0.02)
    parser.add_argument("--max-download-attempts", type=int, default=6)
    parser.add_argument("--require-translation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.workers = max(1, min(int(args.workers), 2))
    args.ocr_concurrency = max(1, min(int(args.ocr_concurrency), 2))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = now_ms()
    try:
        report = execute(args)
    except Exception as exc:
        report = {
            "status": "fail",
            "promotion_ready": False,
            "source_revision": os.getenv("GITHUB_SHA") or None,
            "seed": int(args.seed),
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    report["wall_ms"] = round(now_ms() - started, 3)
    path = REPORT_DIR / "real-e2e-asura-report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
