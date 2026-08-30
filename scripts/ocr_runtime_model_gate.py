from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def _normalize(text: str) -> str:
    return "".join(str(text or "").split())


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _crop_polygon(image: np.ndarray, polygon, pad: int = 20) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    h, w = image.shape[:2]
    x1 = max(0, int(pts[:, 0].min()) - pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    x2 = min(w, int(pts[:, 0].max()) + 1 + pad)
    y2 = min(h, int(pts[:, 1].max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Polygon produced an empty crop")
    return image[y1:y2, x1:x2]


def _build_slices(raw_dir: Path, output_dir: Path) -> list[Path]:
    from app.downloader.slicer import slice_image

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(path for path in raw_dir.iterdir() if path.is_file())
    if not raw_paths:
        raise ValueError(f"No Chapter 210 raw images found in {raw_dir}")
    slices: list[Path] = []
    for index, source in enumerate(raw_paths):
        slices.extend(slice_image(source, output_dir, f"p{index:03d}"))
    return slices


def _identity_gate(lang: str) -> dict:
    from app.ocr.identity import engine_identity

    previous = os.environ.get("MANGA_OCR_TARGET_SELECTION")
    try:
        engine_identity.cache_clear()
        os.environ["MANGA_OCR_TARGET_SELECTION"] = "centered"
        centered = engine_identity(lang)
        engine_identity.cache_clear()
        os.environ["MANGA_OCR_TARGET_SELECTION"] = "all"
        all_mode = engine_identity(lang)
    finally:
        if previous is None:
            os.environ.pop("MANGA_OCR_TARGET_SELECTION", None)
        else:
            os.environ["MANGA_OCR_TARGET_SELECTION"] = previous
        engine_identity.cache_clear()
    if centered == all_mode:
        raise AssertionError(f"OCR cache identity did not change for {lang}")
    return {"centered": centered, "all": all_mode, "changed": True}


def _read(engine, bgr: np.ndarray, lang: str):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    started = time.perf_counter()
    result = engine.read_detailed(rgb, lang)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def _english_probe(engine, slices_dir: Path, ppocr_summary: dict) -> dict:
    chosen = None
    for row in ppocr_summary.get("rows", []):
        source_name = Path(str(row.get("source") or "")).name
        texts = list(row.get("texts") or [])
        polygons = list(row.get("polygons") or [])
        for text, polygon in zip(texts, polygons):
            if source_name == "p000_00.png" and str(text).strip() == "210":
                chosen = (source_name, str(text), polygon)
                break
        if chosen:
            break
    if chosen is None:
        raise AssertionError("Could not locate stable English Chapter 210 probe")

    source_name, reference, polygon = chosen
    image = _read_bgr(slices_dir / source_name)
    crop = _crop_polygon(image, polygon, pad=28)
    result, latency = _read(engine, crop, "en")
    if result.model != "PP-OCRv6_small_rec":
        raise AssertionError(f"English routed to {result.model!r}")
    if not str(result.text or "").strip():
        raise AssertionError("English real-model OCR returned blank text")
    return {
        "source": source_name,
        "reference": reference,
        "text": result.text,
        "model": result.model,
        "confidence": result.confidence,
        "quality": result.quality,
        "region_count": result.region_count,
        "latency_ms": latency,
    }


def _korean_probe(engine, slices_dir: Path, ko_summary: dict) -> dict:
    attempts: list[dict] = []
    for item in ko_summary.get("hangul_examples", [])[:8]:
        source_name = Path(str(item.get("source") or "")).name
        source_path = slices_dir / source_name
        if not source_path.is_file():
            continue
        image = _read_bgr(source_path)
        crop = _crop_polygon(image, item.get("polygon"), pad=24)
        result, latency = _read(engine, crop, "ko")
        attempt = {
            "source": source_name,
            "expected_research_text": item.get("ko_text"),
            "text": result.text,
            "model": result.model,
            "confidence": result.confidence,
            "quality": result.quality,
            "region_count": result.region_count,
            "latency_ms": latency,
        }
        attempts.append(attempt)
        if result.model != "korean_PP-OCRv5_mobile_rec":
            raise AssertionError(f"Korean routed to {result.model!r}")
        if _HANGUL_RE.search(str(result.text or "")):
            return {"hangul_recovered": True, "attempts": attempts}
    raise AssertionError(
        "Korean production route ran but did not recover Hangul on known Chapter 210 examples: "
        + json.dumps(attempts, ensure_ascii=False)
    )


def _japanese_probe(engine, japanese_root: Path) -> dict:
    benchmark = _load_json(japanese_root / "mangaocr.json")
    rows = list(benchmark.get("rows") or [])
    preferred = next((row for row in rows if row.get("exact")), rows[0] if rows else None)
    if preferred is None:
        raise AssertionError("Japanese held-out artifact has no rows")
    image_path = japanese_root / "images" / f"{preferred['id']}.png"
    image = _read_bgr(image_path)
    result, latency = _read(engine, image, "ja")
    if result.model != "manga-ocr":
        raise AssertionError(f"Japanese routed to {result.model!r}")
    if _normalize(result.text) != _normalize(preferred.get("ground_truth") or ""):
        raise AssertionError(
            f"Japanese held-out regression on {preferred['id']}: "
            f"expected={preferred.get('ground_truth')!r} actual={result.text!r}"
        )
    return {
        "id": preferred["id"],
        "ground_truth": preferred.get("ground_truth"),
        "text": result.text,
        "model": result.model,
        "quality": result.quality,
        "latency_ms": latency,
        "exact": True,
    }


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from app.ocr.multi_lang_ocr import MultiLangOCR

    engine = MultiLangOCR()
    if engine._paddle_target_mode != "centered":
        raise AssertionError(f"Default OCR target mode is {engine._paddle_target_mode!r}, expected centered")
    if engine._paddle.tier != "small":
        raise AssertionError(f"Default PP-OCRv6 tier is {engine._paddle.tier!r}, expected small")
    if engine._paddle.textline_orientation:
        raise AssertionError("Text-line orientation unexpectedly enabled by default")

    slices_dir = args.work_dir / "chapter210-slices"
    slices = _build_slices(args.chapter_raw_dir, slices_dir)
    if len(slices) != 94:
        raise AssertionError(f"Chapter 210 slicer produced {len(slices)} slices, expected 94")

    ppocr = _load_json(args.chapter_ocr_dir / "ppocrv6" / "summary.json")
    ko = _load_json(args.chapter_ocr_dir / "korean_fallback" / "summary.json")

    identities = {
        "en": _identity_gate("en"),
        "ko": _identity_gate("ko"),
    }
    en = _english_probe(engine, slices_dir, ppocr)
    ko_result = _korean_probe(engine, slices_dir, ko)
    ja = _japanese_probe(engine, args.japanese_dir)

    latencies = [en["latency_ms"], ja["latency_ms"]]
    latencies.extend(item["latency_ms"] for item in ko_result["attempts"])
    return {
        "status": "pass",
        "chapter210_slices": len(slices),
        "defaults": {
            "tier": engine._paddle.tier,
            "target_mode": engine._paddle_target_mode,
            "textline_orientation": engine._paddle.textline_orientation,
        },
        "cache_identity": identities,
        "english": en,
        "korean": ko_result,
        "japanese": ja,
        "mean_probe_latency_ms": statistics.fmean(latencies),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-model production OCR route gate")
    parser.add_argument("--chapter-raw-dir", required=True, type=Path)
    parser.add_argument("--chapter-ocr-dir", required=True, type=Path)
    parser.add_argument("--japanese-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        report = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
