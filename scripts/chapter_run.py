"""Process a whole chapter through the app pipeline and report speed and text left behind."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for b in boxes:
        if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
            mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 127
    return mask


def cpu_model() -> str:
    """The runner's CPU, as float results of the models can differ between CPU types."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def stage_seconds(manifest: dict) -> dict:
    """Seconds each processing stage took, summed over the slices, with the shared seam pass and the run's wall time."""
    totals: dict[str, float] = {}

    def add(name: str, value) -> None:
        if isinstance(value, (int, float)):
            totals[name] = totals.get(name, 0.0) + value

    for page in manifest.get("pages", []):
        metrics = page.get("processing_metrics") or {}
        for name, value in (metrics.get("timing_ms") or {}).items():
            add(f"{name}_ms", value)
        detector, inpaint = metrics.get("detector") or {}, metrics.get("auto_inpaint") or {}
        add("text_model_ms", detector.get("text_model_ms"))
        add("second_pass_boxes", detector.get("second_pass_boxes"))
        for name in ("lama_model_ms", "lama_model_runs", "lama_regions", "smart_fill_regions",
                     "session_lock_wait_ms", "ort_global_lock_wait_ms"):
            add(name, inpaint.get(name))
    run = manifest.get("last_processing_run") or {}
    for name, value in (run.get("shared_seam") or {}).items():
        add(f"seam_{name}", value)
    add("wall_ms", run.get("wall_ms"))
    add("workers", run.get("workers"))
    return {(name[:-3] + "_s" if name.endswith("_ms") else name):
            (round(value / 1000.0, 1) if name.endswith("_ms") else value) for name, value in totals.items()}


def pair(original: np.ndarray, clean: np.ndarray, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in (("original", original), ("clean", clean)):
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chapter-id", default="c1a90001")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = build_processing_pipeline()
    started = time.perf_counter()
    pages = pipeline.download_chapter(args.url, args.chapter_id, workers=2)["pages"]
    download_s = time.perf_counter() - started
    started = time.perf_counter()
    pipeline.process_pages(args.chapter_id, list(range(len(pages))))
    process_s = time.perf_counter() - started
    print(f"{len(pages)} slices processed in {process_s:.1f} s", flush=True)

    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)

    def found(image: np.ndarray) -> np.ndarray:
        return text_mask(image.shape[:2], kiuyha.text_boxes(image))

    blocks = left = 0
    lefts, per_slice = [], []
    for number, page in enumerate(load_manifest_raw(args.chapter_id)["pages"]):
        if not page.get("clean"):
            continue
        core = page.get("stitch_core") or {}
        original, clean = read_image(Path(page["original"])), read_image(Path(page["clean"]))
        y0, y1 = int(core.get("core_y1", 0)), int(core.get("core_y2", original.shape[0]))
        before, after = np.ascontiguousarray(original[y0:y1]), np.ascontiguousarray(clean[y0:y1])
        text, rest = found(before), found(after)
        count, labels = cv2.connectedComponents(cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)))
        per_slice.append((count - 1, number, page, y0, y1))
        for i in range(1, count):
            block = (labels == i) & text
            if block.sum() < 150:
                continue
            blocks += 1
            if (rest & block).sum() >= 0.3 * block.sum():
                left += 1
                ys, xs = np.nonzero(block)
                box = (max(0, int(xs.min()) - 40), max(0, int(ys.min()) - 40),
                       min(before.shape[1], int(xs.max()) + 40), min(before.shape[0], int(ys.max()) + 40))
                lefts.append({"slice": number, "box": box, "px": int(block.sum())})
                pair(before, after, box, args.out / f"left-{len(lefts):02d}.jpg")

    for rank, (_, number, page, y0, y1) in enumerate(sorted(per_slice, key=lambda s: -s[0])[:8], 1):
        before, after = (read_image(Path(page[key]))[y0:y1] for key in ("original", "clean"))
        side = np.concatenate([before, np.full((before.shape[0], 12, 3), 255, np.uint8), after], axis=1)
        side = cv2.resize(side, None, fx=900 / side.shape[1], fy=900 / side.shape[1])
        cv2.imwrite(str(args.out / f"sample-{rank}-slice{number:03d}.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 85])

    report = {
        "url": args.url, "slices": len(pages), "cpu": cpu_model(),
        "download_s": round(download_s, 1), "process_s": round(process_s, 1),
        "per_slice_s": round(process_s / max(1, len(pages)), 2),
        "time_s": stage_seconds(load_manifest_raw(args.chapter_id)),
        "blocks": blocks, "blocks_left": left, "left": lefts,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "left"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
