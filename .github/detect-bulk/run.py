#!/usr/bin/env python3
"""Bulk text detection: the committed yolo26s soup vs the run-8 single.

Per chapter: download once with the app's own downloader, rejoin site-cut
strips, slice with the app's slicer, then detect every slice with BOTH
models (soup .pt committed in this directory, run-8 single from its run
artifact) through ultralytics (imgsz=1280, conf=0.25). Detection only — no
translate, clean, or render. Raw, strip, and slice files are deleted per
chapter so disk stays bounded; only the stats JSONL and a few sample crops
(with soup boxes in red, run-8 boxes in blue) survive.

Downloads are guarded for datacenter-runner IPs: chapters run strictly one
after another with a pause between them, a failed chapter is retried a few
times with growing backoff, and a run of consecutive failures stops the
shard instead of hammering a site that already said no. The app's own
downloader already sends a browser User-Agent and retries single images.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.downloader.registry import download_chapter  # noqa: E402
from app.downloader.slicer import slice_image  # noqa: E402
from join import group_fragments, join_fragments  # noqa: E402

CHAPTER_DELAY_SECONDS = 1.5
CHAPTER_MAX_ATTEMPTS = 3
CHAPTER_RETRY_BACKOFF_BASE = 5.0
MAX_CONSECUTIVE_FAILURES = 10


def shard_bounds(n, shard, shards):
    base, rem = divmod(n, shards)
    start = shard * base + min(shard, rem)
    return start, start + base + (1 if shard < rem else 0)


def safe_name(series):
    return "".join(c if c.isalnum() else "_" for c in series)


def save_sample(slice_path, boxes_soup, boxes_r8, out_path):
    img = cv2.imread(str(slice_path))
    if img is None:
        return False
    for x1, y1, x2, y2 in boxes_soup:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    for x1, y1, x2, y2 in boxes_r8:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
    h, w = img.shape[:2]
    if w > 900:
        img = cv2.resize(img, (900, int(h * 900 / w)))
    return bool(cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85]))


def predict_boxes(model, path):
    res = model.predict(str(path), imgsz=1280, conf=0.25, verbose=False)[0]
    return res.boxes.xyxy.tolist() if res.boxes is not None else []


def run_chapter(models, ch, work, samples_dir, sample_budget):
    model_soup, model_r8 = models
    row = {
        "series": ch["series"],
        "chapter": ch["chapter"],
        "url": ch["url"],
        "n_frags": 0,
        "n_strips": 0,
        "n_slices": 0,
        "n_boxes_soup": 0,
        "n_boxes_r8": 0,
        "ms_download": 0.0,
        "download_attempts": 0,
        "ms_detect_soup": 0.0,
        "ms_detect_r8": 0.0,
        "ms_total": 0.0,
        "error": None,
    }
    t_total = time.perf_counter()
    cdir = work / f"{safe_name(ch['series'])}_{ch['chapter']}"
    try:
        raw_dir = cdir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        t_dl = time.perf_counter()
        frags = None
        last_error = None
        for attempt in range(1, CHAPTER_MAX_ATTEMPTS + 1):
            try:
                frags = download_chapter(ch["url"], raw_dir)
                if frags:
                    break
                last_error = "no fragments downloaded"
            except Exception as exc:  # network / 429 / 5xx / page fetch failures
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < CHAPTER_MAX_ATTEMPTS:
                time.sleep(CHAPTER_RETRY_BACKOFF_BASE * attempt)
        row["download_attempts"] = attempt
        row["ms_download"] = (time.perf_counter() - t_dl) * 1000.0
        if not frags:
            if last_error and last_error != "no fragments downloaded":
                raise ValueError(last_error)
            frags = sorted(p for p in raw_dir.iterdir() if p.is_file())
        if not frags:
            raise ValueError("no fragments downloaded")
        row["n_frags"] = len(frags)
        joined_dir = cdir / "joined"
        sliced_dir = cdir / "sliced"
        joined_dir.mkdir(exist_ok=True)
        sliced_dir.mkdir(exist_ok=True)
        groups = group_fragments(frags)
        row["n_strips"] = len(groups)
        samples_left = sample_budget[0]
        for si, grp in enumerate(groups):
            strip = joined_dir / f"strip_{si:02d}.png"
            join_fragments(grp, strip)
            metas = slice_image(strip, sliced_dir, f"{si:02d}", return_metadata=True)
            for k, m in enumerate(metas):
                p = Path(m["path"])
                if not p.name.startswith(f"{si:02d}_") or not p.exists():
                    continue
                t0 = time.perf_counter()
                boxes_soup = predict_boxes(model_soup, p)
                row["ms_detect_soup"] += (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                boxes_r8 = predict_boxes(model_r8, p)
                row["ms_detect_r8"] += (time.perf_counter() - t0) * 1000.0
                row["n_slices"] += 1
                row["n_boxes_soup"] += len(boxes_soup)
                row["n_boxes_r8"] += len(boxes_r8)
                if (boxes_soup or boxes_r8) and samples_left > 0:
                    out = samples_dir / f"{safe_name(ch['series'])}_{ch['chapter']}_s{si:02d}_{k:02d}.jpg"
                    if save_sample(p, boxes_soup, boxes_r8, out):
                        samples_left -= 1
        sample_budget[0] = samples_left
    except Exception as exc:  # one bad chapter must not kill the shard
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
    row["ms_total"] = (time.perf_counter() - t_total) * 1000.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--model-soup", required=True)
    ap.add_argument("--model-r8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--work", default="bulk-work")
    ap.add_argument("--sample-budget", type=int, default=6)
    args = ap.parse_args()

    chapters = [json.loads(line) for line in Path(args.chapters).read_text().splitlines() if line.strip()]
    start, stop = shard_bounds(len(chapters), args.shard, args.shards)
    mine = chapters[start:stop]
    if not mine:
        raise SystemExit(f"empty shard {args.shard}/{args.shards} of {len(chapters)} chapters")

    from ultralytics import YOLO

    models = (YOLO(args.model_soup), YOLO(args.model_r8))
    samples_dir = Path(args.samples)
    samples_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    sample_budget = [args.sample_budget]

    print(f"shard {args.shard}/{args.shards}: chapters {start}..{stop - 1} ({len(mine)} chapters)", flush=True)
    consecutive_failures = 0
    failed = []
    stopped_early = None
    n_ok = 0
    with open(args.out, "a", encoding="utf-8") as fh:
        for i, ch in enumerate(mine):
            row = run_chapter(models, ch, work, samples_dir, sample_budget)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[{i + 1}/{len(mine)}] {ch['series']} ch{ch['chapter']}: "
                f"{row['n_slices']} slices soup={row['n_boxes_soup']} r8={row['n_boxes_r8']} "
                f"{row['ms_total']:.0f}ms" + (f" ERROR {row['error']}" if row["error"] else ""),
                flush=True,
            )
            if row["error"]:
                consecutive_failures += 1
                failed.append({
                    "series": row["series"],
                    "chapter": row["chapter"],
                    "url": row["url"],
                    "error": str(row["error"])[:300],
                })
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    stopped_early = (
                        f"suspected block: {consecutive_failures} consecutive chapter failures"
                    )
                    print(f"STOPPING SHARD EARLY: {stopped_early}", flush=True)
                    break
            else:
                consecutive_failures = 0
                n_ok += 1
            if i + 1 < len(mine):
                time.sleep(CHAPTER_DELAY_SECONDS)
        fh.write(json.dumps({
            "summary": True,
            "shard": args.shard,
            "n_attempted": i + 1,
            "n_ok": n_ok,
            "n_failed": len(failed),
            "failed_chapters": failed,
            "stopped_early": stopped_early,
        }, ensure_ascii=False) + "\n")
        fh.flush()
    print(f"shard {args.shard} done: {n_ok} ok, {len(failed)} failed"
          + (f", STOPPED EARLY: {stopped_early}" if stopped_early else ""), flush=True)


if __name__ == "__main__":
    main()
