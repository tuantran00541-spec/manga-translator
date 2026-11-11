"""Clean one chapter from a zip of page images (no download, no API).

Ingests the zip through the pipeline's upload path, then runs the local
detect + inpaint clean over every slice. Writes the chapter manifest and a
small summary JSON to --out for the evidence step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    zip_bytes = args.zip.read_bytes()
    pipeline = build_processing_pipeline()
    started = time.perf_counter()
    manifest = pipeline.create_chapter_from_uploads(
        args.chapter_id, [(args.zip.name, zip_bytes)]
    )
    pages = manifest.get("pages", [])
    ingest_s = time.perf_counter() - started
    pipeline.process_pages(
        args.chapter_id,
        list(range(len(pages))),
        **({"workers": args.workers} if args.workers else {}),
    )
    total_s = time.perf_counter() - started

    manifest = load_manifest_raw(args.chapter_id)
    pages = manifest.get("pages", [])
    cleaned = sum(1 for p in pages if p.get("clean"))
    boxes = sum(len(p.get("boxes", [])) for p in pages)
    run = manifest.get("last_processing_run") or {}
    summary = {
        "chapter_id": args.chapter_id,
        "zip": args.zip.name,
        "slices": len(pages),
        "cleaned": cleaned,
        "boxes": boxes,
        "ingest_s": round(ingest_s, 1),
        "total_s": round(total_s, 1),
        "wall_s": round((run.get("wall_ms") or 0) / 1000.0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if cleaned == len(pages) and pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
