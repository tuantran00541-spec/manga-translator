#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.downloader.registry import download_chapter
from app.downloader.slicer import slice_image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", default="benchmark-results/chapter210")
    args = parser.parse_args()

    root = Path(args.output).resolve()
    raw_dir = root / "raw"
    slices_dir = root / "slices"
    root.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    paths = download_chapter(args.url, raw_dir)
    if not paths:
        raise RuntimeError("Downloader returned no chapter images")

    pages = []
    slices = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
        pages.append(
            {
                "index": index,
                "path": str(path.relative_to(root)),
                "filename": path.name,
                "width": width,
                "height": height,
                "format": fmt,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

        metadata = slice_image(path, slices_dir, f"p{index:03d}", return_metadata=True)
        for slice_index, item in enumerate(metadata):
            slice_path = Path(item["path"]).resolve()
            with Image.open(slice_path) as slice_image_file:
                slice_width, slice_height = slice_image_file.size
            slices.append(
                {
                    "page_index": index,
                    "slice_index": slice_index,
                    "path": str(slice_path),
                    "filename": slice_path.name,
                    "width": slice_width,
                    "height": slice_height,
                    "source_y1": int(item.get("source_y1", 0)),
                    "source_y2": int(item.get("source_y2", height)),
                    "core_y1": int(item.get("core_y1", 0)),
                    "core_y2": int(item.get("core_y2", slice_height)),
                    "core_source_y1": int(item.get("core_source_y1", 0)),
                    "core_source_y2": int(item.get("core_source_y2", height)),
                    "unsafe_before": bool(item.get("unsafe_before", False)),
                    "unsafe_after": bool(item.get("unsafe_after", False)),
                }
            )

    payload = {
        "url": args.url,
        "page_count": len(pages),
        "total_bytes": sum(item["bytes"] for item in pages),
        "slice_count": len(slices),
        "max_slice_height": max((item["height"] for item in slices), default=0),
        "pages": pages,
        "slices": slices,
    }
    (root / "chapter_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "all_slices.txt").write_text(
        "\n".join(item["path"] for item in slices) + "\n", encoding="utf-8"
    )

    print("@@CHAPTER@@" + json.dumps({
        "page_count": payload["page_count"],
        "total_bytes": payload["total_bytes"],
        "slice_count": payload["slice_count"],
        "max_slice_height": payload["max_slice_height"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
