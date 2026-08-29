#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from app.downloader.registry import download_chapter


def _sample_indices(count: int, wanted: int) -> list[int]:
    if count <= 0:
        return []
    wanted = max(1, min(wanted, count))
    if wanted == 1:
        return [count // 2]
    out = {round(i * (count - 1) / (wanted - 1)) for i in range(wanted)}
    return sorted(out)


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
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.output).resolve()
    raw_dir = root / "raw"
    root.mkdir(parents=True, exist_ok=True)

    paths = download_chapter(args.url, raw_dir)
    if not paths:
        raise RuntimeError("Downloader returned no chapter images")

    pages = []
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

    selected = _sample_indices(len(pages), args.samples)
    selected_paths = [str((raw_dir / paths[i].name).resolve()) for i in selected]
    payload = {
        "url": args.url,
        "page_count": len(pages),
        "total_bytes": sum(item["bytes"] for item in pages),
        "selected_indices": selected,
        "selected_paths": selected_paths,
        "pages": pages,
    }
    (root / "chapter_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "selected_pages.txt").write_text(
        "\n".join(selected_paths) + "\n", encoding="utf-8"
    )

    print("@@CHAPTER@@" + json.dumps({
        "page_count": payload["page_count"],
        "total_bytes": payload["total_bytes"],
        "selected_indices": selected,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
