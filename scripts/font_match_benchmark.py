"""Run a repeatable latency/quality smoke benchmark for the native matcher."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from app.render.font_matcher import match_fonts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--text", default="Sample lettering")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    start = time.perf_counter()
    matches = match_fonts(image, None, args.text, top_k=args.top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(json.dumps({"elapsed_ms": round(elapsed_ms, 2), "matches": [item.as_dict() for item in matches]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
