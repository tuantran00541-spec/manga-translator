#!/usr/bin/env python3
"""Probe whether an extra 640 text-segmenter session degrades resident detector sessions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ort_utils import make_session


def _zero_feed(session, size: int):
    meta = session.get_inputs()[0]
    return {meta.name: np.zeros((1, 3, size, size), dtype=np.float32)}


def _bench(session, feed, repeats: int):
    # Warm once outside the measured series.
    session.run(None, feed)
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(None, feed)
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "runs_ms": [round(v, 3) for v in values],
        "median_ms": round(float(statistics.median(values)), 3),
        "mean_ms": round(float(statistics.mean(values)), 3),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resident", choices=["none", "openvino", "cpu"], required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    original_provider = os.environ.get("MANGA_ORT_PROVIDER")
    os.environ["MANGA_ORT_PROVIDER"] = "openvino"
    bubble = make_session(Path("models/bubble_yolo.onnx"))
    text = make_session(Path("models/text_segmenter.onnx"))

    resident = None
    if args.resident != "none":
        os.environ["MANGA_ORT_PROVIDER"] = (
            "openvino" if args.resident == "openvino" else "cpu"
        )
        resident = make_session(Path("models/text_segmenter_640.onnx"))
        resident_feed = _zero_feed(resident, 640)
        resident.run(None, resident_feed)

    os.environ["MANGA_ORT_PROVIDER"] = "openvino"
    result = {
        "resident": args.resident,
        "cpu_count": os.cpu_count(),
        "bubble_providers": bubble.get_providers(),
        "text_providers": text.get_providers(),
        "resident_providers": resident.get_providers() if resident is not None else [],
        "bubble": _bench(bubble, _zero_feed(bubble, 1024), args.repeats),
        "text": _bench(text, _zero_feed(text, 1024), args.repeats),
    }
    if resident is not None:
        result["resident_640"] = _bench(resident, resident_feed, args.repeats)

    if original_provider is None:
        os.environ.pop("MANGA_ORT_PROVIDER", None)
    else:
        os.environ["MANGA_ORT_PROVIDER"] = original_provider

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
