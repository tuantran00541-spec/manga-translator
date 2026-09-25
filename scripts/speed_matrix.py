from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refactor_parity  # noqa: E402

ENV_KEYS = (
    "MANGA_ORT_PROVIDER",
    "MANGA_ORT_OPENVINO_SCOPE",
    "MANGA_ORT_OPENVINO_THREADS",
    "MANGA_ORT_OPENVINO_STREAMS",
    "MANGA_ORT_INTRA_OP_THREADS",
    "MANGA_USE_DYNAMIC_LAMA",
)


def run(url: str, out: Path, workers: int) -> None:
    started = time.perf_counter()
    refactor_parity.run(url, out, workers)
    from app.manifest_utils import load_manifest_raw

    pages = load_manifest_raw(refactor_parity.CHAPTER_ID)["pages"]
    detect = inpaint = lama_runs = 0.0
    for page in pages:
        metrics = page.get("processing_metrics") or {}
        timing = metrics.get("timing_ms") or {}
        detect += float(timing.get("detect") or 0)
        inpaint += float(timing.get("auto_inpaint") or 0)
        lama_runs += int((metrics.get("auto_inpaint") or {}).get("lama_model_runs") or 0)
    run_json = json.loads((out / refactor_parity.RUN_JSON).read_text())
    run_json.update(
        workers=workers,
        cpu_count=os.cpu_count(),
        env={key: os.environ.get(key, "") for key in ENV_KEYS},
        detect_s_sum=round(detect / 1000, 1),
        inpaint_s_sum=round(inpaint / 1000, 1),
        lama_model_runs=int(lama_runs),
        total_s=round(time.perf_counter() - started, 1),
    )
    (out / refactor_parity.RUN_JSON).write_text(json.dumps(run_json, indent=1))


def _diff(base: Path, head: Path, index: int) -> dict | None:
    a = cv2.imread(str(base / f"clean_{index:03d}.png"))
    b = cv2.imread(str(head / f"clean_{index:03d}.png"))
    if a is None or b is None or a.shape != b.shape:
        return None
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return {
        "max": int(diff.max()),
        "over2": int((diff > 2).sum()),
        "over16": int((diff > 16).sum()),
        "mean": float(diff.mean()),
    }


def summarize(runs: list[Path], base_name: str, out: Path) -> int:
    base = next(path for path in runs if path.name == base_name)
    base_json = json.loads((base / refactor_parity.RUN_JSON).read_text())
    rows = []
    for path in runs:
        data = json.loads((path / refactor_parity.RUN_JSON).read_text())
        identical = over16_slices = 0
        worst = 0
        over16_pixels = 0
        box_delta = 0
        for sa, sb in zip(base_json["slices"], data["slices"]):
            box_delta += abs(int(sa.get("boxes") or 0) - int(sb.get("boxes") or 0))
            if sa.get("sha256") and sa.get("sha256") == sb.get("sha256"):
                identical += 1
                continue
            stats = _diff(base, path, sa["index"])
            if stats is None:
                over16_slices += 1
                continue
            worst = max(worst, stats["max"])
            over16_pixels += stats["over16"]
            if stats["over16"]:
                over16_slices += 1
        rows.append(
            {
                "config": path.name,
                "workers": data.get("workers"),
                "env": data.get("env"),
                "clean_s": data["clean_s"],
                "speedup": round(base_json["clean_s"] / data["clean_s"], 2) if data["clean_s"] else None,
                "detect_s_sum": data.get("detect_s_sum"),
                "inpaint_s_sum": data.get("inpaint_s_sum"),
                "lama_model_runs": data.get("lama_model_runs"),
                "slices": len(data["slices"]),
                "identical": identical,
                "slices_visibly_different": over16_slices,
                "pixels_over16": over16_pixels,
                "worst_pixel_diff": worst,
                "box_count_delta": box_delta,
                "boxes": sum(int(s.get("boxes") or 0) for s in data["slices"]),
                "cleanup_verified": sum(1 for s in data["slices"] if s.get("cleanup_verified")),
            }
        )
    report = {"base": base_name, "cpu_count": base_json.get("cpu_count"), "runs": rows}
    out.write_text(json.dumps(report, indent=1))
    print(f"{'config':14} {'clean_s':>8} {'speedup':>7} {'ident':>6} {'visdiff':>7} {'worst':>5} {'boxes':>6} {'verified':>8}")
    for row in rows:
        print(
            f"{row['config']:14} {row['clean_s']:>8} {row['speedup']:>7} {row['identical']:>6} "
            f"{row['slices_visibly_different']:>7} {row['worst_pixel_diff']:>5} {row['boxes']:>6} {row['cleanup_verified']:>8}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("summarize")
    s.add_argument("runs", type=Path, nargs="+")
    s.add_argument("--base", default="base")
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.url, refactor_parity.confined(args.out), args.workers)
        return 0
    return summarize(
        [refactor_parity.confined(path) for path in args.runs],
        args.base,
        refactor_parity.confined(args.out),
    )


if __name__ == "__main__":
    raise SystemExit(main())
