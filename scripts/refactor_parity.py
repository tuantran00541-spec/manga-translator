"""Process one real chapter and record every clean slice, for before/after parity checks.

    python scripts/refactor_parity.py run <chapter_url> --out <dir>
    python scripts/refactor_parity.py compare <base_dir> <head_dir> --out <report.json>
    python scripts/refactor_parity.py unexecuted <coverage.json> --out <report.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAPTER_ID = "b0d17c01"


def run(url: str, out: Path, workers: int) -> None:
    from app.config import PROCESSED_DIR, RAW_DIR
    from app.manifest_utils import load_manifest_raw
    from app.processing_pipeline_factory import build_processing_pipeline

    shutil.rmtree(RAW_DIR / CHAPTER_ID, ignore_errors=True)
    shutil.rmtree(PROCESSED_DIR / CHAPTER_ID, ignore_errors=True)
    pipeline = build_processing_pipeline()
    manifest = pipeline.download_chapter(url, CHAPTER_ID, workers=workers)
    started = time.perf_counter()
    pipeline.process_pages(CHAPTER_ID, list(range(len(manifest["pages"]))), workers=workers)
    elapsed = time.perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    slices = []
    for index, page in enumerate(load_manifest_raw(CHAPTER_ID)["pages"]):
        record = {"index": index, "boxes": len(page.get("boxes") or [])}
        if page.get("clean"):
            target = out / f"clean_{index:03d}.png"
            shutil.copy2(page["clean"], target)
            record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            record["cleanup_verified"] = bool(page.get("cleanup_verified"))
            metrics = page.get("processing_metrics") or {}
            record["residue_verify_ms"] = (metrics.get("timing_ms") or {}).get("residue_verify")
            record["residue_regions"] = len(page.get("residue_regions") or [])
            record["residue_repair"] = metrics.get("residue_repair")
        slices.append(record)
    (out / "run.json").write_text(json.dumps({"clean_s": round(elapsed, 1), "slices": slices}, indent=1))
    print(f"processed {len(slices)} slices in {elapsed:.0f}s")


def compare(base: Path, head: Path, report: Path, crops: Path | None = None) -> int:
    a = json.loads((base / "run.json").read_text())
    b = json.loads((head / "run.json").read_text())
    rows, identical = [], 0
    for sa, sb in zip(a["slices"], b["slices"]):
        row = {"index": sa["index"], "boxes": [sa["boxes"], sb["boxes"]]}
        if sa.get("sha256") and sa.get("sha256") == sb.get("sha256"):
            identical += 1
            row["identical"] = True
        else:
            ia = cv2.imread(str(base / f"clean_{sa['index']:03d}.png"))
            ib = cv2.imread(str(head / f"clean_{sa['index']:03d}.png"))
            row["identical"] = False
            if ia is not None and ib is not None and ia.shape == ib.shape:
                diff = np.abs(ia.astype(int) - ib.astype(int))
                row["max_diff"] = int(diff.max())
                row["changed_pixels"] = int((diff.max(axis=2) > 0).sum())
                if crops is not None and row["changed_pixels"]:
                    ys, xs = np.nonzero(diff.max(axis=2) > 0)
                    y1, y2 = max(0, ys.min() - 60), min(ia.shape[0], ys.max() + 60)
                    x1, x2 = max(0, xs.min() - 60), min(ia.shape[1], xs.max() + 60)
                    gap = np.full((y2 - y1, 10, 3), (0, 0, 255), np.uint8)
                    pair = np.hstack([ia[y1:y2, x1:x2], gap, ib[y1:y2, x1:x2]])
                    crops.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crops / f"slice{sa['index']:03d}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                row["shape_or_missing"] = True
        rows.append(row)
    verify = [s for s in b["slices"] if s.get("residue_verify_ms") is not None]
    summary = {
        "residue_verify_ms_total": round(sum(s["residue_verify_ms"] or 0 for s in verify)),
        "residue_regions_left": sum(s.get("residue_regions") or 0 for s in b["slices"]),
        "residue_repairs": sum(1 for s in b["slices"] if s.get("residue_repair")),
        "slices": [len(a["slices"]), len(b["slices"])],
        "identical": identical,
        "clean_s": [a["clean_s"], b["clean_s"]],
        "different": [row for row in rows if not row["identical"]],
    }
    report.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("slices", "identical", "clean_s")}))
    print(f"different slices: {len(summary['different'])}")
    return 0


def unexecuted(coverage_json: Path, report: Path) -> int:
    import ast

    data = json.loads(coverage_json.read_text())["files"]
    out = {}
    for name, info in sorted(data.items()):
        path = Path(name)
        if not path.is_file():
            continue
        executed = set(info["executed_lines"])
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dead = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = range(node.body[0].lineno, node.end_lineno + 1)
                if not executed.intersection(body):
                    dead.append(f"{node.lineno}:{node.name}")
        summary = info["summary"]
        out[str(path)] = {
            "statements": summary["num_statements"],
            "covered_percent": round(summary["percent_covered"], 1),
            "never_called": sorted(dead, key=lambda item: int(item.split(":")[0])),
        }
    report.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} files; {sum(len(v['never_called']) for v in out.values())} functions never called")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("head", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--crops", type=Path)
    u = sub.add_parser("unexecuted")
    u.add_argument("coverage_json", type=Path)
    u.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "unexecuted":
        return unexecuted(args.coverage_json, args.out)
    if args.command == "run":
        run(args.url, args.out, args.workers)
        return 0
    return compare(args.base, args.head, args.out, args.crops)


if __name__ == "__main__":
    raise SystemExit(main())
