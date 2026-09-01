from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED_ROOT = _REPO_ROOT / "data" / "processed"

COLUMNS = (
    "total",
    "active",
    "safe",
    "review",
    "ocr",
    "segmenter",
    "flat_fallback",
    "mser",
    "manual",
    "overlap_only",
    "geometry_override",
)
METRIC_COLUMNS = (
    "detect_ms",
    "inpaint_ms",
    "lama_runs",
    "smart_fill",
)


def _manifest_path(target: str) -> Path:
    candidate = Path(target)
    if candidate.suffix.lower() == ".json":
        path = candidate if candidate.is_absolute() else _REPO_ROOT / candidate
    else:
        path = _PROCESSED_ROOT / candidate / "manifest.json"
    root = _PROCESSED_ROOT.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest path must stay under data/processed") from exc
    return resolved


def _page_stats(page: dict) -> dict[str, int | float | str | bool | dict]:
    boxes = [box for box in (page.get("boxes") or []) if isinstance(box, dict)]
    active = [box for box in boxes if not box.get("removed")]
    metrics = page.get("processing_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    timing = metrics.get("timing_ms") if isinstance(metrics.get("timing_ms"), dict) else {}
    auto = metrics.get("auto_inpaint") if isinstance(metrics.get("auto_inpaint"), dict) else {}
    manual = metrics.get("manual_inpaint") if isinstance(metrics.get("manual_inpaint"), dict) else {}
    detect_ms = float(timing.get("detect") or 0.0)
    inpaint_ms = float(timing.get("auto_inpaint") or 0.0) + float(
        timing.get("manual_inpaint") or 0.0
    )
    return {
        "total": len(boxes),
        "active": len(active),
        "safe": sum(bool(box.get("safe_to_inpaint")) for box in active),
        "review": sum(bool(box.get("needs_review")) for box in active),
        "ocr": sum(bool(box.get("ocr_eligible")) for box in active),
        "segmenter": sum(box.get("mask_source") == "text_segmenter" for box in active),
        "flat_fallback": sum(box.get("mask_source") == "bubble_flat_contrast" for box in active),
        "mser": sum(str(box.get("source_model") or "") == "opencv_mser" for box in active),
        "manual": sum(bool(box.get("manual")) or box.get("origin") == "manual" for box in active),
        "overlap_only": sum(bool(box.get("overlap_context_only")) for box in active),
        "geometry_override": sum(bool(box.get("geometry_overridden")) for box in active),
        "detection_state": str(page.get("detection_state") or "unknown"),
        "needs_review": bool(page.get("needs_review")),
        "metrics_available": bool(metrics),
        "detect_ms": round(detect_ms, 3),
        "inpaint_ms": round(inpaint_ms, 3),
        "lama_runs": int(auto.get("lama_model_runs") or 0)
        + int(manual.get("lama_model_runs") or 0),
        "smart_fill": int(auto.get("smart_fill_regions") or 0)
        + int(manual.get("smart_fill_regions") or 0),
        "processing_metrics": metrics,
    }


def _sum_stats(rows: list[dict[str, int | float | str | bool | dict]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        key: sum(int(row.get(key) or 0) for row in rows) for key in COLUMNS
    }
    for key in METRIC_COLUMNS:
        values = [row.get(key) or 0 for row in rows]
        totals[key] = round(sum(float(value) for value in values), 3)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize detector authority and observed processing timing for one chapter."
    )
    parser.add_argument(
        "target",
        help="Chapter id or explicit manifest path under data/processed/.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    path = _manifest_path(args.target)
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))  # NOSONAR(S8707): _manifest_path confines explicit paths to data/processed.
    pages = manifest.get("pages") or []
    rows = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        row = {"page": index, **_page_stats(page)}
        rows.append(row)

    totals = _sum_stats(rows)
    report = {
        "chapter_id": manifest.get("chapter_id"),
        "manifest": path.as_posix(),
        "pages": rows,
        "totals": totals,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    header = ("page", *COLUMNS, *METRIC_COLUMNS, "state", "page_review")
    print("\t".join(header))
    for row in rows:
        values = [
            str(row["page"]),
            *(str(row[key]) for key in COLUMNS),
            *(str(row[key]) for key in METRIC_COLUMNS),
            str(row["detection_state"]),
            "1" if row["needs_review"] else "0",
        ]
        print("\t".join(values))

    total_values = [
        "TOTAL",
        *(str(totals[key]) for key in COLUMNS),
        *(str(totals[key]) for key in METRIC_COLUMNS),
        "-",
        "-",
    ]
    print("\t".join(total_values))


if __name__ == "__main__":
    main()
