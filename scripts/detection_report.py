from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def _manifest_path(target: str) -> Path:
    path = Path(target)
    if path.is_file():
        return path
    return Path("data") / "processed" / target / "manifest.json"


def _page_stats(page: dict) -> dict[str, int | str | bool]:
    boxes = [box for box in (page.get("boxes") or []) if isinstance(box, dict)]
    active = [box for box in boxes if not box.get("removed")]
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
    }


def _sum_stats(rows: list[dict[str, int | str | bool]]) -> dict[str, int]:
    return {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in COLUMNS
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize persisted detector/inpaint provenance for one processed chapter."
    )
    parser.add_argument(
        "target",
        help="Chapter id (data/processed/<id>/manifest.json) or an explicit manifest path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    path = _manifest_path(args.target)
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
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

    header = ("page", *COLUMNS, "state", "page_review")
    print("\t".join(header))
    for row in rows:
        values = [
            str(row["page"]),
            *(str(row[key]) for key in COLUMNS),
            str(row["detection_state"]),
            "1" if row["needs_review"] else "0",
        ]
        print("\t".join(values))

    total_values = ["TOTAL", *(str(totals[key]) for key in COLUMNS), "-", "-"]
    print("\t".join(total_values))


if __name__ == "__main__":
    main()
