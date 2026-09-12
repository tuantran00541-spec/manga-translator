from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.manifest_utils import load_manifest_raw
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.service import OCRService, _ocr_crop_bounds, ocr_target_skip_reason


class _ManifestOnlyPipeline:
    """OCRService callback shim for checkpoint jobs.

    save_manifest_raw() is the durable source of truth. The desktop pipeline's
    output-dir mirroring is not needed inside an isolated Actions checkpoint.
    """

    @staticmethod
    def _sync_output_dir(chapter_id: str, manifest: dict, page_indices: list[int]) -> None:
        return None


def _restore_data_layout(root: Path, chapter_id: str) -> None:
    raw_src = root / "raw"
    processed_src = root / "processed"
    raw_dst = Path("data/raw") / chapter_id
    processed_dst = Path("data/processed") / chapter_id
    if raw_dst.exists():
        shutil.rmtree(raw_dst)
    if processed_dst.exists():
        shutil.rmtree(processed_dst)
    raw_dst.parent.mkdir(parents=True, exist_ok=True)
    processed_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_src, raw_dst)
    shutil.copytree(processed_src, processed_dst)

    manifest_path = processed_dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != chapter_id:
        raise RuntimeError("checkpoint chapter id mismatch")
    for page in manifest.get("pages", []):
        original = Path(str(page.get("original") or ""))
        clean = Path(str(page.get("clean") or ""))
        if original.name:
            sliced = raw_dst / "sliced" / original.name
            direct = raw_dst / original.name
            page["original"] = str(sliced if sliced.is_file() else direct)
        if clean.name:
            page["clean"] = str(processed_dst / clean.name)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _active_boxes(manifest: dict) -> list[tuple[int, dict, dict]]:
    rows = []
    for page_index, page in enumerate(manifest.get("pages", [])):
        if page.get("skipped"):
            continue
        for box in page.get("boxes", []) or []:
            if not isinstance(box, dict) or box.get("removed"):
                continue
            rows.append((page_index, page, box))
    return rows


def _box_record(page_index: int, page: dict, box: dict) -> dict:
    return {
        "page_index": page_index,
        "source_page": page.get("source_page"),
        "slice_index": page.get("slice_index"),
        "box_id": str(box.get("id") or ""),
        "region": {k: int(box.get(k) or 0) for k in ("x1", "y1", "x2", "y2")},
        "semantic_type": box.get("semantic_type"),
        "class_name": box.get("class_name"),
        "source_role": box.get("source_role"),
        "safe_to_inpaint": box.get("safe_to_inpaint"),
        "ocr_eligible": box.get("ocr_eligible"),
        "needs_review": box.get("needs_review"),
        "ocr_text": str(box.get("ocr_text") or ""),
        "ocr_confidence": box.get("ocr_confidence"),
        "ocr_quality": box.get("ocr_quality"),
        "ocr_quality_reason": box.get("ocr_quality_reason"),
        "ocr_coverage": box.get("ocr_coverage"),
        "ocr_model": box.get("ocr_model"),
        "ocr_orientation": box.get("ocr_orientation"),
        "ocr_region_count": box.get("ocr_region_count"),
        "ocr_target_mode": box.get("ocr_target_mode"),
        "ocr_retry_applied": bool(box.get("ocr_retry_applied")),
        "skip_reason": ocr_target_skip_reason(box),
    }


def _is_suspicious(row: dict) -> list[str]:
    reasons = []
    if row.get("skip_reason"):
        return reasons
    text = str(row.get("ocr_text") or "").strip()
    if not text:
        reasons.append("empty")
    quality = str(row.get("ocr_quality") or "unknown")
    if quality in {"review", "reject", "unknown"}:
        reasons.append(f"quality:{quality}")
    conf = row.get("ocr_confidence")
    try:
        if conf is not None and float(conf) < 0.84:
            reasons.append("low-confidence")
    except (TypeError, ValueError):
        reasons.append("bad-confidence")
    cov = row.get("ocr_coverage")
    try:
        if cov is not None and float(cov) < 0.80:
            reasons.append("low-coverage")
    except (TypeError, ValueError):
        reasons.append("bad-coverage")
    if int(row.get("ocr_region_count") or 0) == 0:
        reasons.append("zero-regions")
    return list(dict.fromkeys(reasons))


def _crop_for_row(row: dict, manifest: dict) -> Image.Image:
    page = manifest["pages"][int(row["page_index"])]
    with Image.open(page["original"]) as im:
        rgb = im.convert("RGB")
    box = next(
        b for b in page.get("boxes", [])
        if str(b.get("id") or "") == str(row["box_id"])
    )
    import numpy as np
    shape = (rgb.height, rgb.width, 3)
    x1, y1, x2, y2 = _ocr_crop_bounds(shape, box)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = (0, 0, min(rgb.width, 32), min(rgb.height, 32))
    return rgb.crop((x1, y1, x2, y2))


def _card(row: dict, manifest: dict, *, suspicious: bool) -> Image.Image:
    crop = _crop_for_row(row, manifest)
    crop.thumbnail((360, 230), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (760, 280), "white")
    card.paste(crop, ((360 - crop.width) // 2, 34 + (230 - crop.height) // 2))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    header = (
        f"p{row['page_index']:03d} src={row['source_page']} sl={row['slice_index']} "
        f"id={str(row['box_id'])[:12]} q={row.get('ocr_quality')} "
        f"conf={row.get('ocr_confidence')} cov={row.get('ocr_coverage')}"
    )
    draw.text((6, 7), header[:125], fill="black", font=font)
    text = str(row.get("ocr_text") or "<EMPTY>")
    lines = []
    for paragraph in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(paragraph, width=48) or [""])
    y = 42
    if suspicious:
        reasons = ", ".join(row.get("suspicious_reasons") or [])
        draw.text((382, 24), ("REVIEW: " + reasons)[:58], fill="black", font=font)
        y = 48
    for line in lines[:18]:
        draw.text((382, y), line, fill="black", font=font)
        y += 13
    return card


def _save_sheets(rows: list[dict], manifest: dict, out_dir: Path, prefix: str, per_sheet: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), per_sheet):
        chunk = rows[start:start + per_sheet]
        cards = [_card(row, manifest, suspicious=bool(row.get("suspicious_reasons"))) for row in chunk]
        if not cards:
            continue
        cols = 2
        cw, ch = cards[0].size
        rows_n = (len(cards) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows_n * ch), "white")
        for i, card in enumerate(cards):
            sheet.paste(card, ((i % cols) * cw, (i // cols) * ch))
        sheet.save(out_dir / f"{prefix}-{start:03d}-{start + len(chunk) - 1:03d}.jpg", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--chapter-id", default="f1cd0121")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    root = Path(args.root)
    chapter_id = args.chapter_id
    _restore_data_layout(root, chapter_id)

    service = OCRService(MultiLangOCR(), _ManifestOnlyPipeline())
    plan = service.plan_chapter(chapter_id)
    results = []
    errors = []
    for idx, (page_index, box_id) in enumerate(plan, start=1):
        try:
            result = service.inspect_box_id(
                chapter_id, page_index, box_id, args.lang, force=True
            )
            results.append(result)
        except Exception as exc:
            errors.append({
                "page_index": page_index,
                "box_id": box_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
        if idx == 1 or idx % 20 == 0 or idx == len(plan):
            print(f"OCR {idx}/{len(plan)} results={len(results)} errors={len(errors)}", flush=True)

    manifest = load_manifest_raw(chapter_id)
    all_rows = [_box_record(pi, page, box) for pi, page, box in _active_boxes(manifest)]
    suspects = []
    for row in all_rows:
        reasons = _is_suspicious(row)
        if reasons:
            row = dict(row)
            row["suspicious_reasons"] = reasons
            suspects.append(row)

    skipped = [row for row in all_rows if row.get("skip_reason")]
    recognized = [row for row in all_rows if not row.get("skip_reason") and str(row.get("ocr_text") or "").strip()]
    quality_counts = Counter(str(row.get("ocr_quality") or "missing") for row in all_rows if not row.get("skip_reason"))
    skip_counts = Counter(str(row.get("skip_reason")) for row in skipped)

    by_scene = defaultdict(list)
    for row in all_rows:
        if row.get("skip_reason"):
            continue
        by_scene[(int(row.get("source_page") or 0), int(row.get("slice_index") or 0))].append(row)
    scene_index = []
    for (source_page, slice_index), rows in sorted(by_scene.items()):
        rows.sort(key=lambda r: (r["region"]["y1"], r["region"]["x1"], r["box_id"]))
        scene_index.append({
            "source_page": source_page,
            "slice_index": slice_index,
            "objects": rows,
        })

    review_dir = root / "ocr-review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "ocr-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "ocr-errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "ocr-objects.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "ocr-suspects.json").write_text(json.dumps(suspects, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "ocr-skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "scene-ocr-index.json").write_text(json.dumps(scene_index, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_sheets(suspects, manifest, review_dir / "suspect-sheets", "suspect", per_sheet=10)
    _save_sheets(recognized, manifest, review_dir / "all-ocr-sheets", "ocr", per_sheet=12)

    summary = {
        "checkpoint": "03-ocr-review",
        "chapter_id": chapter_id,
        "lang": args.lang,
        "source_policy": "RAW_ORIGINAL_ONLY",
        "planned_targets": len(plan),
        "completed_targets": len(results),
        "errors": len(errors),
        "active_detector_boxes": len(all_rows),
        "recognized_targets": len(recognized),
        "empty_or_suspicious_targets": len(suspects),
        "skipped_targets": len(skipped),
        "quality_counts": dict(quality_counts),
        "skip_reason_counts": dict(skip_counts),
        "suspect_pages": sorted({int(row["page_index"]) for row in suspects}),
        "next_action": "STOP_FOR_HUMAN_OCR_REVIEW_AND_CURATION",
    }
    (root / "ocr-review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "HUMAN_CHECKPOINT_03.txt").write_text(
        "CHECKPOINT: OCR REVIEW\n"
        "STOP here. OCR was read from RAW/original images only.\n"
        "Review suspect sheets plus scene-ocr-index against RAW. Check missing lines, duplicates, reading order, fragments, free-text misses and non-story false positives before translation.\n",
        encoding="utf-8",
    )

    # Promote the OCR-mutated production manifest and sidecars back into the
    # durable checkpoint for the next curation stage.
    updated_processed = Path("data/processed") / chapter_id
    if (root / "processed").exists():
        shutil.rmtree(root / "processed")
    shutil.copytree(updated_processed, root / "processed")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(f"OCR completed with {len(errors)} runtime errors")


if __name__ == "__main__":
    main()
