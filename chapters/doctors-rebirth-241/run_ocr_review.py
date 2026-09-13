from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from app.manifest_utils import load_manifest_raw, save_manifest_raw
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.service import OCRService, _ocr_crop_bounds, ocr_target_skip_reason
from app.text_objects import ensure_page_text_objects

EXPECTED_SLICES = 68
EXPECTED_SOURCE_PAGES = 17
LOGO_BOX_ID = "box_b04869771481418a"


class _SyncOnlyPipeline:
    def _sync_output_dir(self, chapter_id: str, manifest: dict, page_indices: list[int]) -> None:
        return None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _portable_ascii(value: str, limit: int = 95) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip())
    return value.encode("ascii", "replace").decode("ascii")[:limit]


def _garbled(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if "�" in value:
        return True
    alnum = sum(ch.isalnum() for ch in value)
    visible = sum(not ch.isspace() for ch in value)
    if visible >= 4 and alnum / max(1, visible) < 0.35:
        return True
    return bool(re.search(r"(.)\1{5,}", value))


def _global_region(page: dict, box: dict) -> tuple[float, float, float, float]:
    core = page.get("stitch_core") or {}
    source_y1 = float(core.get("source_y1") or 0)
    return (
        float(box.get("x1") or 0),
        source_y1 + float(box.get("y1") or 0),
        float(box.get("x2") or 0),
        source_y1 + float(box.get("y2") or 0),
    )


def _center_owned(page: dict, box: dict) -> bool:
    core = page.get("stitch_core") or {}
    if not core:
        return True
    cy = (float(box.get("y1") or 0) + float(box.get("y2") or 0)) / 2.0
    return float(core.get("core_y1") or 0) <= cy < float(core.get("core_y2") or 10**9)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def _prepare_runtime(chapter_id: str, import_root: Path, repaired_root: Path) -> None:
    raw_src = import_root / "raw"
    processed_src = repaired_root / "processed"
    if not raw_src.is_dir() or not (processed_src / "manifest.json").is_file():
        raise RuntimeError("checkpoint roots are incomplete")

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
    manifest = _read_json(manifest_path)
    pages = manifest.get("pages", [])
    if manifest.get("chapter_id") != chapter_id:
        raise RuntimeError(f"clean checkpoint chapter id mismatch: {manifest.get('chapter_id')}")
    if len(pages) != EXPECTED_SLICES:
        raise RuntimeError(f"expected {EXPECTED_SLICES} reviewed slices, got {len(pages)}")
    if len({int(p.get("source_page")) for p in pages}) != EXPECTED_SOURCE_PAGES:
        raise RuntimeError("source page count changed after CLEAN")

    logo_found = False
    for page_index, page in enumerate(pages):
        source_page = int(page.get("source_page"))
        slice_index = int(page.get("slice_index"))
        raw_path = raw_dst / "sliced" / f"{source_page:03d}_{slice_index:02d}.png"
        clean_path = processed_dst / f"clean_{source_page:03d}_{slice_index:02d}.png"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if not clean_path.is_file():
            raise FileNotFoundError(clean_path)
        page["original"] = raw_path.as_posix()
        page["clean"] = clean_path.as_posix()
        if page_index in (0, 67) and not page.get("skipped"):
            raise RuntimeError(f"non-story preservation page {page_index} lost skipped marker")
        if page_index not in (0, 67) and page.get("skipped"):
            raise RuntimeError(f"unexpected story slice marked skipped: {page_index}")
        for box in page.get("boxes", []) or []:
            if str(box.get("id")) == LOGO_BOX_ID:
                logo_found = True
                if box.get("ocr_eligible") is not False:
                    raise RuntimeError("decorative Doctor's Rebirth logo became OCR eligible")
    if not logo_found:
        raise RuntimeError("decorative Doctor's Rebirth logo box missing")

    save_manifest_raw(chapter_id, manifest)


def _run_ocr(chapter_id: str, lang: str):
    service = OCRService(MultiLangOCR(), _SyncOnlyPipeline())
    plan = service.plan_chapter(chapter_id)
    if not plan:
        raise RuntimeError("OCR plan is empty")
    results, failures = [], []
    print(f"OCR planned targets: {len(plan)}", flush=True)
    for index, (page_index, box_id) in enumerate(plan, start=1):
        try:
            results.append(service.inspect_box_id(chapter_id, page_index, box_id, lang, force=True))
        except Exception as exc:
            failures.append({
                "page_index": page_index,
                "box_id": box_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        if index == 1 or index % 20 == 0 or index == len(plan):
            print(f"OCR {index}/{len(plan)}; failures={len(failures)}", flush=True)
    return results, failures


def _reconcile_text_objects(chapter_id: str) -> int:
    manifest = load_manifest_raw(chapter_id)
    created = 0
    for page in manifest.get("pages", []):
        count, _ = ensure_page_text_objects(page)
        created += count
    save_manifest_raw(chapter_id, manifest)
    return created


def _collect_rows(chapter_id: str, result_by_id: dict[str, dict]):
    manifest = load_manifest_raw(chapter_id)
    rows = []
    for page_index, page in enumerate(manifest.get("pages", [])):
        for box_index, box in enumerate(page.get("boxes", []) or []):
            if not isinstance(box, dict):
                continue
            box_id = str(box.get("id") or "")
            result = result_by_id.get(box_id, {})
            global_region = _global_region(page, box)
            active = (
                not page.get("skipped")
                and not box.get("removed")
                and box.get("ocr_eligible") is not False
                and not ocr_target_skip_reason(box)
            )
            rows.append({
                "page_index": page_index,
                "box_index": box_index,
                "box_id": box_id,
                "source_page": int(page.get("source_page") or 0),
                "slice_index": int(page.get("slice_index") or 0),
                "page_skipped": bool(page.get("skipped")),
                "global_x1": round(global_region[0], 1),
                "global_y1": round(global_region[1], 1),
                "global_x2": round(global_region[2], 1),
                "global_y2": round(global_region[3], 1),
                "center_owned": _center_owned(page, box),
                "semantic_type": box.get("semantic_type"),
                "class_name": box.get("class_name"),
                "source_role": box.get("source_role"),
                "safe_to_inpaint": box.get("safe_to_inpaint"),
                "ocr_eligible": bool(active),
                "skip_reason": ocr_target_skip_reason(box) or ("page-skipped" if page.get("skipped") else None),
                "text": str(box.get("ocr_text") or ""),
                "confidence": box.get("ocr_confidence", result.get("confidence")),
                "quality": box.get("ocr_quality", result.get("quality")),
                "quality_reason": box.get("ocr_quality_reason", result.get("quality_reason")),
                "target_mode": box.get("ocr_target_mode", result.get("target_mode")),
                "retry_applied": box.get("ocr_retry_applied", result.get("retry_applied")),
                "region_count": box.get("ocr_region_count", result.get("region_count")),
            })
    rows.sort(key=lambda r: (r["source_page"], r["global_y1"], r["global_x1"], r["box_id"]))
    return rows


def _duplicate_suspects(rows):
    grouped = defaultdict(list)
    for row in rows:
        text = _normalize_text(row["text"])
        if row["ocr_eligible"] and len(text) >= 2:
            grouped[(row["source_page"], text)].append(row)
    pairs, ids = [], set()
    for (source_page, text), items in grouped.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["slice_index"] == b["slice_index"]:
                    continue
                ar = (a["global_x1"], a["global_y1"], a["global_x2"], a["global_y2"])
                br = (b["global_x1"], b["global_y1"], b["global_x2"], b["global_y2"])
                acx, acy = (ar[0]+ar[2])/2, (ar[1]+ar[3])/2
                bcx, bcy = (br[0]+br[2])/2, (br[1]+br[3])/2
                overlap = _iou(ar, br)
                near = abs(acy-bcy) <= 320 and abs(acx-bcx) <= 220
                if overlap >= 0.15 or near:
                    ids.update((a["box_id"], b["box_id"]))
                    pairs.append({
                        "source_page": source_page,
                        "normalized_text": text,
                        "a": a["box_id"],
                        "b": b["box_id"],
                        "a_owned": a["center_owned"],
                        "b_owned": b["center_owned"],
                        "global_iou": round(overlap, 4),
                    })
    return pairs, ids


def _review_candidates(rows, duplicate_ids):
    candidates = []
    for row in rows:
        if not row["ocr_eligible"]:
            continue
        reasons = []
        text = str(row["text"] or "").strip()
        conf = row.get("confidence")
        quality = str(row.get("quality") or "unknown")
        semantic = " ".join(str(row.get(k) or "").lower() for k in ("semantic_type", "class_name", "source_role"))
        if not text:
            reasons.append("empty")
        if quality != "good":
            reasons.append(f"quality:{quality}")
        try:
            if conf is not None and float(conf) < 0.82:
                reasons.append("low_confidence")
        except (TypeError, ValueError):
            reasons.append("invalid_confidence")
        if _garbled(text):
            reasons.append("garbled")
        if not row["center_owned"]:
            reasons.append("outside_stitch_core")
        if row["box_id"] in duplicate_ids:
            reasons.append("duplicate_suspect")
        if any(token in semantic for token in ("free_text", "narration", "recovery")) and (quality != "good" or not text or not row["center_owned"]):
            reasons.append("free_text_suspect")
        if reasons:
            item = dict(row)
            item["review_reasons"] = sorted(set(reasons))
            candidates.append(item)
    return candidates


def _write_table(rows, out_dir: Path):
    (out_dir / "ocr-table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "ocr-table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _make_candidate_sheets(chapter_id: str, candidates, out_dir: Path):
    manifest = load_manifest_raw(chapter_id)
    by_id = {}
    for page in manifest.get("pages", []):
        for box in page.get("boxes", []) or []:
            if isinstance(box, dict) and box.get("id"):
                by_id[str(box["id"])] = (page, box)
    proof_dir = out_dir / "proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    cell_w, cell_h, per_sheet = 360, 300, 20
    for start in range(0, len(candidates), per_sheet):
        batch = candidates[start:start+per_sheet]
        cols = 4
        rows_n = max(1, math.ceil(len(batch)/cols))
        canvas = Image.new("RGB", (cols*cell_w, rows_n*cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for n, item in enumerate(batch):
            page, box = by_id[item["box_id"]]
            raw = cv2.imread(str(page["original"]), cv2.IMREAD_COLOR)
            if raw is None:
                crop = None
            else:
                x1, y1, x2, y2 = _ocr_crop_bounds(raw.shape, box)
                crop = raw[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            if crop is not None and crop.size:
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                image.thumbnail((cell_w-16, 190))
            else:
                image = Image.new("RGB", (cell_w-16, 80), "white")
            x = (n % cols)*cell_w + 8
            y = (n // cols)*cell_h + 8
            canvas.paste(image, (x, y+64))
            draw.text((x,y), _portable_ascii(f"p{item['page_index']} {item['box_id']}",55), fill="black")
            draw.text((x,y+18), _portable_ascii(",".join(item["review_reasons"]),55), fill="black")
            draw.text((x,y+36), _portable_ascii(f"q={item.get('quality')} c={item.get('confidence')} :: {item.get('text','')}",55), fill="black")
        name = f"ocr-review-{start//per_sheet:02d}.jpg"
        canvas.save(proof_dir / name, quality=90)
        sheets.append(f"proof/{name}")
    return sheets


def _scene_rows(rows):
    return [r for r in rows if r["ocr_eligible"] and r["center_owned"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    _prepare_runtime(args.chapter_id, args.import_root, args.repaired_root)
    results, failures = _run_ocr(args.chapter_id, args.lang)
    created = _reconcile_text_objects(args.chapter_id)

    result_by_id = {str(item.get("box_id")): item for item in results}
    rows = _collect_rows(args.chapter_id, result_by_id)
    duplicates, duplicate_ids = _duplicate_suspects(rows)
    candidates = _review_candidates(rows, duplicate_ids)
    _write_table(rows, args.output)
    (args.output / "ocr-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "ocr-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "duplicate-suspects.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "review-candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "scene-reading-order.json").write_text(json.dumps(_scene_rows(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    sheets = _make_candidate_sheets(args.chapter_id, candidates, args.output)

    manifest = load_manifest_raw(args.chapter_id)
    pages = manifest.get("pages", [])
    if any("clean_" in str(page.get("original") or "") for page in pages):
        raise RuntimeError("OCR source invariant violated: page.original points at CLEAN")
    if not pages[0].get("skipped") or not pages[67].get("skipped"):
        raise RuntimeError("non-story skip invariant was lost during OCR")
    if any(
        str(box.get("id")) == LOGO_BOX_ID and box.get("ocr_eligible") is not False
        for page in pages for box in (page.get("boxes") or []) if isinstance(box, dict)
    ):
        raise RuntimeError("decorative title logo resurrected as OCR target")
    (args.output / "processed-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    active_rows = [row for row in rows if row["ocr_eligible"]]
    quality_counts = Counter(str(row.get("quality") or "unknown") for row in active_rows)
    recognized = sum(bool(str(row.get("text") or "").strip()) for row in active_rows)
    summary = {
        "checkpoint": "03-ocr-review",
        "chapter_id": args.chapter_id,
        "language": args.lang,
        "ocr_source": "RAW",
        "ocr_source_invariant_verified": True,
        "total_detected_boxes": len(rows),
        "active_ocr_boxes": len(active_rows),
        "center_owned_active_boxes": len(_scene_rows(rows)),
        "recognized": recognized,
        "empty": len(active_rows)-recognized,
        "failed": len(failures),
        "quality_counts": dict(quality_counts),
        "text_objects_created": created,
        "skipped_non_story_pages": [0, 67],
        "decorative_title_box_ocr_excluded": True,
        "duplicate_suspect_pairs": len(duplicates),
        "review_candidate_count": len(candidates),
        "review_sheets": sheets,
        "status": "REVIEW_REQUIRED",
        "next_action": "HUMAN_OCR_REVIEW_AND_CURATION",
    }
    (args.output / "ocr-review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "HUMAN_CHECKPOINT.txt").write_text(
        "CHECKPOINT: OCR HUMAN REVIEW\n"
        "OCR source is RAW. Review low-confidence, empty, garbled, duplicate, outside-core and free-text suspects.\n"
        "Read scene-reading-order.json in source-page order and return to RAW whenever scene coherence has a gap.\n"
        "Do not advance until active story OCR is curated and duplicates/fragments are reconciled.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(f"OCR technical failures require review: {len(failures)}")


if __name__ == "__main__":
    main()
