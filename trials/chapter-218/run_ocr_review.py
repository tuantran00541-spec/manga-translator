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
import numpy as np
from PIL import Image, ImageDraw

from app.manifest_utils import load_manifest_raw, save_manifest_raw
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.service import OCRService, ocr_crop_from_box, ocr_target_skip_reason
from app.text_objects import DEFAULT_TEXT_OBJECT_STYLE, ensure_page_text_objects


class _SyncOnlyPipeline:
    """OCRService only needs output syncing after each manifest commit.

    Checkpoint jobs persist the manifest directly and package it explicitly, so
    no render/output mirroring is required here. This avoids loading detector or
    inpaint sessions during the OCR-only gate.
    """

    def _sync_output_dir(self, chapter_id: str, manifest: dict, page_indices: list[int]) -> None:
        return None


def _read_json(path: Path) -> dict:
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
    if re.search(r"(.)\1{5,}", value):
        return True
    return False


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


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def _seed_non_story_tombstones(manifest: dict, drop_ids: set[str]) -> None:
    seen: set[str] = set()
    for page in manifest.get("pages", []):
        objects = page.setdefault("text_objects", [])
        if not isinstance(objects, list):
            objects = []
            page["text_objects"] = objects
        for box in page.get("boxes", []) or []:
            box_id = str(box.get("id") or "")
            if box_id not in drop_ids:
                continue
            seen.add(box_id)
            box["ocr_eligible"] = False
            box["human_review_drop"] = True
            box["story_role"] = "non_story_promo_credit"
            box["human_review_origin"] = "human_review_drop"
            tombstone_id = f"text_{box_id[4:] if box_id.startswith('box_') else box_id}"
            if any(isinstance(obj, dict) and obj.get("id") == tombstone_id for obj in objects):
                continue
            region = {
                "x1": int(box["x1"]), "y1": int(box["y1"]),
                "x2": int(box["x2"]), "y2": int(box["y2"]),
            }
            objects.append({
                "id": tombstone_id,
                "shape": "rectangle",
                "region": region,
                "source_boxes": [box_id],
                "ocr_text": "",
                "translation": "",
                "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
                "origin": "human_review_drop",
                "auto_generated": False,
                "source_missing": True,
                "reason": "non_story_promo_credit",
            })
    missing = sorted(drop_ids - seen)
    if missing:
        raise RuntimeError(f"tombstone seed box ids missing from clean manifest: {missing}")


def _assert_tombstones(manifest: dict, drop_ids: set[str]) -> int:
    resurrection = 0
    found: set[str] = set()
    for page in manifest.get("pages", []):
        for obj in page.get("text_objects", []) or []:
            if not isinstance(obj, dict):
                continue
            refs = {str(v) for v in obj.get("source_boxes", []) if isinstance(v, str)}
            hits = refs & drop_ids
            if not hits:
                continue
            found |= hits
            if obj.get("auto_generated") or not obj.get("source_missing") or obj.get("origin") != "human_review_drop":
                resurrection += 1
    if found != drop_ids:
        raise RuntimeError(f"tombstone refs missing after reconciliation: {sorted(drop_ids-found)}")
    return resurrection


def _prepare_runtime(chapter_id: str, import_root: Path, clean_root: Path, repair_json: Path) -> set[str]:
    raw_src = import_root / "raw"
    clean_src = clean_root / "processed"
    if not raw_src.is_dir() or not (clean_src / "manifest.json").is_file():
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
    shutil.copytree(clean_src, processed_dst)

    manifest = _read_json(processed_dst / "manifest.json")
    if manifest.get("chapter_id") != chapter_id:
        raise RuntimeError("clean checkpoint chapter id mismatch")
    if len(manifest.get("pages", [])) != 69:
        raise RuntimeError("expected 69 reviewed slices")

    for page in manifest.get("pages", []):
        source_page = int(page.get("source_page"))
        slice_index = int(page.get("slice_index"))
        raw_path = raw_dst / "sliced" / f"{source_page:03d}_{slice_index:02d}.png"
        clean_path = processed_dst / f"clean_{source_page:03d}_{slice_index:02d}.png"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        page["original"] = raw_path.as_posix()
        if clean_path.is_file():
            page["clean"] = clean_path.as_posix()

    repair = _read_json(repair_json)
    drop_ids = {str(v) for v in repair.get("drop_from_story_object_ids", []) if v}
    if len(drop_ids) != 6:
        raise RuntimeError(f"expected six durable promo/credit drops, got {len(drop_ids)}")
    _seed_non_story_tombstones(manifest, drop_ids)
    save_manifest_raw(chapter_id, manifest)
    return drop_ids


def _run_ocr(chapter_id: str, lang: str) -> tuple[list[dict], list[dict]]:
    service = OCRService(MultiLangOCR(), _SyncOnlyPipeline())
    plan = service.plan_chapter(chapter_id)
    results: list[dict] = []
    failures: list[dict] = []
    print(f"OCR planned targets: {len(plan)}", flush=True)
    for index, (page_index, box_id) in enumerate(plan, start=1):
        try:
            result = service.inspect_box_id(chapter_id, page_index, box_id, lang, force=True)
            results.append(result)
        except Exception as exc:  # review artifact must preserve failures instead of hiding them
            failures.append({"page_index": page_index, "box_id": box_id, "error_type": type(exc).__name__, "error": str(exc)})
        if index == 1 or index % 20 == 0 or index == len(plan):
            print(f"OCR {index}/{len(plan)}; failures={len(failures)}", flush=True)
    return results, failures


def _reconcile_text_objects(chapter_id: str, drop_ids: set[str]) -> tuple[int, int]:
    manifest = load_manifest_raw(chapter_id)
    created = 0
    for page in manifest.get("pages", []):
        count, _ = ensure_page_text_objects(page)
        created += count
    save_manifest_raw(chapter_id, manifest)

    # Resurrection test: reconciliation must be idempotent and must not revive
    # explicit non-story tombstones.
    manifest = load_manifest_raw(chapter_id)
    for page in manifest.get("pages", []):
        ensure_page_text_objects(page)
    save_manifest_raw(chapter_id, manifest)
    manifest = load_manifest_raw(chapter_id)
    resurrection_count = _assert_tombstones(manifest, drop_ids)
    return created, resurrection_count


def _collect_rows(chapter_id: str, result_by_id: dict[str, dict]) -> list[dict]:
    manifest = load_manifest_raw(chapter_id)
    rows: list[dict] = []
    for page_index, page in enumerate(manifest.get("pages", [])):
        for box_index, box in enumerate(page.get("boxes", []) or []):
            if not isinstance(box, dict):
                continue
            box_id = str(box.get("id") or "")
            text = str(box.get("ocr_text") or "")
            result = result_by_id.get(box_id, {})
            global_region = _global_region(page, box)
            row = {
                "page_index": page_index,
                "box_index": box_index,
                "box_id": box_id,
                "source_page": int(page.get("source_page") or 0),
                "slice_index": int(page.get("slice_index") or 0),
                "global_x1": round(global_region[0], 1),
                "global_y1": round(global_region[1], 1),
                "global_x2": round(global_region[2], 1),
                "global_y2": round(global_region[3], 1),
                "center_owned": _center_owned(page, box),
                "semantic_type": box.get("semantic_type"),
                "class_name": box.get("class_name"),
                "source_role": box.get("source_role"),
                "safe_to_inpaint": box.get("safe_to_inpaint"),
                "ocr_eligible": box.get("ocr_eligible") is not False and not box.get("removed"),
                "skip_reason": ocr_target_skip_reason(box),
                "text": text,
                "confidence": box.get("ocr_confidence", result.get("confidence")),
                "quality": box.get("ocr_quality", result.get("quality")),
                "quality_reason": box.get("ocr_quality_reason", result.get("quality_reason")),
                "target_mode": box.get("ocr_target_mode", result.get("target_mode")),
                "retry_applied": box.get("ocr_retry_applied", result.get("retry_applied")),
                "region_count": box.get("ocr_region_count", result.get("region_count")),
                "human_review_drop": bool(box.get("human_review_drop")),
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["source_page"], r["global_y1"], r["global_x1"], r["box_id"]))
    return rows


def _duplicate_suspects(rows: list[dict]) -> tuple[list[dict], set[str]]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        text = _normalize_text(row["text"])
        if row["ocr_eligible"] and len(text) >= 2:
            grouped[(row["source_page"], text)].append(row)
    pairs: list[dict] = []
    ids: set[str] = set()
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
                        "a": a["box_id"], "b": b["box_id"],
                        "a_owned": a["center_owned"], "b_owned": b["center_owned"],
                        "global_iou": round(overlap, 4),
                    })
    return pairs, ids


def _review_candidates(rows: list[dict], duplicate_ids: set[str]) -> list[dict]:
    candidates: list[dict] = []
    for row in rows:
        if not row["ocr_eligible"] or row["human_review_drop"]:
            continue
        reasons: list[str] = []
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
        if any(token in semantic for token in ("free_text", "narration", "recovery")) and (
            quality != "good" or not text or not row["center_owned"]
        ):
            reasons.append("free_text_suspect")
        if reasons:
            candidate = dict(row)
            candidate["review_reasons"] = sorted(set(reasons))
            candidates.append(candidate)
    return candidates


def _write_table(rows: list[dict], out_dir: Path) -> None:
    (out_dir / "ocr-table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "ocr-table.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _make_candidate_sheets(chapter_id: str, candidates: list[dict], out_dir: Path) -> list[str]:
    manifest = load_manifest_raw(chapter_id)
    by_id: dict[str, tuple[dict, dict]] = {}
    for page in manifest.get("pages", []):
        for box in page.get("boxes", []) or []:
            if isinstance(box, dict) and box.get("id"):
                by_id[str(box["id"])] = (page, box)

    sheets: list[str] = []
    proof_dir = out_dir / "proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 360, 300
    per_sheet = 20
    for start in range(0, len(candidates), per_sheet):
        batch = candidates[start:start+per_sheet]
        cols = 4
        rows_n = max(1, math.ceil(len(batch)/cols))
        canvas = Image.new("RGB", (cols*cell_w, rows_n*cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for n, item in enumerate(batch):
            page, box = by_id[item["box_id"]]
            raw = cv2.imread(str(page["original"]), cv2.IMREAD_COLOR)
            crop = ocr_crop_from_box(raw, box)
            if crop.size:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(crop_rgb)
                image.thumbnail((cell_w-16, 190))
            else:
                image = Image.new("RGB", (cell_w-16, 80), "white")
            x = (n % cols)*cell_w + 8
            y = (n // cols)*cell_h + 8
            canvas.paste(image, (x, y+64))
            label1 = f"p{item['page_index']} {item['box_id']}"
            label2 = ",".join(item["review_reasons"])
            label3 = f"q={item.get('quality')} c={item.get('confidence')} :: {_portable_ascii(item.get('text',''))}"
            draw.text((x,y), _portable_ascii(label1, 55), fill="black")
            draw.text((x,y+18), _portable_ascii(label2, 55), fill="black")
            draw.text((x,y+36), _portable_ascii(label3, 55), fill="black")
        name = f"ocr-review-{start//per_sheet:02d}.jpg"
        canvas.save(proof_dir / name, quality=90)
        sheets.append(f"proof/{name}")
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--repair-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    drop_ids = _prepare_runtime(args.chapter_id, args.import_root, args.clean_root, args.repair_json)
    results, failures = _run_ocr(args.chapter_id, args.lang)
    created, resurrection_count = _reconcile_text_objects(args.chapter_id, drop_ids)

    result_by_id = {str(item.get("box_id")): item for item in results}
    rows = _collect_rows(args.chapter_id, result_by_id)
    duplicates, duplicate_ids = _duplicate_suspects(rows)
    candidates = _review_candidates(rows, duplicate_ids)
    _write_table(rows, args.output)
    (args.output / "ocr-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "ocr-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "duplicate-suspects.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "review-candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    sheets = _make_candidate_sheets(args.chapter_id, candidates, args.output)

    manifest = load_manifest_raw(args.chapter_id)
    if any("clean_" in str(page.get("original") or "") for page in manifest.get("pages", [])):
        raise RuntimeError("OCR source invariant violated: a page.original points at CLEAN")
    (args.output / "processed-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    quality_counts = Counter(str(row.get("quality") or "unknown") for row in rows if row["ocr_eligible"])
    recognized = sum(bool(str(row.get("text") or "").strip()) for row in rows if row["ocr_eligible"])
    active = sum(bool(row["ocr_eligible"]) for row in rows)
    summary = {
        "checkpoint": "03-ocr-review",
        "chapter_id": args.chapter_id,
        "language": args.lang,
        "ocr_source": "RAW",
        "ocr_source_invariant_verified": True,
        "active_ocr_boxes": active,
        "recognized": recognized,
        "empty": active-recognized,
        "failed": len(failures),
        "quality_counts": dict(quality_counts),
        "text_objects_created": created,
        "seeded_non_story_tombstones": sorted(drop_ids),
        "resurrection_count": resurrection_count,
        "duplicate_suspect_pairs": len(duplicates),
        "review_candidate_count": len(candidates),
        "review_sheets": sheets,
        "status": "REVIEW_REQUIRED",
        "next_action": "HUMAN_OCR_REVIEW_AND_CURATION"
    }
    (args.output / "ocr-review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "HUMAN_CHECKPOINT.txt").write_text(
        "CHECKPOINT: OCR HUMAN REVIEW\n"
        "OCR source is RAW. Review low-confidence, empty, garbled, duplicate, outside-core and free-text suspects.\n"
        "Do not advance until active story OCR is curated, duplicates/fragments reconciled, and resurrection_count=0.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(f"OCR technical failures require review: {len(failures)}")
    if resurrection_count:
        raise SystemExit(f"non-story tombstone resurrection detected: {resurrection_count}")


if __name__ == "__main__":
    main()
