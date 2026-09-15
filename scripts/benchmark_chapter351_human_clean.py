from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from app.manifest_utils import load_manifest_raw
from app.optimized_pipeline import OptimizedChapterPipeline
from app.security import validate_chapter_id

CHAPTER_ID = "c3513510"
CHAPTER_URL = "https://asurascans.com/comics/logging-10000-years-into-the-future-53fc8424/chapter/351"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = (int(a.get(k) or 0) for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (int(b.get(k) or 0) for k in ("x1", "y1", "x2", "y2"))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(max(1, area_a + area_b - inter))


def _greedy_matches(refs: list[dict], candidates: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    options = []
    for ri, ref in enumerate(refs):
        for ci, cand in enumerate(candidates):
            score = _iou(ref, cand)
            if score >= threshold:
                options.append((score, ri, ci))
    options.sort(reverse=True)
    used_r: set[int] = set()
    used_c: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, ri, ci in options:
        if ri in used_r or ci in used_c:
            continue
        used_r.add(ri)
        used_c.add(ci)
        matches.append((ri, ci, score))
    return matches


def _restore_import(import_root: Path) -> dict:
    chapter_id = validate_chapter_id(CHAPTER_ID)
    manifest_path = import_root / "manifest.json"
    raw_source = import_root / "raw"
    if not manifest_path.is_file() or not raw_source.is_dir():
        raise RuntimeError(f"invalid import checkpoint: {import_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != chapter_id or manifest.get("source_url") != CHAPTER_URL:
        raise RuntimeError("reviewed import checkpoint identity mismatch")

    raw_dst = Path("data/raw") / chapter_id
    processed_dst = Path("data/processed") / chapter_id
    if raw_dst.exists():
        shutil.rmtree(raw_dst)
    if processed_dst.exists():
        shutil.rmtree(processed_dst)
    raw_dst.parent.mkdir(parents=True, exist_ok=True)
    processed_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_source, raw_dst)

    for page in manifest.get("pages", []):
        original = str(page.get("original") or "")
        if "/raw/" in original:
            rel = original.split("/raw/", 1)[1]
        elif original.startswith("raw/"):
            rel = original[4:]
        else:
            raise RuntimeError(f"cannot normalize import path: {original}")
        page["original"] = (raw_dst / rel).as_posix()
        for key in (
            "clean", "boxes", "detections", "processing_metrics", "detection_state",
            "detection_issues", "unverified_regions", "deferred_regions", "residue_regions",
            "cleanup_verified", "needs_review", "process_revision", "clean_revision",
            "artifact_transaction_id",
        ):
            page.pop(key, None)

    (processed_dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _current_boxes(manifest: dict) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    safe: dict[int, list[dict]] = {}
    review: dict[int, list[dict]] = {}
    for pi, page in enumerate(manifest.get("pages", [])):
        boxes = page.get("boxes") or page.get("detections") or []
        safe[pi] = [b for b in boxes if bool(b.get("safe_to_inpaint"))]
        review[pi] = [b for b in boxes if not bool(b.get("safe_to_inpaint"))]
    return safe, review


def _box_metrics(base_root: Path, repair_root: Path, current: dict) -> dict:
    rows = json.loads((base_root / "box-index.json").read_text(encoding="utf-8"))
    repair = json.loads((repair_root / "clean-repair.json").read_text(encoding="utf-8"))
    tombstones = set(repair.get("drop_from_story_object_ids") or [])
    ref_safe: dict[int, list[dict]] = {}
    ref_review: dict[int, list[dict]] = {}
    forbidden: dict[int, list[dict]] = {}
    for row in rows:
        pi = int(row["page_index"])
        item = {**(row.get("region") or {}), "id": row.get("box_id"), "semantic_type": row.get("semantic_type"), "confidence": row.get("confidence")}
        if row.get("box_id") in tombstones:
            forbidden.setdefault(pi, []).append(item)
        elif bool(row.get("safe_to_inpaint")):
            ref_safe.setdefault(pi, []).append(item)
        else:
            ref_review.setdefault(pi, []).append(item)

    cur_safe, cur_review = _current_boxes(current)
    match_summary = {}
    for threshold in (0.30, 0.50, 0.75):
        nr = nc = nm = 0
        for pi in range(len(current.get("pages", []))):
            refs, curs = ref_safe.get(pi, []), cur_safe.get(pi, [])
            nr += len(refs)
            nc += len(curs)
            nm += len(_greedy_matches(refs, curs, threshold))
        match_summary[f"iou_{str(threshold).replace('.', '_')}"] = {
            "reference_story_safe": nr,
            "current_safe": nc,
            "matched": nm,
            "recall_pct": round(100.0 * nm / max(1, nr), 2),
            "pseudo_precision_pct": round(100.0 * nm / max(1, nc), 2),
        }

    page_rows, forbidden_hits, all_iou = [], [], []
    for pi in range(len(current.get("pages", []))):
        refs, curs = ref_safe.get(pi, []), cur_safe.get(pi, [])
        loose = _greedy_matches(refs, curs, 0.10)
        all_iou.extend(score for _, _, score in loose)
        matches030 = _greedy_matches(refs, curs, 0.30)
        matched_ref = {ri for ri, _, _ in matches030}
        missing = [refs[i] for i in range(len(refs)) if i not in matched_ref]
        for ci, cand in enumerate(curs):
            for old in forbidden.get(pi, []):
                score = _iou(cand, old)
                if score >= 0.30:
                    forbidden_hits.append({
                        "page_index": pi,
                        "current_box_index": ci,
                        "iou": round(score, 4),
                        "forbidden_reference_id": old.get("id"),
                        "current": {k: cand.get(k) for k in ("x1", "y1", "x2", "y2", "confidence", "semantic_type", "class_name")},
                    })
        page_rows.append({
            "page_index": pi,
            "reference_safe": len(refs),
            "current_safe": len(curs),
            "matched_iou_0_30": len(matches030),
            "missing_reference_safe": len(missing),
            "extra_current_safe": max(0, len(curs) - len(matches030)),
            "reference_review": len(ref_review.get(pi, [])),
            "current_review": len(cur_review.get(pi, [])),
            "missing_reference_boxes": missing[:10],
        })

    return {
        "reference_total_boxes": len(rows),
        "human_tombstones": len(tombstones),
        "reference_story_safe_boxes": sum(map(len, ref_safe.values())),
        "reference_review_boxes": sum(map(len, ref_review.values())),
        "current_safe_boxes": sum(map(len, cur_safe.values())),
        "current_review_boxes": sum(map(len, cur_review.values())),
        "match": match_summary,
        "mean_iou_for_matches_ge_0_10": round(float(np.mean(all_iou)), 4) if all_iou else 0.0,
        "forbidden_non_story_destructive_hits": forbidden_hits,
        "pages": page_rows,
    }


def _reference_clean(base_root: Path, repair_root: Path, clean_name: str) -> Path:
    overlay = repair_root / "overlay" / "processed" / clean_name
    return overlay if overlay.is_file() else base_root / "processed" / clean_name


def _image_metrics(base_root: Path, repair_root: Path, current: dict) -> dict:
    kernel = np.ones((7, 7), np.uint8)
    total_ref_changed = total_covered = total_cur_changed = total_extra = total_pixels = 0
    weighted_mae = 0.0
    pages = []
    for pi, page in enumerate(current.get("pages", [])):
        raw_path = Path(str(page.get("original")))
        cur_path = Path(str(page.get("clean")))
        clean_name = f"clean_{raw_path.stem}.png"
        ref_path = _reference_clean(base_root, repair_root, clean_name)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        cur = cv2.imread(str(cur_path), cv2.IMREAD_COLOR)
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if raw is None or cur is None or ref is None or raw.shape != cur.shape or raw.shape != ref.shape:
            raise RuntimeError(f"invalid comparison images for page {pi}")
        ref_delta = np.max(cv2.absdiff(raw, ref), axis=2) > 8
        cur_delta = np.max(cv2.absdiff(raw, cur), axis=2) > 8
        guard = cv2.dilate(ref_delta.astype(np.uint8), kernel, iterations=1).astype(bool)
        ref_changed = int(np.count_nonzero(ref_delta))
        covered = int(np.count_nonzero(ref_delta & cur_delta))
        cur_changed = int(np.count_nonzero(cur_delta))
        extra = int(np.count_nonzero(cur_delta & ~guard))
        pixels = int(raw.shape[0] * raw.shape[1])
        mae = float(np.mean(cv2.absdiff(cur, ref)))
        total_ref_changed += ref_changed
        total_covered += covered
        total_cur_changed += cur_changed
        total_extra += extra
        total_pixels += pixels
        weighted_mae += mae * pixels
        pages.append({
            "page_index": pi,
            "clean_name": clean_name,
            "reference_changed_pixels": ref_changed,
            "current_changed_pixels": cur_changed,
            "reference_change_coverage_pct": round(100.0 * covered / ref_changed, 2) if ref_changed else None,
            "extra_change_area_pct": round(100.0 * extra / max(1, pixels), 4),
            "extra_share_of_current_changes_pct": round(100.0 * extra / max(1, cur_changed), 2),
            "approved_clean_mae": round(mae, 4),
            "current_clean_sha256": _sha256(cur_path),
            "reference_clean_sha256": _sha256(ref_path),
            "pixel_exact": bool(np.array_equal(cur, ref)),
        })
    return {
        "threshold_abs_rgb": 8,
        "guard_dilation_px": 3,
        "reference_changed_pixels": total_ref_changed,
        "current_changed_pixels": total_cur_changed,
        "reference_change_coverage_pct": round(100.0 * total_covered / max(1, total_ref_changed), 2),
        "extra_change_area_pct": round(100.0 * total_extra / max(1, total_pixels), 4),
        "extra_share_of_current_changes_pct": round(100.0 * total_extra / max(1, total_cur_changed), 2),
        "approved_clean_weighted_mae": round(weighted_mae / max(1, total_pixels), 4),
        "pixel_exact_pages": sum(1 for r in pages if r["pixel_exact"]),
        "worst_reference_coverage": sorted([r for r in pages if r["reference_change_coverage_pct"] is not None], key=lambda r: r["reference_change_coverage_pct"])[:10],
        "worst_extra_change": sorted(pages, key=lambda r: r["extra_change_area_pct"], reverse=True)[:10],
        "worst_approved_clean_mae": sorted(pages, key=lambda r: r["approved_clean_mae"], reverse=True)[:10],
        "pages": pages,
    }


def _timing_metrics(manifest: dict, batches: list[dict], reference_summary: dict) -> dict:
    wall_s = sum(float(r["elapsed_s"]) for r in batches)
    n = len(manifest.get("pages", []))
    detector_ms, inpaint_ms, residue_ms, total_ms = [], [], [], []
    for page in manifest.get("pages", []):
        timing = ((page.get("processing_metrics") or {}).get("timing_ms") or {})
        for key, dst in (("detect", detector_ms), ("auto_inpaint", inpaint_ms), ("residue_verify", residue_ms), ("total", total_ms)):
            if timing.get(key) is not None:
                dst.append(float(timing[key]))
    ref_wall = sum(float(b.get("elapsed_s") or 0) for b in reference_summary.get("batches", []))
    return {
        "current_batch_wall_s": round(wall_s, 3),
        "current_s_per_slice": round(wall_s / max(1, n), 3),
        "current_slices_per_min": round(60.0 * n / max(0.001, wall_s), 3),
        "reference_approved_batch_wall_s": round(ref_wall, 3),
        "reference_s_per_slice": round(ref_wall / max(1, int(reference_summary.get("processable") or n)), 3),
        "wall_reduction_vs_reference_pct": round(100.0 * (ref_wall - wall_s) / max(0.001, ref_wall), 2) if ref_wall else None,
        "mean_detector_ms": round(float(np.mean(detector_ms)), 3) if detector_ms else None,
        "mean_auto_inpaint_ms": round(float(np.mean(inpaint_ms)), 3) if inpaint_ms else None,
        "mean_residue_verify_ms": round(float(np.mean(residue_ms)), 3) if residue_ms else None,
        "mean_page_total_ms": round(float(np.mean(total_ms)), 3) if total_ms else None,
        "batches": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--base-clean-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/chapter351-human-clean/report.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    reference_summary = json.loads((args.base_clean_root / "clean-review-summary.json").read_text(encoding="utf-8"))
    if reference_summary.get("source_url") != CHAPTER_URL or int(reference_summary.get("slices") or 0) != 40:
        raise RuntimeError("unexpected human clean reference")
    imported = _restore_import(args.import_root)
    if len(imported.get("pages", [])) != 40:
        raise RuntimeError(f"expected 40 reviewed slices, got {len(imported.get('pages', []))}")

    pipeline = OptimizedChapterPipeline()
    indices = [i for i, page in enumerate(imported.get("pages", [])) if not page.get("skipped")]
    batches = []
    started_all = time.perf_counter()
    for start in range(0, len(indices), args.batch_size):
        batch = indices[start:start + args.batch_size]
        t0 = time.perf_counter()
        pipeline.process_pages(CHAPTER_ID, batch, workers=args.workers)
        elapsed = time.perf_counter() - t0
        batches.append({"indices": batch, "elapsed_s": round(elapsed, 3)})
        print(f"processed {start + len(batch)}/{len(indices)} in {elapsed:.1f}s", flush=True)

    current = load_manifest_raw(CHAPTER_ID)
    boxes = _box_metrics(args.base_clean_root, args.repair_root, current)
    images = _image_metrics(args.base_clean_root, args.repair_root, current)
    timing = _timing_metrics(current, batches, reference_summary)
    timing["current_total_loop_wall_s"] = round(time.perf_counter() - started_all, 3)
    report = {
        "benchmark": "chapter351-human-approved-clean-reference",
        "chapter_id": CHAPTER_ID,
        "chapter_url": CHAPTER_URL,
        "reference": {
            "import_run_id": 34700742617,
            "import_artifact": "01-import-c3513510",
            "base_clean_run_id": 34700955145,
            "base_clean_artifact": "02-clean-review-c3513510",
            "repair_run_id": 34702416332,
            "repair_artifact": "02b-clean-local-repair-c3513510",
            "human_review": "PASS; zero story-text/artwork-damage/residue blockers after local repair",
        },
        "timing": timing,
        "boxes": boxes,
        "images": images,
        "flags": {
            "human_forbidden_destructive_hit": bool(boxes["forbidden_non_story_destructive_hits"]),
            "reference_safe_recall_below_99_at_iou_030": boxes["match"]["iou_0_3"]["recall_pct"] < 99.0,
            "reference_change_coverage_below_99": images["reference_change_coverage_pct"] < 99.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CH351_HUMAN_CLEAN=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
