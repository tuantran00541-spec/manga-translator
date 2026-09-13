from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from app.manifest_utils import save_manifest_raw
from app.ocr.consistency import ConsistencyOCRService
from app.ocr.identity import clear_ocr_cache
from app.ocr.multi_lang_ocr import MultiLangOCR


CHAPTER_ID = "c241d001"
EXPECTED_SLICES = 68
EXPECTED_SOURCE_PAGES = 17
BASELINE = {
    "planned": 105,
    "recognized": 101,
    "empty": 4,
    "good": 80,
    "review": 18,
    "reject": 7,
    "crop_edge_text": 13,
    "incomplete_coverage": 5,
    "paddle_selective_retry": 21,
    "manual_corrections": 4,
}
HUMAN_CORRECTIONS = {
    "box_f3766e782eaa4479": "I KNOW IT\nSOUNDS INSANE,\nBUT IT'S TRUE.",
    "box_0eb41947eb9742b1": 'IS TO SAY,\n"GIVE UP, EVEN\nNOW."',
    "box_703644a6bbd4473a": "BUT I\nWANTED TO\nWIN.",
    "box_1bf8f8b8b4fd4bdf": (
        "PIERCING\nFINGER STRIKE,\nJIN CHEONHEE-STYLE\nAPPLICATION."
    ),
}
PUNCTUATION_STORY_IDS = {
    "box_ebbb5475e55d4a32",
    "box_a5b4216f8a584c31",
}


class _SyncOnlyPipeline:
    def _sync_output_dir(
        self, chapter_id: str, manifest: dict, page_indices: list[int]
    ) -> None:
        return None


def _normalize(value: str) -> str:
    value = str(value or "").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _prepare_runtime(import_root: Path, repaired_root: Path) -> None:
    raw_src = import_root / "raw"
    processed_src = repaired_root / "processed"
    if not raw_src.is_dir():
        raise SystemExit(f"missing raw import tree: {raw_src}")
    if not (processed_src / "manifest.json").is_file():
        raise SystemExit(f"missing repaired manifest: {processed_src / 'manifest.json'}")

    raw_dst = Path("data/raw") / CHAPTER_ID
    processed_dst = Path("data/processed") / CHAPTER_ID
    for path in (raw_dst, processed_dst):
        if path.exists():
            shutil.rmtree(path)
    raw_dst.parent.mkdir(parents=True, exist_ok=True)
    processed_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_src, raw_dst)
    shutil.copytree(processed_src, processed_dst)

    manifest = json.loads((processed_dst / "manifest.json").read_text(encoding="utf-8"))
    pages = manifest.get("pages") or []
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit(f"chapter id drift: {manifest.get('chapter_id')}")
    if len(pages) != EXPECTED_SLICES:
        raise SystemExit(f"slice count drift: {len(pages)}")
    if len({int(page.get("source_page")) for page in pages}) != EXPECTED_SOURCE_PAGES:
        raise SystemExit("source page count drift")

    for page_index, page in enumerate(pages):
        source_page = int(page.get("source_page"))
        slice_value = page.get("slice_index")
        slice_index = 0 if slice_value is None else int(slice_value)
        raw_path = raw_dst / "sliced" / f"{source_page:03d}_{slice_index:02d}.png"
        clean_path = processed_dst / f"clean_{source_page:03d}_{slice_index:02d}.png"
        if not raw_path.is_file():
            raise SystemExit(f"missing RAW slice: {raw_path}")
        if not clean_path.is_file():
            raise SystemExit(f"missing CLEAN slice: {clean_path}")
        page["original"] = raw_path.as_posix()
        page["clean"] = clean_path.as_posix()
        if page_index in {0, 67}:
            if not page.get("skipped"):
                raise SystemExit(f"non-story page {page_index} lost skipped state")
        elif page.get("skipped"):
            raise SystemExit(f"story page {page_index} unexpectedly skipped")

        for box in page.get("boxes") or []:
            if isinstance(box, dict):
                clear_ocr_cache(box)

    save_manifest_raw(CHAPTER_ID, manifest)


def _run_ocr() -> tuple[list[dict], list[dict]]:
    service = ConsistencyOCRService(MultiLangOCR(), _SyncOnlyPipeline())
    plan = service.plan_chapter(CHAPTER_ID)
    if len(plan) != BASELINE["planned"]:
        raise SystemExit(f"OCR plan drift: {len(plan)} != {BASELINE['planned']}")

    results: list[dict] = []
    failures: list[dict] = []
    print(f"planned={len(plan)}", flush=True)
    for index, (page_index, box_id) in enumerate(plan, start=1):
        try:
            result = service.inspect_box_id(
                CHAPTER_ID, page_index, box_id, "en", force=True
            )
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "page_index": page_index,
                    "box_id": box_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if index == 1 or index % 20 == 0 or index == len(plan):
            print(
                f"ocr={index}/{len(plan)} failures={len(failures)}",
                flush=True,
            )
    return results, failures


def _summarize(results: list[dict], failures: list[dict]) -> dict:
    quality = Counter(str(item.get("quality") or "unknown") for item in results)
    reasons = Counter(
        str(item.get("quality_reason") or "none") for item in results
    )
    recognized = sum(bool(str(item.get("text") or "").strip()) for item in results)
    context_retry = sum(bool(item.get("context_retry_applied")) for item in results)
    any_retry = sum(bool(item.get("retry_applied")) for item in results)

    by_id = {str(item.get("box_id")): item for item in results}
    corrections = {}
    exact_after = 0
    for box_id, expected in HUMAN_CORRECTIONS.items():
        actual = str((by_id.get(box_id) or {}).get("text") or "")
        exact = _normalize(actual) == _normalize(expected)
        exact_after += int(exact)
        corrections[box_id] = {
            "text": actual,
            "expected_human_after": expected,
            "matches_human_after": exact,
            "quality": (by_id.get(box_id) or {}).get("quality"),
            "quality_reason": (by_id.get(box_id) or {}).get("quality_reason"),
            "context_retry_applied": bool(
                (by_id.get(box_id) or {}).get("context_retry_applied")
            ),
        }

    punctuation = {}
    for box_id in sorted(PUNCTUATION_STORY_IDS):
        item = by_id.get(box_id) or {}
        punctuation[box_id] = {
            "text": str(item.get("text") or ""),
            "quality": item.get("quality"),
            "quality_reason": item.get("quality_reason"),
        }

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "planned": len(results) + len(failures),
        "recognized": recognized,
        "empty": len(results) - recognized,
        "failed": len(failures),
        "quality": dict(sorted(quality.items())),
        "quality_reasons": dict(sorted(reasons.items())),
        "crop_edge_text": reasons.get("crop-edge-text", 0),
        "incomplete_coverage": reasons.get("incomplete-coverage", 0),
        "punctuation_only_review": reasons.get("punctuation-only", 0),
        "context_retry_applied": context_retry,
        "any_retry_applied": any_retry,
        "manual_correction_targets_matching_human_after": exact_after,
        "manual_correction_targets_total": len(HUMAN_CORRECTIONS),
        "manual_correction_targets": corrections,
        "punctuation_story_targets": punctuation,
        "baseline": BASELINE,
        "delta": {
            "recognized": recognized - BASELINE["recognized"],
            "empty": (len(results) - recognized) - BASELINE["empty"],
            "good": quality.get("good", 0) - BASELINE["good"],
            "review": quality.get("review", 0) - BASELINE["review"],
            "reject": quality.get("reject", 0) - BASELINE["reject"],
            "crop_edge_text": reasons.get("crop-edge-text", 0)
            - BASELINE["crop_edge_text"],
            "incomplete_coverage": reasons.get("incomplete-coverage", 0)
            - BASELINE["incomplete_coverage"],
        },
        "failures": failures,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _prepare_runtime(args.import_root, args.repaired_root)
    results, failures = _run_ocr()
    summary = _summarize(results, failures)
    (args.output / "ocr-consistency-benchmark.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "ocr-consistency-results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("OCR benchmark had runtime failures")


if __name__ == "__main__":
    main()
