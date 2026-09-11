#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

CHAPTER_ID = "c2182180"
PAGE_INDEX = 64
SELECTED_FONT = "Mac-dinh-3"

# Human-selected from the 06B candidate proof. These are the nine skill-grid
# objects only; dialogue and all other pages remain untouched.
REPAIRS = {
    "text_e432a813a33b417e": {"font_size": 59, "lines": ["KỸ NĂNG SỞ HỮU"]},
    "text_64b56e211cab46fb": {"font_size": 25, "lines": ["KỸ THUẬT MA PHÁP", "CAO CẤP", "LV. 1"]},
    "text_54b93201c49b4ac7": {"font_size": 35, "lines": ["TRUYỀN MANA", "LV. 6"]},
    "text_5562636cb2f34195": {"font_size": 22, "lines": ["HỎA THUẬT", "LV. 5"]},
    "text_d5b806e496784aac": {"font_size": 34, "lines": ["TRIỆU HỒI", "LV. 6"]},
    "text_b478707b63c54344": {"font_size": 30, "lines": ["PHỤ MA TRUNG", "CẤP", "LV. 4"]},
    "text_6b6e678650b64f71": {"font_size": 30, "lines": ["NGHIÊN CỨU", "TRUNG CẤP", "LV. 8"]},
    "text_a3771febd50a45ce": {"font_size": 35, "lines": ["KHÁNG MA PHÁP", "LV. 3"]},
    "text_ca9d8fbd11e94caa": {"font_size": 25, "lines": ["ĐIỀM TĨNH", "LV. 3"]},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typeset-root", required=True)
    ap.add_argument("--candidate-report", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.typeset_root)
    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_json(src / "processed-manifest-typeset.json")
    plan = read_json(src / "typography-plan.json")
    summary = read_json(src / "typeset-preflight-summary.json")
    candidate = read_json(Path(args.candidate_report))

    if manifest.get("chapter_id") != CHAPTER_ID or plan.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("chapter identity mismatch")
    if summary.get("status") != "PASS" or summary.get("active_story_objects") != 98:
        raise SystemExit(f"source typeset gate not PASS: {summary}")
    if int(candidate.get("page_index", -1)) != PAGE_INDEX:
        raise SystemExit("candidate proof page mismatch")

    selected = next((x for x in candidate.get("candidate_fonts") or [] if x.get("font") == SELECTED_FONT), None)
    if not selected:
        raise SystemExit(f"selected font missing from candidate proof: {SELECTED_FONT}")
    candidate_rows = {str(x.get("id")): x for x in selected.get("objects") or []}
    if set(REPAIRS) != set(candidate_rows):
        raise SystemExit(f"candidate target mismatch: expected {sorted(REPAIRS)}, got {sorted(candidate_rows)}")
    for oid, spec in REPAIRS.items():
        row = candidate_rows[oid]
        if int(row.get("font_size", -1)) != spec["font_size"] or list(row.get("lines") or []) != spec["lines"]:
            raise SystemExit(f"candidate proof changed for {oid}: {row}")

    before_manifest = copy.deepcopy(manifest)
    changed_manifest = []
    page = (manifest.get("pages") or [])[PAGE_INDEX]
    by_id = {str(o.get("id") or ""): o for o in page.get("text_objects") or [] if isinstance(o, dict)}
    for oid, spec in REPAIRS.items():
        obj = by_id.get(oid)
        if obj is None or obj.get("source_missing"):
            raise SystemExit(f"repair object missing/inactive: {oid}")
        style = obj.setdefault("style", {})
        old = copy.deepcopy(style)
        style.update({
            "font": SELECTED_FONT,
            "fontSize": str(spec["font_size"]),
            "bold": False,
            "strokeWidth": "2",
            "horizontalAlign": "left",
            "verticalAlign": "top",
        })
        changed_manifest.append({"id": oid, "before": old, "after": copy.deepcopy(style)})

    # Fail closed: no render-facing object outside the explicit repair set may change.
    original_pages = before_manifest.get("pages") or []
    current_pages = manifest.get("pages") or []
    if len(original_pages) != len(current_pages):
        raise SystemExit("page count changed during repair")
    for pi, (old_page, new_page) in enumerate(zip(original_pages, current_pages)):
        old_objs = {str(o.get("id") or ""): o for o in old_page.get("text_objects") or [] if isinstance(o, dict)}
        new_objs = {str(o.get("id") or ""): o for o in new_page.get("text_objects") or [] if isinstance(o, dict)}
        if set(old_objs) != set(new_objs):
            raise SystemExit(f"text object set changed on page {pi}")
        for oid in old_objs:
            old_obj = copy.deepcopy(old_objs[oid])
            new_obj = copy.deepcopy(new_objs[oid])
            if pi == PAGE_INDEX and oid in REPAIRS:
                old_obj["style"] = new_obj.get("style")
            if old_obj != new_obj:
                raise SystemExit(f"unexpected object mutation page={pi} id={oid}")

    plan_objects = plan.get("objects") or {}
    for oid, spec in REPAIRS.items():
        meta = plan_objects.get(oid)
        if not isinstance(meta, dict) or int(meta.get("page_index", -1)) != PAGE_INDEX:
            raise SystemExit(f"typography object mismatch: {oid}")
        style = meta.setdefault("style", {})
        style.update({
            "font": SELECTED_FONT,
            "fontSize": str(spec["font_size"]),
            "bold": False,
            "strokeWidth": "2",
            "horizontalAlign": "left",
            "verticalAlign": "top",
        })
        meta["wrapped_lines"] = spec["lines"]
        sizing = meta.setdefault("sizing", {})
        sizing["selected_font_size"] = spec["font_size"]
        sizing["wrapped_line_count"] = len(spec["lines"])
        meta["human_local_typeset_repair"] = {
            "reason": "visual_review_status_grid_collision_and_decorative_font",
            "selected_font": SELECTED_FONT,
            "candidate_proof": "06b-status-ui-font-candidates-c2182180",
        }

    sizes = []
    font_usage = Counter()
    for oid, meta in plan_objects.items():
        if not isinstance(meta, dict):
            continue
        style = meta.get("style") or {}
        fs = str(style.get("fontSize") or "")
        if fs.isdigit():
            sizes.append(int(fs))
        font_usage[str(style.get("font") or "default")] += 1
    if len(sizes) != 98:
        raise SystemExit(f"expected 98 fixed sizes after repair, got {len(sizes)}")

    plan["font_usage"] = dict(font_usage)
    plan["font_size_range_selected"] = {"minimum": min(sizes), "maximum": max(sizes)}
    plan["size_histogram"] = {str(k): v for k, v in sorted(Counter(sizes).items(), reverse=True)}
    plan["readability_warning_objects"] = [
        x for x in (plan.get("readability_warning_objects") or [])
        if str(x.get("id")) not in REPAIRS
    ]
    counts = plan.setdefault("counts", {})
    counts["readability_warnings"] = len(plan["readability_warning_objects"])
    plan["local_repair"] = {
        "status": "PASS_PENDING_RENDER_VISUAL_REVIEW",
        "page_index": PAGE_INDEX,
        "selected_font": SELECTED_FONT,
        "target_object_ids": sorted(REPAIRS),
        "reason": "human visual review found inter-column collisions and poor status-grid legibility",
        "candidate_run_id": 34582722992,
        "candidate_artifact": "06b-status-ui-font-candidates-c2182180",
    }

    summary["checkpoint"] = "05b-typeset-local-repair"
    summary["minimum_font_size"] = min(sizes)
    summary["maximum_font_size"] = max(sizes)
    summary["readability_warning_count"] = len(plan["readability_warning_objects"])
    summary["status"] = "PASS"
    summary["next_action"] = "RERENDER_AND_HUMAN_VISUAL_REVIEW"
    summary["local_repair_object_count"] = len(REPAIRS)
    summary["local_repair_font"] = SELECTED_FONT

    repair_review = {
        "checkpoint": "05b-typeset-local-repair",
        "chapter_id": CHAPTER_ID,
        "status": "PASS_PENDING_RENDER_VISUAL_REVIEW",
        "page_index": PAGE_INDEX,
        "selected_font": SELECTED_FONT,
        "target_object_ids": sorted(REPAIRS),
        "changed_objects": changed_manifest,
        "source_typeset_run_id": 34580885638,
        "source_typeset_artifact": "05-typeset-preflight-c2182180",
        "candidate_run_id": 34582722992,
        "candidate_artifact": "06b-status-ui-font-candidates-c2182180",
        "human_reason": "Mac-dinh-3 was visually selected for compact Vietnamese legibility and clean separation between skill-grid columns.",
        "next_action": "RERENDER_FROM_APPROVED_CLEAN_STACK_AND_REVIEW_PAGE_14_STATUS_GRID",
    }

    write_json(out / "processed-manifest-typeset.json", manifest)
    write_json(out / "typography-plan.json", plan)
    write_json(out / "typeset-preflight-summary.json", summary)
    write_json(out / "typeset-local-repair.json", repair_review)
    print(json.dumps({
        "status": "PASS",
        "repaired": len(REPAIRS),
        "font": SELECTED_FONT,
        "minimum_font_size": min(sizes),
        "warnings_remaining": len(plan["readability_warning_objects"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
