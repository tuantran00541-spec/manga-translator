"""Score a finished chapter, so each change to A.I mode is measured against the last run.

``python -m app.ai_mode.scorecard BASE.json NEW.json`` prints what got better and what got worse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from app.ai_mode.verify import leftovers, text_mask
from app.image_io import read_image
from app.region_policy import page_preserve_regions, text_object_in_preserve_region

# Metrics where a smaller number is better; every other compared metric is better when larger.
LOWER_IS_BETTER = {
    "clean.text_left", "clean.ghosts", "text.untranslated", "text.needs_review", "text.kept_original",
    "fit.does_not_fit", "fit.under_20px", "fit.split_word", "fit.missing_glyphs",
    "spend.cost_usd", "spend.requests", "spend.prompt_tokens", "spend.images", "spend.wall_s",
    "polish.flagged", "polish.still_flagged",
}


def _slice_view(page: dict) -> tuple[int, int]:
    """Rows of a slice that are exported, so text in an overlap is counted once."""
    core = page.get("stitch_core") if isinstance(page.get("stitch_core"), dict) else {}
    return int(core.get("core_y1", 0)), int(core.get("core_y2", 1 << 30))


def clean_score(pages: list[dict], detector) -> dict:
    """Text and ghosts the cleanup left on each exported slice, found by the detector models."""
    items = []
    checked = 0
    for index, page in enumerate(pages):
        if page.get("skipped") or not page.get("clean") or not page.get("original"):
            continue
        original, clean = read_image(Path(page["original"])), read_image(Path(page["clean"]))
        if original is None or clean is None or original.shape != clean.shape:
            continue
        top, bottom = _slice_view(page)
        bottom = min(bottom, original.shape[0])
        before, after = np.ascontiguousarray(original[top:bottom]), np.ascontiguousarray(clean[top:bottom])
        keep = [(int(r["x1"]), int(r["y1"]) - top, int(r["x2"]), int(r["y2"]) - top)
                for r in page_preserve_regions(page)]
        checked += 1
        for item in leftovers(before, after, text_mask(before.shape[:2], detector.text_boxes(before)), keep):
            x1, y1, x2, y2 = item.box
            items.append({"slice": index + 1, "kind": item.kind, "box": [x1, y1 + top, x2, y2 + top],
                          "px": item.px, "share": item.share})
    return {
        "slices": checked,
        "text_left": sum(item["kind"] == "text" for item in items),
        "ghosts": sum(item["kind"] == "ghost" for item in items),
        "items": items,
    }


def text_score(pages: list[dict]) -> dict:
    """How many text objects were translated, left blank, flagged or given back their original pixels."""
    counts = {"objects": 0, "translated": 0, "untranslated": 0, "needs_review": 0, "kept_original": 0}
    for page in pages:
        if page.get("skipped"):
            continue
        for obj in page.get("text_objects") or []:
            if (not isinstance(obj, dict) or obj.get("removed") or obj.get("joined_into")
                    or obj.get("overlap_dropped") or obj.get("source_missing")):
                continue
            counts["objects"] += 1
            if text_object_in_preserve_region(page, obj):
                counts["kept_original"] += 1
            elif str(obj.get("translation") or "").strip():
                counts["translated"] += 1
            else:
                counts["untranslated"] += 1
            counts["needs_review"] += bool(obj.get("needs_review"))
    return counts


def spend_score(run: dict) -> dict:
    """What the run cost, from the A.I cost report: money, requests, tokens, images and time per stage."""
    stages = run.get("transport") or {}
    gateway = run.get("gateway_job") or {}
    return {
        "cost_usd": round(float(gateway.get("cost_usd") or 0.0), 5),
        "requests": int(gateway.get("requests") or 0),
        "prompt_tokens": int(gateway.get("prompt_tokens") or 0),
        "images": sum(int(s.get("images") or 0) for s in stages.values()),
        "wall_s": float(run.get("wall_s") or 0.0),
        "by_stage": {name: {k: s.get(k) for k in ("requests", "images", "prompt_tokens", "completion_tokens")}
                     for name, s in stages.items()},
        "stage_s": {s.get("key"): s.get("elapsed_s") for s in run.get("stages") or []},
    }


def score_chapter(pages: list[dict], detector, run: dict | None = None) -> dict:
    """The whole scorecard of one chapter; ``run`` is the A.I cost report when there is one."""
    card = {"clean": clean_score(pages, detector), "text": text_score(pages)}
    if run:
        card["spend"] = spend_score(run)
        polished = (run.get("report") or {}).get("polish") or {}
        card["polish"] = {k: polished.get(k, 0) for k in ("judged", "flagged", "rewritten", "still_flagged")}
        readability = run.get("readability") or {}
        card["fit"] = {k: readability.get(k) for k in ("does_not_fit", "under_20px", "split_word", "missing_glyphs")}
    return card


def _flat(card: dict) -> dict[str, float]:
    flat = {}
    for group, values in card.items():
        for key, value in (values or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{group}.{key}"] = value
    return flat


def compare(base: dict, new: dict) -> list[str]:
    """One line per metric that changed, marked better or worse."""
    old, now = _flat(base), _flat(new)
    lines = []
    for key in sorted(set(old) & set(now)):
        if old[key] == now[key]:
            continue
        lower = now[key] < old[key]
        verdict = "better" if lower == (key in LOWER_IS_BETTER) else "worse"
        lines.append(f"{key}: {old[key]} -> {now[key]} ({verdict})")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.ai_mode.scorecard BASE.json NEW.json")
        return 2
    base, new = (json.loads(Path(path).read_text(encoding="utf-8")) for path in argv)
    for line in compare(base.get("scorecard", base), new.get("scorecard", new)) or ["no metric changed"]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
