from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path("chapters/logging-10000-years-into-the-future-351")
CONFIG = ROOT / "typography-review-repair.json"
BASE_RENDERER = ROOT / "run_typography_render.py"
TRANSLATED_MANIFEST = Path("checkpoint/04-translation/translated-manifest.json")
OUT = Path("checkpoint/05-render")


def load_renderer():
    spec = importlib.util.spec_from_file_location("chapter351_typography_renderer", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load chapter 351 typography renderer")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_translation_visual_repairs(config: dict) -> None:
    manifest = json.loads(TRANSLATED_MANIFEST.read_text(encoding="utf-8"))
    by_box = {}
    for page in manifest.get("pages") or []:
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            refs = [str(v) for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
            if len(refs) == 1:
                by_box[refs[0]] = obj

    applied = []
    for box_id, change in (config.get("translation_visual_repairs") or {}).items():
        obj = by_box.get(box_id)
        if obj is None:
            raise SystemExit(f"visual translation repair target missing: {box_id}")
        before = str(change["before"])
        after = str(change["after"])
        current = str(obj.get("translation") or "")
        if current != before:
            raise SystemExit(f"translation drift before visual repair {box_id}: {current!r} != {before!r}")
        obj["translation_before_visual_review_repair"] = current
        obj["translation"] = after
        obj["visual_review_translation_repair"] = True
        obj["visual_review_translation_repair_reason"] = str(change.get("reason") or "")
        applied.append(box_id)

    TRANSLATED_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"applied {len(applied)} local visual translation repairs: {applied}")


def verify_repaired_candidate(config: dict) -> None:
    summary = json.loads((OUT / "typography-summary.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "typography-plan.json").read_text(encoding="utf-8"))
    expected = config.get("role_font_overrides") or {}
    for role, font in expected.items():
        rows = [row for row in plan if row.get("role") == role]
        if not rows:
            raise SystemExit(f"expected typography role missing after repair: {role}")
        wrong = [row["box_id"] for row in rows if row.get("font") != font]
        if wrong:
            raise SystemExit(f"font repair failed for {role}: expected {font}, wrong={wrong}")
    if summary.get("font_fallback_count") != 0:
        raise SystemExit(f"unexpected font fallback remains after visual repair: {summary}")
    if summary.get("objects_at_or_below_30px") != 0:
        raise SystemExit(f"unexpected small text after visual repair: {summary}")

    shutil.copy2(CONFIG, OUT / "typography-review-repair.json")
    report = {
        "checkpoint": "05b-human-typography-review-repair",
        "chapter_id": "c3513510",
        "status": "RERENDERED_REVIEW_REQUIRED",
        "font_fallback_count": summary.get("font_fallback_count"),
        "role_font_overrides_verified": expected,
        "next_action": "HUMAN_VISUAL_REVIEW_REPAIRED_CANDIDATE"
    }
    (OUT / "typography-review-repair-applied.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    apply_translation_visual_repairs(config)
    renderer = load_renderer()
    renderer.ROLE_FONT.update(config.get("role_font_overrides") or {})
    renderer.main()
    verify_repaired_candidate(config)


if __name__ == "__main__":
    main()
