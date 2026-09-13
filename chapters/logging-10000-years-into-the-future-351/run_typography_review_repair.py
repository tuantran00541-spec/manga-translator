from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

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


def install_clean_visual_repairs(config: dict, renderer) -> None:
    repairs = list(config.get("clean_visual_repairs") or [])
    if not repairs:
        return

    original_restore = renderer.restore_clean_stack

    def restore_with_human_clean_repairs() -> None:
        original_restore()
        applied = []
        for repair in repairs:
            filename = str(repair.get("file") or "")
            if not filename:
                raise SystemExit(f"clean visual repair missing file: {repair}")
            path = renderer.WORK / filename
            if not path.is_file():
                raise SystemExit(f"clean visual repair target missing: {path}")

            image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)
            fill = tuple(int(v) for v in (repair.get("fill") or [255, 255, 255]))
            rects = repair.get("rectangles") or []
            if not rects:
                raise SystemExit(f"clean visual repair has no rectangles: {repair}")

            for rect in rects:
                if len(rect) != 4:
                    raise SystemExit(f"malformed clean repair rectangle: {rect}")
                x1, y1, x2, y2 = [int(v) for v in rect]
                if x2 <= x1 or y2 <= y1:
                    raise SystemExit(f"empty clean repair rectangle: {rect}")
                draw.rectangle((x1, y1, x2, y2), fill=fill)

            image.save(path)

            # The repaired areas are fully inside a white speech bubble. After the
            # local patch no source-ink pixel should remain inside those rectangles.
            verify = Image.open(path).convert("L")
            dark_pixels = 0
            for rect in rects:
                x1, y1, x2, y2 = [int(v) for v in rect]
                crop = verify.crop((x1, y1, x2 + 1, y2 + 1))
                dark_pixels += sum(1 for px in crop.getdata() if px < 200)
            if dark_pixels:
                raise SystemExit(
                    f"clean visual repair verification failed for {filename}: {dark_pixels} dark pixels remain"
                )

            applied.append({
                "file": filename,
                "rectangles": rects,
                "reason": str(repair.get("reason") or ""),
                "dark_pixels_after": dark_pixels,
            })

        report = {
            "checkpoint": "05a-local-clean-visual-repair",
            "chapter_id": "c3513510",
            "status": "PASS",
            "repairs": applied,
        }
        (OUT / "clean-visual-repair-applied.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))

    renderer.restore_clean_stack = restore_with_human_clean_repairs


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

    clean_repairs = list(config.get("clean_visual_repairs") or [])
    clean_report = None
    if clean_repairs:
        clean_report_path = OUT / "clean-visual-repair-applied.json"
        if not clean_report_path.is_file():
            raise SystemExit("configured clean visual repair was not applied")
        clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
        if clean_report.get("status") != "PASS" or len(clean_report.get("repairs") or []) != len(clean_repairs):
            raise SystemExit(f"clean visual repair contract failed: {clean_report}")

    shutil.copy2(CONFIG, OUT / "typography-review-repair.json")
    report = {
        "checkpoint": "05b-human-typography-review-repair",
        "chapter_id": "c3513510",
        "status": "RERENDERED_REVIEW_REQUIRED",
        "font_fallback_count": summary.get("font_fallback_count"),
        "role_font_overrides_verified": expected,
        "clean_visual_repairs_verified": len(clean_report.get("repairs") or []) if clean_report else 0,
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
    install_clean_visual_repairs(config, renderer)
    renderer.main()
    verify_repaired_candidate(config)


if __name__ == "__main__":
    main()
