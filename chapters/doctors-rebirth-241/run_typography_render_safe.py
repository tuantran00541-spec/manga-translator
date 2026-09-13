from __future__ import annotations

from pathlib import Path

RUNNER = Path("chapters/doctors-rebirth-241/run_typography_render.py")
source = RUNNER.read_text(encoding="utf-8")

anchor = "    plan_by_object = {row[\"object_id\"]: row for row in plan}\n    rendered_lookup: dict[tuple[int, int], Path] = {}\n"
replacement = "    plan_by_object = {row[\"object_id\"]: row for row in plan}\n    raster_fallbacks: list[dict] = []\n    rendered_lookup: dict[tuple[int, int], Path] = {}\n"
if source.count(anchor) != 1:
    raise SystemExit("chapter 241 render runner drifted before raster fallback anchor")
source = source.replace(anchor, replacement)

old_render = '''            image = render_text_in_box(
                image, obj["translation"],
                (region["x1"], region["y1"], region["x2"], region["y2"]),
                font_name=style["font"], font_size=style["fontSize"], padding=padding,
                fill="auto", is_bold=style["bold"], stroke_width=2, stroke_color="auto",
                bg_color="transparent", horizontal_align="center", vertical_align="middle",
            )
'''
new_render = '''            candidate = image.copy()
            try:
                image = render_text_in_box(
                    candidate, obj["translation"],
                    (region["x1"], region["y1"], region["x2"], region["y2"]),
                    font_name=style["font"], font_size=style["fontSize"], padding=padding,
                    fill="auto", is_bold=style["bold"], stroke_width=2, stroke_color="auto",
                    bg_color="transparent", horizontal_align="center", vertical_align="middle",
                )
            except OSError as exc:
                if "array allocation size too large" not in str(exc) or style["font"] == "Mac-dinh-3":
                    raise
                original_font = str(style["font"])
                original_size = int(style["fontSize"])
                fallback_font = "Mac-dinh-3"
                raw_w = int(region["x2"]) - int(region["x1"])
                raw_h = int(region["y2"]) - int(region["y1"])
                box_w = raw_w - 2 * padding
                box_h = raw_h - 2 * padding
                draw = ImageDraw.Draw(image)
                fallback_path = str(get_font_path(fallback_font))
                fallback_size = None
                fallback_lines = []
                for size in [s for s in ROLE_SIZES[row["role"]] if 28 <= s <= original_size]:
                    ok, wrapped = _fits(draw, obj["translation"], box_w, box_h, fallback_path, size, 2)
                    if ok:
                        fallback_size = size
                        fallback_lines = wrapped
                        break
                if fallback_size is None:
                    raise SystemExit(
                        f"raster fallback for {row['box_id']} cannot fit Mac-dinh-3 at >=28px; "
                        f"original={original_font} {original_size}px text={obj['translation']!r}"
                    ) from exc
                candidate = image.copy()
                image = render_text_in_box(
                    candidate, obj["translation"],
                    (region["x1"], region["y1"], region["x2"], region["y2"]),
                    font_name=fallback_font, font_size=fallback_size, padding=padding,
                    fill="auto", is_bold=style["bold"], stroke_width=2, stroke_color="auto",
                    bg_color="transparent", horizontal_align="center", vertical_align="middle",
                )
                fallback_reason = (
                    f"{original_font} rasterization failed in Pillow: array allocation size too large; "
                    f"human-safe fallback to {fallback_font}"
                )
                raster_fallbacks.append({
                    "box_id": row["box_id"], "source_page": row["source_page"],
                    "slice_index": row["slice_index"], "text": obj["translation"],
                    "original_font": original_font, "original_font_size": original_size,
                    "fallback_font": fallback_font, "fallback_font_size": fallback_size,
                    "region": dict(region), "reason": fallback_reason,
                })
                style["font"] = fallback_font
                style["fontSize"] = fallback_size
                row["font"] = fallback_font
                row["font_size"] = fallback_size
                row["font_fallback_reason"] = fallback_reason
                row["line_count"] = len(fallback_lines)
'''
if source.count(old_render) != 1:
    raise SystemExit("chapter 241 render runner drifted at render call site")
source = source.replace(old_render, new_render)

old_summary = '''        "font_fallback_count": sum(1 for row in plan if row.get("font_fallback_reason")),
        "font_counts": dict(Counter(row["font"] for row in plan)),
        "seam_shift_repairs": len(shifted), "proof_sheets": proof_names,
'''
new_summary = '''        "font_fallback_count": sum(1 for row in plan if row.get("font_fallback_reason")),
        "raster_fallback_count": len(raster_fallbacks),
        "font_counts": dict(Counter(row["font"] for row in plan)),
        "seam_shift_repairs": len(shifted), "proof_sheets": proof_names,
'''
if source.count(old_summary) != 1:
    raise SystemExit("chapter 241 render runner drifted at summary block")
source = source.replace(old_summary, new_summary)

old_write = '''    (OUT / "seam-shift-repairs.json").write_text(json.dumps(shifted, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "render-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
'''
new_write = '''    (OUT / "seam-shift-repairs.json").write_text(json.dumps(shifted, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "raster-fallbacks.json").write_text(json.dumps(raster_fallbacks, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "render-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
'''
if source.count(old_write) != 1:
    raise SystemExit("chapter 241 render runner drifted at artifact write block")
source = source.replace(old_write, new_write)

namespace = {
    "__name__": "__main__",
    "__file__": str(RUNNER),
}
exec(compile(source, str(RUNNER), "exec"), namespace, namespace)
