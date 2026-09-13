from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

CHAPTER_ID = "c241d001"
SERIES = "Doctors Rebirth"
SERIES_SLUG = "doctors-rebirth-53fc8424"
CHAPTER_NUMBER = 241
BRANCH = "chapter/doctors-rebirth+241"
SOURCE_URL = "https://asurascans.com/comics/doctors-rebirth-53fc8424/chapter/241"
RENDER_RUN_ID = 34744936501
RENDER_ARTIFACT = "05-render-candidate-c241d001"
RENDER_ARTIFACT_ID = 10313930879
RENDER_ARTIFACT_DIGEST = "sha256:3f10c19e602687a6eacecf50beebe5369f9340c9b68db41d77ff62d858604f56"
RENDER_HEAD_SHA = "d19d9eb51e6fc910aed04adc0c7517bd49b04066"
FINAL_NAME = "Doctors-Rebirth-Chapter-241-VI-FINAL.zip"

DURABLE = Path("chapters/doctors-rebirth-241")
SRC = Path("checkpoint/06-approved-render")
OUT = Path("checkpoint/07-final")

EXPECTED_DIMS = [
    (1200, 800),
    (900, 16000),
    (900, 1295),
    (900, 13891),
    (900, 14548),
    (900, 16000),
    (900, 6340),
    (900, 8858),
    (900, 15549),
    (900, 16000),
    (900, 1763),
    (900, 15827),
    (900, 16000),
    (900, 1301),
    (900, 16000),
    (900, 1734),
    (900, 1307),
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_durable_gates() -> tuple[dict, dict]:
    visual = load_json(DURABLE / "visual-review.json")
    if visual.get("status") != "PASS" or visual.get("final_export_allowed") is not True:
        raise SystemExit(f"visual review is not FINAL-ready: {visual}")
    if visual.get("render_run_id") != RENDER_RUN_ID:
        raise SystemExit("visual review render run drift")
    if visual.get("render_artifact_id") != RENDER_ARTIFACT_ID:
        raise SystemExit("visual review render artifact id drift")
    if visual.get("render_artifact_digest") != RENDER_ARTIFACT_DIGEST:
        raise SystemExit("visual review render digest drift")
    if visual.get("render_head_sha") != RENDER_HEAD_SHA:
        raise SystemExit("visual review render head drift")
    proof = visual.get("proof_review") or {}
    if proof.get("object_proof_sheets_reviewed") != 7:
        raise SystemExit(f"object proof coverage incomplete: {proof}")
    if proof.get("stitch_seams_reviewed") != 51:
        raise SystemExit(f"seam proof coverage incomplete: {proof}")
    if proof.get("full_page_overview_reviewed") is not True:
        raise SystemExit("full page overview not reviewed")
    if int((visual.get("checks") or {}).get("visual_blockers_total", -1)) != 0:
        raise SystemExit(f"visual blockers remain: {visual.get('checks')}")
    typo = visual.get("typography") or {}
    if int(typo.get("font_fallback_count", -1)) != 0:
        raise SystemExit(f"visual review still records font fallback: {typo}")
    if int(typo.get("raster_fallback_count", -1)) != 0:
        raise SystemExit(f"visual review still records raster fallback: {typo}")
    if int(typo.get("font_size_auto_objects", -1)) != 0:
        raise SystemExit(f"visual review still records autosize: {typo}")

    state = load_json(DURABLE / "chapter-state.json")
    if state.get("current_stage") != "VISUAL_REVIEW_PASS":
        raise SystemExit(f"chapter state not at visual PASS: {state.get('current_stage')}")
    if state.get("last_passed_checkpoint") != "VISUAL_REVIEW":
        raise SystemExit(f"last checkpoint drift: {state.get('last_passed_checkpoint')}")
    current = state.get("current_artifact") or {}
    if current.get("run_id") != RENDER_RUN_ID or current.get("artifact_id") != RENDER_ARTIFACT_ID:
        raise SystemExit(f"state points to wrong render artifact: {current}")
    if current.get("artifact_digest") != RENDER_ARTIFACT_DIGEST:
        raise SystemExit("state render artifact digest drift")
    if state.get("blockers") != []:
        raise SystemExit(f"chapter state still has blockers: {state.get('blockers')}")
    return visual, state


def validate_render_artifact() -> tuple[dict, list[dict], dict, int, int, int, int]:
    render = load_json(SRC / "render-summary.json")
    if render.get("status") != "REVIEW_REQUIRED":
        raise SystemExit(f"unexpected render status: {render.get('status')}")
    expected_counts = (76, 76, 68, 17)
    actual_counts = (
        int(render.get("active_story_objects", -1)),
        int(render.get("rendered_story_objects", -1)),
        int(render.get("rendered_slices", -1)),
        int(render.get("source_pages", -1)),
    )
    if actual_counts != expected_counts:
        raise SystemExit(f"render counts drifted: {actual_counts} != {expected_counts}")
    if int(render.get("autosize_active_objects", -1)) != 0:
        raise SystemExit(f"autosize objects remain: {render}")
    if int(render.get("font_fallback_count", -1)) != 0:
        raise SystemExit(f"font fallback remains: {render}")
    if int(render.get("raster_fallback_count", -1)) != 0:
        raise SystemExit(f"raster fallback remains: {render}")
    if int(render.get("font_size_min_px", 0)) < 28:
        raise SystemExit(f"font below hard minimum: {render}")

    plan_path = SRC / "typography-plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"missing typography plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, list) or len(plan) != 76:
        raise SystemExit(f"typography plan count drift: {len(plan) if isinstance(plan, list) else type(plan)}")
    sizes = [int(row["font_size"]) for row in plan]
    if min(sizes) != 30 or max(sizes) != 80:
        raise SystemExit(f"typography size range drift: min={min(sizes)} max={max(sizes)}")
    if any(row.get("font_fallback_reason") for row in plan):
        raise SystemExit("typography plan contains fallback reasons")

    raster_path = SRC / "raster-fallbacks.json"
    if not raster_path.is_file():
        raise SystemExit("missing raster-fallbacks.json")
    raster = json.loads(raster_path.read_text(encoding="utf-8"))
    if raster != []:
        raise SystemExit(f"raster fallbacks remain: {raster}")

    manifest = load_json(SRC / "translated-manifest-rendered.json")
    active = persistent_tombstones = untranslated = auto = 0
    for page in manifest.get("pages") or []:
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            if obj.get("source_missing"):
                persistent_tombstones += 1
                continue
            active += 1
            if not str(obj.get("translation") or "").strip():
                untranslated += 1
            if str((obj.get("style") or {}).get("fontSize")).strip().lower() == "auto":
                auto += 1
    if (active, persistent_tombstones, untranslated, auto) != (76, 134, 0, 0):
        raise SystemExit(
            "manifest gate failed: "
            f"active={active} tombstones={persistent_tombstones} untranslated={untranslated} auto={auto}"
        )
    return render, plan, manifest, active, persistent_tombstones, untranslated, auto


def build_final_archive() -> tuple[Path, list[dict]]:
    final_dir = SRC / "final_pages"
    pages = sorted(final_dir.glob("*.png"))
    expected_names = [f"{i:03d}.png" for i in range(17)]
    if [p.name for p in pages] != expected_names:
        raise SystemExit(f"final page set/order drift: {[p.name for p in pages]}")

    OUT.mkdir(parents=True, exist_ok=True)
    final_zip = OUT / FINAL_NAME
    page_meta: list[dict] = []
    expected_archive_names = [f"page_{i:03d}.png" for i in range(1, 18)]

    with zipfile.ZipFile(final_zip, "w") as archive:
        for source_idx, (path, expected_dim, archive_name) in enumerate(
            zip(pages, EXPECTED_DIMS, expected_archive_names)
        ):
            data = path.read_bytes()
            with Image.open(path) as image:
                image.load()
                actual_dim = (image.width, image.height)
                if image.format != "PNG":
                    raise SystemExit(f"not PNG: {path}")
            if actual_dim != expected_dim:
                raise SystemExit(f"{path.name} dimensions {actual_dim}, expected {expected_dim}")

            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
            page_meta.append(
                {
                    "source_page_index": source_idx,
                    "file": archive_name,
                    "width": actual_dim[0],
                    "height": actual_dim[1],
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )

    with zipfile.ZipFile(final_zip, "r") as archive:
        if archive.namelist() != expected_archive_names:
            raise SystemExit(f"archive order drift: {archive.namelist()}")
        bad = archive.testzip()
        if bad is not None:
            raise SystemExit(f"ZIP CRC failure: {bad}")
        for row in page_meta:
            data = archive.read(row["file"])
            if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
                raise SystemExit(f"ZIP member integrity mismatch: {row['file']}")
    return final_zip, page_meta


def main() -> None:
    visual, _state = validate_durable_gates()
    render, plan, _manifest, active, persistent_tombstones, untranslated, auto = validate_render_artifact()
    final_zip, page_meta = build_final_archive()
    archive_bytes = final_zip.read_bytes()

    report = {
        "chapter_id": CHAPTER_ID,
        "series": SERIES,
        "series_slug": SERIES_SLUG,
        "chapter_number": CHAPTER_NUMBER,
        "status": "FINAL_PASS",
        "branch": BRANCH,
        "source_url": SOURCE_URL,
        "source_render_run_id": RENDER_RUN_ID,
        "source_render_artifact": RENDER_ARTIFACT,
        "source_render_artifact_id": RENDER_ARTIFACT_ID,
        "source_render_artifact_digest": RENDER_ARTIFACT_DIGEST,
        "source_render_head_sha": RENDER_HEAD_SHA,
        "source_pages": 17,
        "render_slices": 68,
        "active_story_objects": active,
        "human_tombstones": int(visual.get("human_tombstones", -1)),
        "persistent_tombstones": persistent_tombstones,
        "active_untranslated_story_objects": untranslated,
        "import_gate": "PASS",
        "clean_gate": "PASS_AFTER_LOCAL_REPAIRS",
        "ocr_gate": "PASS_AFTER_HUMAN_CURATION",
        "translation_gate": "PASS",
        "typeset_gate": "PASS_AFTER_HUMAN_GEOMETRY_PUNCTUATION_FONT_AND_BREATHING_REPAIRS",
        "visual_gate": "PASS",
        "visual_blockers_remaining": 0,
        "normal_dialogue_font": "Mac-dinh-3",
        "scene_emotion_context_aware_typography": True,
        "font_size_auto_objects": auto,
        "font_size_min_px": min(int(row["font_size"]) for row in plan),
        "font_size_max_px": max(int(row["font_size"]) for row in plan),
        "font_fallback_count": int(render.get("font_fallback_count", -1)),
        "raster_fallback_count": int(render.get("raster_fallback_count", -1)),
        "geometry_repairs": 1,
        "punctuation_visual_repairs": 8,
        "reviewed_font_assignments": 3,
        "optical_size_repairs": 1,
        "stitch_seams_reviewed": 51,
        "archive": final_zip.name,
        "archive_png_count": 17,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_bytes(archive_bytes),
        "page_order": [row["file"] for row in page_meta],
        "final_pages": page_meta,
        "final_export_allowed": True,
    }
    (OUT / "final-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_CHECKPOINT.txt").write_text(
        "FINAL PASS: durable human visual review + deterministic ZIP integrity gates passed.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"final_pages", "page_order"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
