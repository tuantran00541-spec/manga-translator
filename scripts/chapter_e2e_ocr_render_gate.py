from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

from PIL import Image

from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.service import OCRService
from app.optimized_pipeline import OptimizedChapterPipeline
from app.routers.export import _core_range, _source_core_metadata, _stitch_png_to_file, _validate_stitch_group
from app.render.page_renderer import render_text_objects
from app.schemas import RenderRequest
from app.text_objects import ensure_page_text_objects


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _valid_region(value: object, *, width: int, height: int) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        x1, y1, x2, y2 = (
            int(value["x1"]),
            int(value["y1"]),
            int(value["x2"]),
            int(value["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _seed_text_objects(pipeline: OptimizedChapterPipeline, chapter_id: str) -> int:
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        created = 0
        changed = False
        for page in manifest.get("pages", []):
            page_created, page_changed = ensure_page_text_objects(page)
            created += page_created
            changed = changed or page_changed
        if changed:
            save_manifest_raw(chapter_id, manifest)
    if changed:
        pipeline._sync_output_dir(
            chapter_id,
            manifest,
            list(range(len(manifest.get("pages", [])))),
        )
    return created


def _validate_metadata(manifest: dict) -> tuple[list[dict], list[dict], list[dict]]:
    recognized: list[dict] = []
    missing: list[dict] = []
    object_missing: list[dict] = []

    for page_index, page in enumerate(manifest.get("pages", [])):
        original_path = Path(str(page.get("original") or ""))
        if not original_path.is_file():
            continue
        with Image.open(original_path) as source:
            width, height = source.size

        box_by_id = {
            str(box.get("id")): box
            for box in page.get("boxes", [])
            if isinstance(box, dict) and box.get("id")
        }

        for box in page.get("boxes", []):
            if not isinstance(box, dict):
                continue
            text = str(box.get("ocr_text") or "").strip()
            if not text:
                continue
            record = {
                "page_index": page_index,
                "box_id": str(box.get("id") or ""),
                "text": text,
                "color": box.get("ocr_text_color"),
                "font_size": box.get("ocr_font_size"),
                "region": box.get("ocr_text_region"),
            }
            recognized.append(record)
            reasons: list[str] = []
            color = str(box.get("ocr_text_color") or "")
            if not _HEX_COLOR.fullmatch(color):
                reasons.append("color")
            try:
                size = int(box.get("ocr_font_size") or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= 0:
                reasons.append("font_size")
            if not _valid_region(box.get("ocr_text_region"), width=width, height=height):
                reasons.append("text_region")
            if reasons:
                missing.append({**record, "missing": reasons})

        for obj in page.get("text_objects", []):
            if not isinstance(obj, dict) or not obj.get("auto_generated"):
                continue
            text = str(obj.get("ocr_text") or "").strip()
            if not text:
                continue
            reasons: list[str] = []
            if not _HEX_COLOR.fullmatch(str(obj.get("ocr_text_color") or "")):
                reasons.append("color")
            try:
                size = int(obj.get("ocr_font_size") or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= 0:
                reasons.append("font_size")
            if not _valid_region(obj.get("ocr_text_region"), width=width, height=height):
                reasons.append("text_region")
            source_boxes = [
                box_by_id.get(str(box_id))
                for box_id in (obj.get("source_boxes") or [])
            ]
            if source_boxes and not any(source_boxes):
                reasons.append("source_box")
            if reasons:
                object_missing.append(
                    {
                        "page_index": page_index,
                        "object_id": str(obj.get("id") or ""),
                        "text": text,
                        "missing": reasons,
                    }
                )

    return recognized, missing, object_missing


def _render_samples(
    manifest: dict,
    chapter_id: str,
    output_dir: Path,
    *,
    max_pages: int,
) -> tuple[int, int, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir.parent / "original"
    clean_dir = output_dir.parent / "clean"
    original_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = 0
    rendered_objects = 0
    evidence: list[dict] = []

    for page_index, page in enumerate(manifest.get("pages", [])):
        if rendered_pages >= max_pages:
            break
        objects = [
            obj
            for obj in (page.get("text_objects") or [])
            if isinstance(obj, dict)
            and not obj.get("source_missing")
            and str(obj.get("ocr_text") or "").strip()
            and obj.get("ocr_text_color")
            and obj.get("ocr_font_size")
            and isinstance(obj.get("ocr_text_region"), dict)
        ]
        if not objects:
            continue
        clean_path = Path(str(page.get("clean") or ""))
        if not clean_path.is_file():
            continue

        with Image.open(clean_path) as source:
            image = source.convert("RGB")

        translations = {
            str(obj["id"]): str(obj.get("ocr_text") or "").strip()
            for obj in objects
            if obj.get("id")
        }
        if not translations:
            continue

        req = RenderRequest(
            chapter_id=chapter_id,
            page_index=page_index,
            translations=translations,
        )
        count = render_text_objects(
            image,
            req,
            objects,
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        if count <= 0:
            continue

        out_path = output_dir / f"page_{page_index:03d}_rendered.png"
        original_out = original_dir / f"page_{page_index:03d}_original.png"
        clean_out = clean_dir / f"page_{page_index:03d}_clean.png"
        image.save(out_path, format="PNG")
        with Image.open(Path(str(page.get("original") or ""))) as source:
            source.convert("RGB").save(original_out, format="PNG")
        with Image.open(clean_path) as clean_source:
            clean_source.convert("RGB").save(clean_out, format="PNG")
        rendered_pages += 1
        rendered_objects += count
        evidence.append(
            {
                "page_index": page_index,
                "rendered_objects": count,
                "rendered_path": out_path.as_posix(),
                "original_path": original_out.as_posix(),
                "clean_path": clean_out.as_posix(),
                "objects": [
                    {
                        "id": str(obj.get("id") or ""),
                        "text": str(obj.get("ocr_text") or ""),
                        "color": obj.get("ocr_text_color"),
                        "font_size": obj.get("ocr_font_size"),
                        "region": obj.get("ocr_text_region"),
                    }
                    for obj in objects
                ],
            }
        )

    return rendered_pages, rendered_objects, evidence


def _complete_text_objects(page: dict) -> list[dict]:
    return [
        obj
        for obj in (page.get("text_objects") or [])
        if isinstance(obj, dict)
        and obj.get("id")
        and not obj.get("source_missing")
        and str(obj.get("ocr_text") or "").strip()
        and obj.get("ocr_text_color")
        and obj.get("ocr_font_size")
        and isinstance(obj.get("ocr_text_region"), dict)
    ]


def _render_complete_slice(
    page: dict,
    *,
    chapter_id: str,
    page_index: int,
    output_path: Path,
) -> int:
    clean_path = Path(str(page.get("clean") or ""))
    if not clean_path.is_file():
        raise FileNotFoundError(f"clean slice missing for page {page_index}")
    objects = _complete_text_objects(page)
    if not objects:
        shutil.copyfile(clean_path, output_path)
        return 0

    with Image.open(clean_path) as source:
        image = source.convert("RGB")
    translations = {
        str(obj["id"]): str(obj.get("ocr_text") or "").strip()
        for obj in objects
    }
    req = RenderRequest(
        chapter_id=chapter_id,
        page_index=page_index,
        translations=translations,
    )
    count = render_text_objects(
        image,
        req,
        objects,
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    image.save(output_path, format="PNG")
    return int(count)


def _write_zip_from_directory(directory: Path, output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(directory).as_posix())


def _build_complete_chapter(
    manifest: dict,
    chapter_id: str,
    artifact_root: Path,
) -> dict:
    complete_root = artifact_root / "complete"
    rendered_slice_dir = complete_root / "_rendered_slices"
    clean_pages_dir = complete_root / "clean"
    rendered_pages_dir = complete_root / "rendered-source-text"
    metadata_dir = complete_root / "metadata"
    for directory in (
        rendered_slice_dir,
        clean_pages_dir,
        rendered_pages_dir,
        metadata_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    groups: dict[int, list[dict]] = {}
    rendered_objects = 0
    metadata_records: list[dict] = []
    translation_template: list[dict] = []

    for page_index, page in enumerate(manifest.get("pages", [])):
        if page.get("skipped"):
            continue
        clean_path = Path(str(page.get("clean") or ""))
        if not clean_path.is_file():
            raise FileNotFoundError(f"clean slice missing for page {page_index}")

        rendered_slice = rendered_slice_dir / f"slice_{page_index:03d}.png"
        rendered_objects += _render_complete_slice(
            page,
            chapter_id=chapter_id,
            page_index=page_index,
            output_path=rendered_slice,
        )

        source_page = int(page.get("source_page", page_index))
        source_range, source_height = _source_core_metadata(page)
        item = {
            "page_index": page_index,
            "source_page": source_page,
            "slice_index": int(page.get("slice_index", 0)),
            "clean_path": clean_path,
            "rendered_path": rendered_slice,
            "core_range": _core_range(page),
            "source_core_range": source_range,
            "source_height": source_height,
        }
        groups.setdefault(source_page, []).append(item)

        for obj in _complete_text_objects(page):
            record = {
                "page_index": page_index,
                "source_page": source_page,
                "slice_index": int(page.get("slice_index", 0)),
                "object_id": str(obj.get("id") or ""),
                "source_boxes": [str(value) for value in (obj.get("source_boxes") or [])],
                "ocr_text": str(obj.get("ocr_text") or ""),
                "ocr_text_color": obj.get("ocr_text_color"),
                "ocr_font_size": obj.get("ocr_font_size"),
                "ocr_text_region": obj.get("ocr_text_region"),
                "ocr_quality": obj.get("ocr_quality"),
                "ocr_confidence": obj.get("ocr_confidence"),
            }
            metadata_records.append(record)
            translation_template.append(
                {
                    "object_id": record["object_id"],
                    "page_index": page_index,
                    "source_text": record["ocr_text"],
                    "translation": "",
                    "style": {
                        "color": record["ocr_text_color"],
                        "font_size": record["ocr_font_size"],
                        "region": record["ocr_text_region"],
                    },
                }
            )

    stitched_pages = 0
    for output_index, source_page in enumerate(sorted(groups), start=1):
        items = sorted(
            groups[source_page],
            key=lambda item: (int(item["slice_index"]), int(item["page_index"])),
        )
        _validate_stitch_group(source_page, items)
        core_ranges = [item["core_range"] for item in items]
        _stitch_png_to_file(
            [Path(item["clean_path"]) for item in items],
            clean_pages_dir / f"page_{output_index:03d}.png",
            core_ranges,
        )
        _stitch_png_to_file(
            [Path(item["rendered_path"]) for item in items],
            rendered_pages_dir / f"page_{output_index:03d}.png",
            core_ranges,
        )
        stitched_pages += 1

    (metadata_dir / "ocr_metadata.json").write_text(
        json.dumps(metadata_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "translations-template.json").write_text(
        json.dumps(translation_template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_copy = json.loads(json.dumps(manifest, default=str))
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest_copy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    clean_zip = complete_root / "war-of-extinction-ch1-clean.zip"
    rendered_zip = complete_root / "war-of-extinction-ch1-rendered-source-text.zip"
    _write_zip_from_directory(clean_pages_dir, clean_zip)
    _write_zip_from_directory(rendered_pages_dir, rendered_zip)

    package_zip = complete_root / "war-of-extinction-ch1-complete-package.zip"
    with zipfile.ZipFile(package_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for directory, prefix in (
            (clean_pages_dir, "clean"),
            (rendered_pages_dir, "rendered-source-text"),
            (metadata_dir, "metadata"),
        ):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=f"{prefix}/{path.relative_to(directory).as_posix()}",
                    )

    shutil.rmtree(rendered_slice_dir, ignore_errors=True)
    return {
        "stitched_source_pages": stitched_pages,
        "rendered_objects_full_chapter": rendered_objects,
        "metadata_objects": len(metadata_records),
        "clean_zip": clean_zip.as_posix(),
        "rendered_zip": rendered_zip.as_posix(),
        "package_zip": package_zip.as_posix(),
    }


def run(url: str, output: Path, *, workers: int, max_render_pages: int) -> dict:
    pipeline = OptimizedChapterPipeline()
    chapter_id = hashlib.sha256(
        f"chapter-e2e:{url}:{os.getenv('GITHUB_RUN_ID', time.time_ns())}".encode()
    ).hexdigest()[:8]
    report: dict = {
        "source_sha": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "chapter_url": url,
        "chapter_id": chapter_id,
        "workers": workers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        manifest = pipeline.download_chapter(url, chapter_id, workers=workers)
        report["download_and_slice_s"] = round(time.perf_counter() - started, 3)
        page_indices = list(range(len(manifest.get("pages", []))))
        report["slices"] = len(page_indices)
        report["source_pages"] = len(
            {page.get("source_page") for page in manifest.get("pages", [])}
        )
        if not page_indices:
            raise RuntimeError("chapter produced zero slices")

        started = time.perf_counter()
        pipeline.process_pages(chapter_id, page_indices, workers=workers)
        report["process_s"] = round(time.perf_counter() - started, 3)

        created = _seed_text_objects(pipeline, chapter_id)
        report["text_objects_created"] = created

        ocr_service = OCRService(MultiLangOCR(), pipeline)
        targets = ocr_service.plan_chapter(chapter_id)
        report["ocr_targets"] = len(targets)
        if not targets:
            raise RuntimeError("processed chapter produced zero OCR targets")

        ocr_errors: list[dict] = []
        ocr_fallback_targets = 0
        started = time.perf_counter()
        for ordinal, (page_index, box_id) in enumerate(targets, start=1):
            try:
                result = ocr_service.inspect_box_id(
                    chapter_id,
                    page_index,
                    box_id,
                    "en",
                    force=True,
                )
                if str(result.get("target_mode") or "") == "hybrid-fallback":
                    ocr_fallback_targets += 1
                if ordinal % 20 == 0 or ordinal == len(targets):
                    print(
                        f"OCR {ordinal}/{len(targets)} page={page_index} box={box_id} "
                        f"text={str(result.get('text') or '')[:80]!r}",
                        flush=True,
                    )
            except Exception as exc:
                ocr_errors.append(
                    {
                        "page_index": page_index,
                        "box_id": box_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        report["ocr_s"] = round(time.perf_counter() - started, 3)
        report["ocr_errors"] = ocr_errors
        report["ocr_fallback_targets"] = ocr_fallback_targets

        manifest = load_manifest_raw(chapter_id)
        recognized, missing, object_missing = _validate_metadata(manifest)
        report["recognized_text_boxes"] = len(recognized)
        report["metadata_complete_boxes"] = len(recognized) - len(missing)
        report["metadata_missing_boxes"] = missing
        report["text_object_metadata_missing"] = object_missing
        report["sample_recognized"] = recognized[:25]

        render_dir = output.parent / "rendered"
        started = time.perf_counter()
        rendered_pages, rendered_objects, evidence = _render_samples(
            manifest,
            chapter_id,
            render_dir,
            max_pages=max_render_pages,
        )
        report["render_s"] = round(time.perf_counter() - started, 3)
        report["rendered_pages"] = rendered_pages
        report["rendered_objects"] = rendered_objects
        report["render_evidence"] = evidence

        complete_started = time.perf_counter()
        report["complete_build"] = _build_complete_chapter(
            manifest,
            chapter_id,
            output.parent,
        )
        report["complete_build_s"] = round(time.perf_counter() - complete_started, 3)

        report["total_s"] = round(
            report["download_and_slice_s"]
            + report["process_s"]
            + report["ocr_s"]
            + report["render_s"]
            + report["complete_build_s"],
            3,
        )

        failures: list[str] = []
        if ocr_errors:
            failures.append(f"{len(ocr_errors)} OCR target(s) raised errors")
        if not recognized:
            failures.append("OCR produced no recognized text")
        if missing:
            failures.append(
                f"{len(missing)}/{len(recognized)} recognized boxes lack color/size/position metadata"
            )
        if object_missing:
            failures.append(
                f"{len(object_missing)} auto text object(s) failed metadata propagation"
            )
        if rendered_objects <= 0:
            failures.append("renderer produced no metadata-driven sample text")
        complete = report.get("complete_build") or {}
        if int(complete.get("stitched_source_pages") or 0) != int(report.get("source_pages") or 0):
            failures.append("complete chapter stitch did not reproduce every source page")
        if int(complete.get("metadata_objects") or 0) != len(recognized):
            failures.append("complete chapter metadata object count does not match recognized OCR")
        report["failures"] = failures
        report["passed"] = not failures
        if failures:
            raise RuntimeError("; ".join(failures))
        return report
    finally:
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real chapter detector -> inpaint -> OCR metadata -> render E2E gate"
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-render-pages", type=int, default=8)
    args = parser.parse_args()
    run(
        args.url,
        args.output,
        workers=max(1, args.workers),
        max_render_pages=max(1, args.max_render_pages),
    )


if __name__ == "__main__":
    main()
