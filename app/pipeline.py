import os
import base64
import copy
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from app.downloader.registry import download_chapter as fetch_chapter_images
from app.downloader.slicer import slice_image
from app.detector.combined_detector import CombinedTextDetector
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter
from app.config import RAW_DIR, PROCESSED_DIR
from app.logging_config import logger
from app.manifest_utils import (
    assign_stable_detector_box_ids,
    bump_page_revision,
    capture_processing_state,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    is_processing_state_current,
    load_manifest_raw,
    new_box_id,
    save_manifest_raw,
)
from app.security import MAX_IMAGE_PIXELS, MAX_UPLOAD_FILE_BYTES, MAX_UPLOAD_FILES, MAX_UPLOAD_TOTAL_BYTES, validate_upload_image


def _advise_file_cache_drop(file_obj) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        os.posix_fadvise(file_obj.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError, ValueError):
        pass


def read_image(path: Path) -> np.ndarray:
    with path.open("rb") as source_file:
        data = np.fromfile(source_file, dtype=np.uint8)
        _advise_file_cache_drop(source_file)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    h, w = img.shape[:2]
    if w * h > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large at {path}: {w}x{h}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".png"
    success, buf = cv2.imencode(ext, image)
    if success:
        with path.open("wb") as output_file:
            buf.tofile(output_file)
            output_file.flush()
            _advise_file_cache_drop(output_file)
    else:
        cv2.imwrite(str(path), image)
        try:
            with path.open("rb") as output_file:
                _advise_file_cache_drop(output_file)
        except OSError:
            pass


def _encode_mask(mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    success, buf = cv2.imencode(".png", mask)
    if not success:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_mask(mask_b64: str | None) -> np.ndarray | None:
    if not mask_b64:
        return None
    try:
        raw = base64.b64decode(mask_b64)
    except (ValueError, TypeError):
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


_TEXT_OBJECT_STYLE_KEYS = {
    "color", "font", "fontSize", "bold",
    "strokeWidth", "strokeColor", "bgColor", "cornerRadius",
    "horizontalAlign", "verticalAlign",
}

DEFAULT_TEXT_OBJECT_STYLE = {
    "color": "auto",
    "font": "default",
    "fontSize": "auto",
    "bold": False,
    "strokeWidth": "auto",
    "strokeColor": "auto",
    "bgColor": "transparent",
    "cornerRadius": "0",
    "horizontalAlign": "center",
    "verticalAlign": "middle",
}


def _normalize_region(region: dict, w: int, h: int) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (
            int(region["x1"]), int(region["y1"]),
            int(region["x2"]), int(region["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid region payload")
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class ChapterPipeline:
    def __init__(self):
        self._detector = None
        self._inpainter = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = CombinedTextDetector()
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            self._inpainter = Inpainter()
        return self._inpainter

    def download_chapter(self, chapter_url: str, chapter_id: str, workers: int = 2) -> dict:
        raw_dir = RAW_DIR / chapter_id
        raw_paths = fetch_chapter_images(chapter_url, raw_dir)
        logger.info(f"Chapter {chapter_id}: downloaded {len(raw_paths)} raw images")
        return self._build_chapter_from_raw_paths(
            chapter_id, raw_paths, source_url=chapter_url, workers=workers
        )

    def create_chapter_from_uploads(
        self, chapter_id: str, uploads: list[tuple[str, bytes]], workers: int = 2
    ) -> dict:
        import io
        import re
        import zipfile

        def natural_sort_key(s: str):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

        raw_dir = RAW_DIR / chapter_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        extracted_files = []
        total_extracted_bytes = 0
        for filename, data in uploads:
            if len(extracted_files) >= MAX_UPLOAD_FILES or total_extracted_bytes >= MAX_UPLOAD_TOTAL_BYTES:
                if total_extracted_bytes >= MAX_UPLOAD_TOTAL_BYTES:
                    logger.warning(f"Vượt quá giới hạn tổng dung lượng {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB")
                else:
                    logger.warning(f"Vượt quá giới hạn {MAX_UPLOAD_FILES} files")
                break
            is_zip = filename.lower().endswith((".zip", ".cbz"))
            if not is_zip:
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        is_zip = True
                except Exception:
                    is_zip = False

            if is_zip:
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        namelist = sorted(z.namelist(), key=natural_sort_key)
                        for name in namelist:
                            if len(extracted_files) >= MAX_UPLOAD_FILES:
                                logger.warning(f"Đã đạt giới hạn {MAX_UPLOAD_FILES} ảnh từ ZIP")
                                break
                            if total_extracted_bytes >= MAX_UPLOAD_TOTAL_BYTES:
                                logger.warning(f"Đã đạt giới hạn tổng dung lượng {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB từ ZIP")
                                break
                            if name.startswith("__MACOSX/") or name.startswith("."):
                                continue
                            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                                continue
                            info = z.getinfo(name)
                            if info.file_size > MAX_UPLOAD_FILE_BYTES:
                                logger.warning(
                                    f"Skip {name}: giải nén vượt {MAX_UPLOAD_FILE_BYTES // (1024*1024)}MB"
                                )
                                continue
                            if total_extracted_bytes + info.file_size > MAX_UPLOAD_TOTAL_BYTES:
                                logger.warning(
                                    f"Skip {name}: tổng dung lượng vượt {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB"
                                )
                                break
                            safe_name = name.replace("\\", "/")
                            clean_name = Path(safe_name).name
                            if not clean_name:
                                continue
                            img_bytes = z.read(name)
                            if len(img_bytes) > 0:
                                extracted_files.append((clean_name, img_bytes))
                                total_extracted_bytes += len(img_bytes)
                except Exception as e:
                    logger.warning(f"Failed to extract zip file {filename}: {e}")
            else:
                if total_extracted_bytes + len(data) > MAX_UPLOAD_TOTAL_BYTES:
                    logger.warning(f"Skip {filename}: tổng dung lượng vượt {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB")
                    break
                extracted_files.append((filename, data))
                total_extracted_bytes += len(data)

        if not extracted_files:
            raise ValueError("Không tìm thấy file ảnh hợp lệ nào trong dữ liệu tải lên")

        raw_paths = []
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp"}
        for idx, (filename, data) in enumerate(extracted_files[:MAX_UPLOAD_FILES]):
            fmt = validate_upload_image(data, filename)
            ext = ext_map.get(fmt, ".png")
            out_path = raw_dir / f"{idx:03d}{ext}"
            out_path.write_bytes(data)
            raw_paths.append(out_path)

        logger.info(f"Chapter {chapter_id}: saved {len(raw_paths)} uploaded images")
        return self._build_chapter_from_raw_paths(
            chapter_id, raw_paths, source_url=None, workers=workers
        )

    def _build_chapter_from_raw_paths(
        self,
        chapter_id: str,
        raw_paths: list[Path],
        source_url: str | None,
        workers: int = 2,
    ) -> dict:
        sliced_dir = RAW_DIR / chapter_id / "sliced"
        processed_dir = PROCESSED_DIR / chapter_id
        sliced_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        slice_results: dict[int, list] = {}
        if raw_paths:
            max_workers = max(1, min(int(workers or 2), 8, len(raw_paths)))

            def _slice_one(item):
                idx, raw_path = item
                return idx, slice_image(raw_path, sliced_dir, f"{idx:03d}", return_metadata=True)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_slice_one, (i, p)): i for i, p in enumerate(raw_paths)
                }
                for future in as_completed(futures):
                    idx, slice_paths = future.result()
                    slice_results[idx] = slice_paths

        pages = []
        for source_index in range(len(raw_paths)):
            for slice_index, slice_item in enumerate(slice_results[source_index]):
                if isinstance(slice_item, dict):
                    slice_path = Path(slice_item["path"])
                    stitch_core = {
                        k: slice_item.get(k) for k in (
                            "source_y1", "source_y2", "core_y1", "core_y2",
                            "core_source_y1", "core_source_y2", "unsafe_before",
                            "unsafe_after", "source_height"
                        )
                    }
                else:
                    slice_path = Path(slice_item)
                    stitch_core = None
                page = {
                    "original": slice_path.as_posix(),
                    "clean": None,
                    "boxes": [],
                    "skipped": False,
                    "excluded_regions": [],
                    "source_page": source_index,
                    "slice_index": slice_index,
                }
                if stitch_core is not None:
                    page["stitch_core"] = stitch_core
                pages.append(page)

        manifest = {
            "chapter_id": chapter_id,
            "source_url": source_url,
            "pages": pages,
            "workflow": {"stage": "preview", "page_index": 0},
        }
        with get_manifest_lock(chapter_id):
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def process_pages(self, chapter_id: str, page_indices: list[int], workers: int = 2) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            work_items = []
            for idx in page_indices:
                if 0 <= idx < len(manifest.get("pages", [])):
                    page = manifest["pages"][idx]
                    if not page.get("skipped", False):
                        state_snapshot = capture_processing_state(manifest, idx, processed_dir)
                        work_items.append((
                            idx,
                            Path(page["original"]),
                            copy.deepcopy(page.get("excluded_regions", [])),
                            copy.deepcopy(page.get("boxes", [])),
                            copy.deepcopy(page.get("stitch_core")),
                            state_snapshot,
                        ))

        if not work_items:
            return manifest

        _ = self.detector
        _ = self.inpainter

        max_workers = max(1, min(int(workers or 2), 8, len(work_items)))
        results: dict[int, tuple[dict, dict | None]] = {}
        errors: list[tuple[int, Exception]] = []

        # If only one page is active, use the otherwise-idle CPU cores to run
        # bubble/text detection concurrently. With multiple page workers the
        # outer executor already supplies model-level concurrency, so keeping
        # each page sequential avoids oversubscribing CPU threads.
        parallel_detectors = max_workers == 1

        def _process_one(item) -> tuple[int, dict, dict | None]:
            idx, img_path, excluded, existing_boxes, stitch_core, snapshot = item
            return idx, self._process_page(
                img_path,
                processed_dir,
                excluded_regions=excluded,
                existing_boxes=existing_boxes,
                stitch_core=stitch_core,
                parallel_detectors=parallel_detectors,
            ), snapshot

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_one, item): item[0] for item in work_items
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    page_idx, page_data, snapshot = future.result()
                    results[page_idx] = (page_data, snapshot)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Chapter {} page {} operation 'process_page' failed: {}",
                        chapter_id,
                        idx,
                        exc,
                    )
                    errors.append((idx, exc))

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            committed_indices = []
            for idx, (page_data, snapshot) in results.items():
                tmp_clean_posix = page_data.get("tmp_clean")
                tmp_clean_path = Path(tmp_clean_posix) if tmp_clean_posix else None
                tmp_auto_clean_posix = page_data.get("tmp_auto_clean")
                tmp_auto_clean_path = Path(tmp_auto_clean_posix) if tmp_auto_clean_posix else None
                try:
                    if 0 <= idx < len(manifest.get("pages", [])) and is_processing_state_current(
                        manifest, idx, snapshot, processed_dir
                    ):
                        if tmp_clean_path and tmp_clean_path.exists():
                            orig_path = Path(manifest["pages"][idx]["original"])
                            final_clean_path = processed_dir / f"clean_{orig_path.name}"
                            os.replace(tmp_clean_path, final_clean_path)
                            manifest["pages"][idx]["clean"] = final_clean_path.as_posix()

                        if tmp_auto_clean_path and tmp_auto_clean_path.exists():
                            orig_path = Path(manifest["pages"][idx]["original"])
                            os.replace(tmp_auto_clean_path, self._auto_clean_path(processed_dir, orig_path))

                        target_page = manifest["pages"][idx]
                        existing_boxes = target_page.get("boxes", [])
                        detected_boxes = assign_stable_detector_box_ids(page_data["boxes"], existing_boxes)
                        manual_boxes = [b for b in existing_boxes if b.get("manual")]
                        target_page["boxes"] = detected_boxes + manual_boxes
                        target_page["detection_state"] = page_data.get("detection_state", "verified")
                        target_page["detection_issues"] = list(page_data.get("detection_issues") or [])
                        target_page["unverified_regions"] = list(page_data.get("unverified_regions") or [])
                        target_page["needs_review"] = bool(page_data.get("needs_review"))
                        bump_page_revision(target_page, "process_revision")
                        bump_page_revision(target_page, "clean_revision")

                        if "manual_mask" in page_data:
                            manifest["pages"][idx]["manual_mask"] = page_data["manual_mask"]

                        invalidate_page_render(manifest, idx)
                        committed_indices.append(idx)
                    else:
                        logger.warning(
                            "Chapter %s page %s: page state changed during processing, discarding stale output",
                            chapter_id,
                            idx,
                        )
                finally:
                    for tmp_path in (tmp_clean_path, tmp_auto_clean_path):
                        if tmp_path and tmp_path.exists():
                            try:
                                tmp_path.unlink()
                            except OSError:
                                pass

            if committed_indices:
                manifest["workflow"] = {
                    "stage": "review",
                    "page_index": min(committed_indices),
                }
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, committed_indices)

        if errors:
            failed_indices = [e[0] for e in errors]
            first_exc = errors[0][1]
            if not results:
                raise RuntimeError(
                    f"Chapter {chapter_id}: All requested pages failed to process. First error (page {failed_indices[0]}): {first_exc}"
                ) from first_exc
            else:
                raise RuntimeError(
                    f"Chapter {chapter_id}: Failed to process {len(errors)} page(s) (indices: {failed_indices}). First error: {first_exc}"
                ) from first_exc

        return manifest

    def mark_skipped(self, chapter_id: str, page_indices: list[int], skipped: bool) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            for idx in page_indices:
                if 0 <= idx < len(manifest["pages"]):
                    page = manifest["pages"][idx]
                    changed = bool(page.get("skipped", False)) != bool(skipped)
                    page["skipped"] = skipped
                    if skipped:
                        if page.get("clean") != page.get("original") or page.get("boxes"):
                            changed = True
                        page["clean"] = page["original"]
                        page["boxes"] = []
                    if changed:
                        bump_page_revision(page, "clean_revision")
                    invalidate_page_render(manifest, idx)
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, page_indices)
        return manifest

    @staticmethod
    def _sync_output_dir(chapter_id: str, manifest: dict, page_indices: list[int] | None = None) -> None:
        from app.config import OUTPUT_DIR
        out_dir = OUTPUT_DIR / chapter_id
        out_dir.mkdir(parents=True, exist_ok=True)

        indices = page_indices if page_indices is not None else range(len(manifest["pages"]))

        for i in indices:
            if i < 0 or i >= len(manifest["pages"]):
                continue
            page = manifest["pages"][i]
            if page.get("rendered"):
                continue

            # OUTPUT_DIR contains committed translation renders only. Before text
            # rendering, /api/image and /api/download already fall back to the
            # canonical clean/original image, so copying clean PNGs here only
            # duplicates disk usage and can leave stale output artifacts behind.
            target_path = out_dir / f"page_{i:03d}.png"
            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove stale output %s: %s", target_path, exc)

    def add_manual_box(self, chapter_id: str, page_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                page = manifest["pages"][page_index]
                img_path = Path(page["original"])

            image = read_image(img_path)
            h, w = image.shape[:2]

            nx1, nx2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
            ny1, ny2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
            if nx2 <= nx1 or ny2 <= ny1:
                return manifest

            new_box = {
                "id": new_box_id(), "origin": "manual",
                "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                "confidence": 1.0, "mask": None, "manual": True,
            }
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                boxes_snapshot = copy.deepcopy(target_page.get("boxes", []))
                boxes_snapshot.append(copy.deepcopy(new_box))
                manual_mask_posix = target_page.get("manual_mask")

            clean_path_posix = self._do_reinpaint(
                processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
            )

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_page.setdefault("boxes", []).append(new_box)
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def update_box(self, chapter_id: str, page_index: int, box_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                page = manifest["pages"][page_index]
                boxes = page.get("boxes", [])
                if box_index < 0 or box_index >= len(boxes):
                    raise ValueError(f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}")
                img_path = Path(page["original"])

            image = read_image(img_path)
            height, width = image.shape[:2]
            nx1, nx2 = sorted((int(x1), int(x2)))
            ny1, ny2 = sorted((int(y1), int(y2)))
            if nx1 < 0 or ny1 < 0 or nx2 > width or ny2 > height or (nx2 - nx1) < 10 or (ny2 - ny1) < 10:
                raise ValueError(f"Chapter {chapter_id} page {page_index}: Box coordinates out of bounds or smaller than 10px")

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError(f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}")

                boxes_snapshot = copy.deepcopy(target_boxes)
                snapshot_box = boxes_snapshot[box_index]
                if snapshot_box.get("origin") == "detector" and not snapshot_box.get("manual"):
                    if not snapshot_box.get("geometry_overridden"):
                        snapshot_box["detector_anchor"] = {k: snapshot_box.get(k) for k in ("x1", "y1", "x2", "y2")}
                    snapshot_box["geometry_overridden"] = True
                snapshot_box.update({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2, "mask": None})
                manual_mask_posix = target_page.get("manual_mask")

            clean_path_posix = self._do_reinpaint(
                processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
            )

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError(f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}")
                target_box = target_boxes[box_index]
                if target_box.get("origin") == "detector" and not target_box.get("manual"):
                    if not target_box.get("geometry_overridden"):
                        target_box["detector_anchor"] = {k: target_box.get(k) for k in ("x1", "y1", "x2", "y2")}
                    target_box["geometry_overridden"] = True
                target_box.update({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2, "mask": None})
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def remove_box(self, chapter_id: str, page_index: int, box_index: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                boxes = page.get("boxes", [])
                if box_index < 0 or box_index >= len(boxes):
                    raise ValueError("Invalid box index")

                img_path = Path(page["original"])
                manual_mask_posix = page.get("manual_mask")
                boxes_snapshot = copy.deepcopy(boxes)
                boxes_snapshot[box_index]["removed"] = True

            image = read_image(img_path)
            clean_path_posix = self._do_reinpaint(
                processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
            )

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError("Invalid box index")

                target_boxes[box_index]["removed"] = True
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def repaint_mask(self, chapter_id: str, page_index: int, mask: np.ndarray) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                page = manifest["pages"][page_index]
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                manual_mask_posix = page.get("manual_mask")

            image = read_image(img_path)
            img_h, img_w = image.shape[:2]

            # If this is the first manual/Gemini repaint, the existing clean image
            # is already the automatic inpaint baseline. Seed the cache by copying
            # it instead of running all automatic LaMa regions again.
            auto_clean_path = self._auto_clean_path(processed_dir, img_path)
            if not auto_clean_path.exists() and not manual_mask_posix:
                clean_posix = page.get("clean")
                clean_path = Path(clean_posix) if clean_posix else None
                if clean_path is not None and clean_path.exists():
                    try:
                        cached = read_image(clean_path)
                        if cached.shape[:2] == (img_h, img_w):
                            self._write_auto_clean_cache(processed_dir, img_path, cached)
                    except Exception as exc:
                        logger.warning("Could not seed auto-clean cache from %s: %s", clean_path, exc)

            bin_mask = (mask > 10).astype(np.uint8) * 255 if mask is not None else None
            if bin_mask is not None and bin_mask.shape[:2] != (img_h, img_w):
                bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                bin_mask = (bin_mask > 10).astype(np.uint8) * 255

            manual_mask_path = Path(manual_mask_posix) if manual_mask_posix else processed_dir / f"manual_mask_{img_path.name}"
            existing_mask = None
            if manual_mask_path.exists():
                try:
                    raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
                    decoded = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
                    if decoded is not None and np.any(decoded > 10):
                        if decoded.shape[:2] != (img_h, img_w):
                            decoded = cv2.resize(decoded, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        existing_mask = (decoded > 10).astype(np.uint8) * 255
                except Exception as exc:
                    logger.warning("Could not read existing manual mask at %s: %s", manual_mask_path, exc)

            if existing_mask is not None and bin_mask is not None:
                accumulated_mask = np.maximum(existing_mask, bin_mask)
            elif bin_mask is not None:
                accumulated_mask = bin_mask
            elif existing_mask is not None:
                accumulated_mask = existing_mask
            else:
                accumulated_mask = None

            tmp_mask_path = processed_dir / f"manual_mask_{img_path.name}.{uuid.uuid4().hex}.tmp.png"
            if accumulated_mask is not None:
                write_image(tmp_mask_path, accumulated_mask)
                inpaint_mask_posix = tmp_mask_path.as_posix()
            else:
                inpaint_mask_posix = None

            try:
                clean_path_posix = self._do_reinpaint(
                    processed_dir, img_path, image, boxes_snapshot, inpaint_mask_posix,
                    reuse_auto_clean=True,
                )
                if accumulated_mask is not None:
                    final_mask_path = processed_dir / f"manual_mask_{img_path.name}"
                    os.replace(tmp_mask_path, final_mask_path)
                    manual_mask_posix = final_mask_path.as_posix()
                else:
                    manual_mask_posix = None
            finally:
                if tmp_mask_path.exists():
                    try:
                        tmp_mask_path.unlink()
                    except OSError:
                        pass

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                target_page = manifest["pages"][page_index]
                if manual_mask_posix:
                    target_page["manual_mask"] = manual_mask_posix
                else:
                    target_page.pop("manual_mask", None)
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def reset_manual_mask(self, chapter_id: str, page_index: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))

            manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"

            image = read_image(img_path)
            clean_path_posix = self._do_reinpaint(
                processed_dir, img_path, image, boxes_snapshot, manual_mask_posix=None,
                reuse_auto_clean=True,
                apply_manual_mask=False,
            )

            if manual_mask_path.exists():
                try:
                    manual_mask_path.unlink()
                except OSError as exc:
                    raise RuntimeError(f"Cannot remove manual mask: {exc}") from exc

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                target_page = manifest["pages"][page_index]
                target_page.pop("manual_mask", None)
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def create_text_object(
        self, chapter_id: str, page_index: int, shape: str, region: dict
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            img_path = Path(pages[page_index]["original"])

        image = read_image(img_path)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = _normalize_region(region, w, h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            raise ValueError(f"Chapter {chapter_id}: Region too small (min 10px)")

        obj = {
            "id": uuid.uuid4().hex,
            "shape": shape,
            "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "source_boxes": [],
            "ocr_text": "",
            "translation": "",
            "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
        }

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            target.setdefault("text_objects", []).append(obj)
            target.setdefault("width", w)
            target.setdefault("height", h)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def update_text_object(
        self, chapter_id: str, page_index: int, text_object_id: str, changes: dict
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            objs = target.setdefault("text_objects", [])
            obj = next((o for o in objs if o.get("id") == text_object_id), None)
            if obj is None:
                raise ValueError(f"Chapter {chapter_id}: Text object not found {text_object_id!r}")

            if "shape" in changes and changes["shape"] is not None:
                if changes["shape"] not in ("rectangle", "ellipse"):
                    raise ValueError("Invalid shape")
                obj["shape"] = changes["shape"]

            if "region" in changes and changes["region"] is not None:
                region = changes["region"]
                try:
                    x1, y1, x2, y2 = (
                        int(region["x1"]), int(region["y1"]),
                        int(region["x2"]), int(region["y2"]),
                    )
                except (KeyError, TypeError, ValueError):
                    raise ValueError("Invalid region payload")
                if x1 < 0 or y1 < 0:
                    raise ValueError("Region coordinates must be non-negative")
                if x1 >= x2 or y1 >= y2:
                    raise ValueError("Region must have x1 < x2 and y1 < y2")
                if target.get("width") is not None and target.get("height") is not None:
                    pw, ph = int(target["width"]), int(target["height"])
                    x1 = max(0, min(x1, pw))
                    x2 = max(0, min(x2, pw))
                    y1 = max(0, min(y1, ph))
                    y2 = max(0, min(y2, ph))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    raise ValueError("Region too small (min 10px)")
                obj["region"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            if "ocr_text" in changes:
                obj["ocr_text"] = changes["ocr_text"] or ""
            if "translation" in changes:
                obj["translation"] = changes["translation"] or ""
            if "style" in changes and changes["style"] is not None:
                style = changes["style"]
                if not isinstance(style, dict):
                    raise ValueError("Invalid style payload")
                cleaned: dict = {}
                for k, v in style.items():
                    if k not in _TEXT_OBJECT_STYLE_KEYS:
                        raise ValueError(f"Invalid style key {k!r}")
                    if k == "bold":
                        cleaned[k] = bool(v)
                    elif k == "horizontalAlign":
                        if v not in ("left", "center", "right"):
                            raise ValueError("Invalid horizontalAlign value")
                        cleaned[k] = v
                    elif k == "verticalAlign":
                        if v not in ("top", "middle", "bottom"):
                            raise ValueError("Invalid verticalAlign value")
                        cleaned[k] = v
                    else:
                        cleaned[k] = v
                merged = dict(DEFAULT_TEXT_OBJECT_STYLE)
                merged.update(cleaned)
                obj["style"] = merged

            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def delete_text_object(
        self, chapter_id: str, page_index: int, text_object_id: str
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            objs = target.get("text_objects") or []
            idx = next((i for i, o in enumerate(objs) if o.get("id") == text_object_id), None)
            if idx is None:
                raise ValueError(f"Chapter {chapter_id}: Text object not found {text_object_id!r}")
            del objs[idx]
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    @staticmethod
    def _auto_clean_path(processed_dir: Path, img_path: Path) -> Path:
        return processed_dir / f"auto_clean_{img_path.name}"

    def _write_auto_clean_cache(self, processed_dir: Path, img_path: Path, image: np.ndarray) -> Path:
        auto_path = self._auto_clean_path(processed_dir, img_path)
        tmp_path = processed_dir / f"auto_clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_image(tmp_path, image)
        os.replace(tmp_path, auto_path)
        return auto_path

    def _load_auto_clean_cache(self, processed_dir: Path, img_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
        auto_path = self._auto_clean_path(processed_dir, img_path)
        if not auto_path.exists():
            return None
        try:
            cached = read_image(auto_path)
        except Exception as exc:
            logger.warning("Could not read auto-clean cache at %s: %s", auto_path, exc)
            return None
        if cached.shape[:2] != expected_shape:
            logger.warning(
                "Ignoring stale auto-clean cache at %s: expected shape %s, got %s",
                auto_path, expected_shape, cached.shape[:2],
            )
            return None
        return cached

    def _do_reinpaint(
        self,
        processed_dir: Path,
        img_path: Path,
        image: np.ndarray,
        boxes: list[dict],
        manual_mask_posix: str | None = None,
        *,
        reuse_auto_clean: bool = False,
        apply_manual_mask: bool = True,
    ) -> str:
        boxes_objects = []
        for b in boxes:
            if b.get("removed"):
                continue
            box_h = b["y2"] - b["y1"]
            box_w = b["x2"] - b["x1"]
            mask_arr = _decode_mask(b.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(mask_arr, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_arr = None
            boxes_objects.append(
                BubbleBox(
                    b["x1"], b["y1"], b["x2"], b["y2"], b.get("confidence", 1.0),
                    mask_arr,
                )
            )

        clean_image = None
        if reuse_auto_clean:
            clean_image = self._load_auto_clean_cache(processed_dir, img_path, image.shape[:2])

        if clean_image is None:
            clean_image = self.inpainter.inpaint(image, boxes_objects)
            self._write_auto_clean_cache(processed_dir, img_path, clean_image)
        else:
            clean_image = clean_image.copy()

        manual_mask_path = Path(manual_mask_posix) if manual_mask_posix else processed_dir / f"manual_mask_{img_path.name}"
        if apply_manual_mask and manual_mask_path.exists():
            raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
            manual_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if manual_mask is not None and np.any(manual_mask > 10):
                h, w = clean_image.shape[:2]
                if manual_mask.shape[:2] != (h, w):
                    try:
                        manual_mask = cv2.resize(manual_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        manual_mask = None
                if manual_mask is not None and np.any(manual_mask > 10):
                    manual_mask = (manual_mask > 10).astype(np.uint8) * 255
                    clean_image = self.inpainter.inpaint_mask(clean_image, manual_mask)

        clean_path = processed_dir / f"clean_{img_path.name}"
        tmp_clean_path = processed_dir / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_image(tmp_clean_path, clean_image)
        os.replace(tmp_clean_path, clean_path)
        return clean_path.as_posix()

    def _process_page(
        self,
        img_path: Path,
        processed_dir: Path,
        excluded_regions: list[dict] | None = None,
        existing_boxes: list[dict] | None = None,
        stitch_core: dict | None = None,
        *,
        parallel_detectors: bool = False,
    ) -> dict:
        image = read_image(img_path)
        detected = self.detector.detect(image, parallel=parallel_detectors)
        if excluded_regions:
            detected = [b for b in detected if not self._box_in_excluded(b, excluded_regions)]

        existing_boxes = copy.deepcopy(existing_boxes or [])
        detector_records = [
            {
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "confidence": b.confidence, "_mask_array": b.mask,
                "source_model": b.source_model, "class_id": b.class_id,
                "class_name": b.class_name, "semantic_type": b.semantic_type,
                "mask_source": b.mask_source, "safe_to_inpaint": bool(b.safe_to_inpaint),
                "ocr_eligible": bool(b.ocr_eligible), "needs_review": bool(b.needs_review),
            }
            for b in detected
        ]
        if isinstance(stitch_core, dict):
            try:
                core_y1 = int(stitch_core.get("core_y1", 0))
                core_y2 = int(stitch_core.get("core_y2", image.shape[0]))
            except (TypeError, ValueError):
                core_y1, core_y2 = 0, image.shape[0]
            for record in detector_records:
                cy = (int(record["y1"]) + int(record["y2"])) / 2.0
                if cy < core_y1 or cy >= core_y2:
                    record["overlap_context_only"] = True
                    record["ocr_eligible"] = False

        assign_stable_detector_box_ids(detector_records, existing_boxes)
        old_by_id = {str(b.get("id")): b for b in existing_boxes if isinstance(b, dict) and b.get("id")}

        effective_boxes: list[BubbleBox] = []
        for record in detector_records:
            old = old_by_id.get(str(record.get("id")))
            if old is not None:
                if old.get("removed"):
                    record["removed"] = True
                if old.get("geometry_overridden"):
                    record["geometry_overridden"] = True
                    if isinstance(old.get("detector_anchor"), dict):
                        record["detector_anchor"] = copy.deepcopy(old["detector_anchor"])
                    for key in ("x1", "y1", "x2", "y2"):
                        record[key] = int(old[key])
                    record["_mask_array"] = None
                    record["safe_to_inpaint"] = False
                    record["ocr_eligible"] = False
                    record["needs_review"] = True
            if not record.get("removed") and (record.get("safe_to_inpaint") or record.get("geometry_overridden")):
                _effective = BubbleBox(
                    int(record["x1"]), int(record["y1"]), int(record["x2"]), int(record["y2"]),
                    float(record.get("confidence", 1.0)), record.get("_mask_array"),
                    source_model=str(record.get("source_model") or "unknown"),
                    class_id=int(record.get("class_id") or 0),
                    class_name=str(record.get("class_name") or "unknown"),
                    semantic_type=str(record.get("semantic_type") or "unknown"),
                    mask_source=str(record.get("mask_source") or "none"),
                    safe_to_inpaint=True, ocr_eligible=bool(record.get("ocr_eligible")),
                    needs_review=bool(record.get("needs_review")),
                )
                if record.get("geometry_overridden"):
                    _effective.allow_rectangle_fallback = True
                effective_boxes.append(_effective)

        for old in existing_boxes:
            if not isinstance(old, dict) or not old.get("manual") or old.get("removed"):
                continue
            box_h = int(old.get("y2", 0)) - int(old.get("y1", 0))
            box_w = int(old.get("x2", 0)) - int(old.get("x1", 0))
            if box_h <= 0 or box_w <= 0:
                continue
            mask_arr = _decode_mask(old.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(mask_arr, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_arr = None
            effective_boxes.append(BubbleBox(
                int(old["x1"]), int(old["y1"]), int(old["x2"]), int(old["y2"]),
                float(old.get("confidence", 1.0)), mask_arr,
            ))

        clean_image = self.inpainter.inpaint(image, effective_boxes)

        # Never mutate the canonical auto-clean cache inside a worker. The page
        # may change while detection/inpaint is running; write a job-local cache
        # candidate and only promote it after the processing-state check passes.
        auto_clean_path = self._auto_clean_path(processed_dir, img_path)
        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
        tmp_auto_clean_path = None
        if auto_clean_path.exists() or manual_mask_path.exists():
            tmp_auto_clean_path = processed_dir / f"auto_clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
            write_image(tmp_auto_clean_path, clean_image)

        manual_mask_posix = None
        if manual_mask_path.exists():
            raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
            manual_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if manual_mask is not None and np.any(manual_mask > 10):
                h, w = clean_image.shape[:2]
                if manual_mask.shape[:2] != (h, w):
                    try:
                        manual_mask = cv2.resize(manual_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        manual_mask = None
                if manual_mask is not None and np.any(manual_mask > 10):
                    manual_mask = (manual_mask > 10).astype(np.uint8) * 255
                    clean_image = self.inpainter.inpaint_mask(clean_image.copy(), manual_mask)
                    manual_mask_posix = manual_mask_path.as_posix()

        tmp_clean_path = processed_dir / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_image(tmp_clean_path, clean_image)

        unverified_regions = [
            {k: record.get(k) for k in ("x1", "y1", "x2", "y2", "confidence", "source_model", "class_name", "semantic_type")}
            for record in detector_records
            if record.get("needs_review") or not record.get("safe_to_inpaint")
        ]
        detection_issues = []
        if unverified_regions:
            detection_issues.append("unverified_regions")
        if not detector_records:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) > 24.0:
                detection_issues.append("content_heavy_zero_box")
        detection_state = "needs_review" if detection_issues else "verified"

        res = {
            "tmp_clean": tmp_clean_path.as_posix(),
            "boxes": [
                {
                    **{k: v for k, v in record.items() if k != "_mask_array"},
                    "mask": _encode_mask(record.get("_mask_array")),
                }
                for record in detector_records
            ],
            "detection_state": detection_state,
            "detection_issues": detection_issues,
            "unverified_regions": unverified_regions,
            "needs_review": bool(detection_issues),
        }
        if tmp_auto_clean_path is not None:
            res["tmp_auto_clean"] = tmp_auto_clean_path.as_posix()
        if manual_mask_posix:
            res["manual_mask"] = manual_mask_posix
        return res

    @staticmethod
    def _box_in_excluded(box, excluded_regions: list[dict]) -> bool:
        box_cx = (box.x1 + box.x2) / 2
        box_cy = (box.y1 + box.y2) / 2
        for r in excluded_regions:
            x1 = r.get("x1", 0)
            y1 = r.get("y1", 0)
            x2 = r.get("x2", 0)
            y2 = r.get("y2", 0)
            if min(x1, x2) <= box_cx <= max(x1, x2) and min(y1, y2) <= box_cy <= max(y1, y2):
                return True
        return False
