import os
import base64
import copy
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from app.downloader.registry import download_chapter as fetch_chapter_images
from app.downloader.slicer import OVERLAP_CONTEXT, slice_image
from app.detector.combined_detector import CombinedTextDetector
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.mask_geometry import geometry_dict, remap_local_mask_page_space
from app.config import RAW_DIR, PROCESSED_DIR
from app.logging_config import logger
from app.mask_store import decode_mask_value
from app.parameters import (
    DETECTION_CONTENT_STD_MIN,
    DETECTOR_FINAL_NMS_IOU,
    MANUAL_MASK_THRESHOLD,
    PIPELINE_DEFAULT_WORKERS,
    PIPELINE_PROCESS_WORKER_LIMIT,
    PIPELINE_SLICE_WORKER_LIMIT,
)
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
    PageArtifactTransaction,
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
        if not cv2.imwrite(str(path), image):
            raise ValueError(f"Could not write image at {path}")
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
        self._detector_init_lock = threading.Lock()
        self._inpainter_init_lock = threading.Lock()

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = CombinedTextDetector()
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            with self._inpainter_init_lock:
                if self._inpainter is None:
                    self._inpainter = Inpainter()
        return self._inpainter

    @staticmethod
    def _apply_box_geometry(box: dict, new_geometry: dict[str, int]) -> None:
        """Geometry-safe edit that understands legacy and sidecar detector masks."""
        source_geometry = geometry_dict(box)
        source_mask = decode_mask_value(box.get("mask"))
        detector_origin = box.get("origin") == "detector" and not box.get("manual")

        if detector_origin:
            if not box.get("geometry_overridden"):
                box["detector_anchor"] = copy.deepcopy(source_geometry)
            box["geometry_overridden"] = True

        remapped = (
            remap_local_mask_page_space(source_mask, source_geometry, new_geometry)
            if source_mask is not None
            else None
        )

        box.update(new_geometry)
        if remapped is None or not np.any(remapped > 127):
            box["mask"] = None
        else:
            box["mask"] = _encode_mask(remapped)

    def download_chapter(
        self,
        chapter_url: str,
        chapter_id: str,
        workers: int = PIPELINE_DEFAULT_WORKERS,
    ) -> dict:
        raw_dir = RAW_DIR / chapter_id
        raw_paths = fetch_chapter_images(chapter_url, raw_dir)
        logger.info(f"Chapter {chapter_id}: downloaded {len(raw_paths)} raw images")
        return self._build_chapter_from_raw_paths(
            chapter_id, raw_paths, source_url=chapter_url, workers=workers
        )

    def create_chapter_from_uploads(
        self,
        chapter_id: str,
        uploads: list[tuple[str, bytes]],
        workers: int = PIPELINE_DEFAULT_WORKERS,
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
        workers: int = PIPELINE_DEFAULT_WORKERS,
    ) -> dict:
        sliced_dir = RAW_DIR / chapter_id / "sliced"
        processed_dir = PROCESSED_DIR / chapter_id
        sliced_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        slice_results: dict[int, list] = {}
        if raw_paths:
            max_workers = max(
                1,
                min(
                    int(workers or PIPELINE_DEFAULT_WORKERS),
                    PIPELINE_SLICE_WORKER_LIMIT,
                    len(raw_paths),
                ),
            )

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

    @staticmethod
    def _map_seam_detection_to_page(
        box: BubbleBox,
        *,
        seam_source_y1: int,
        page_source_y1: int,
        page_height: int,
    ) -> BubbleBox | None:
        """Translate one seam-strip detection into an overlapping slice.

        Physical slices keep their existing overlap so OCR/review/render geometry
        remains backward compatible. The detector strip is shared, however, so
        the same +/-384px seam context is not inferred twice by adjacent pages.
        """
        source_y1 = int(seam_source_y1) + int(box.y1)
        source_y2 = int(seam_source_y1) + int(box.y2)
        page_source_y2 = int(page_source_y1) + int(page_height)
        clipped_source_y1 = max(source_y1, int(page_source_y1))
        clipped_source_y2 = min(source_y2, page_source_y2)
        if clipped_source_y2 <= clipped_source_y1:
            return None

        clip_top = clipped_source_y1 - source_y1
        clip_bottom = source_y2 - clipped_source_y2
        clipped_mask = box.mask
        if clipped_mask is not None:
            mask_h = clipped_mask.shape[0]
            mask_end = mask_h - clip_bottom if clip_bottom else mask_h
            clipped_mask = clipped_mask[clip_top:mask_end, :].copy()
            if clipped_mask.shape[0] != clipped_source_y2 - clipped_source_y1:
                return None

        if box.safe_to_inpaint and (
            clipped_mask is None or not np.any(clipped_mask > 0)
        ):
            return None

        return replace(
            box,
            y1=clipped_source_y1 - int(page_source_y1),
            y2=clipped_source_y2 - int(page_source_y1),
            mask=clipped_mask,
        )

    @staticmethod
    def _shift_detection_y(box: BubbleBox, offset_y: int) -> BubbleBox:
        return replace(box, y1=box.y1 + offset_y, y2=box.y2 + offset_y)

    def _shared_seam_detections(
        self, chapter_id: str, work_items: list[tuple]
    ) -> tuple[dict[int, list[BubbleBox]], set[int], dict[str, float | int]]:
        """Detect each unsafe overlap seam once, then share it across pages.

        The slicer deliberately stores +/-384px overlap around unsafe cuts. That
        protects bubbles crossing a cut, but letting each physical slice run its
        full detector duplicates the same context and can push a 1400px core to
        1800-2300px, triggering multiple 1024px detector windows. Instead, each
        page detects only its non-overlapping core while every unsafe seam gets
        one 768px detector pass shared by all requested adjacent pages.

        The physical slice files and their coordinates are unchanged, preserving
        OCR/review/render behavior. If required seam context cannot be read or
        detected, affected pages fail safe to ``needs_review``.
        """
        consumers: dict[tuple[int, int], list[tuple[int, Path, dict]]] = {}
        unavailable_pages: set[int] = set()
        shared_metrics: dict[str, float | int] = {
            "attempts": 0,
            "failures": 0,
            "wall_ms": 0.0,
            "bubble_model_ms": 0.0,
            "text_model_ms": 0.0,
            "mser_ms": 0.0,
            "detector_total_ms": 0.0,
        }

        for item in work_items:
            (
                idx, img_path, _excluded, _existing_boxes, stitch_core,
                _snapshot,
            ) = item
            if not isinstance(stitch_core, dict):
                continue
            try:
                source_page = int(stitch_core.get("source_page", -1))
            except (TypeError, ValueError):
                source_page = -1
            if source_page < 0:
                try:
                    source_page = int(stitch_core["_source_page"])
                except (KeyError, TypeError, ValueError):
                    source_page = -1
            try:
                core_source_y1 = int(stitch_core["core_source_y1"])
                core_source_y2 = int(stitch_core["core_source_y2"])
                source_height = int(stitch_core["source_height"])
                source_y1 = int(stitch_core["source_y1"])
                source_y2 = int(stitch_core["source_y2"])
                core_y1 = int(stitch_core["core_y1"])
                core_y2 = int(stitch_core["core_y2"])
            except (KeyError, TypeError, ValueError):
                if stitch_core.get("unsafe_before") or stitch_core.get("unsafe_after"):
                    unavailable_pages.add(idx)
                continue
            if source_page < 0 or source_y2 <= source_y1 or core_y2 <= core_y1:
                if stitch_core.get("unsafe_before") or stitch_core.get("unsafe_after"):
                    unavailable_pages.add(idx)
                continue

            if bool(stitch_core.get("unsafe_before")) and core_source_y1 > 0:
                consumers.setdefault((source_page, core_source_y1), []).append(
                    (idx, img_path, stitch_core)
                )
            if bool(stitch_core.get("unsafe_after")) and core_source_y2 < source_height:
                consumers.setdefault((source_page, core_source_y2), []).append(
                    (idx, img_path, stitch_core)
                )

        if not consumers:
            return {}, unavailable_pages, shared_metrics

        by_page: dict[int, list[BubbleBox]] = {}
        for (source_page, cut_y), seam_consumers in sorted(consumers.items()):
            provider_idx, provider_path, provider_core = seam_consumers[0]
            seam_started_at = time.perf_counter()
            detector_metrics: dict[str, float | int] = {}
            try:
                provider_image = read_image(provider_path)
                provider_source_y1 = int(provider_core["source_y1"])
                provider_h = provider_image.shape[0]
                seam_center = int(cut_y) - provider_source_y1
                seam_local_y1 = max(0, seam_center - OVERLAP_CONTEXT)
                seam_local_y2 = min(provider_h, seam_center + OVERLAP_CONTEXT)
                if seam_local_y2 <= seam_local_y1:
                    raise ValueError("empty seam detector strip")
                seam_image = provider_image[seam_local_y1:seam_local_y2, :]
                seam_source_y1 = provider_source_y1 + seam_local_y1
                seam_boxes = self.detector.detect(seam_image, parallel=True)
                detector_metrics = self.detector.last_metrics()
            except Exception as exc:
                shared_metrics["failures"] = int(shared_metrics["failures"]) + 1
                logger.warning(
                    "Chapter {} source page {} seam {}: shared context detection failed: {}",
                    chapter_id, source_page, cut_y, exc,
                )
                unavailable_pages.update(idx for idx, _path, _core in seam_consumers)
                continue
            finally:
                shared_metrics["attempts"] = int(shared_metrics["attempts"]) + 1
                shared_metrics["wall_ms"] = float(shared_metrics["wall_ms"]) + (
                    (time.perf_counter() - seam_started_at) * 1000.0
                )
                for source_name, target_name in (
                    ("bubble_model_ms", "bubble_model_ms"),
                    ("text_model_ms", "text_model_ms"),
                    ("mser_ms", "mser_ms"),
                    ("total_ms", "detector_total_ms"),
                ):
                    shared_metrics[target_name] = float(
                        shared_metrics[target_name]
                    ) + float(detector_metrics.get(source_name) or 0.0)
                if "provider_image" in locals():
                    del provider_image

            for idx, target_path, target_core in seam_consumers:
                try:
                    target_source_y1 = int(target_core["source_y1"])
                    target_height = int(target_core["source_y2"]) - target_source_y1
                except (KeyError, TypeError, ValueError):
                    unavailable_pages.add(idx)
                    continue
                if target_height <= 0:
                    unavailable_pages.add(idx)
                    continue
                target = by_page.setdefault(idx, [])
                for box in seam_boxes:
                    mapped = self._map_seam_detection_to_page(
                        box,
                        seam_source_y1=seam_source_y1,
                        page_source_y1=target_source_y1,
                        page_height=target_height,
                    )
                    if mapped is not None:
                        target.append(mapped)

        for idx, boxes in list(by_page.items()):
            by_page[idx] = CombinedTextDetector._apply_final_nms(
                boxes, iou_threshold=DETECTOR_FINAL_NMS_IOU
            )
        return by_page, unavailable_pages, shared_metrics

    def _commit_processed_page(
        self,
        chapter_id: str,
        processed_dir: Path,
        page_index: int,
        page_data: dict,
        snapshot: dict | None,
    ) -> bool:
        """Atomically publish one completed page as soon as its worker finishes."""
        tmp_clean_value = page_data.get("tmp_clean")
        tmp_clean_path = Path(tmp_clean_value) if tmp_clean_value else None
        tmp_auto_clean_value = page_data.get("tmp_auto_clean")
        tmp_auto_clean_path = Path(tmp_auto_clean_value) if tmp_auto_clean_value else None

        try:
            with get_page_lock(chapter_id, page_index), get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                pages = manifest.get("pages", [])
                if not (
                    0 <= page_index < len(pages)
                    and is_processing_state_current(
                        manifest, page_index, snapshot, processed_dir
                    )
                ):
                    logger.warning(
                        "Chapter {} page {}: page state changed during processing, discarding stale output",
                        chapter_id,
                        page_index,
                    )
                    return False

                target_page = pages[page_index]
                orig_path = Path(target_page["original"])
                final_clean_path = processed_dir / f"clean_{orig_path.name}"
                auto_clean_path = self._auto_clean_path(processed_dir, orig_path)
                target_clean_revision = int(target_page.get("clean_revision") or 0) + 1
                with PageArtifactTransaction(
                    processed_dir,
                    page_index,
                    [final_clean_path, auto_clean_path],
                    target_clean_revision,
                ) as artifact_tx:
                    if tmp_clean_path and tmp_clean_path.exists():
                        os.replace(tmp_clean_path, final_clean_path)
                        target_page["clean"] = final_clean_path.as_posix()

                    if tmp_auto_clean_path and tmp_auto_clean_path.exists():
                        os.replace(tmp_auto_clean_path, auto_clean_path)

                    existing_boxes = target_page.get("boxes", [])
                    detected_boxes = assign_stable_detector_box_ids(
                        page_data["boxes"], existing_boxes
                    )
                    manual_boxes = [b for b in existing_boxes if b.get("manual")]
                    committed_boxes = detected_boxes + manual_boxes
                    target_page["boxes"] = committed_boxes
                    target_page["detection_state"] = page_data.get(
                        "detection_state", "verified"
                    )
                    target_page["detection_issues"] = list(
                        page_data.get("detection_issues") or []
                    )
                    target_page["unverified_regions"] = list(
                        page_data.get("unverified_regions") or []
                    )
                    target_page["needs_review"] = bool(page_data.get("needs_review"))
                    target_page["processing_metrics"] = dict(
                        page_data.get("processing_metrics") or {}
                    )
                    bump_page_revision(target_page, "process_revision")
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during commit")

                    for mask_field in ("manual_mask", "manual_lama_mask"):
                        if mask_field in page_data:
                            target_page[mask_field] = page_data[mask_field]
                        else:
                            target_page.pop(mask_field, None)

                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
                    return True
        finally:
            for tmp_path in (tmp_clean_path, tmp_auto_clean_path):
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = PIPELINE_DEFAULT_WORKERS,
    ) -> dict:
        """Process pages with shared seams and durable per-page progress."""
        run_started_at = time.perf_counter()
        processed_dir = PROCESSED_DIR / chapter_id
        unique_indices = list(dict.fromkeys(page_indices))

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            manifest_pages = manifest.get("pages", [])
            work_items = []
            for idx in unique_indices:
                if 0 <= idx < len(manifest_pages):
                    page = manifest_pages[idx]
                    if page.get("skipped", False):
                        continue
                    state_snapshot = capture_processing_state(
                        manifest, idx, processed_dir
                    )
                    stitch_core = copy.deepcopy(page.get("stitch_core"))
                    if isinstance(stitch_core, dict):
                        stitch_core["_source_page"] = int(
                            page.get("source_page", -1)
                        )
                    work_items.append(
                        (
                            idx,
                            Path(page["original"]),
                            copy.deepcopy(page.get("excluded_regions", [])),
                            copy.deepcopy(page.get("boxes", [])),
                            stitch_core,
                            state_snapshot,
                        )
                    )

        if not work_items:
            return manifest

        _ = self.detector
        _ = self.inpainter

        max_workers = max(
            1,
            min(
                int(workers or PIPELINE_DEFAULT_WORKERS),
                PIPELINE_PROCESS_WORKER_LIMIT,
                len(work_items),
            ),
        )
        shared_seam_detections, seam_context_unavailable, shared_seam_metrics = (
            self._shared_seam_detections(chapter_id, work_items)
        )
        committed_indices: list[int] = []
        errors: list[tuple[int, Exception]] = []
        parallel_detectors = max_workers == 1

        def _process_one(item) -> tuple[int, dict, dict | None]:
            (
                idx,
                img_path,
                excluded,
                existing_boxes,
                stitch_core,
                snapshot,
            ) = item
            return (
                idx,
                self._process_page(
                    img_path,
                    processed_dir,
                    excluded_regions=excluded,
                    existing_boxes=existing_boxes,
                    stitch_core=stitch_core,
                    supplemental_detections=shared_seam_detections.get(idx),
                    seam_context_unavailable=idx in seam_context_unavailable,
                    parallel_detectors=parallel_detectors,
                ),
                snapshot,
            )

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="page-process"
        ) as pool:
            futures = {
                pool.submit(_process_one, item): item[0] for item in work_items
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    page_idx, page_data, snapshot = future.result()
                    if self._commit_processed_page(
                        chapter_id,
                        processed_dir,
                        page_idx,
                        page_data,
                        snapshot,
                    ):
                        committed_indices.append(page_idx)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Chapter {} page {} operation 'process_page' failed: {}",
                        chapter_id,
                        idx,
                        exc,
                    )
                    errors.append((idx, exc))

        failed_indices = [item[0] for item in errors]
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if committed_indices:
                manifest["workflow"] = {
                    "stage": "review",
                    "page_index": min(committed_indices),
                }
            inpaint_metric_names = (
                "lama_model_runs",
                "lama_model_ms",
                "session_lock_wait_ms",
                "ort_global_lock_wait_ms",
                "smart_fill_regions",
            )
            inpaint_totals = {name: 0 for name in inpaint_metric_names}
            for page_index in committed_indices:
                if not 0 <= page_index < len(manifest.get("pages", [])):
                    continue
                page_metrics = manifest["pages"][page_index].get(
                    "processing_metrics", {}
                )
                for section in ("auto_inpaint", "manual_inpaint"):
                    section_metrics = page_metrics.get(section, {})
                    if not isinstance(section_metrics, dict):
                        continue
                    for name in inpaint_metric_names:
                        inpaint_totals[name] += int(section_metrics.get(name, 0) or 0)
            inpainter = self.inpainter
            session_loaded = bool(getattr(inpainter, "session_loaded", False))
            selected_dynamic = bool(
                getattr(inpainter, "dynamic_lama", False)
                if session_loaded
                else getattr(inpainter, "_prefer_dynamic", False)
            )
            serialized_inference = bool(
                getattr(inpainter, "serialized_inference", not selected_dynamic)
            )
            manifest["last_processing_run"] = {
                "requested_page_indices": [item[0] for item in work_items],
                "committed_page_indices": sorted(committed_indices),
                "failed_page_indices": failed_indices,
                "workers": max_workers,
                "wall_ms": round((time.perf_counter() - run_started_at) * 1000.0, 3),
                "shared_seam": {
                    name: round(float(value), 3)
                    if name.endswith("_ms")
                    else int(value)
                    for name, value in shared_seam_metrics.items()
                },
                "inpaint": {
                    "model": Path(getattr(inpainter, "lama_model_path", "")).name
                    or None,
                    "dynamic": selected_dynamic,
                    "serialized_inference": serialized_inference,
                    "serialization_scope": (
                        "global" if serialized_inference else None
                    ),
                    **inpaint_totals,
                },
            }
            save_manifest_raw(chapter_id, manifest)

        if errors:
            first_exc = errors[0][1]
            if not committed_indices:
                raise RuntimeError(
                    f"Chapter {chapter_id}: All requested pages failed to process. "
                    f"First error (page {failed_indices[0]}): {first_exc}"
                ) from first_exc
            raise RuntimeError(
                f"Chapter {chapter_id}: Failed to process {len(errors)} page(s) "
                f"(indices: {failed_indices}). First error: {first_exc}"
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
                        if page.get("clean") is not None or page.get("boxes"):
                            changed = True
                        # ``clean`` is always a processed artifact. A skipped page
                        # intentionally has none, so render/export falls back to
                        # its raw original without violating managed-path roots.
                        page["clean"] = None
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

            target_path = out_dir / f"page_{i:03d}.png"
            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove stale output {}: {}", target_path, exc)

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
                manual_lama_mask_posix = target_page.get("manual_lama_mask")
                target_clean_revision = int(
                    target_page.get("clean_revision") or 0
                ) + 1

            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                )

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                    target_page = manifest["pages"][page_index]
                    target_page.setdefault("boxes", []).append(new_box)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def update_box(
        self,
        chapter_id: str,
        page_index: int,
        box_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                page = manifest["pages"][page_index]
                boxes = page.get("boxes", [])
                if box_index < 0 or box_index >= len(boxes):
                    raise ValueError(
                        f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                    )
                img_path = Path(page["original"])
                expected_box_id = boxes[box_index].get("id")

            image = read_image(img_path)
            height, width = image.shape[:2]
            nx1, nx2 = sorted((int(x1), int(x2)))
            ny1, ny2 = sorted((int(y1), int(y2)))
            if (
                nx1 < 0
                or ny1 < 0
                or nx2 > width
                or ny2 > height
                or (nx2 - nx1) < 10
                or (ny2 - ny1) < 10
            ):
                raise ValueError(
                    f"Chapter {chapter_id} page {page_index}: Box coordinates out of bounds or smaller than 10px"
                )

            new_geometry = {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2}
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError(
                        f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                    )
                if expected_box_id and target_boxes[box_index].get("id") != expected_box_id:
                    raise RuntimeError(
                        f"Chapter {chapter_id} page {page_index}: Box changed while geometry edit was pending"
                    )

                boxes_snapshot = copy.deepcopy(target_boxes)
                self._apply_box_geometry(boxes_snapshot[box_index], new_geometry)
                manual_mask_posix = target_page.get("manual_mask")
                manual_lama_mask_posix = target_page.get("manual_lama_mask")
                target_clean_revision = int(
                    target_page.get("clean_revision") or 0
                ) + 1

            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                )

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                    target_page = manifest["pages"][page_index]
                    target_boxes = target_page.get("boxes", [])
                    if box_index < 0 or box_index >= len(target_boxes):
                        raise ValueError(
                            f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                        )
                    if expected_box_id and target_boxes[box_index].get("id") != expected_box_id:
                        raise RuntimeError(
                            f"Chapter {chapter_id} page {page_index}: Box changed during geometry repaint"
                        )

                    self._apply_box_geometry(target_boxes[box_index], new_geometry)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
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
                manual_lama_mask_posix = page.get("manual_lama_mask")
                boxes_snapshot = copy.deepcopy(boxes)
                boxes_snapshot[box_index]["removed"] = True
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            image = read_image(img_path)
            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
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
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def repaint_mask(
        self,
        chapter_id: str,
        page_index: int,
        mask: np.ndarray,
        *,
        force_lama: bool = False,
    ) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                page = manifest["pages"][page_index]
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1
                manual_mask_posix = page.get("manual_mask")
                manual_lama_mask_posix = page.get("manual_lama_mask")
                clean_posix = page.get("clean")

            image = read_image(img_path)
            img_h, img_w = image.shape[:2]

            manual_mask_path = (
                Path(manual_mask_posix)
                if manual_mask_posix
                else self._manual_mask_path(processed_dir, img_path)
            )
            manual_lama_mask_path = (
                Path(manual_lama_mask_posix)
                if manual_lama_mask_posix
                else self._manual_mask_path(
                    processed_dir, img_path, force_lama=True
                )
            )

            bin_mask = (
                (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
                if mask is not None
                else None
            )
            if bin_mask is not None and bin_mask.shape[:2] != (img_h, img_w):
                bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                bin_mask = (
                    (bin_mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
                )

            target_field = "manual_lama_mask" if force_lama else "manual_mask"
            target_path = manual_lama_mask_path if force_lama else manual_mask_path
            target_prefix = "manual_lama_mask" if force_lama else "manual_mask"
            existing_mask = self._read_manual_mask(target_path, (img_h, img_w))

            if existing_mask is not None and bin_mask is not None:
                accumulated_mask = np.maximum(existing_mask, bin_mask)
            elif bin_mask is not None:
                accumulated_mask = bin_mask
            elif existing_mask is not None:
                accumulated_mask = existing_mask
            else:
                accumulated_mask = None

            tmp_mask_path = processed_dir / (
                f"{target_prefix}_{img_path.name}.{uuid.uuid4().hex}.tmp.png"
            )
            if accumulated_mask is not None:
                write_image(tmp_mask_path, accumulated_mask)

            final_mask_path = self._manual_mask_path(
                processed_dir, img_path, force_lama=force_lama
            )
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [final_mask_path],
            ) as artifact_tx:
                auto_clean_path = self._auto_clean_path(processed_dir, img_path)
                if (
                    not auto_clean_path.exists()
                    and not manual_mask_path.exists()
                    and not manual_lama_mask_path.exists()
                ):
                    clean_path = Path(clean_posix) if clean_posix else None
                    if clean_path is not None and clean_path.exists():
                        try:
                            cached = read_image(clean_path)
                            if cached.shape[:2] == (img_h, img_w):
                                self._write_auto_clean_cache(
                                    processed_dir, img_path, cached
                                )
                        except Exception as exc:
                            logger.warning(
                                "Could not seed auto-clean cache from {}: {}",
                                clean_path,
                                exc,
                            )

                try:
                    standard_mask_for_repaint = (
                        tmp_mask_path.as_posix()
                        if accumulated_mask is not None and not force_lama
                        else manual_mask_posix
                    )
                    lama_mask_for_repaint = (
                        tmp_mask_path.as_posix()
                        if accumulated_mask is not None and force_lama
                        else manual_lama_mask_posix
                    )
                    clean_path_posix = self._do_reinpaint(
                        processed_dir,
                        img_path,
                        image,
                        boxes_snapshot,
                        manual_mask_posix=standard_mask_for_repaint,
                        manual_lama_mask_posix=lama_mask_for_repaint,
                        reuse_auto_clean=True,
                    )
                    if accumulated_mask is not None:
                        os.replace(tmp_mask_path, final_mask_path)
                        target_mask_posix = final_mask_path.as_posix()
                    else:
                        target_mask_posix = None
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
                    if target_mask_posix:
                        target_page[target_field] = target_mask_posix
                    else:
                        target_page.pop(target_field, None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
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
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            manual_mask_path = self._manual_mask_path(processed_dir, img_path)
            manual_lama_mask_path = self._manual_mask_path(
                processed_dir, img_path, force_lama=True
            )

            image = read_image(img_path)
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [manual_mask_path, manual_lama_mask_path],
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=None,
                    manual_lama_mask_posix=None,
                    reuse_auto_clean=True,
                    apply_manual_mask=False,
                )

                for mask_path in (manual_mask_path, manual_lama_mask_path):
                    if mask_path.exists():
                        try:
                            mask_path.unlink()
                        except OSError as exc:
                            raise RuntimeError(
                                f"Cannot remove manual mask: {exc}"
                            ) from exc

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError("Invalid page index")
                    target_page = manifest["pages"][page_index]
                    target_page.pop("manual_mask", None)
                    target_page.pop("manual_lama_mask", None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
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

    def _page_artifact_transaction(
        self,
        processed_dir: Path,
        img_path: Path,
        page_index: int,
        target_clean_revision: int,
        extra_paths: list[Path] | None = None,
    ) -> PageArtifactTransaction:
        paths = [
            processed_dir / f"clean_{img_path.name}",
            self._auto_clean_path(processed_dir, img_path),
        ]
        paths.extend(extra_paths or [])
        return PageArtifactTransaction(
            processed_dir,
            page_index,
            paths,
            target_clean_revision,
        )

    @staticmethod
    def _manual_mask_path(
        processed_dir: Path, img_path: Path, *, force_lama: bool = False
    ) -> Path:
        prefix = "manual_lama_mask" if force_lama else "manual_mask"
        return processed_dir / f"{prefix}_{img_path.name}"

    @staticmethod
    def _read_manual_mask(
        mask_path: Path, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        if not mask_path.exists():
            return None
        try:
            raw = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read manual mask at {}: {}", mask_path, exc)
            return None
        if mask is None or not np.any(mask > MANUAL_MASK_THRESHOLD):
            return None
        if mask.shape[:2] != expected_shape:
            try:
                mask = cv2.resize(
                    mask,
                    (expected_shape[1], expected_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            except Exception as exc:
                logger.warning("Could not resize manual mask at {}: {}", mask_path, exc)
                return None
        if not np.any(mask > MANUAL_MASK_THRESHOLD):
            return None
        return (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255

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
            logger.warning("Could not read auto-clean cache at {}: {}", auto_path, exc)
            return None
        if cached.shape[:2] != expected_shape:
            logger.warning(
                "Ignoring stale auto-clean cache at {}: expected shape {}, got {}",
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
        manual_lama_mask_posix: str | None = None,
        *,
        reuse_auto_clean: bool = False,
        apply_manual_mask: bool = True,
    ) -> str:
        """Repaint using masks stored inline or in managed PNG sidecars."""
        boxes_objects = []
        for box in boxes:
            if box.get("removed"):
                continue

            confidence = float(box.get("confidence", 1.0))
            geometry_overridden = bool(box.get("geometry_overridden"))
            explicit_manual = bool(
                box.get("manual")
                or box.get("origin") == "manual"
                or confidence >= 1.0
            )
            safe_to_inpaint = bool(box.get("safe_to_inpaint"))
            overlap_context_only = bool(box.get("overlap_context_only"))

            # Persisted review-only detector masks are evidence, not erase
            # authority. Preserve the same overlap rule as initial processing.
            if overlap_context_only and not geometry_overridden:
                continue
            if not (safe_to_inpaint or geometry_overridden or explicit_manual):
                continue

            box_h = int(box["y2"]) - int(box["y1"])
            box_w = int(box["x2"]) - int(box["x1"])
            if box_h <= 0 or box_w <= 0:
                continue
            mask_arr = decode_mask_value(box.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(
                        mask_arr,
                        (box_w, box_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except Exception:
                    mask_arr = None

            box_object = BubbleBox(
                int(box["x1"]),
                int(box["y1"]),
                int(box["x2"]),
                int(box["y2"]),
                confidence,
                mask_arr,
                source_model=str(box.get("source_model") or "unknown"),
                class_id=int(box.get("class_id") or 0),
                class_name=str(box.get("class_name") or "unknown"),
                semantic_type=str(box.get("semantic_type") or "unknown"),
                mask_source=str(box.get("mask_source") or "none"),
                safe_to_inpaint=safe_to_inpaint,
                ocr_eligible=bool(box.get("ocr_eligible")),
                needs_review=bool(box.get("needs_review")),
            )
            if geometry_overridden or explicit_manual:
                box_object.allow_rectangle_fallback = True
            boxes_objects.append(box_object)

        clean_image = None
        if reuse_auto_clean:
            clean_image = self._load_auto_clean_cache(
                processed_dir, img_path, image.shape[:2]
            )

        if clean_image is None:
            clean_image = self.inpainter.inpaint(image, boxes_objects)
            self._write_auto_clean_cache(processed_dir, img_path, clean_image)
        else:
            clean_image = clean_image.copy()

        manual_mask_path = (
            Path(manual_mask_posix)
            if manual_mask_posix
            else self._manual_mask_path(processed_dir, img_path)
        )
        manual_lama_mask_path = (
            Path(manual_lama_mask_posix)
            if manual_lama_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=True)
        )
        # Keep repaint authority attached to the persisted mask. Standard marks
        # retain Smart Fill eligibility; explicit LaMa marks always run LaMa,
        # including after box edits and later page reprocessing.
        mask_passes = (
            (manual_mask_path, False),
            (manual_lama_mask_path, True),
        )
        if apply_manual_mask:
            for mask_path, force_lama in mask_passes:
                manual_mask = self._read_manual_mask(
                    mask_path, clean_image.shape[:2]
                )
                if manual_mask is not None:
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask, force_lama=force_lama
                    )

        clean_path = processed_dir / f"clean_{img_path.name}"
        tmp_clean_path = (
            processed_dir
            / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        )
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
        supplemental_detections: list[BubbleBox] | None = None,
        seam_context_unavailable: bool = False,
        *,
        parallel_detectors: bool = False,
    ) -> dict:
        started_at = time.perf_counter()
        read_started_at = started_at
        image = read_image(img_path)
        read_ms = (time.perf_counter() - read_started_at) * 1000.0
        detector_kwargs = {"parallel": parallel_detectors}

        core_bounds: tuple[int, int] | None = None
        if isinstance(stitch_core, dict):
            try:
                core_y1 = max(0, min(int(stitch_core.get("core_y1", 0)), image.shape[0]))
                core_y2 = max(core_y1, min(int(stitch_core.get("core_y2", image.shape[0])), image.shape[0]))
            except (TypeError, ValueError):
                core_y1, core_y2 = 0, image.shape[0]
            if core_y2 > core_y1:
                core_bounds = (core_y1, core_y2)

        use_core_detector = (
            core_bounds is not None
            and core_bounds != (0, image.shape[0])
        )
        detect_started_at = time.perf_counter()
        if use_core_detector:
            core_y1, core_y2 = core_bounds
            core_image = image[core_y1:core_y2, :]
            detected = [
                self._shift_detection_y(box, core_y1)
                for box in self.detector.detect(core_image, **detector_kwargs)
            ]
        else:
            detected = self.detector.detect(image, **detector_kwargs)
        detector_metrics = self.detector.last_metrics()

        if supplemental_detections:
            detected = CombinedTextDetector._apply_final_nms(
                detected + list(supplemental_detections),
                iou_threshold=DETECTOR_FINAL_NMS_IOU,
            )
        if excluded_regions:
            detected = [b for b in detected if not self._box_in_excluded(b, excluded_regions)]
        detect_ms = (time.perf_counter() - detect_started_at) * 1000.0

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
                    # Geometry explicitly confirmed by the user remains a valid
                    # non-destructive OCR target even though rectangle inpaint
                    # authority is handled separately below.
                    record["ocr_eligible"] = True
                    record["needs_review"] = True

            if core_bounds is not None:
                core_y1, core_y2 = core_bounds
                record_y1 = int(record["y1"])
                record_y2 = int(record["y2"])
                overlap_only = record_y2 <= core_y1 or record_y1 >= core_y2
                if overlap_only:
                    record["overlap_context_only"] = True
                    record["ocr_eligible"] = False
                else:
                    record.pop("overlap_context_only", None)

            skip_auto_overlap_inpaint = bool(
                record.get("overlap_context_only")
                and not record.get("geometry_overridden")
            )
            if (
                not record.get("removed")
                and not skip_auto_overlap_inpaint
                and (record.get("safe_to_inpaint") or record.get("geometry_overridden"))
            ):
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
            mask_arr = decode_mask_value(old.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(mask_arr, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_arr = None
            effective_boxes.append(BubbleBox(
                int(old["x1"]), int(old["y1"]), int(old["x2"]), int(old["y2"]),
                float(old.get("confidence", 1.0)), mask_arr,
            ))

        auto_inpaint_started_at = time.perf_counter()
        clean_image = self.inpainter.inpaint(image, effective_boxes)
        auto_inpaint_ms = (time.perf_counter() - auto_inpaint_started_at) * 1000.0
        auto_inpaint_metrics = self.inpainter.last_metrics()

        auto_clean_path = self._auto_clean_path(processed_dir, img_path)
        manual_mask_path = self._manual_mask_path(processed_dir, img_path)
        manual_lama_mask_path = self._manual_mask_path(
            processed_dir, img_path, force_lama=True
        )
        tmp_auto_clean_path = None
        if (
            auto_clean_path.exists()
            or manual_mask_path.exists()
            or manual_lama_mask_path.exists()
        ):
            tmp_auto_clean_path = processed_dir / f"auto_clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
            write_image(tmp_auto_clean_path, clean_image)

        manual_mask_posix = None
        manual_lama_mask_posix = None
        manual_inpaint_ms = 0.0
        manual_inpaint_metrics: dict[str, int] = {}
        manual_passes = (
            ("manual_mask", manual_mask_path, False),
            ("manual_lama_mask", manual_lama_mask_path, True),
        )
        for mask_field, mask_path, force_lama in manual_passes:
            manual_mask = self._read_manual_mask(mask_path, clean_image.shape[:2])
            if manual_mask is None:
                continue
            manual_inpaint_started_at = time.perf_counter()
            clean_image = self.inpainter.inpaint_mask(
                clean_image.copy(), manual_mask, force_lama=force_lama
            )
            manual_inpaint_ms += (
                time.perf_counter() - manual_inpaint_started_at
            ) * 1000.0
            for metric_name, metric_value in self.inpainter.last_metrics().items():
                manual_inpaint_metrics[metric_name] = (
                    manual_inpaint_metrics.get(metric_name, 0) + int(metric_value)
                )
            if mask_field == "manual_mask":
                manual_mask_posix = mask_path.as_posix()
            else:
                manual_lama_mask_posix = mask_path.as_posix()

        tmp_clean_path = processed_dir / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_started_at = time.perf_counter()
        write_image(tmp_clean_path, clean_image)
        write_ms = (time.perf_counter() - write_started_at) * 1000.0

        unverified_regions = [
            {k: record.get(k) for k in ("x1", "y1", "x2", "y2", "confidence", "source_model", "class_name", "semantic_type")}
            for record in detector_records
            if record.get("needs_review") or not record.get("safe_to_inpaint")
        ]
        detection_issues = []
        if seam_context_unavailable:
            detection_issues.append("seam_context_unavailable")
        if unverified_regions:
            detection_issues.append("unverified_regions")
        if not detector_records:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) > DETECTION_CONTENT_STD_MIN:
                detection_issues.append("content_heavy_zero_box")
        detection_state = "needs_review" if detection_issues else "verified"
        inpainter = self.inpainter
        inpaint_model_path = getattr(inpainter, "lama_model_path", None)
        inpaint_session_loaded = bool(getattr(inpainter, "session_loaded", False))
        inpaint_dynamic = bool(
            getattr(inpainter, "dynamic_lama", False)
            if inpaint_session_loaded
            else getattr(inpainter, "_prefer_dynamic", False)
        )
        serialized_inference = bool(
            getattr(inpainter, "serialized_inference", not inpaint_dynamic)
        )
        processing_metrics = {
            "timing_ms": {
                "read": round(read_ms, 3),
                "detect": round(detect_ms, 3),
                "auto_inpaint": round(auto_inpaint_ms, 3),
                "manual_inpaint": round(manual_inpaint_ms, 3),
                "write": round(write_ms, 3),
                "total": round((time.perf_counter() - started_at) * 1000.0, 3),
            },
            "detector": {
                **detector_metrics,
                "records": len(detector_records),
                "authorized": len(effective_boxes),
                "review_only": len(unverified_regions),
            },
            "auto_inpaint": auto_inpaint_metrics,
            "manual_inpaint": manual_inpaint_metrics,
            "model": {
                "active": Path(inpaint_model_path).name if inpaint_model_path else None,
                "dynamic": inpaint_dynamic,
                "serialized_inference": serialized_inference,
                "serialization_scope": (
                    "global" if serialized_inference else None
                ),
                "session_loaded": inpaint_session_loaded,
            },
        }

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
            "processing_metrics": processing_metrics,
        }
        if tmp_auto_clean_path is not None:
            res["tmp_auto_clean"] = tmp_auto_clean_path.as_posix()
        if manual_mask_posix:
            res["manual_mask"] = manual_mask_posix
        if manual_lama_mask_posix:
            res["manual_lama_mask"] = manual_lama_mask_posix
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
