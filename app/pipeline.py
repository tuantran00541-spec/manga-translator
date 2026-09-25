import copy
import threading
import time
from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from app.downloader.registry import download_chapter as fetch_chapter_images
from app.downloader.slicer import OVERLAP_CONTEXT, slice_image
from app.detector.bubble_detector import BubbleBox, apply_final_nms
from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.mask_geometry import geometry_dict, remap_local_mask_page_space
from app.image_io import encode_mask as _encode_mask, read_image
from app.one_shot_cleanup import OneShotProductionDetector
from app.page_processing import PageProcessingMixin
from app.pipeline_editing import PipelineEditingMixin
from app.config import RAW_DIR, PROCESSED_DIR
from app.logging_config import logger
from app.mask_store import decode_mask_value
from app.parameters import (
    DETECTOR_FINAL_NMS_IOU,
    PIPELINE_DEFAULT_WORKERS,
    PIPELINE_SLICE_WORKER_LIMIT,
)
from app.manifest_utils import (
    assign_stable_detector_box_ids,
    atomic_replace,
    bump_page_revision,
    capture_processing_state,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    is_processing_state_current,
    load_manifest_raw,
    PageArtifactTransaction,
    save_manifest_raw,
)
from app.security import MAX_UPLOAD_FILE_BYTES, MAX_UPLOAD_FILES, MAX_UPLOAD_TOTAL_BYTES, validate_upload_image


class StaleProcessingStateError(RuntimeError):
    pass


class ChapterPipeline(PageProcessingMixin, PipelineEditingMixin):
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
                    self._detector = OneShotProductionDetector()
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
                    "process_required": True,
                    "preserve_regions": [],
                    "source_page": source_index,
                    "slice_index": slice_index,
                }
                if isinstance(slice_item, dict):
                    try:
                        width = int(slice_item.get("width") or 0)
                        height = int(slice_item.get("height") or 0)
                    except (TypeError, ValueError):
                        width = height = 0
                    if width > 0:
                        page["width"] = width
                    if height > 0:
                        page["height"] = height
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
        consumers: dict[tuple[int, int], list[tuple[int, Path, dict]]] = {}
        unavailable_pages: set[int] = set()
        shared_metrics: dict[str, float | int] = {
            "attempts": 0,
            "failures": 0,
            "wall_ms": 0.0,
            "bubble_model_ms": 0.0,
            "text_model_ms": 0.0,
            "mser_ms": 0.0,
            "text_grayscale_fallback_runs": 0,
            "text_grayscale_fallback_ms": 0.0,
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
                    (
                        "text_grayscale_fallback_runs",
                        "text_grayscale_fallback_runs",
                    ),
                    (
                        "text_grayscale_fallback_ms",
                        "text_grayscale_fallback_ms",
                    ),
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
            by_page[idx] = apply_final_nms(
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
                        atomic_replace(tmp_clean_path, final_clean_path)
                        target_page["clean"] = final_clean_path.as_posix()

                    if tmp_auto_clean_path and tmp_auto_clean_path.exists():
                        atomic_replace(tmp_auto_clean_path, auto_clean_path)

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
                    target_page["deferred_regions"] = list(
                        page_data.get("deferred_regions") or []
                    )
                    target_page["residue_regions"] = list(
                        page_data.get("residue_regions") or []
                    )
                    target_page["cleanup_verified"] = bool(
                        page_data.get("cleanup_verified", False)
                    )
                    target_page["needs_review"] = bool(page_data.get("needs_review"))
                    target_page["processing_metrics"] = dict(
                        page_data.get("processing_metrics") or {}
                    )
                    target_page["process_required"] = False
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
                    artifact_tx.mark_manifest_commit(target_page)
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
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict:
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
                            copy.deepcopy(page.get("preserve_regions", [])),
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
                8,
                len(work_items),
            ),
        )
        shared_seam_detections, seam_context_unavailable, shared_seam_metrics = (
            self._shared_seam_detections(chapter_id, work_items)
        )
        committed_indices: list[int] = []
        discarded_stale_indices: list[int] = []
        errors: list[tuple[int, Exception]] = []
        parallel_detectors = max_workers == 1

        def _process_one(item) -> tuple[int, dict, dict | None]:
            (
                idx,
                img_path,
                preserve_regions,
                existing_boxes,
                stitch_core,
                snapshot,
            ) = item
            return (
                idx,
                self._process_page(
                    img_path,
                    processed_dir,
                    preserve_regions=preserve_regions,
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
                    else:
                        discarded_stale_indices.append(page_idx)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Chapter {} page {} operation 'process_page' failed: {}",
                        chapter_id,
                        idx,
                        exc,
                    )
                    errors.append((idx, exc))
                finally:
                    if progress_callback is not None:
                        progress_callback(idx)

        failed_indices = [item[0] for item in errors]
        discarded_stale_indices = sorted(set(discarded_stale_indices))
        if errors:
            processing_outcome = "partial_failed" if committed_indices else "failed"
        elif discarded_stale_indices:
            processing_outcome = "partial_stale" if committed_indices else "stale_only"
        else:
            processing_outcome = "completed"

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
                "discarded_stale_page_indices": discarded_stale_indices,
                "discarded_stale_pages": [
                    {"page_index": page_index, "reason": "processing_state_changed"}
                    for page_index in discarded_stale_indices
                ],
                "outcome": processing_outcome,
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

        if discarded_stale_indices and not committed_indices:
            raise StaleProcessingStateError(
                f"Chapter {chapter_id}: all requested processing output became stale "
                f"before commit (indices: {discarded_stale_indices})"
            )

        return manifest

    def mark_skipped(self, chapter_id: str, page_indices: list[int], skipped: bool) -> dict:
        indices = sorted({int(idx) for idx in page_indices if int(idx) >= 0})
        with ExitStack() as stack:
            for idx in indices:
                stack.enter_context(get_page_lock(chapter_id, idx))
            stack.enter_context(get_manifest_lock(chapter_id))
            manifest = load_manifest_raw(chapter_id)
            for idx in indices:
                if 0 <= idx < len(manifest["pages"]):
                    page = manifest["pages"][idx]
                    changed = bool(page.get("skipped", False)) != bool(skipped)
                    page["skipped"] = skipped
                    if skipped:
                        if page.get("clean") is not None or page.get("boxes"):
                            changed = True
                        page["clean"] = None
                        page["boxes"] = []
                        page["process_required"] = False
                    else:
                        page["process_required"] = True
                    if changed:
                        bump_page_revision(page, "clean_revision")
                    invalidate_page_render(manifest, idx)
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, indices)
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
