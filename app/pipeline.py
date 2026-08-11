import shutil
import base64
import copy
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
from app.manifest_utils import load_manifest_raw, save_manifest_raw, get_manifest_lock
from app.security import MAX_IMAGE_PIXELS, MAX_UPLOAD_FILE_BYTES, MAX_UPLOAD_FILES, MAX_UPLOAD_TOTAL_BYTES, validate_upload_image


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
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
        buf.tofile(str(path))
    else:
        cv2.imwrite(str(path), image)


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
                return idx, slice_image(raw_path, sliced_dir, f"{idx:03d}")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_slice_one, (i, p)): i for i, p in enumerate(raw_paths)
                }
                for future in as_completed(futures):
                    idx, slice_paths = future.result()
                    slice_results[idx] = slice_paths

        pages = []
        for source_index in range(len(raw_paths)):
            for slice_index, slice_path in enumerate(slice_results[source_index]):
                pages.append({
                    "original": slice_path.as_posix(),
                    "clean": None,
                    "boxes": [],
                    "skipped": False,
                    "excluded_regions": [],
                    "source_page": source_index,
                    "slice_index": slice_index,
                })

        manifest = {"chapter_id": chapter_id, "source_url": source_url, "pages": pages}
        with get_manifest_lock(chapter_id):
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def process_pages(self, chapter_id: str, page_indices: list[int], workers: int = 2) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        # 1) Read work list under lock, then release — do NOT hold lock during inference.
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            work_items = [
                (
                    idx,
                    Path(manifest["pages"][idx]["original"]),
                    manifest["pages"][idx].get("excluded_regions", []),
                )
                for idx in page_indices
                if 0 <= idx < len(manifest["pages"]) and not manifest["pages"][idx]["skipped"]
            ]

        if not work_items:
            return manifest

        # Warm models once so workers don't race on first lazy load.
        _ = self.detector
        _ = self.inpainter

        max_workers = max(1, min(int(workers or 2), 8, len(work_items)))
        results: dict[int, dict] = {}
        errors: list[tuple[int, Exception]] = []

        def _process_one(item: tuple[int, Path, list[dict]]) -> tuple[int, dict]:
            idx, img_path, excluded = item
            return idx, self._process_page(img_path, processed_dir, excluded_regions=excluded)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_one, item): item[0] for item in work_items
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    page_idx, page_data = future.result()
                    results[page_idx] = page_data
                except Exception as exc:
                    logger.error(
                        "Chapter %s page %s operation 'process_page' failed: %s",
                        chapter_id,
                        idx,
                        exc,
                        exc_info=True,
                    )
                    errors.append((idx, exc))

        # 2) Merge results under lock (reload in case another request edited boxes).
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            for idx, page_data in results.items():
                if 0 <= idx < len(manifest["pages"]):
                    existing_boxes = manifest["pages"][idx].get("boxes", [])
                    manual_boxes = [b for b in existing_boxes if b.get("manual")]
                    merged_boxes = page_data["boxes"] + manual_boxes
                    manifest["pages"][idx]["clean"] = page_data["clean"]
                    manifest["pages"][idx]["boxes"] = merged_boxes
                    if "manual_mask" in page_data:
                        manifest["pages"][idx]["manual_mask"] = page_data["manual_mask"]
                    manifest["pages"][idx]["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, list(results.keys()))

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
                    manifest["pages"][idx]["skipped"] = skipped
                    if skipped:
                        manifest["pages"][idx]["clean"] = manifest["pages"][idx]["original"]
                        manifest["pages"][idx]["boxes"] = []
                    manifest["pages"][idx]["rendered"] = False
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
            clean_p = Path(page["clean"]) if page.get("clean") else None
            src_p = clean_p if (clean_p and clean_p.exists()) else Path(page["original"])

            if src_p and src_p.exists():
                shutil.copyfile(src_p, target_path)

    def add_manual_box(self, chapter_id: str, page_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

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

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
            target_page = manifest["pages"][page_index]
            target_page.setdefault("boxes", []).append({
                "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                "confidence": 1.0, "mask": None, "manual": True,
            })
            boxes_snapshot = copy.deepcopy(target_page["boxes"])
            manual_mask_posix = target_page.get("manual_mask")

        clean_path_posix = self._do_reinpaint(
            processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
        )

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
            target_page = manifest["pages"][page_index]
            target_page["clean"] = clean_path_posix
            target_page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def update_box(self, chapter_id: str, page_index: int, box_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

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

            target_boxes[box_index].update({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2, "mask": None})
            boxes_snapshot = copy.deepcopy(target_boxes)
            manual_mask_posix = target_page.get("manual_mask")

        clean_path_posix = self._do_reinpaint(
            processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
        )

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
            target_page = manifest["pages"][page_index]
            target_page["clean"] = clean_path_posix
            target_page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def remove_box(self, chapter_id: str, page_index: int, box_index: int) -> dict:
        """Mark a box removed, repaint the page, and persist the new state."""
        processed_dir = PROCESSED_DIR / chapter_id

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
            target_page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def repaint_mask(self, chapter_id: str, page_index: int, mask: np.ndarray) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            page = manifest["pages"][page_index]
            img_path = Path(page["original"])
            boxes_snapshot = copy.deepcopy(page.get("boxes", []))

        image = read_image(img_path)
        bin_mask = (mask > 10).astype(np.uint8) * 255 if mask is not None else None
        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
        if bin_mask is not None:
            write_image(manual_mask_path, bin_mask)
        manual_mask_posix = manual_mask_path.as_posix()
        clean_path_posix = self._do_reinpaint(
            processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
        )

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target_page = manifest["pages"][page_index]
            target_page["manual_mask"] = manual_mask_posix
            target_page["clean"] = clean_path_posix
            target_page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def reset_manual_mask(self, chapter_id: str, page_index: int) -> dict:
        """Remove the persisted manual mask and rebuild the clean page."""
        processed_dir = PROCESSED_DIR / chapter_id

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError("Invalid page index")
            page = manifest["pages"][page_index]
            img_path = Path(page["original"])
            boxes_snapshot = copy.deepcopy(page.get("boxes", []))

        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
        if manual_mask_path.exists():
            try:
                manual_mask_path.unlink()
            except OSError as exc:
                raise RuntimeError(f"Cannot remove manual mask: {exc}") from exc

        image = read_image(img_path)
        clean_path_posix = self._do_reinpaint(
            processed_dir, img_path, image, boxes_snapshot, manual_mask_posix=None
        )

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError("Invalid page index")
            target_page = manifest["pages"][page_index]
            target_page.pop("manual_mask", None)
            target_page["clean"] = clean_path_posix
            target_page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def _do_reinpaint(
        self,
        processed_dir: Path,
        img_path: Path,
        image: np.ndarray,
        boxes: list[dict],
        manual_mask_posix: str | None = None,
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

        clean_image = self.inpainter.inpaint(image, boxes_objects)

        manual_mask_path = Path(manual_mask_posix) if manual_mask_posix else processed_dir / f"manual_mask_{img_path.name}"
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
                    clean_image = self.inpainter.inpaint_mask(clean_image, manual_mask)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)
        return clean_path.as_posix()

    def _process_page(self, img_path: Path, processed_dir: Path, excluded_regions: list[dict] | None = None) -> dict:
        image = read_image(img_path)
        boxes = self.detector.detect(image)
        if excluded_regions:
            boxes = [b for b in boxes if not self._box_in_excluded(b, excluded_regions)]
        clean_image = self.inpainter.inpaint(image, boxes)

        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
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
                    clean_image = self.inpainter.inpaint_mask(clean_image, manual_mask)
                    manual_mask_posix = manual_mask_path.as_posix()

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)

        logger.debug(f"Processed {img_path.name}: {len(boxes)} boxes detected (after excluded filtering)")
        res = {
            "clean": clean_path.as_posix(),
            "boxes": [
                {
                    "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                    "confidence": b.confidence,
                    "mask": _encode_mask(b.mask),
                }
                for b in boxes
            ],
        }
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
