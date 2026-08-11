import shutil
import base64
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
                    logger.exception("Page %s failed: %s", idx, exc)
                    errors.append((idx, exc))

        if not results and errors:
            raise RuntimeError(f"Xử lý trang thất bại: {errors[0][1]}") from errors[0][1]

        # 2) Merge results under lock (reload in case another request edited boxes).
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            for idx, page_data in results.items():
                if 0 <= idx < len(manifest["pages"]):
                    manifest["pages"][idx]["clean"] = page_data["clean"]
                    manifest["pages"][idx]["boxes"] = page_data["boxes"]
                    manifest["pages"][idx]["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, list(results.keys()))

        if errors:
            logger.warning(
                "Chapter %s: %d/%d pages failed",
                chapter_id,
                len(errors),
                len(work_items),
            )
        return manifest

    def mark_skipped(self, chapter_id: str, page_indices: list[int], skipped: bool) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            for idx in page_indices:
                if 0 <= idx < len(manifest["pages"]):
                    manifest["pages"][idx]["skipped"] = skipped
                    if skipped:
                        manifest["pages"][idx]["clean"] = manifest["pages"][idx]["original"]
                        manifest["pages"][idx]["boxes"] = []
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
            if page_index < 0 or page_index >= len(manifest["pages"]):
                return manifest
            page = manifest["pages"][page_index]

            img_path = Path(page["original"])
            image = read_image(img_path)
            h, w = image.shape[:2]

            x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
            y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
            if x2 <= x1 or y2 <= y1:
                return manifest

            page["boxes"].append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": 1.0, "mask": None, "manual": True,
            })
            page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def update_box(self, chapter_id: str, page_index: int, box_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest["pages"]):
                raise ValueError("Invalid page index")
            page = manifest["pages"][page_index]
            if box_index < 0 or box_index >= len(page.get("boxes", [])):
                raise ValueError("Invalid box index")
            image = read_image(Path(page["original"]))
            h, w = image.shape[:2]
            x1, x2 = sorted((int(x1), int(x2)))
            y1, y2 = sorted((int(y1), int(y2)))
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
                raise ValueError("Box coordinates out of bounds")
            box = page["boxes"][box_index]
            box.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            page["rendered"] = False
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def repaint_mask(self, chapter_id: str, page_index: int, mask: np.ndarray) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest["pages"]):
                raise ValueError("Invalid page index")
            page = manifest["pages"][page_index]
            img_path = Path(page["original"])
            image = read_image(img_path)
            manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
            write_image(manual_mask_path, mask)
            self._reinpaint_page(page, image, processed_dir, img_path)
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
        return manifest

    def _reinpaint_page(self, page: dict, image, processed_dir: Path, img_path: Path) -> None:
        boxes = [
            BubbleBox(
                b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"],
                _decode_mask(b.get("mask")),
            )
            for b in page["boxes"]
            if not b.get("removed")
        ]
        clean_image = self.inpainter.inpaint(image, boxes)

        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
        if manual_mask_path.exists():
            raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
            manual_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if manual_mask is not None and manual_mask.any():
                # Keep all manual brush strokes in one mask and process them
                # in one inpaint invocation. This avoids sequential LaMa
                # inference for every disconnected painted region.
                manual_mask = (manual_mask > 127).astype(np.uint8) * 255
                clean_image = self.inpainter.inpaint_mask(clean_image, manual_mask)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)
        page["clean"] = clean_path.as_posix()
        page["rendered"] = False



    def _process_page(self, img_path: Path, processed_dir: Path, excluded_regions: list[dict] | None = None) -> dict:
        image = read_image(img_path)
        boxes = self.detector.detect(image)
        if excluded_regions:
            boxes = [b for b in boxes if not self._box_in_excluded(b, excluded_regions)]
        clean_image = self.inpainter.inpaint(image, boxes)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)

        logger.debug(f"Processed {img_path.name}: {len(boxes)} boxes detected (after excluded filtering)")
        return {
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
