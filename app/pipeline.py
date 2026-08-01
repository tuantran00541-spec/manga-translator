import json
import os
import shutil
import base64
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from filelock import FileLock, Timeout
from app.downloader.registry import download_chapter as fetch_chapter_images
from app.downloader.slicer import slice_image
from app.detector.combined_detector import CombinedTextDetector
from app.detector.bubble_detector import BubbleBox
from app.inpaint.lama_inpainter import Inpainter
from app.config import RAW_DIR, PROCESSED_DIR
from app.logging_config import logger
from app.security import MAX_IMAGE_PIXELS, validate_upload_image


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


_MANIFEST_LOCK_TIMEOUT = 30


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

    @staticmethod
    @contextmanager
    def _manifest_lock(chapter_id: str):
        processed_dir = PROCESSED_DIR / chapter_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(processed_dir / "manifest.lock")
        try:
            with lock.acquire(timeout=_MANIFEST_LOCK_TIMEOUT):
                yield
        except Timeout:
            raise RuntimeError(
                f"Manifest lock timeout for chapter {chapter_id}"
            )

    def download_chapter(self, chapter_url: str, chapter_id: str) -> dict:
        raw_dir = RAW_DIR / chapter_id
        raw_paths = fetch_chapter_images(chapter_url, raw_dir)
        logger.info(f"Chapter {chapter_id}: downloaded {len(raw_paths)} raw images")
        return self._build_chapter_from_raw_paths(chapter_id, raw_paths, source_url=chapter_url)

    def create_chapter_from_uploads(self, chapter_id: str, uploads: list[tuple[str, bytes]]) -> dict:
        import io
        import re
        import zipfile

        def natural_sort_key(s: str):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

        raw_dir = RAW_DIR / chapter_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        extracted_files = []
        for filename, data in uploads:
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
                            if name.startswith("__MACOSX/") or name.startswith("."):
                                continue
                            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                                img_bytes = z.read(name)
                                clean_name = Path(name).name
                                if clean_name and len(img_bytes) > 0:
                                    extracted_files.append((clean_name, img_bytes))
                except Exception as e:
                    logger.warning(f"Failed to extract zip {filename}: {e}")
            else:
                extracted_files.append((filename, data))

        extracted_files.sort(key=lambda x: natural_sort_key(x[0]))

        raw_paths = []
        for idx, (filename, data) in enumerate(extracted_files):
            try:
                fmt = validate_upload_image(data, filename)
            except Exception as e:
                logger.warning(f"Skip invalid upload {filename}: {e}")
                continue
            ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp"}
            ext = ext_map.get(fmt, ".png")
            out_path = raw_dir / f"page_{idx:03d}{ext}"
            out_path.write_bytes(data)
            raw_paths.append(out_path)

        if not raw_paths:
            raise ValueError("No valid images in upload")

        logger.info(f"Chapter {chapter_id}: created from {len(raw_paths)} uploaded images")
        return self._build_chapter_from_raw_paths(chapter_id, raw_paths, source_url=None)

    def _build_chapter_from_raw_paths(self, chapter_id: str, raw_paths: list[Path], source_url: str | None) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id
        processed_dir.mkdir(parents=True, exist_ok=True)

        slice_results: list[list[Path]] = [[] for _ in raw_paths]

        if raw_paths:
            max_workers = min(2, len(raw_paths))

            def _slice_one(item):
                i, p = item
                return i, slice_image(p, processed_dir)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_slice_one, (i, p)) for i, p in enumerate(raw_paths)]
                for fut in as_completed(futures):
                    i, paths = fut.result()
                    slice_results[i] = paths

        pages = []
        for source_index, paths in enumerate(slice_results):
            for slice_index, slice_path in enumerate(paths):
                pages.append({
                    "original": slice_path.as_posix(),
                    "clean": None,
                    "boxes": [],
                    "skipped": False,
                    "rendered": False,
                    "source_index": source_index,
                    "slice_index": slice_index,
                })

        if not pages:
            for p in raw_paths:
                pages.append({
                    "original": p.as_posix(),
                    "clean": None,
                    "boxes": [],
                    "skipped": False,
                    "rendered": False,
                })

        manifest = {
            "chapter_id": chapter_id,
            "source_url": source_url or "",
            "pages": pages,
        }
        with self._manifest_lock(chapter_id):
            self._save_manifest(processed_dir, manifest)
        return manifest

    def process_pages(self, chapter_id: str, page_indices: list[int]) -> dict:
        with self._manifest_lock(chapter_id):
            processed_dir = PROCESSED_DIR / chapter_id
            manifest = self._load_manifest(processed_dir)
            work_items = []
            for idx in page_indices:
                if idx < 0 or idx >= len(manifest["pages"]):
                    continue
                page = manifest["pages"][idx]
                if page.get("skipped"):
                    continue
                work_items.append((idx, Path(page["original"])))

            if work_items:
                max_workers = min(2, len(work_items))

                def _process_one(item: tuple[int, Path]) -> tuple[int, dict]:
                    i, p = item
                    return i, self._process_page(p, processed_dir)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_process_one, item) for item in work_items]
                    for fut in as_completed(futures):
                        i, result = fut.result()
                        manifest["pages"][i].update(result)
                        manifest["pages"][i]["skipped"] = False

            self._save_manifest(processed_dir, manifest)
            self._sync_output_dir(chapter_id, manifest, page_indices)
            return manifest

    def mark_skipped(self, chapter_id: str, page_indices: list[int], skipped: bool) -> dict:
        with self._manifest_lock(chapter_id):
            processed_dir = PROCESSED_DIR / chapter_id
            manifest = self._load_manifest(processed_dir)
            for idx in page_indices:
                if 0 <= idx < len(manifest["pages"]):
                    manifest["pages"][idx]["skipped"] = skipped
            self._save_manifest(processed_dir, manifest)
            self._sync_output_dir(chapter_id, manifest, page_indices)
            return manifest

    @staticmethod
    def _sync_output_dir(chapter_id: str, manifest: dict, page_indices: list[int] | None = None) -> None:
        from app.config import OUTPUT_DIR
        out_dir = OUTPUT_DIR / chapter_id
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = page_indices if page_indices is not None else range(len(manifest["pages"]))
        for idx in indices:
            if idx < 0 or idx >= len(manifest["pages"]):
                continue
            page = manifest["pages"][idx]
            dest = out_dir / f"page_{idx:03d}.png"
            if page.get("skipped"):
                src = Path(page["original"])
            elif page.get("clean"):
                src = Path(page["clean"])
            else:
                src = Path(page["original"])
            if src.exists():
                shutil.copyfile(src, dest)

    def add_manual_box(self, chapter_id: str, page_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        with self._manifest_lock(chapter_id):
            processed_dir = PROCESSED_DIR / chapter_id
            manifest = self._load_manifest(processed_dir)
            page = manifest["pages"][page_index]
            page.setdefault("boxes", []).append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": 1.0, "mask": None, "manual": True,
            })
            img = read_image(Path(page["original"]))
            self._reinpaint_page(page, img, processed_dir, Path(page["original"]))
            self._save_manifest(processed_dir, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def remove_box(self, chapter_id: str, page_index: int, box_index: int) -> dict:
        with self._manifest_lock(chapter_id):
            processed_dir = PROCESSED_DIR / chapter_id
            manifest = self._load_manifest(processed_dir)
            page = manifest["pages"][page_index]
            if 0 <= box_index < len(page.get("boxes", [])):
                page["boxes"][box_index]["removed"] = True
                img = read_image(Path(page["original"]))
                self._reinpaint_page(page, img, processed_dir, Path(page["original"]))
            self._save_manifest(processed_dir, manifest)
            self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def repaint_mask(self, chapter_id: str, page_index: int, mask_png: bytes) -> dict:
        with self._manifest_lock(chapter_id):
            processed_dir = PROCESSED_DIR / chapter_id
            manifest = self._load_manifest(processed_dir)
            page = manifest["pages"][page_index]
            arr = np.frombuffer(mask_png, dtype=np.uint8)
            mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Invalid mask image")
            image = read_image(Path(page["original"]))
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            result = image.copy()
            num_labels, labels = cv2.connectedComponents((mask > 127).astype(np.uint8))
            for label in range(1, num_labels):
                component_mask = ((labels == label).astype(np.uint8) * 255)
                if component_mask.sum() // 255 < 100:
                    continue
                result = self.inpainter.inpaint_mask(result, component_mask)

            img_path = Path(page["original"])
            out_path = processed_dir / f"clean_{img_path.name}"
            write_image(out_path, result)
            page["clean"] = out_path.as_posix()
            page["rendered"] = False

            self._save_manifest(processed_dir, manifest)
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

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)
        page["clean"] = clean_path.as_posix()
        page["rendered"] = False

    @staticmethod
    def _load_manifest(processed_dir: Path) -> dict:
        manifest_path = processed_dir / "manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_manifest(processed_dir: Path, manifest: dict) -> None:
        tmp_path = processed_dir / "manifest.json.tmp"
        final_path = processed_dir / "manifest.json"
        tmp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, final_path)

    def _process_page(self, img_path: Path, processed_dir: Path) -> dict:
        image = read_image(img_path)
        boxes = self.detector.detect(image)
        clean_image = self.inpainter.inpaint(image, boxes)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)

        logger.debug(f"Processed {img_path.name}: {len(boxes)} boxes detected")
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
