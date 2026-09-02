import copy
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from fastapi import HTTPException
from PIL import Image
from filelock import FileLock, Timeout

from app.config import PROCESSED_DIR
from app.inpaint.mask_geometry import reconcile_detector_geometry_override
from app.mask_store import externalize_page_masks, prune_page_masks
from app.ocr.identity import OCR_CACHE_FIELDS, clear_ocr_cache, geometry_signature
from app.parameters import (
    DETECTOR_STABLE_ID_IOU_MIN,
    MANIFEST_LOCK_TIMEOUT_SECONDS,
    PAGE_LOCK_TIMEOUT_SECONDS,
    STALE_TEMP_MAX_AGE_SECONDS,
)
from app.security import validate_chapter_id

MANIFEST_SCHEMA_VERSION = 3

_STALE_TEMP_PATTERNS = (
    "manifest.json.*.tmp",
    "clean_*.tmp.png",
    "auto_clean_*.tmp.png",
    "manual_mask_*.tmp.png",
    "manual_lama_mask_*.tmp.png",
    "page_*.rendering.*.tmp",
    "masks/**/*.tmp.png",
)


def cleanup_stale_temp_artifacts(
    directory: Path,
    *,
    max_age_seconds: float = STALE_TEMP_MAX_AGE_SECONDS,
    now: float | None = None,
) -> int:
    """Remove only known orphan temp files older than the safety window."""
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be >= 0")
    if not directory.exists() or not directory.is_dir():
        return 0

    cutoff = (time.time() if now is None else float(now)) - float(max_age_seconds)
    removed = 0
    seen: set[Path] = set()
    for pattern in _STALE_TEMP_PATTERNS:
        for path in directory.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_mtime > cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def new_box_id() -> str:
    return f"box_{uuid.uuid4().hex[:16]}"


def _legacy_box_id(chapter_id: str, page_index: int, box_index: int, box: dict) -> str:
    fingerprint = {
        "chapter_id": chapter_id,
        "page_index": page_index,
        "box_index": box_index,
        "x1": box.get("x1"), "y1": box.get("y1"),
        "x2": box.get("x2"), "y2": box.get("y2"),
        "manual": bool(box.get("manual", False)),
    }
    raw = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"box_{hashlib.sha256(raw).hexdigest()[:16]}"


def _try_image_dimensions(path_value: str | None) -> tuple[int, int] | None:
    if not path_value:
        return None
    try:
        with Image.open(Path(path_value)) as image:
            w, h = image.size
        if w > 0 and h > 0:
            return int(w), int(h)
    except (OSError, ValueError):
        return None
    return None


def _try_file_revision(path_value: str | None) -> tuple[int, int, int] | None:
    if not path_value:
        return None
    try:
        st = Path(path_value).stat()
        return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns))
    except OSError:
        return None


def _normalize_box_ocr_cache(
    box: dict,
    *,
    source_revision: int,
    original_revision: tuple[int, int, int] | None,
) -> bool:
    if not any(key in box for key in OCR_CACHE_FIELDS):
        return False
    if box.get("ocr_source") == "manual":
        return False
    valid = box.get("ocr_source") == "machine" and original_revision is not None
    if valid:
        try:
            cached_source = int(box.get("ocr_source_revision"))
            cached_file = tuple(int(value) for value in box.get("ocr_file_revision", ()))
            cached_geometry = tuple(int(value) for value in box.get("ocr_geometry", ()))
        except (TypeError, ValueError):
            valid = False
        else:
            valid = (
                bool(box.get("ocr_engine"))
                and cached_source == int(source_revision)
                and cached_file == original_revision
                and cached_geometry == geometry_signature(box)
            )
    if valid:
        return False
    clear_ocr_cache(box)
    return True


def normalize_manifest_schema(manifest: dict) -> bool:
    """Upgrade legacy manifests in memory without changing user-visible content."""
    changed = False
    if int(manifest.get("schema_version") or 0) < MANIFEST_SCHEMA_VERSION:
        manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
        changed = True

    chapter_id = str(manifest.get("chapter_id") or "")
    for page_index, page in enumerate(manifest.get("pages", [])):
        if not isinstance(page, dict):
            continue

        if page.get("width") is None or page.get("height") is None:
            dims = _try_image_dimensions(page.get("original"))
            if dims is not None:
                page["width"], page["height"] = dims
                changed = True

        # Older skip handling stored the raw original in ``clean``. The two
        # fields have distinct managed roots, so normalize that legacy sentinel
        # to the absence of a processed artifact.
        if page.get("skipped") and page.get("clean") == page.get("original"):
            page["clean"] = None
            changed = True

        revision_defaults = {
            "source_revision": 1,
            "process_revision": 1 if (page.get("clean") or page.get("boxes")) else 0,
            "clean_revision": 1 if page.get("clean") else 0,
            "render_revision": 1 if page.get("rendered") else 0,
        }
        for key, default in revision_defaults.items():
            if page.get(key) is None:
                page[key] = default
                changed = True

        source_revision = int(page.get("source_revision") or 0)
        boxes = page.get("boxes") or []
        has_ocr_cache = any(
            isinstance(box, dict) and any(key in box for key in OCR_CACHE_FIELDS)
            for box in boxes
        )
        original_revision = (
            _try_file_revision(page.get("original")) if has_ocr_cache else None
        )
        for box_index, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue
            if not box.get("id"):
                box["id"] = _legacy_box_id(chapter_id, page_index, box_index, box)
                changed = True
            expected_origin = "manual" if box.get("manual") else "detector"
            if box.get("origin") not in ("manual", "detector"):
                box["origin"] = expected_origin
                changed = True
            if _normalize_box_ocr_cache(
                box,
                source_revision=source_revision,
                original_revision=original_revision,
            ):
                changed = True

        box_ids = [b.get("id") if isinstance(b, dict) else None for b in boxes]
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            refs = obj.get("source_boxes")
            if not isinstance(refs, list):
                continue
            migrated: list[str] = []
            for ref in refs:
                if isinstance(ref, int):
                    if 0 <= ref < len(box_ids) and box_ids[ref]:
                        migrated.append(str(box_ids[ref]))
                elif isinstance(ref, str) and ref:
                    migrated.append(ref)
            if refs != migrated:
                obj["source_boxes"] = migrated
                changed = True
    return changed


def bump_page_revision(page: dict, field: str) -> int:
    if field not in {"source_revision", "process_revision", "clean_revision", "render_revision"}:
        raise ValueError(f"Unsupported page revision field: {field}")
    try:
        current = int(page.get(field) or 0)
    except (TypeError, ValueError):
        current = 0
    page[field] = current + 1
    return page[field]


def _box_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = (float(a.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(b.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assign_stable_detector_box_ids(
    new_boxes: list[dict],
    existing_boxes: list[dict],
    min_iou: float = DETECTOR_STABLE_ID_IOU_MIN,
) -> list[dict]:
    """Assign stable detector IDs and reconcile persisted geometry overrides.

    During the pre-inpaint call, detector records carry ``_mask_array`` and
    ``existing_boxes`` is a job-local deep copy. At that point a matched geometry
    override can safely remap the fresh segmentation mask into page-space edited
    geometry and neutralize the legacy branch that used to discard the mask.
    """
    candidates = [
        b for b in existing_boxes
        if isinstance(b, dict) and b.get("origin", "manual" if b.get("manual") else "detector") == "detector" and b.get("id")
    ]
    used: set[str] = set()
    for box in new_boxes:
        box["origin"] = "detector"
        if box.get("id"):
            used.add(str(box["id"]))
            continue
        best = None
        best_iou = min_iou
        for old in candidates:
            old_id = str(old.get("id"))
            if old_id in used:
                continue
            match_geometry = old.get("detector_anchor") if isinstance(old.get("detector_anchor"), dict) else old
            score = _box_iou(box, match_geometry)
            if score >= best_iou:
                best_iou = score
                best = old
        if best is not None:
            box["id"] = str(best["id"])
            reconcile_detector_geometry_override(box, best)
            used.add(str(best["id"]))
        else:
            box["id"] = new_box_id()
            used.add(str(box["id"]))
    return new_boxes


@contextmanager
def get_manifest_lock(chapter_id: str):
    lock_path = PROCESSED_DIR / chapter_id / "manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=MANIFEST_LOCK_TIMEOUT_SECONDS):
            yield
    except Timeout:
        raise RuntimeError(f"Manifest lock timeout for chapter {chapter_id}")


@contextmanager
def get_page_lock(chapter_id: str, page_index: int):
    lock_path = PROCESSED_DIR / chapter_id / f"page_{page_index:03d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=PAGE_LOCK_TIMEOUT_SECONDS):
            yield
    except Timeout:
        raise RuntimeError(f"Page lock timeout for chapter {chapter_id} page {page_index}")


def load_manifest_raw(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    manifest_path = PROCESSED_DIR / chapter_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"Chapter {chapter_id} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalize_manifest_schema(manifest)
    return manifest


def save_manifest_raw(chapter_id: str, manifest: dict) -> None:
    normalize_manifest_schema(manifest)
    processed_dir = PROCESSED_DIR / chapter_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = processed_dir / f"manifest.json.{uuid.uuid4().hex}.tmp"
    final_path = processed_dir / "manifest.json"
    created_sidecars: set[Path] = set()
    pages = manifest.get("pages", [])
    try:
        for page_index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            externalize_page_masks(
                processed_dir,
                page_index,
                page.get("boxes") or [],
                created_paths=created_sidecars,
            )

        with tmp_path.open("w", encoding="utf-8") as output:
            json.dump(
                manifest,
                output,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        for sidecar in created_sidecars:
            try:
                if sidecar.is_file() and not sidecar.is_symlink():
                    sidecar.unlink()
            except OSError:
                pass
        raise

    # The new manifest is now the source of truth. Pruning before os.replace()
    # could invalidate sidecar references held by the previous manifest.
    for page_index, page in enumerate(pages):
        if isinstance(page, dict):
            prune_page_masks(processed_dir, page_index, page.get("boxes") or [])


def urlify_manifest(manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    chapter_id = result["chapter_id"]
    for i, page in enumerate(result["pages"]):
        page["original"] = f"/api/image/{chapter_id}/{i}/original"
        if page.get("clean"):
            page["clean"] = f"/api/image/{chapter_id}/{i}/clean"
        for box in page.get("boxes", []):
            box.pop("mask", None)
    return result


def invalidate_page_render(manifest: dict, page_index: int) -> None:
    pages = manifest.get("pages", [])
    if 0 <= page_index < len(pages):
        pages[page_index]["rendered"] = False


def _get_manual_mask_state(
    processed_dir: Path, img_path: Path, *, force_lama: bool = False
) -> tuple[bool, int, int]:
    if not img_path.name:
        return (False, 0, 0)
    prefix = "manual_lama_mask" if force_lama else "manual_mask"
    mask_path = processed_dir / f"{prefix}_{img_path.name}"
    try:
        st = mask_path.stat()
        return (True, st.st_mtime_ns, st.st_size)
    except OSError:
        return (False, 0, 0)


def capture_processing_state(manifest: dict, page_index: int, processed_dir: Path) -> dict | None:
    """Capture snapshot of canonical inputs for page detection and inpainting.

    Derived outputs (such as 'clean') are deliberately excluded to prevent
    self-referential validation.
    """
    pages = manifest.get("pages", [])
    if page_index < 0 or page_index >= len(pages):
        return None
    page = pages[page_index]
    img_path = Path(page.get("original", ""))
    return {
        "original": page.get("original"),
        "skipped": page.get("skipped", False),
        "excluded_regions": copy.deepcopy(page.get("excluded_regions", [])),
        "boxes": copy.deepcopy(page.get("boxes", [])),
        "manual_mask": page.get("manual_mask"),
        "manual_lama_mask": page.get("manual_lama_mask"),
        "disk_mask_state": _get_manual_mask_state(processed_dir, img_path),
        "disk_lama_mask_state": _get_manual_mask_state(
            processed_dir, img_path, force_lama=True
        ),
    }


def is_processing_state_current(
    manifest: dict, page_index: int, snapshot: dict | None, processed_dir: Path
) -> bool:
    """Check whether a page's canonical processing inputs still match the snapshot."""
    if snapshot is None:
        return False
    current = capture_processing_state(manifest, page_index, processed_dir)
    return current == snapshot
