from __future__ import annotations

import base64
import binascii
import os
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR
from app.logging_config import logger

MASK_REF_PREFIX = "@mask:"
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _processed_root() -> Path:
    return PROCESSED_DIR.resolve()


def _validated_mask_path(value: str) -> Path | None:
    if not isinstance(value, str) or not value.startswith(MASK_REF_PREFIX):
        return None
    rel = value[len(MASK_REF_PREFIX):].strip()
    if not rel:
        return None
    try:
        candidate = (_processed_root() / Path(rel)).resolve()
        if not candidate.is_relative_to(_processed_root()):
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def is_mask_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(MASK_REF_PREFIX)


def decode_mask_value(value) -> np.ndarray | None:
    """Decode either a legacy base64 PNG mask or a managed sidecar reference."""
    if not value:
        return None

    raw: bytes
    if is_mask_ref(value):
        path = _validated_mask_path(value)
        if path is None or not path.is_file() or path.is_symlink():
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None
    elif isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, TypeError, binascii.Error):
            return None
    else:
        return None

    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        return None
    return mask


def _legacy_png_bytes(value) -> bytes | None:
    if not isinstance(value, str) or not value or is_mask_ref(value):
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError, binascii.Error):
        return None
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.ndim != 2:
        return None
    return raw


def _ref_for_path(path: Path) -> str:
    relative = path.resolve().relative_to(_processed_root())
    return MASK_REF_PREFIX + relative.as_posix()


def _safe_box_id(box: dict, fallback_index: int) -> str:
    raw = str(box.get("id") or f"box_{fallback_index:04d}")
    safe = _SAFE_ID_RE.sub("_", raw).strip("._")
    return (safe or f"box_{fallback_index:04d}")[:128]


def externalize_page_masks(
    processed_dir: Path,
    page_index: int,
    boxes: list[dict],
) -> int:
    """Move legacy base64 masks out of manifest state into atomic PNG sidecars.

    The function is deliberately best-effort: an I/O failure leaves the original
    base64 value untouched so correctness is never traded for manifest size.
    Existing managed references are retained. Orphan PNGs in this page's sidecar
    directory are removed only after the current live reference set is known.
    """
    try:
        chapter_dir = processed_dir.resolve()
        if not chapter_dir.is_relative_to(_processed_root()):
            return 0
    except (OSError, RuntimeError, ValueError):
        return 0

    mask_dir = processed_dir / "masks" / f"page_{int(page_index):03d}"
    live_paths: set[Path] = set()
    externalized = 0

    for fallback_index, box in enumerate(boxes):
        if not isinstance(box, dict):
            continue
        value = box.get("mask")
        if not value:
            continue

        if is_mask_ref(value):
            existing = _validated_mask_path(value)
            if existing is not None and existing.is_file() and not existing.is_symlink():
                live_paths.add(existing)
            else:
                # A broken reference must fail safe: no segmentation evidence is
                # preferable to silently treating geometry as a destructive mask.
                box["mask"] = None
            continue

        raw = _legacy_png_bytes(value)
        if raw is None:
            box["mask"] = None
            continue

        try:
            mask_dir.mkdir(parents=True, exist_ok=True)
            safe_id = _safe_box_id(box, fallback_index)
            final_path = mask_dir / f"{safe_id}.png"
            tmp_path = mask_dir / f".{safe_id}.{uuid.uuid4().hex[:10]}.tmp.png"
            tmp_path.write_bytes(raw)
            os.replace(tmp_path, final_path)
            box["mask"] = _ref_for_path(final_path)
            live_paths.add(final_path.resolve())
            externalized += 1
        except OSError as exc:
            logger.warning(
                "Could not externalize mask for page %s box %s: %s",
                page_index,
                box.get("id"),
                exc,
            )
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    if mask_dir.is_dir():
        for candidate in mask_dir.glob("*.png"):
            try:
                resolved = candidate.resolve()
                if resolved not in live_paths and not candidate.is_symlink():
                    candidate.unlink()
            except OSError:
                continue
        try:
            if not any(mask_dir.iterdir()):
                mask_dir.rmdir()
        except OSError:
            pass

    return externalized
