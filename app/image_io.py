from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image

from app.security import MAX_IMAGE_PIXELS


def _advise_file_cache_drop(file_obj) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        os.posix_fadvise(file_obj.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError, ValueError):
        pass


def _check_header_dimensions(raw_bytes: bytes, path: Path) -> None:
    """Inspect image header dimensions before OpenCV allocates full pixel buffers."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image too large at {path}") from exc
    except Exception:
        # Let OpenCV attempt decoding formats PIL does not handle; OpenCV's
        # post-decode pixel check still guards the final array size.
        return
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions at {path}: {width}x{height}")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large at {path}: {width}x{height}")


def read_image(path: Path) -> np.ndarray:
    with path.open("rb") as source_file:
        raw_bytes = source_file.read()
        _advise_file_cache_drop(source_file)
    if not raw_bytes:
        raise ValueError(f"Could not read empty image at {path}")
    _check_header_dimensions(raw_bytes, path)
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large at {path}: {width}x{height}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError(f"Cannot write empty image to {path}")
    extension = path.suffix or ".png"
    params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if extension.lower() == ".png" else []
    success, buffer = cv2.imencode(extension, image, params)
    if not success:
        raise ValueError(f"Could not encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as output_file:
            output_file.write(memoryview(buffer))
            output_file.flush()
            _advise_file_cache_drop(output_file)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def encode_mask(mask: np.ndarray | None) -> str | None:
    if mask is None or getattr(mask, "size", 0) == 0:
        return None
    success, buffer = cv2.imencode(".png", mask)
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")
