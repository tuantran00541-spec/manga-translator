from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np

from app.security import MAX_IMAGE_PIXELS


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
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large at {path}: {width}x{height}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    success, buffer = cv2.imencode(extension, image)
    if success:
        with path.open("wb") as output_file:
            buffer.tofile(output_file)
            output_file.flush()
            _advise_file_cache_drop(output_file)
        return
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image at {path}")
    try:
        with path.open("rb") as output_file:
            _advise_file_cache_drop(output_file)
    except OSError:
        pass


def encode_mask(mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    success, buffer = cv2.imencode(".png", mask)
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")
