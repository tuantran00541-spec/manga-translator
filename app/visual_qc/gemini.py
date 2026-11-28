from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from app.security import MAX_IMAGE_PIXELS
from app.visual_qc.gemini_interactions import DEFAULT_GEMINI_MODEL

__all__ = ["DEFAULT_GEMINI_MODEL", "_encode_for_gemini", "_read_image"]

MAX_QC_SIDE = 2048


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    h, w = image.shape[:2]
    if w * h > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large at {path}: {w}x{h}")
    return image


def _encode_for_gemini(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, MAX_QC_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not encode page for Gemini")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_image_strips(image: np.ndarray, label: str) -> list[tuple[str, str]]:
    """Encode overlapping vertical strips while retaining original page coordinates."""
    height, width = image.shape[:2]
    if height <= 2400:
        return [(label, _encode_for_gemini(image))]
    strip_height = max(512, min(2048, round(width * 2.0)))
    overlap = min(384, strip_height // 4)
    parts = []
    top = 0
    while top < height:
        bottom = min(height, top + strip_height)
        strip_label = f"{label} (original y={top}..{bottom}, original size {width}x{height})"
        parts.append((strip_label, _encode_for_gemini(image[top:bottom])))
        if bottom == height:
            break
        top = bottom - overlap
    return parts
