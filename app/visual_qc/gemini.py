from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from app.security import MAX_IMAGE_PIXELS

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_VISUAL_QC_MODEL", "gemini-3.7-flash")
DEFAULT_TIMEOUT_SECONDS = 60
MAX_QC_SIDE = 2048


@dataclass(frozen=True)
class VisualQCIssue:
    issue_type: str
    confidence: float
    label: str
    box_2d: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "residual_text",
                            "partial_text",
                            "inpaint_artifact",
                            "over_erased_art",
                        ],
                    },
                    "confidence": {"type": "number"},
                    "label": {"type": "string"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "mask": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                },
                "required": ["issue_type", "confidence", "label", "box_2d", "mask"],
            },
        }
    },
    "required": ["issues"],
}

_PROMPT = """
You are a visual quality-control inspector for comic/manga/manhua/webtoon text removal.
You receive two images of the SAME page in this order:
1) ORIGINAL: before text removal/inpainting.
2) CLEANED: after the application's text removal/inpainting.

Find ONLY regions in CLEANED that genuinely need another inpaint pass.
Prioritize:
- residual_text: source-language glyphs/letters still clearly visible where text was intended to be removed;
- partial_text: fragments/strokes of removed text left behind;
- inpaint_artifact: obvious smears, repeated texture, broken fills, or unnatural reconstruction caused by removal;
- over_erased_art: artwork visibly damaged by the removal operation. Mark only the damaged patch that should be repaired.

Do NOT flag normal character line art, screentones, speed lines, borders, decorative patterns, or legitimate artwork merely because they resemble text.
Be conservative. If unsure, omit the issue.
For every issue return:
- box_2d as [ymin, xmin, ymax, xmax], normalized to 0..1000 over the whole CLEANED image;
- mask as a polygon of [x, y] points normalized to 0..1000 INSIDE that box, tightly covering only pixels that should be repainted;
- confidence from 0.0 to 1.0.
Return an empty issues array when the page looks clean.
""".strip()


class GeminiVisualQC:
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        self.model = model
        self.timeout_seconds = timeout_seconds

    def inspect(self, original_path: Path, cleaned_path: Path, api_key: str) -> list[VisualQCIssue]:
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is not configured")
        original = _read_image(original_path)
        cleaned = _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned images must have the same dimensions")

        original_b64 = _encode_for_gemini(original)
        cleaned_b64 = _encode_for_gemini(cleaned)
        payload = {
            "model": self.model,
            "store": False,
            "input": [
                {"type": "text", "text": _PROMPT},
                {"type": "text", "text": "ORIGINAL image:"},
                {"type": "image", "data": original_b64, "mime_type": "image/jpeg"},
                {"type": "text", "text": "CLEANED image to inspect:"},
                {"type": "image", "data": cleaned_b64, "mime_type": "image/jpeg"},
            ],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": _RESPONSE_SCHEMA,
            },
            "generation_config": {"thinking_level": "low"},
        }

        try:
            response = requests.post(
                GEMINI_INTERACTIONS_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key.strip(),
                    "Api-Revision": "2026-05-20",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        if not response.ok:
            detail = _safe_error_detail(response)
            raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {detail}")

        try:
            body = response.json()
            raw_text = _extract_output_text(body)
            parsed = json.loads(raw_text)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Gemini returned an invalid structured response") from exc

        h, w = cleaned.shape[:2]
        return _parse_issues(parsed, w, h)


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


def _extract_output_text(body: dict) -> str:
    for step in reversed(body.get("steps") or []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content") or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("No text model output found")


def _safe_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
            if payload.get("message"):
                return str(payload["message"])[:500]
    except ValueError:
        pass
    text = (response.text or "").strip()
    return text[:500] if text else "unknown error"


def _parse_issues(parsed: dict, width: int, height: int) -> list[VisualQCIssue]:
    allowed_types = {"residual_text", "partial_text", "inpaint_artifact", "over_erased_art"}
    out: list[VisualQCIssue] = []
    for raw in parsed.get("issues") or []:
        if not isinstance(raw, dict):
            continue
        issue_type = raw.get("issue_type")
        if issue_type not in allowed_types:
            continue
        try:
            confidence_raw = float(raw.get("confidence", 0.0))
            box = [float(v) for v in raw.get("box_2d", [])]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence_raw) or len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        confidence = max(0.0, min(1.0, confidence_raw))
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, v)) for v in box]
        if ymax <= ymin or xmax <= xmin:
            continue

        x1 = int(round(xmin / 1000.0 * width))
        y1 = int(round(ymin / 1000.0 * height))
        x2 = int(round(xmax / 1000.0 * width))
        y2 = int(round(ymax / 1000.0 * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        polygon: list[tuple[int, int]] = []
        for point in raw.get("mask") or []:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                mx = float(point[0])
                my = float(point[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mx) or not math.isfinite(my):
                continue
            mx = max(0.0, min(1000.0, mx))
            my = max(0.0, min(1000.0, my))
            px = int(round(x1 + (mx / 1000.0) * (x2 - x1)))
            py = int(round(y1 + (my / 1000.0) * (y2 - y1)))
            polygon.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
        if len(polygon) < 3:
            continue

        out.append(
            VisualQCIssue(
                issue_type=issue_type,
                confidence=confidence,
                label=str(raw.get("label") or issue_type)[:200],
                box_2d=(y1, x1, y2, x2),
                polygon=polygon,
            )
        )
    return out
