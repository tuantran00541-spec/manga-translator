from __future__ import annotations

import base64
from typing import Any

import cv2

from app.visual_qc.contact_sheet import ContactSheet

_ALLOWED_MODES = {"global-clean", "region-clean", "region-pair"}

_REGION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "flagged", "ambiguous"]},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_type": {"type": "string", "enum": ["residual_text", "partial_erase", "smear", "over_erased_art", "suspicious_fill", "unknown"]},
                                "confidence": {"type": "number"},
                                "box_2d": {"type": "array", "items": {"type": "integer"}},
                                "reason": {"type": "string"},
                                "recommended_action": {"type": "string", "enum": ["repaint", "review", "review_original", "deep_qc", "none"]},
                            },
                            "required": ["issue_type", "confidence", "box_2d", "reason", "recommended_action"],
                        },
                    },
                },
                "required": ["region_id", "status", "issues"],
            },
        }
    },
    "required": ["regions"],
}


def _encode_jpeg(image) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not encode visual QC contact sheet")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _prompt(sheet: ContactSheet, mode: str) -> str:
    ids = ", ".join(item.region_id for item in sheet.items)
    common = f"""
You are a visual quality-control inspector for comic/manga/manhua/webtoon text removal.
The image is a contact sheet. Inspect every labeled region exactly once.
Expected region IDs: {ids}.
Return one structured entry for each known region_id and never invent IDs.
For each issue, box_2d is [ymin, xmin, ymax, xmax] normalized 0..1000 RELATIVE TO THAT REGION'S CLEAN crop, not the whole contact sheet.
Use status pass when the region is clean, flagged when a real issue exists, and ambiguous when the available evidence is insufficient.
Be conservative: do not flag legitimate line art, screentones, borders, speed lines, decorative patterns, or intentional colored/stylized artwork.
""".strip()
    if mode in {"global-clean", "region-clean"}:
        scope = """
Each labeled panel is a CLEAN image only. Look for visible residual source text, partial erasure, obvious smear/repeated texture, or suspicious fill discontinuity.
Because ORIGINAL is not provided, do not claim over_erased_art from clean-only evidence. If art damage is plausible but cannot be established, return status ambiguous and recommended_action deep_qc.
""".strip()
    else:
        scope = """
Each labeled cell contains ORIGINAL on the left and CLEAN on the right for the same region. Compare them directly.
Detect residual text, partial erasure, smear/inpaint artifacts, suspicious fills, and over_erased_art where valid artwork was removed or damaged.
Coordinates must still be normalized relative to the CLEAN crop only.
""".strip()
    return common + "\n\n" + scope


def build_region_batch_payload(sheet: ContactSheet, *, model: str, mode: str) -> dict:
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported visual QC batch mode: {mode}")
    if not sheet.items:
        raise ValueError("Contact sheet has no region mapping")
    return {
        "model": model,
        "store": False,
        "input": [
            {"type": "text", "text": _prompt(sheet, mode)},
            {"type": "image", "data": _encode_jpeg(sheet.image), "mime_type": "image/jpeg"},
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": _REGION_RESPONSE_SCHEMA},
        "generation_config": {"thinking_level": "low"},
    }
