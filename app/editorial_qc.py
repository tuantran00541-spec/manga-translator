from __future__ import annotations

from pathlib import Path
import hashlib

import app.config as config
from app.editorial_layout import measure_text_layout
from app.render.identity import render_artifact_is_current

SCRIPT_STATUS_DRAFT = "draft"
SCRIPT_STATUS_REVIEWED = "reviewed"
SCRIPT_STATUS_SKIP = "skip"
SCRIPT_STATUSES = {SCRIPT_STATUS_DRAFT, SCRIPT_STATUS_REVIEWED, SCRIPT_STATUS_SKIP}


def script_review_fingerprint(obj: dict) -> str:
    source = str(obj.get("ocr_text") or "")
    translation = str(obj.get("translation") or "")
    payload = f"{source}\0{translation}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_script_status(value: object) -> str:
    value = str(value or SCRIPT_STATUS_DRAFT).strip().lower()
    return value if value in SCRIPT_STATUSES else SCRIPT_STATUS_DRAFT


def effective_script_status(obj: dict) -> str:
    status = normalize_script_status(obj.get("script_status"))
    if status == SCRIPT_STATUS_REVIEWED:
        expected = script_review_fingerprint(obj)
        if str(obj.get("script_review_fingerprint") or "") != expected:
            return SCRIPT_STATUS_DRAFT
    return status


def _issue(code: str, message: str, *, object_id: str | None = None, severity: str = "error") -> dict:
    item = {"code": code, "message": message, "severity": severity}
    if object_id:
        item["object_id"] = object_id
    return item


def _valid_region(obj: dict, page: dict) -> tuple[bool, tuple[int, int, int, int] | None]:
    region = obj.get("region") or {}
    try:
        x1, y1, x2, y2 = (int(region[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False, None
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return False, None
    width = page.get("width")
    height = page.get("height")
    if width is not None and x2 > int(width):
        return False, None
    if height is not None and y2 > int(height):
        return False, None
    return True, (x1, y1, x2, y2)


def page_editorial_issues(manifest: dict, page_index: int, *, output_dir: Path | None = None) -> list[dict]:
    pages = manifest.get("pages") or []
    if page_index < 0 or page_index >= len(pages):
        return [_issue("invalid_page", "Trang không tồn tại.")]
    page = pages[page_index]
    if page.get("skipped"):
        return []

    issues: list[dict] = []
    has_cleanup_warning = bool(
        page.get("needs_review")
        or page.get("detection_state") == "needs_review"
        or (page.get("detection_issues") or [])
    )
    clean_revision = int(page.get("clean_revision") or 0)
    clean_approved_revision = int(page.get("clean_review_approved_revision") or 0)
    if has_cleanup_warning and (clean_revision <= 0 or clean_approved_revision != clean_revision):
        issues.append(_issue("cleanup_review", "Trang vẫn còn cảnh báo từ bước làm sạch cần được người biên tập xác nhận."))

    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        object_id = str(obj["id"])
        status = effective_script_status(obj)
        source_text = str(obj.get("ocr_text") or "").strip()
        translation = str(obj.get("translation") or "").strip()

        if obj.get("source_missing"):
            issues.append(_issue("source_missing", "Vùng chữ không còn khớp với vùng nhận diện nguồn.", object_id=object_id))

        if status == SCRIPT_STATUS_SKIP:
            continue

        if not source_text and obj.get("auto_generated"):
            issues.append(_issue("missing_source_text", "Vùng chữ tự động chưa có nội dung OCR/source.", object_id=object_id))
        if not translation:
            issues.append(_issue("missing_translation", "Vùng chữ chưa có bản dịch.", object_id=object_id))
        if status != SCRIPT_STATUS_REVIEWED:
            issues.append(_issue("script_unreviewed", "Bản dịch chưa được đánh dấu đã soát.", object_id=object_id))

        valid_region, box = _valid_region(obj, page)
        if not valid_region or box is None:
            issues.append(_issue("invalid_geometry", "Khung chữ có tọa độ không hợp lệ.", object_id=object_id))
            continue
        if translation:
            style = obj.get("style") or {}
            layout = measure_text_layout(
                translation,
                box,
                font_size=style.get("fontSize", "auto"),
                font_name=str(style.get("font") or "default"),
                stroke_width=style.get("strokeWidth", "auto"),
            )
            if not layout.get("fits"):
                issues.append(
                    _issue(
                        "text_overflow",
                        f"Bản dịch không vừa khung chữ ở cỡ {layout.get('font_size', 0)}px.",
                        object_id=object_id,
                    )
                )

    root = output_dir or config.OUTPUT_DIR
    output_path = root / str(manifest.get("chapter_id") or "") / f"page_{page_index:03d}.png"
    if not render_artifact_is_current(manifest, page_index, output_path):
        issues.append(_issue("render_stale", "Trang chưa được kết xuất hoặc bản kết xuất đã cũ."))

    return issues


def build_final_qc_report(manifest: dict, *, output_dir: Path | None = None) -> dict:
    pages_report: list[dict] = []
    required = 0
    approved = 0
    blocking = 0

    for page_index, page in enumerate(manifest.get("pages") or []):
        if page.get("skipped"):
            pages_report.append(
                {
                    "page_index": page_index,
                    "skipped": True,
                    "approved": True,
                    "render_revision": int(page.get("render_revision") or 0),
                    "issues": [],
                }
            )
            continue

        required += 1
        issues = page_editorial_issues(manifest, page_index, output_dir=output_dir)
        render_revision = int(page.get("render_revision") or 0)
        approved_revision = int(page.get("final_qc_approved_render_revision") or 0)
        is_approved = bool(render_revision > 0 and approved_revision == render_revision and not issues)
        if is_approved:
            approved += 1
        blocking += len([issue for issue in issues if issue.get("severity") == "error"])
        pages_report.append(
            {
                "page_index": page_index,
                "skipped": False,
                "approved": is_approved,
                "render_revision": render_revision,
                "approved_render_revision": approved_revision,
                "issues": issues,
            }
        )

    ready = required > 0 and approved == required and blocking == 0
    if required == 0:
        ready = True
    return {
        "chapter_id": str(manifest.get("chapter_id") or ""),
        "ready_for_export": ready,
        "summary": {
            "pages_total": len(manifest.get("pages") or []),
            "pages_required": required,
            "pages_approved": approved,
            "blocking_issues": blocking,
        },
        "pages": pages_report,
    }
