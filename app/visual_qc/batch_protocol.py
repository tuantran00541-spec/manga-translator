from __future__ import annotations

from dataclasses import dataclass
import math

from app.visual_qc.regions import QCRegion

_ALLOWED_ISSUE_TYPES = {
    "residual_text", "partial_erase", "partial_text", "smear", "inpaint_artifact",
    "over_erased_art", "suspicious_fill", "unknown",
}
_ALLOWED_ACTIONS = {"repaint", "review", "review_original", "deep_qc", "none"}
_ALLOWED_STATUSES = {"pass", "flagged", "ambiguous"}


@dataclass(frozen=True)
class RegionBatchIssue:
    page_index: int
    region_id: str
    issue_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    reason: str
    recommended_action: str


@dataclass(frozen=True)
class RegionBatchDecision:
    page_index: int
    region_id: str
    status: str
    issues: tuple[RegionBatchIssue, ...]


def _map_relative_box(box_2d: object, region: QCRegion) -> tuple[int, int, int, int] | None:
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None
    try:
        values = [float(v) for v in box_2d]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, v)) for v in values]
    if ymax <= ymin or xmax <= xmin:
        return None
    rx1, ry1, rx2, ry2 = region.bbox
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    x1 = int(round(rx1 + xmin / 1000.0 * rw))
    y1 = int(round(ry1 + ymin / 1000.0 * rh))
    x2 = int(round(rx1 + xmax / 1000.0 * rw))
    y2 = int(round(ry1 + ymax / 1000.0 * rh))
    x1 = max(rx1, min(rx2 - 1, x1))
    y1 = max(ry1, min(ry2 - 1, y1))
    x2 = max(x1 + 1, min(rx2, x2))
    y2 = max(y1 + 1, min(ry2, y2))
    return x1, y1, x2, y2


def _parse_issue(raw_issue: object, region: QCRegion) -> RegionBatchIssue | None:
    if not isinstance(raw_issue, dict):
        return None
    issue_type = str(raw_issue.get("issue_type") or "")
    if issue_type not in _ALLOWED_ISSUE_TYPES:
        return None
    try:
        confidence = float(raw_issue.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    bbox = _map_relative_box(raw_issue.get("box_2d"), region)
    if bbox is None:
        return None
    action = str(raw_issue.get("recommended_action") or "review")
    if action not in _ALLOWED_ACTIONS:
        action = "review"
    return RegionBatchIssue(
        region.page_index,
        region.region_id,
        issue_type,
        max(0.0, min(1.0, confidence)),
        bbox,
        str(raw_issue.get("reason") or "")[:500],
        action,
    )


def _parse_region_issues(raw_region: dict, region: QCRegion) -> tuple[RegionBatchIssue, ...]:
    return tuple(
        issue
        for raw_issue in (raw_region.get("issues") or [])
        if (issue := _parse_issue(raw_issue, region)) is not None
    )


def parse_region_batch_decisions(parsed: dict, regions_by_id: dict[str, QCRegion]) -> list[RegionBatchDecision]:
    if not isinstance(parsed, dict):
        return []
    out: list[RegionBatchDecision] = []
    for raw_region in parsed.get("regions") or []:
        if not isinstance(raw_region, dict):
            continue
        region_id = str(raw_region.get("region_id") or "")
        region = regions_by_id.get(region_id)
        if region is None:
            continue
        status = str(raw_region.get("status") or "ambiguous")
        if status not in _ALLOWED_STATUSES:
            status = "ambiguous"
        issues = _parse_region_issues(raw_region, region)
        if issues and status == "pass":
            status = "flagged"
        out.append(RegionBatchDecision(region.page_index, region.region_id, status, issues))
    return out
