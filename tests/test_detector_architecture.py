from dataclasses import fields

import numpy as np

from app.detector.evidence import (
    AuthorityLevel,
    AuthorityPolicy,
    DetectionEvidence,
    EvidenceKind,
)
from app.detector.model_adapter import ModelCapabilities
from app.detector.page_context import PageContext
from app.detector.proposal_planner import ProposalPlanner


def _evidence(*, source: str, mask=True, confidence=0.9, bbox=(2, 2, 8, 8)):
    data = np.ones((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=np.uint8) if mask else None
    return DetectionEvidence(
        bbox=bbox,
        confidence=confidence,
        semantic="text",
        source=source,
        evidence_kind=EvidenceKind.MASK if mask else EvidenceKind.BOX,
        mask=data,
        class_id=1,
        class_name="text",
    )


def test_model_capabilities_do_not_encode_cleanup_authority():
    names = {item.name for item in fields(ModelCapabilities)}
    assert "authority" not in names
    assert "destructive" not in names
    assert "destructive_text_mask" not in names


def test_yolo26_mask_is_not_destructive_authority_by_default():
    decision = AuthorityPolicy().decide(_evidence(source="yolo26_segmentation"))
    assert decision.level is AuthorityLevel.REVIEW_ONLY
    assert decision.safe_to_inpaint is False
    assert decision.mask_source == "model"


def test_production_yolov8_text_mask_can_receive_authority():
    decision = AuthorityPolicy().decide(_evidence(source="yolov8_text_segmenter"))
    assert decision.level is AuthorityLevel.VERIFIED_TEXT_MASK
    assert decision.safe_to_inpaint is True
    assert decision.ocr_eligible is True
    assert decision.needs_review is False


def test_page_context_reuses_color_conversions():
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    context = PageContext(image)
    assert context.gray is context.gray
    assert context.rgb is context.rgb

    child = context.child(3, 4, 10, 11)
    assert (child.origin_x, child.origin_y) == (3, 4)
    assert (child.width, child.height) == (7, 7)


def test_planner_requests_full_page_fallback_when_budget_defers_proposal():
    page = PageContext(np.zeros((100, 100, 3), dtype=np.uint8))
    planner = ProposalPlanner(
        pad_x=0,
        pad_y=0,
        max_rois=1,
        max_source_side=30,
        merge_overlap=0.8,
    )
    plan = planner.plan(
        page=page,
        proposals=[
            _evidence(source="opencv_mser", mask=False, bbox=(0, 0, 20, 20), confidence=0.9),
            _evidence(source="yolo26_detect", mask=False, bbox=(70, 70, 90, 90), confidence=0.8),
        ],
    )
    assert len(plan.rois) == 1
    assert len(plan.deferred) == 1
    assert plan.needs_full_page_fallback is True
    assert plan.reason == "proposal_geometry_or_budget"
