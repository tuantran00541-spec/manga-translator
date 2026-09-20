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


def test_legacy_bridge_preserves_decoder_source_role_gate():
    from dataclasses import dataclass

    @dataclass
    class LegacyBox:
        x1: int = 0
        y1: int = 0
        x2: int = 4
        y2: int = 4
        confidence: float = 0.9
        mask: np.ndarray | None = None
        class_id: int = 0
        class_name: str = "text_comic"
        semantic_type: str = "text"
        mask_source: str = "none"
        safe_to_inpaint: bool = False
        ocr_eligible: bool = False
        needs_review: bool = False
        source_role: str = "unknown"

        @property
        def verified_mask(self):
            return self.mask is not None and self.mask.shape == (4, 4) and bool(np.any(self.mask))

    mask = np.ones((4, 4), dtype=np.uint8)
    policy = AuthorityPolicy()

    untrusted = policy.apply_legacy_yolov8_box(
        LegacyBox(mask=mask, source_role="unknown"),
        producer_role="text_segmenter",
    )
    assert untrusted.safe_to_inpaint is False

    trusted = policy.apply_legacy_yolov8_box(
        LegacyBox(mask=mask, source_role="text_segmenter"),
        producer_role="text_segmenter",
    )
    assert trusted.safe_to_inpaint is True

def test_clustered_planner_merges_nearby_proposals_and_tracks_real_coverage():
    page = PageContext(np.zeros((2400, 800, 3), dtype=np.uint8))
    planner = ProposalPlanner(
        pad_x=80,
        pad_y=80,
        max_rois=3,
        max_source_side=1344,
        merge_overlap=0.10,
        merge_gap=64,
    )
    proposals = [
        _evidence(source="bubble", mask=False, bbox=(100, 400, 260, 560), confidence=0.9),
        _evidence(source="bubble", mask=False, bbox=(280, 430, 440, 590), confidence=0.8),
        _evidence(source="opencv_mser", mask=False, bbox=(120, 1500, 500, 1650), confidence=0.7),
    ]

    plan = planner.plan_clustered_shape(
        image_shape=(page.height, page.width),
        proposals=proposals,
        source_pixel_budget=3 * 1344 * 1344,
        tensor_pixel_budget=3 * 1024 * 1024,
        tensor_pixels_per_roi=1024 * 1024,
        tile_overlap=96,
        span_short_axis=True,
    )

    assert plan.group_count == 2
    assert len(plan.rois) == 2
    assert len(plan.covered) == 3
    assert plan.deferred == ()
    assert plan.deferred_roi_count == 0
    assert all(x1 == 0 and x2 == 800 for x1, _y1, x2, _y2 in plan.rois)


def test_clustered_planner_reports_deferred_proposals_not_merely_deferred_tiles():
    planner = ProposalPlanner(
        pad_x=0,
        pad_y=0,
        max_rois=1,
        max_source_side=100,
        merge_overlap=0.10,
        merge_gap=0,
    )
    first = _evidence(source="bubble", mask=False, bbox=(0, 0, 80, 80), confidence=0.9)
    second = _evidence(source="bubble", mask=False, bbox=(0, 160, 80, 240), confidence=0.8)

    plan = planner.plan_clustered_shape(
        image_shape=(300, 80),
        proposals=[first, second],
        source_pixel_budget=100 * 100,
        tensor_pixel_budget=1024 * 1024,
        tensor_pixels_per_roi=1024 * 1024,
        tile_overlap=0,
        span_short_axis=True,
    )

    assert len(plan.rois) == 1
    assert len(plan.covered) == 1
    assert len(plan.deferred) == 1
    assert plan.needs_full_page_fallback is True

