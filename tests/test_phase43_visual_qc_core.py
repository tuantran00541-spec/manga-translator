import numpy as np

from app.visual_qc.batch_protocol import parse_region_batch_response
from app.visual_qc.contact_sheet import build_contact_sheet
from app.visual_qc.regions import QCRegion, extract_candidate_regions, qc_cache_identity, qc_cache_matches


def _page(boxes=None, *, width=1000, height=1400, clean_revision=3, source_revision=1):
    return {
        "width": width,
        "height": height,
        "boxes": boxes or [],
        "clean_revision": clean_revision,
        "source_revision": source_revision,
    }


def test_regions_use_active_stable_box_ids_and_ignore_removed_boxes():
    page = _page([
        {"id": "box_a", "x1": 100, "y1": 100, "x2": 200, "y2": 180, "origin": "detector"},
        {"id": "box_removed", "x1": 500, "y1": 500, "x2": 600, "y2": 600, "removed": True},
    ])
    regions = extract_candidate_regions(page, page_index=2, margin=40, merge_gap=10)
    assert len(regions) == 1
    assert regions[0].page_index == 2
    assert regions[0].source_box_ids == ("box_a",)
    assert regions[0].bbox == (60, 60, 240, 220)
    assert regions[0].region_id == "P0003-R01"


def test_regions_merge_nearby_boxes_and_keep_deterministic_mapping():
    boxes = [
        {"id": "box_b", "x1": 220, "y1": 110, "x2": 300, "y2": 190},
        {"id": "box_a", "x1": 100, "y1": 100, "x2": 180, "y2": 180},
    ]
    page = _page(boxes)
    first = extract_candidate_regions(page, page_index=0, margin=20, merge_gap=30)
    second = extract_candidate_regions(page, page_index=0, margin=20, merge_gap=30)
    assert first == second
    assert len(first) == 1
    assert first[0].source_box_ids == ("box_a", "box_b")
    assert first[0].bbox == (80, 80, 320, 210)


def test_manual_mask_components_are_region_sources_and_clamped():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[2:20, 3:30] = 255
    mask[150:180, 250:290] = 255
    page = _page(width=300, height=200)
    regions = extract_candidate_regions(page, page_index=4, manual_mask=mask, margin=25, merge_gap=0)
    assert len(regions) == 2
    assert regions[0].bbox == (0, 0, 55, 45)
    assert regions[0].source_kinds == ("manual_mask",)
    assert regions[1].bbox == (225, 125, 300, 200)


def test_large_region_is_marked_for_deep_qc_not_dropped():
    page = _page([{"id": "box_big", "x1": 50, "y1": 50, "x2": 950, "y2": 1300}])
    regions = extract_candidate_regions(page, page_index=0, margin=0, deep_area_ratio=0.5)
    assert len(regions) == 1
    assert regions[0].requires_deep_qc is True
    assert regions[0].area_ratio > 0.5


def test_qc_cache_identity_is_revision_model_and_mode_aware():
    page = _page(clean_revision=7, source_revision=2)
    identity = qc_cache_identity(page, model="gemini-3.7-flash", mode="region-clean")
    assert identity == {
        "source_revision": 2,
        "clean_revision": 7,
        "model": "gemini-3.7-flash",
        "mode": "region-clean",
        "pipeline_version": 1,
    }
    assert qc_cache_matches(identity, page, model="gemini-3.7-flash", mode="region-clean")
    assert not qc_cache_matches(identity, {**page, "clean_revision": 8}, model="gemini-3.7-flash", mode="region-clean")
    assert not qc_cache_matches(identity, page, model="another-model", mode="region-clean")


def test_contact_sheet_preserves_region_labels_and_mapping_without_downscale_when_small():
    crops = []
    for idx in range(4):
        image = np.full((180, 240, 3), 240 - idx * 10, dtype=np.uint8)
        region = QCRegion(idx, f"P{idx+1:04d}-R01", (10, 20, 250, 200), (f"box_{idx}",), ("box",), 0.03, False)
        crops.append((region, image))
    sheet = build_contact_sheet(crops, max_side=2048)
    assert sheet.scale == 1.0
    assert [item.region_id for item in sheet.items] == [f"P{i+1:04d}-R01" for i in range(4)]
    assert sheet.image.shape[0] <= 2048
    assert sheet.image.shape[1] <= 2048
    for item in sheet.items:
        x1, y1, x2, y2 = item.sheet_bbox
        assert x2 > x1 and y2 > y1


def test_batch_response_maps_region_relative_bbox_back_to_page_coordinates():
    region = QCRegion(3, "P0004-R02", (100, 200, 500, 600), ("box_a",), ("box",), 0.1, False)
    parsed = {
        "regions": [{
            "region_id": "P0004-R02",
            "status": "flagged",
            "issues": [{
                "issue_type": "residual_text",
                "confidence": 0.9,
                "box_2d": [250, 250, 750, 750],
                "reason": "glyph fragment remains",
                "recommended_action": "repaint",
            }],
        }]
    }
    result = parse_region_batch_response(parsed, {region.region_id: region})
    assert len(result) == 1
    issue = result[0]
    assert issue.page_index == 3
    assert issue.region_id == "P0004-R02"
    assert issue.bbox == (200, 300, 400, 500)
    assert issue.issue_type == "residual_text"
    assert issue.recommended_action == "repaint"


def test_batch_response_ignores_unknown_regions_invalid_boxes_and_nonfinite_confidence():
    region = QCRegion(0, "P0001-R01", (0, 0, 100, 100), (), ("manual_mask",), 0.01, False)
    parsed = {
        "regions": [
            {"region_id": "UNKNOWN", "status": "flagged", "issues": [{"issue_type": "smear", "confidence": 1, "box_2d": [0, 0, 1000, 1000]}]},
            {"region_id": "P0001-R01", "status": "flagged", "issues": [
                {"issue_type": "smear", "confidence": float("nan"), "box_2d": [0, 0, 1000, 1000]},
                {"issue_type": "smear", "confidence": 0.9, "box_2d": [500, 500, 400, 600]},
                {"issue_type": "made_up", "confidence": 0.9, "box_2d": [0, 0, 1000, 1000]},
            ]},
        ]
    }
    assert parse_region_batch_response(parsed, {region.region_id: region}) == []


def test_batch_response_supports_suspicious_fill_and_over_erased_art_without_autorepaint_assumption():
    region = QCRegion(1, "P0002-R01", (50, 60, 250, 260), (), ("box",), 0.02, False)
    parsed = {"regions": [{
        "region_id": region.region_id,
        "status": "flagged",
        "issues": [
            {"issue_type": "suspicious_fill", "confidence": 0.7, "box_2d": [0, 0, 500, 500], "reason": "texture discontinuity", "recommended_action": "review"},
            {"issue_type": "over_erased_art", "confidence": 0.8, "box_2d": [500, 500, 1000, 1000], "reason": "line art missing", "recommended_action": "review_original"},
        ],
    }]}
    issues = parse_region_batch_response(parsed, {region.region_id: region})
    assert [i.issue_type for i in issues] == ["suspicious_fill", "over_erased_art"]
    assert [i.recommended_action for i in issues] == ["review", "review_original"]
