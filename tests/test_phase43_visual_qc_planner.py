import numpy as np

from app.visual_qc.planner import build_chapter_qc_plan, chunk_regions


def test_plan_keeps_global_screening_for_pages_without_detector_boxes():
    manifest = {"pages": [
        {"width": 800, "height": 1200, "clean": "clean0.png", "boxes": [{"id": "box_a", "x1": 100, "y1": 100, "x2": 200, "y2": 180}]},
        {"width": 800, "height": 1200, "clean": "clean1.png", "boxes": []},
        {"width": 800, "height": 1200, "clean": "clean2.png", "boxes": [], "skipped": True},
        {"width": 800, "height": 1200, "clean": None, "boxes": []},
    ]}
    plan = build_chapter_qc_plan(manifest)
    assert [r.page_index for r in plan.global_regions] == [0, 1]
    assert [r.region_id for r in plan.global_regions] == ["P0001-GLOBAL", "P0002-GLOBAL"]
    assert len(plan.candidate_regions) == 1
    assert plan.candidate_regions[0].source_box_ids == ("box_a",)
    assert plan.skipped_pages == (2, 3)


def test_plan_includes_manual_mask_candidate_on_boxless_page():
    manifest = {"pages": [{"width": 300, "height": 200, "clean": "clean.png", "boxes": []}]}
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:100, 120:160] = 255
    plan = build_chapter_qc_plan(manifest, manual_masks={0: mask}, margin=20, merge_gap=0)
    assert len(plan.global_regions) == 1
    assert len(plan.candidate_regions) == 1
    assert plan.candidate_regions[0].source_kinds == ("manual_mask",)


def test_plan_marks_giant_candidates_for_deep_stage():
    manifest = {"pages": [{"width": 1000, "height": 1000, "clean": "clean.png", "boxes": [{"id": "big", "x1": 0, "y1": 0, "x2": 900, "y2": 900}]}]}
    plan = build_chapter_qc_plan(manifest, margin=0, deep_area_ratio=0.5)
    assert plan.deep_region_ids == ("P0001-R01",)


def test_chunk_regions_is_deterministic_and_bounded():
    manifest = {"pages": [{"width": 100, "height": 100, "clean": f"c{i}.png", "boxes": []} for i in range(5)]}
    plan = build_chapter_qc_plan(manifest)
    items = chunk_regions(plan.global_regions, batch_size=2, work_prefix="global")
    assert [len(item.region_ids) for item in items] == [2, 2, 1]
    assert [item.work_id for item in items] == ["global-0001", "global-0002", "global-0003"]
    assert items[0].page_indices == (0, 1)
