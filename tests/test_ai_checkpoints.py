import numpy as np

from app.ai_mode.checkpoints import SHEET_SIDE, TAG_PX, crops_for, pack


def test_each_erased_box_gets_a_padded_crop_and_huge_ones_are_shrunk():
    clean = np.full((3000, 1600, 3), 255, np.uint8)
    crops = crops_for(4, clean, [(100, 100, 300, 200), (0, 0, 1600, 2000), (5, 5, 7, 7)])
    assert [crop.id for crop in crops] == ["5.1", "5.2", "5.3"]
    assert crops[0].image.shape[:2] == (148, 248), "24 px of background round the box"
    assert max(crops[1].image.shape[:2]) == 900
    assert crops[0].rect == (100, 100, 300, 200)


def test_crops_share_sheets_and_none_overlaps_or_is_lost():
    clean = np.full((3000, 1600, 3), 255, np.uint8)
    crops = crops_for(0, clean, [(0, 0, 600, 400)] * 40)
    sheets = pack(crops)
    assert sum(len(placed) for _, placed in sheets) == 40 and len(sheets) < 40
    assert all(sheet.shape[0] <= SHEET_SIDE + TAG_PX and sheet.shape[1] <= SHEET_SIDE for sheet, _ in sheets)
