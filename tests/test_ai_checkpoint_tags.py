import numpy as np

from app.ai_mode.checkpoints import Crop, pack


def test_pack_sheet_is_wide_enough_for_its_tag():
    crop = Crop("1.1", 0, (0, 0, 5, 5), np.zeros((5, 5, 3), dtype=np.uint8))
    sheet, placed = pack([crop])[0]
    assert len(placed) == 1
    assert sheet.shape[1] >= 70
