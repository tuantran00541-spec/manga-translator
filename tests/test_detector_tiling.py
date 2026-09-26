import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.one_shot_cleanup import OneShotTextMaskDetector as Detector


def _box(x1, y1, x2, y2, conf=0.9):
    return BubbleBox(x1, y1, x2, y2, conf, np.full((y2 - y1, x2 - x1), 255, np.uint8),
                     source_role="text_segmenter", safe_to_inpaint=True)


def test_windows_cover_the_slice_with_overlap_at_the_requested_scale():
    assert Detector.tile_windows(2400, 800, 0.0) == [(0, 2400)], "0 keeps one pass over the whole slice"
    windows = Detector.tile_windows(2400, 800, 0.75)
    assert windows[0][0] == 0 and windows[-1][1] == 2400
    assert all(y2 - y1 == 1365 for y1, y2 in windows), "each window reaches the model at 0.75"
    assert all(windows[i][1] - windows[i + 1][0] >= 273 for i in range(len(windows) - 1))
    assert Detector.tile_windows(1000, 800, 1.0) == [(0, 1000)], "a short slice needs no windows"
    assert all(y2 - y1 == 1200 for y1, y2 in Detector.tile_windows(2400, 1200, 1.0)), "width caps the scale"


def test_merge_keeps_the_whole_box_over_the_piece_a_window_edge_cut():
    whole = _box(100, 900, 300, 1100)           # window 1 saw it whole
    piece = _box(100, 900, 300, 1024)           # window 0 cut it at its bottom edge
    neighbour = _box(100, 700, 300, 760)        # a separate line in window 0
    merged = Detector.merge_tiles([(piece, 0, True), (neighbour, 0, False), (whole, 1, False)])
    assert whole in merged and neighbour in merged and piece not in merged


def test_merge_keeps_both_pieces_of_a_block_no_window_saw_whole():
    top, bottom = _box(0, 500, 200, 1024), _box(0, 688, 200, 1500)
    merged = Detector.merge_tiles([(top, 0, True), (bottom, 1, True)])
    assert len(merged) == 1
    box = merged[0]
    assert (box.y1, box.y2) == (500, 1500) and box.mask.shape == (1000, 200)
    assert int(np.count_nonzero(box.mask)) == 1000 * 200, "the mask covers both pieces"


def test_boxes_from_one_window_never_replace_each_other():
    a, b = _box(0, 0, 100, 100), _box(10, 10, 110, 110)
    assert len(Detector.merge_tiles([(a, 0, False), (b, 0, False)])) == 2
