from app.detector.tiled_independent_residue_detector import (
    TiledIndependentRegionResidueSequentialTextDetector,
)


def test_oversized_residue_roi_is_tiled_without_gaps():
    detector = TiledIndependentRegionResidueSequentialTextDetector
    roi = (59, 458, 1536, 1407)

    tiles = detector._tiles_for_roi(roi)

    assert len(tiles) == 2
    assert tiles[0][0] == roi[0]
    assert tiles[-1][2] == roi[2]
    assert all(tile[1] == roi[1] and tile[3] == roi[3] for tile in tiles)
    assert all(max(tile[2] - tile[0], tile[3] - tile[1]) <= 1024 for tile in tiles)
    assert tiles[0][2] - tiles[1][0] >= detector._RESIDUE_TILE_OVERLAP


def test_normal_residue_roi_stays_single_tile():
    detector = TiledIndependentRegionResidueSequentialTextDetector
    roi = (100, 200, 900, 800)
    assert detector._tiles_for_roi(roi) == [roi]


def test_oversized_residue_tile_hard_cap_is_bounded():
    detector = TiledIndependentRegionResidueSequentialTextDetector
    roi = (0, 0, 4096, 4096)
    tiles = detector._tiles_for_roi(roi)
    assert len(tiles) > detector._RESIDUE_MAX_TILES_PER_SOURCE
