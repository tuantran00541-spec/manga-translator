from app.detector.adaptive_tiled_residue_detector import (
    AdaptiveTiledIndependentRegionResidueSequentialTextDetector,
)
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_RESIDUE_VERIFY_MAX_ROIS,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)


def test_oversized_residue_roi_is_tiled_without_gaps():
    detector = AdaptiveTiledIndependentRegionResidueSequentialTextDetector
    roi = (59, 458, 1536, 1407)

    tiles = detector._tiles_for_roi(roi)
    side = int(detector._RESIDUE_TILE_SIDE)
    overlap = int(detector._RESIDUE_TILE_OVERLAP)

    assert len(tiles) == 2
    assert tiles[0][0] == roi[0]
    assert tiles[-1][2] == roi[2]
    assert all(tile[1] == roi[1] and tile[3] == roi[3] for tile in tiles)
    assert all(max(tile[2] - tile[0], tile[3] - tile[1]) <= side for tile in tiles)
    assert tiles[0][2] - tiles[1][0] >= overlap


def test_residue_tile_geometry_tracks_detector_configuration():
    detector = AdaptiveTiledIndependentRegionResidueSequentialTextDetector
    expected_side = min(
        int(DETECTOR_INPUT_SIZE),
        int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE),
    )
    expected_overlap = min(
        max(1, expected_side // 3),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 2,
            int(round(expected_side * 0.16)),
        ),
    )
    expected_tile_cap = max(
        2,
        min(
            int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS),
            (int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS) + 1) // 2,
        ),
    )

    assert detector._RESIDUE_TILE_SIDE == expected_side
    assert detector._RESIDUE_TILE_OVERLAP == expected_overlap
    assert detector._RESIDUE_MAX_TILES_PER_SOURCE == expected_tile_cap


def test_normal_residue_roi_stays_single_tile():
    detector = AdaptiveTiledIndependentRegionResidueSequentialTextDetector
    roi = (100, 200, 900, 800)
    assert detector._tiles_for_roi(roi) == [roi]


def test_oversized_residue_tile_hard_cap_is_bounded():
    detector = AdaptiveTiledIndependentRegionResidueSequentialTextDetector
    roi = (0, 0, 4096, 4096)
    tiles = detector._tiles_for_roi(roi)
    assert len(tiles) > detector._RESIDUE_MAX_TILES_PER_SOURCE
