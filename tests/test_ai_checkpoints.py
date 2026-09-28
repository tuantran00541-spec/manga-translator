from app.ai_mode.checkpoints import parse_clean_review


def test_clean_review_maps_confident_boxes_and_drops_misreads():
    review = parse_clean_review({
        "missed": [{"box_2d": [100, 250, 200, 750], "confidence": 0.9},
                   {"box_2d": [100, 250, 200, 750], "confidence": 0.3}],
        "residue": [{"box_2d": [0, 0, 1000, 1000], "confidence": 0.9}, {"box_2d": [0, 0, 1, 1], "confidence": 0.9}],
        "restore": "nonsense",
    }, 4, 400, 600)
    assert review.page_index == 4
    assert review.missed == ((94, 54, 306, 126),), "0-1000 to pixels plus a margin; unsure boxes dropped"
    assert review.residue == () and review.restore == (), "whole-slice, tiny and malformed boxes are misreads"
