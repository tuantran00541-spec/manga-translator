from app.ai_mode.checkpoints import parse_clean_review


def test_clean_review_maps_confident_boxes_and_drops_misreads():
    review = parse_clean_review({
        "missed": [{"x1": 250, "y1": 100, "x2": 750, "y2": 200, "confidence": 0.9},
                   {"x1": 250, "y1": 100, "x2": 750, "y2": 200, "confidence": 0.3}],
        "residue": [{"x1": 0, "y1": 0, "x2": 1000, "y2": 1000, "confidence": 0.9},
                    {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9}],
        "restore": "nonsense",
    }, 4, 400, 600)
    assert review.page_index == 4
    assert review.missed == ((94, 54, 306, 126),), "0-1000 to pixels plus a margin; unsure boxes dropped"
    assert review.residue == () and review.restore == (), "whole-slice, tiny and malformed boxes are misreads"


def test_clean_review_boxes_name_their_corners():
    # Shadow Slave 1, slices 4, 55 and 69: a model read [ymin, xmin, ...] as x first and boxed a tall strip of art.
    review = parse_clean_review({
        "missed": [{"x1": 250, "y1": 800, "x2": 900, "y2": 950, "confidence": 0.9}],
        "residue": [{"box_2d": [800, 250, 950, 900], "confidence": 0.9}],
    }, 0, 1600, 4000)
    assert review.missed == ((394, 3194, 1446, 3806),), "a caption across the foot of the slice stays across it"
    assert review.residue == (), "an unnamed list is not guessed at"
