from app.ai_mode.checkpoints import parse_clean_review, parse_final_review


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


def test_final_review_keeps_known_ids_and_only_fixes_a_fix_verdict():
    data = {"verdict": "fix", "fixes": [
        {"action": "retranslate", "id": "a"}, {"action": "restore", "id": "a"},
        {"action": "retranslate", "id": "ghost"}, {"action": "repaint", "box_2d": [100, 100, 200, 300]},
        {"action": "rewrite", "id": "b"},
    ]}
    review = parse_final_review(data, 2, 400, 600, {"a", "b"})
    assert not review.ok and review.restore == ("a",) and review.retranslate == ()
    assert review.repaint == ((34, 54, 126, 126),)
    assert parse_final_review({"verdict": "fix", "fixes": []}, 2, 400, 600, {"a"}).ok, "nothing to fix is ok"
    assert parse_final_review({"verdict": "ok", "fixes": data["fixes"]}, 2, 400, 600, {"a"}).ok
