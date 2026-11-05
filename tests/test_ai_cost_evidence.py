from scripts.ai_cost_run import _objects


def test_evidence_keeps_letter_bounds_and_join_relationships():
    bounds = {"x1": 20, "y1": 30, "x2": 100, "y2": 80}
    pages = [{"text_objects": [{"id": "a", "region": bounds, "letter_bounds": bounds,
                                "joined_lines": ["b"]},
                               {"id": "b", "region": bounds, "joined_into": "a"}]}]
    first, second = _objects(pages)[0]["objects"]
    assert first["letter_bounds"] == bounds
    assert first["joined_lines"] == ["b"]
    assert second["joined_into"] == "a"


def test_fixed_size_score_does_not_accept_split_unbreakable_word():
    from scripts.ai_cost_run import _fit_metrics

    obj = {"region": {"x1": 20, "y1": 20, "x2": 120, "y2": 1020},
           "translation": "WWWWWWWWWWWWWWWW", "style": {"fontSize": 40, "font": "default"}}
    assert not _fit_metrics(obj, 400)["fits"]


def test_pair_evidence_does_not_present_stale_render_as_final(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from scripts.ai_cost_run import _pairs
    import app.routers.image as image_router

    raw, stale = tmp_path / "raw.png", tmp_path / "stale.png"
    cv2.imwrite(str(raw), np.full((30, 40, 3), 255, np.uint8))
    cv2.imwrite(str(stale), np.zeros((30, 40, 3), np.uint8))
    monkeypatch.setattr(image_router, "_rendered_file_path", lambda *_: stale)
    monkeypatch.setattr(image_router, "_current_rendered_path", lambda *_: None)
    assert _pairs("a1b2c3d4", [{"original": str(raw)}], tmp_path) == 0
