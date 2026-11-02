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
