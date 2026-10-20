from types import SimpleNamespace

import numpy as np

from app.ai_mode import glossary


def test_read_glossary_labels_slices_by_page_index(monkeypatch):
    captured = {}

    def fake(provider, model, api_key, prompt, images, **kwargs):
        captured["images"] = images
        return type("Result", (), {"data": {}, "estimated_cost_usd": 0.0})()

    monkeypatch.setattr(glossary, "request_vision_json", fake)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    glossary.read_glossary(
        SimpleNamespace(), "model", "key", "Vietnamese", "vi", [(4, image), (5, image)]
    )

    assert [label for label, _ in captured["images"]] == ["SLICE 5", "SLICE 6"]
