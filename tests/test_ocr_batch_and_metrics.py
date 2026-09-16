from __future__ import annotations

import numpy as np

from app.ocr.metrics import (
    character_error_rate,
    line_recall,
    summarize_ocr_rows,
    word_error_rate,
)
from app.ocr.paddle_v6 import PaddleV6OCR


class _BatchPipeline:
    def __init__(self):
        self.batch_inputs = []
        self.single_inputs = []

    def predict(self, *, input):
        if isinstance(input, list):
            self.batch_inputs.append(len(input))
            return [
                {
                    "rec_texts": [f"line {index}"],
                    "rec_scores": [0.99],
                    "rec_polys": [[[20, 20], [100, 20], [100, 42], [20, 42]]],
                }
                for index in range(len(input))
            ]
        self.single_inputs.append(1)
        return [{
            "rec_texts": ["single"],
            "rec_scores": [0.99],
            "rec_polys": [[[20, 20], [100, 20], [100, 42], [20, 42]]],
        }]


def test_ocr_metrics_measure_text_not_confidence_labels():
    assert character_error_rate("Save this", "save  this") == 0.0
    assert word_error_rate("save this", "save that") == 0.5
    assert line_recall("first\nsecond", "first\nmissing") == 0.5
    assert summarize_ocr_rows([
        {"reference": "save this", "hypothesis": "save this"},
        {"reference": "next", "hypothesis": "wrong"},
    ]) == {
        "count": 2,
        "cer": round(5 / 12, 6),
        "wer": round(5 / 6, 6),
        "line_recall": 0.5,
    }


def test_paddle_batch_preserves_order_and_uses_one_backend_call():
    ocr = PaddleV6OCR()
    pipeline = _BatchPipeline()
    ocr._get_pipeline = lambda _key: pipeline
    images = [np.full((100, 140, 3), 255, np.uint8) for _ in range(3)]

    results = ocr.read_batch(images, "en", target_mode="all")

    assert [result.text for result in results] == ["line 0", "line 1", "line 2"]
    assert pipeline.batch_inputs == [3]
    assert pipeline.single_inputs == []
    assert ocr.batch_metrics() == {
        "batch_inference_calls": 1,
        "batch_items": 3,
        "batch_fallbacks": 0,
    }


def test_paddle_batch_malformed_response_falls_back_without_reordering():
    ocr = PaddleV6OCR()
    calls = []

    class _MalformedPipeline:
        def predict(self, *, input):
            calls.append(type(input).__name__)
            if isinstance(input, list):
                return [{"rec_texts": ["only one"]}]
            return [{
                "rec_texts": ["fallback"],
                "rec_scores": [0.99],
                "rec_polys": [[[20, 20], [100, 20], [100, 42], [20, 42]]],
            }]

    ocr._get_pipeline = lambda _key: _MalformedPipeline()
    images = [np.full((100, 140, 3), 255, np.uint8) for _ in range(2)]

    results = ocr.read_batch(images, "en", target_mode="all")

    assert [result.text for result in results] == ["fallback", "fallback"]
    assert calls == ["list", "ndarray", "ndarray"]
    assert ocr.batch_metrics()["batch_fallbacks"] == 1
