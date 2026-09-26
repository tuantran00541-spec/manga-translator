from __future__ import annotations

import numpy as np

from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.paddle_v6 import OCRReadResult


def _result(text: str, quality: str = "good", model: str = "full") -> OCRReadResult:
    return OCRReadResult(text, 0.9, model, "horizontal", 1 if text else 0, quality, "ok")


class FakePaddle:
    def __init__(self, fast: OCRReadResult, full: OCRReadResult):
        self.fast, self.full, self.calls = fast, full, []

    def read_recognition_only(self, image, lang):
        self.calls.append("fast")
        return self.fast

    def read(self, image, lang, *, target_mode="all"):
        self.calls.append("full")
        return self.full


def _ocr(fast: OCRReadResult, full: OCRReadResult) -> tuple[MultiLangOCR, FakePaddle]:
    ocr = MultiLangOCR.__new__(MultiLangOCR)
    ocr._paddle = FakePaddle(fast, full)
    ocr._paddle_target_mode = "all"
    return ocr, ocr._paddle


def test_a_multi_line_bubble_never_goes_through_the_single_line_recognizer():
    ocr, paddle = _ocr(_result("BUTICOOULENTCANVE", model="fast"), _result("BUT I COULDN'T\nCHANGE IT"))
    bubble = np.zeros((180, 240, 3), np.uint8)
    assert ocr.read_detailed(bubble, "en").text == "BUT I COULDN'T\nCHANGE IT"
    assert paddle.calls == ["full"]


def test_a_single_line_crop_keeps_the_fast_path_when_it_reads_well():
    ocr, paddle = _ocr(_result("GAME OVER", model="fast"), _result("unused"))
    line = np.zeros((40, 200, 3), np.uint8)
    assert ocr.read_detailed(line, "en").text == "GAME OVER"
    assert paddle.calls == ["fast"]


def test_a_doubtful_single_line_read_is_checked_with_line_detection():
    ocr, paddle = _ocr(_result("HARDMODE", quality="review", model="fast"), _result("HARD MODE?"))
    line = np.zeros((40, 200, 3), np.uint8)
    assert ocr.read_detailed(line, "en").text == "HARD MODE?"
    assert paddle.calls == ["fast", "full"]


def test_the_fast_read_is_kept_when_line_detection_finds_nothing():
    ocr, _ = _ocr(_result("OK", quality="review", model="fast"), _result(""))
    assert ocr.read_detailed(np.zeros((40, 200, 3), np.uint8), "en").text == "OK"
