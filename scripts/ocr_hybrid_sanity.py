#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np

from app.ocr.identity import engine_identity
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.ocr.paddle_v6 import OCRReadResult
from app.ocr.quality import classify_ocr_quality, should_block_translation
from app.text_objects import ensure_page_text_objects


def _assert_quality() -> None:
    assert classify_ocr_quality("HELLO WORLD", "en", confidence=0.95).status == "good"
    mismatch = classify_ocr_quality("안녕하세요", "en", confidence=0.99)
    assert mismatch.status == "reject" and mismatch.reason == "script-mismatch"
    assert classify_ocr_quality("...?!", "en", confidence=0.99).status == "reject"
    assert classify_ocr_quality("hello", "en", confidence=0.2).status == "reject"
    assert classify_ocr_quality("hello", "en", confidence=0.5).status == "review"


def _assert_manual_override() -> None:
    obj = {
        "auto_generated": True,
        "ocr_text": "안녕하세요",
        "auto_ocr_text": "안녕하세요",
        "ocr_quality": "reject",
    }
    assert should_block_translation(obj)
    obj["ocr_text"] = "Hello there"
    assert not should_block_translation(obj)


def _assert_metadata_sync() -> None:
    page = {
        "boxes": [
            {
                "id": "box_1",
                "x1": 1,
                "y1": 2,
                "x2": 50,
                "y2": 20,
                "ocr_text": "HELLO",
                "ocr_confidence": 0.98,
                "ocr_model": "PP-OCRv6_small_rec",
                "ocr_orientation": "horizontal",
                "ocr_region_count": 1,
                "ocr_quality": "good",
            }
        ]
    }
    created, changed = ensure_page_text_objects(page)
    assert created == 1 and changed
    obj = page["text_objects"][0]
    assert obj["ocr_quality"] == "good"
    assert obj["ocr_confidence"] == 0.98

    # A user correction must not be overwritten or inherit a later machine reject.
    obj["ocr_text"] = "MANUAL FIX"
    page["boxes"][0]["ocr_text"] = "안녕하세요"
    page["boxes"][0]["ocr_quality"] = "reject"
    page["boxes"][0]["ocr_quality_reason"] = "script-mismatch"
    _, changed = ensure_page_text_objects(page)
    assert obj["ocr_text"] == "MANUAL FIX"
    assert obj["ocr_quality"] == "good"


def _assert_japanese_route() -> None:
    class BombPaddle:
        def read(self, *_args, **_kwargs):
            raise AssertionError("Japanese must not enter Paddle runtime")

    engine = MultiLangOCR()
    engine._paddle = BombPaddle()
    engine._manga_ocr = lambda _image: "午後から雨が心配"
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    result = engine.read_detailed(image, "ja")
    assert result.text == "午後から雨が心配"
    assert result.model == "manga-ocr"
    assert result.quality == "good"
    assert "manga-ocr" in engine_identity("ja")


def _assert_orientation_default() -> None:
    os.environ.pop("MANGA_PPOCRV6_TEXTLINE_ORIENTATION", None)
    from app.ocr.paddle_v6 import PaddleV6OCR

    assert PaddleV6OCR().textline_orientation is False


def _assert_target_selection() -> None:
    class CapturePaddle:
        def __init__(self):
            self.modes: list[str] = []

        def read(self, _image, _lang, *, target_mode: str):
            self.modes.append(target_mode)
            return OCRReadResult(
                "HELLO",
                0.99,
                "PP-OCRv6_small_rec",
                "horizontal",
                1,
                "good",
                None,
            )

    original = os.environ.get("MANGA_OCR_TARGET_SELECTION")
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    try:
        os.environ.pop("MANGA_OCR_TARGET_SELECTION", None)
        engine_identity.cache_clear()
        default_engine = MultiLangOCR()
        capture = CapturePaddle()
        default_engine._paddle = capture
        assert default_engine.read(image, "en") == "HELLO"
        assert capture.modes == ["centered"]
        assert "target-centered" in engine_identity("en")

        capture.modes.clear()
        default_engine.read_detailed(image, "en", target_mode="all")
        assert capture.modes == ["all"]

        os.environ["MANGA_OCR_TARGET_SELECTION"] = "all"
        engine_identity.cache_clear()
        rollback_engine = MultiLangOCR()
        rollback_capture = CapturePaddle()
        rollback_engine._paddle = rollback_capture
        rollback_engine.read(image, "en")
        assert rollback_capture.modes == ["all"]
        assert "target-all" in engine_identity("en")
    finally:
        if original is None:
            os.environ.pop("MANGA_OCR_TARGET_SELECTION", None)
        else:
            os.environ["MANGA_OCR_TARGET_SELECTION"] = original
        engine_identity.cache_clear()


def main() -> int:
    _assert_quality()
    _assert_manual_override()
    _assert_metadata_sync()
    _assert_japanese_route()
    _assert_orientation_default()
    _assert_target_selection()
    print("hybrid OCR invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
