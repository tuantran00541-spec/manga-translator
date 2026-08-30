#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np

from app.ocr.hybrid_service import HybridOCRService
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


def _assert_empty_rerun_ownership() -> None:
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
    ensure_page_text_objects(page)
    obj = page["text_objects"][0]
    obj["translation"] = "XIN CHAO"
    obj["translation_source"] = "deepseek"
    obj["translation_model"] = "deepseek-chat"
    obj["translation_input_text"] = "HELLO"
    obj["auto_translation"] = "XIN CHAO"

    # If the object still follows machine OCR, an empty/rejected rerun must clear
    # both stale OCR and a still-machine-owned translation derived from it.
    page["boxes"][0]["ocr_text"] = ""
    page["boxes"][0].pop("ocr_confidence", None)
    page["boxes"][0]["ocr_region_count"] = 0
    page["boxes"][0]["ocr_quality"] = "reject"
    page["boxes"][0]["ocr_quality_reason"] = "empty"
    _, changed = ensure_page_text_objects(page)
    assert changed
    assert obj["ocr_text"] == ""
    assert obj["auto_ocr_text"] == ""
    assert obj["ocr_quality"] == "reject"
    assert obj["ocr_quality_reason"] == "empty"
    assert obj["ocr_region_count"] == 0
    assert "ocr_confidence" not in obj
    assert obj["translation"] == ""
    assert "translation_source" not in obj
    assert "translation_model" not in obj
    assert "translation_input_text" not in obj
    assert "auto_translation" not in obj

    # A manual clear is also a user edit and must not be repopulated by a later
    # machine rerun.
    page["boxes"][0]["ocr_text"] = "MACHINE AGAIN"
    page["boxes"][0]["ocr_quality"] = "good"
    page["boxes"][0].pop("ocr_quality_reason", None)
    obj["auto_ocr_text"] = "OLDER MACHINE TEXT"
    obj["ocr_text"] = ""
    _, changed = ensure_page_text_objects(page)
    assert obj["ocr_text"] == ""
    assert obj["auto_ocr_text"] == "OLDER MACHINE TEXT"
    assert obj["ocr_quality"] == "reject"


def _assert_translation_ownership() -> None:
    page = {
        "boxes": [
            {
                "id": "box_1",
                "x1": 1,
                "y1": 2,
                "x2": 50,
                "y2": 20,
                "ocr_text": "OLD SOURCE",
                "ocr_quality": "good",
            }
        ]
    }
    ensure_page_text_objects(page)
    obj = page["text_objects"][0]

    # A tracked DeepSeek result that has not been edited is machine-owned and
    # must be invalidated when its machine OCR source changes.
    obj.update(
        {
            "translation": "OLD TRANSLATION",
            "translation_source": "deepseek",
            "translation_model": "deepseek-chat",
            "translation_input_text": "OLD SOURCE",
            "auto_translation": "OLD TRANSLATION",
        }
    )
    page["boxes"][0]["ocr_text"] = "NEW SOURCE"
    ensure_page_text_objects(page)
    assert obj["ocr_text"] == "NEW SOURCE"
    assert obj["translation"] == ""
    assert "auto_translation" not in obj

    # A manual OCR correction also invalidates an untouched machine translation.
    obj.update(
        {
            "translation": "AUTO NEW TRANSLATION",
            "translation_source": "deepseek",
            "translation_model": "deepseek-chat",
            "translation_input_text": "NEW SOURCE",
            "auto_translation": "AUTO NEW TRANSLATION",
        }
    )
    obj["ocr_text"] = "MANUAL SOURCE FIX"
    ensure_page_text_objects(page)
    assert obj["ocr_text"] == "MANUAL SOURCE FIX"
    assert obj["auto_ocr_text"] == "NEW SOURCE"
    assert obj["translation"] == ""

    # If the user edits a generated translation too, current text no longer
    # equals auto_translation, so later OCR changes must preserve that user work.
    obj.update(
        {
            "translation": "MANUAL TRANSLATION EDIT",
            "translation_source": "deepseek",
            "translation_model": "deepseek-chat",
            "translation_input_text": "MANUAL SOURCE FIX",
            "auto_translation": "AUTO FOR MANUAL SOURCE",
        }
    )
    obj["ocr_text"] = "SECOND MANUAL SOURCE"
    ensure_page_text_objects(page)
    assert obj["translation"] == "MANUAL TRANSLATION EDIT"
    assert obj["auto_translation"] == "AUTO FOR MANUAL SOURCE"

    # Legacy DeepSeek metadata without an ownership snapshot is intentionally
    # conservative: never delete it because a user may already have edited it.
    obj["translation"] = "LEGACY OR MANUAL"
    obj["translation_source"] = "deepseek"
    obj["translation_input_text"] = "SECOND MANUAL SOURCE"
    obj.pop("auto_translation", None)
    obj["ocr_text"] = "THIRD MANUAL SOURCE"
    ensure_page_text_objects(page)
    assert obj["translation"] == "LEGACY OR MANUAL"


def _assert_grouped_translation_ownership() -> None:
    obj = {
        "ocr_text": "OLD GROUP SOURCE",
        "translation": "OLD GROUP TRANSLATION",
        "translation_source": "deepseek",
        "translation_model": "deepseek-chat",
        "translation_input_text": "OLD GROUP SOURCE",
        "auto_translation": "OLD GROUP TRANSLATION",
    }
    HybridOCRService._stamp_group_object(
        obj,
        source_box_ids=["box_a", "box_b"],
        combined="NEW GROUP SOURCE",
        lang="en",
        engine="test-engine",
        source_revision=4,
        original_revision=(100, 200, 300),
        region={"x1": 1, "y1": 2, "x2": 50, "y2": 60},
    )
    assert obj["ocr_text"] == "NEW GROUP SOURCE"
    assert obj["translation"] == ""
    assert "auto_translation" not in obj
    assert obj["ocr_quality"] == "review"
    assert obj["ocr_quality_reason"] == "grouped-machine-ocr"

    # A user-edited translation on the same grouped object is not machine-owned
    # anymore and must survive a subsequent grouped OCR refresh.
    obj.update(
        {
            "translation": "MANUAL GROUP TRANSLATION",
            "translation_source": "deepseek",
            "translation_input_text": "NEW GROUP SOURCE",
            "auto_translation": "AUTO GROUP TRANSLATION",
        }
    )
    HybridOCRService._stamp_group_object(
        obj,
        source_box_ids=["box_a", "box_b"],
        combined="THIRD GROUP SOURCE",
        lang="en",
        engine="test-engine",
        source_revision=5,
        original_revision=(100, 200, 300),
        region={"x1": 1, "y1": 2, "x2": 50, "y2": 60},
    )
    assert obj["ocr_text"] == "THIRD GROUP SOURCE"
    assert obj["translation"] == "MANUAL GROUP TRANSLATION"
    assert obj["auto_translation"] == "AUTO GROUP TRANSLATION"


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


def _assert_cache_identity_settings() -> None:
    tracked = (
        "MANGA_PPOCRV6_TIER",
        "MANGA_PPOCRV6_TEXTLINE_ORIENTATION",
        "MANGA_OCR_TARGET_SELECTION",
    )
    original = {name: os.environ.get(name) for name in tracked}
    try:
        os.environ["MANGA_PPOCRV6_TEXTLINE_ORIENTATION"] = "0"
        os.environ["MANGA_OCR_TARGET_SELECTION"] = "centered"

        os.environ["MANGA_PPOCRV6_TIER"] = "small"
        engine_identity.cache_clear()
        en_small = engine_identity("en")
        ko_small = engine_identity("ko")
        assert "ppocrv6-small" in en_small
        assert "ppocrv6-small-det" in ko_small
        assert "korean-ppocrv5-mobile-rec" in ko_small

        os.environ["MANGA_PPOCRV6_TIER"] = "medium"
        engine_identity.cache_clear()
        en_medium = engine_identity("en")
        ko_medium = engine_identity("ko")
        assert "ppocrv6-medium" in en_medium
        assert "ppocrv6-medium-det" in ko_medium
        assert en_medium != en_small
        assert ko_medium != ko_small

        os.environ["MANGA_PPOCRV6_TEXTLINE_ORIENTATION"] = "1"
        engine_identity.cache_clear()
        assert engine_identity("en") != en_medium
        assert engine_identity("ko") != ko_medium
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        engine_identity.cache_clear()


def main() -> int:
    _assert_quality()
    _assert_manual_override()
    _assert_metadata_sync()
    _assert_empty_rerun_ownership()
    _assert_translation_ownership()
    _assert_grouped_translation_ownership()
    _assert_japanese_route()
    _assert_orientation_default()
    _assert_target_selection()
    _assert_cache_identity_settings()
    print("hybrid OCR invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
