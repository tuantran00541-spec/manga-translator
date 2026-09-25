from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app.manifest_utils as manifests
import app.ocr.language_detect as language_detect
import app.routers.ocr as ocr_router
from app.ocr.language_detect import (
    ProbeReading,
    classify_readings,
    detect_manifest_language,
    sample_regions,
    script_counts,
    site_language_hint,
)
from app.schemas import ChapterLanguageDetectRequest, ChapterLanguageRequest


def _reading(unified="", unified_conf=0.9, korean="", korean_conf=0.2):
    return ProbeReading(unified, unified_conf, korean, korean_conf)


def test_script_counts_separate_hangul_kana_han_and_latin():
    assert script_counts("안녕 ひらカタ 漢字 Hi!") == {"hangul": 2, "kana": 4, "han": 2, "latin": 2}


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        ([_reading("どうしたの？", 0.95, "도우시", 0.3), _reading("本当にそうだ", 0.9, "", None)], "ja"),
        ([_reading("你到底想干什么", 0.93), _reading("我不知道", 0.9)], "ch"),
        ([_reading("WHAT ARE YOU DOING", 0.97, "WHAT ARE YOU DOING", 0.9)], "en"),
        ([_reading("口口口", 0.2, "무슨 일이야?", 0.94), _reading("", None, "괜찮아요", 0.9)], "korean"),
    ],
)
def test_classify_readings_votes_by_script(readings, expected):
    detection = classify_readings(readings)
    assert detection.lang == expected
    assert detection.reason == "script-vote"


def test_japanese_needs_some_kana_among_kanji():
    readings = [_reading("本当", 0.9), _reading("そうなんですか", 0.9), _reading("大丈夫です", 0.9)]
    assert classify_readings(readings).lang == "ja"


def test_korean_chapter_with_some_english_signs_stays_korean():
    readings = [
        _reading("", None, "어디 가는 거야?", 0.93),
        _reading("", None, "잠깐만 기다려", 0.9),
        _reading("", None, "알았어", 0.88),
        _reading("POLICE DEPARTMENT", 0.95, "POLICE DEPARTMENT", 0.9),
    ]
    assert classify_readings(readings).lang == "korean"


def test_classify_readings_refuses_to_guess_without_enough_text():
    detection = classify_readings([_reading("!?", 0.9), _reading("", None, "", None)])
    assert detection.lang is None
    assert detection.reason == "not-enough-text"


def test_classify_readings_reports_mixed_scripts():
    readings = [_reading("HELLO THERE FRIEND", 0.9), _reading("", None, "안녕하세요 친구", 0.9)]
    detection = classify_readings(readings)
    assert detection.lang is None
    assert detection.reason == "mixed-scripts"


def test_site_hint_matches_known_hosts_only():
    assert site_language_hint("https://asuracomic.net/series/x/chapter/1") == "en"
    assert site_language_hint("https://www.asurascans.com/x") == "en"
    assert site_language_hint("https://notasurascans.com.evil/x") is None
    assert site_language_hint("") is None


def _box(x1, y1, x2, y2, **extra):
    return {"id": f"b{x1}-{y1}", "x1": x1, "y1": y1, "x2": x2, "y2": y2, **extra}


def test_sample_regions_prefers_large_dialogue_boxes_across_pages():
    manifest = {"pages": [
        {"original": "a.png", "boxes": [_box(0, 0, 20, 20), _box(0, 0, 200, 200), _box(0, 0, 90, 90, class_name="sfx")]},
        {"original": "b.png", "skipped": True, "boxes": [_box(0, 0, 300, 300)]},
        {"original": "c.png", "boxes": [_box(0, 0, 100, 100), _box(0, 0, 50, 50, removed=True)]},
    ]}
    picked = sample_regions(manifest, limit=3)
    assert [(page, box["x2"]) for page, box in picked] == [(0, 200), (2, 100), (0, 20)]


@pytest.fixture
def saved_chapter(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    chapter_id = "abcd1234"
    (raw / chapter_id).mkdir(parents=True)
    (processed / chapter_id).mkdir(parents=True)
    image = np.full((400, 300, 3), 255, np.uint8)
    original = raw / chapter_id / "page_000.png"
    cv2.imwrite(str(original), image)
    manifest = {
        "chapter_id": chapter_id,
        "source_url": "https://example.com/ch/1",
        "pages": [{"original": str(original), "clean": str(original), "boxes": [
            _box(10, 10, 150, 90), _box(20, 200, 200, 300),
        ]}],
    }
    (processed / chapter_id / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(manifests, "PROCESSED_DIR", processed)
    monkeypatch.setattr(language_detect, "RAW_DIR", raw)
    return chapter_id


def _fake_reader(unified, korean, japanese=""):
    calls = []

    def read(crop, lang):
        calls.append((crop.shape, lang))
        text = {"ch": unified, "korean": korean, "ja": japanese}[lang]
        return SimpleNamespace(text=text, confidence=0.9 if lang == "ch" else 0.3)

    read.calls = calls
    return read


def test_detect_manifest_language_reads_each_sample_with_both_recognisers(saved_chapter):
    manifest = manifests.load_manifest_raw(saved_chapter)
    reader = _fake_reader("なにをしている", "")
    detection = detect_manifest_language(saved_chapter, manifest, reader)
    assert detection.lang == "ja"
    assert sorted(lang for _, lang in reader.calls) == ["ch", "ch", "korean", "korean"]
    assert all(shape[2] == 3 and shape[0] > 0 for shape, _ in reader.calls)


def test_vertical_japanese_misread_as_kanji_is_rescued_by_manga_ocr(saved_chapter):
    manifest = manifests.load_manifest_raw(saved_chapter)
    reader = _fake_reader("大口日本口", "", japanese="なにしてるの")
    detection = detect_manifest_language(saved_chapter, manifest, reader)
    assert (detection.lang, detection.reason) == ("ja", "manga-ocr-kana")


def test_chinese_stays_chinese_when_manga_ocr_finds_no_kana(saved_chapter):
    manifest = manifests.load_manifest_raw(saved_chapter)
    reader = _fake_reader("你到底想干什么", "", japanese="你到底想干什么")
    assert detect_manifest_language(saved_chapter, manifest, reader).lang == "ch"
    assert [lang for _, lang in reader.calls].count("ja") == 2


def test_korean_detection_skips_the_japanese_reader(saved_chapter):
    manifest = manifests.load_manifest_raw(saved_chapter)
    reader = SimpleNamespace(calls=[])

    def read(crop, lang):
        reader.calls.append(lang)
        return SimpleNamespace(text={"ch": "口口", "korean": "무슨 일이야", "ja": "なに"}[lang],
                               confidence={"ch": 0.2, "korean": 0.95, "ja": None}[lang])

    assert detect_manifest_language(saved_chapter, manifest, read).lang == "korean"
    assert "ja" not in reader.calls


def test_detect_endpoint_persists_result_once_and_manual_choice_wins(saved_chapter, monkeypatch):
    reader = _fake_reader("你到底想干什么", "")
    monkeypatch.setattr(ocr_router, "ocr", SimpleNamespace(read_probe=reader))

    first = asyncio.run(ocr_router.detect_chapter_language(saved_chapter))
    assert first["source_lang"] == "ch"
    assert first["source_lang_origin"] == "auto"
    assert first["detection"]["reason"] == "script-vote"
    stored = manifests.load_manifest_raw(saved_chapter)
    assert (stored["source_lang"], stored["source_lang_origin"]) == ("ch", "auto")

    reader.calls.clear()
    again = asyncio.run(ocr_router.detect_chapter_language(saved_chapter))
    assert again["source_lang"] == "ch" and again["detection"] is None
    assert reader.calls == []

    manual = ocr_router.set_chapter_language(saved_chapter, ChapterLanguageRequest(source_lang="ko"))
    assert (manual["source_lang"], manual["source_lang_origin"]) == ("korean", "manual")
    forced = asyncio.run(ocr_router.detect_chapter_language(
        saved_chapter, ChapterLanguageDetectRequest(force=True),
    ))
    assert forced["source_lang"] == "ch"


def test_detect_endpoint_leaves_language_unset_when_uncertain(saved_chapter, monkeypatch):
    monkeypatch.setattr(ocr_router, "ocr", SimpleNamespace(read_probe=_fake_reader("", "")))
    answer = asyncio.run(ocr_router.detect_chapter_language(saved_chapter))
    assert answer["source_lang"] is None
    assert answer["detection"]["reason"] == "not-enough-text"
    assert "source_lang" not in manifests.load_manifest_raw(saved_chapter)


def test_set_language_rejects_unknown_codes():
    with pytest.raises(ValueError):
        ChapterLanguageRequest(source_lang="fr")
