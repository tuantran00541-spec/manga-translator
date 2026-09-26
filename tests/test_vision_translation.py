from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

import app.config as config
import app.manifest_utils as manifests
import app.routers.render_commit as render_commit
import app.routers.translation as translation_router
from app.ai_providers import PROVIDERS
from app.translation.vision import (
    VisionPageTranslator, VisionTranslationResult, parse_vision_translation,
)

CHAPTER = "a1b2c3d4"


def test_vision_parser_rejects_unknown_duplicate_and_omitted_ids():
    assert parse_vision_translation(
        '{"translations":[{"id":"a","translated_text":"Xin chào"},'
        '{"id":"b","translated_text":""}]}', {"a", "b"}
    ) == {"a": "Xin chào", "b": ""}
    bad = [
        '{"translations":[{"id":"a","translated_text":"x"},{"id":"a","translated_text":"y"}]}',
        '{"translations":[{"id":"unknown","translated_text":"x"}]}',
        '{"translations":[{"id":"a","translated_text":"x"}]}',
        '{"translations":[{"id":"a","translated_text":42},{"id":"b","translated_text":"x"}]}',
    ]
    for raw in bad:
        with pytest.raises(RuntimeError):
            parse_vision_translation(raw, {"a", "b"})


def test_vision_client_sends_original_and_clean_with_existing_ids(tmp_path, monkeypatch):
    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (160, 120), "white").save(original)
    Image.new("RGB", (160, 120), "gray").save(clean)
    sent = []

    class Response:
        status_code = 200
        ok = True

        def json(self):
            return {
                "model": "vision-test",
                "choices": [{"message": {"content": (
                    '{"translations":[{"id":"text_1","translated_text":"Tôi hiểu rồi"}]}'
                )}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            }

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return Response()

    monkeypatch.setattr("app.translation.vision.requests.post", post)
    translator = VisionPageTranslator(PROVIDERS["openai"], "vision-test")
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "I see", "region": [10, 20, 50, 40]}],
        api_key="fake-test-key", source_lang="en", target_lang="vi",
    )
    assert answer.translations == {"text_1": "Tôi hiểu rồi"}
    assert len(sent) == 1
    payload = sent[0][1]["json"]
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert "localization editor" in system["content"] and "VIETNAMESE" in system["content"]
    assert "dialogue.mac-dinh-3" in system["content"] and "narration.mac-dinh-2" in system["content"]
    content = payload["messages"][1]["content"]
    images = [item for item in content if item.get("type") == "image_url"]
    assert len(images) == 2
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in images)
    assert images[0]["image_url"]["url"] != images[1]["image_url"]["url"]
    prompt = content[0]["text"]
    assert '"id":"text_1"' in prompt
    assert '"bbox_xyxy":[10,20,50,40]' in prompt
    assert "fontSize" not in prompt and "strokeColor" not in prompt


@pytest.fixture
def saved_chapter(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    for root in (raw, processed, output):
        (root / CHAPTER).mkdir(parents=True)
    original = raw / CHAPTER / "page.png"
    clean = processed / CHAPTER / "clean.png"
    image = np.full((150, 220, 3), 245, dtype=np.uint8)
    cv2.putText(image, "HELLO", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    assert cv2.imwrite(str(original), image)
    assert cv2.imwrite(str(clean), np.full_like(image, 245))
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)
    monkeypatch.setattr(manifests, "PROCESSED_DIR", processed)
    monkeypatch.setattr(translation_router, "RAW_DIR", raw)
    monkeypatch.setattr(translation_router, "PROCESSED_DIR", processed)
    monkeypatch.setattr(render_commit, "OUTPUT_DIR", output)
    saved_style = {
        "color": "#df1100", "font": "default", "fontSize": "27",
        "bold": True, "strokeWidth": "2", "strokeColor": "#ffffff",
        "bgColor": "transparent", "cornerRadius": "4",
        "horizontalAlign": "right", "verticalAlign": "top",
    }
    saved_region = {"x1": 20, "y1": 23, "x2": 180, "y2": 100}
    manifests.save_manifest_raw(CHAPTER, {
        "chapter_id": CHAPTER,
        "pages": [{
            "original": original.as_posix(), "clean": clean.as_posix(),
            "boxes": [], "text_objects": [{
                "id": "obj_1", "shape": "rectangle",
                "region": saved_region, "style": saved_style,
                "ocr_text": "Hello", "translation": "",
            }],
            "skipped": False, "process_required": False,
            "source_revision": 1, "clean_revision": 1,
            "render_revision": 0, "rendered": False,
        }],
        "workflow": {"stage": "review", "page_index": 0},
    })
    return saved_style, saved_region, output / CHAPTER / "page_000.png"


def test_vision_page_commits_only_translation_and_renders_saved_style(
    saved_chapter, monkeypatch,
):
    style, region, output = saved_chapter
    observed = []
    original_render = render_commit.render_text_objects

    def capture_render(*args, **kwargs):
        observed.append(copy.deepcopy(args[2]))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(render_commit, "render_text_objects", capture_render)
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, original_path, cleaned_path, items, **kwargs: VisionTranslationResult(
            {"obj_1": "Xin chào"}, self.model, {"prompt_tokens": 100}, None
        ),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
        source_lang="en", target_lang="vi",
    )
    answer = asyncio.run(translation_router.translate_page_with_images(request))
    info = answer["translation_run"]
    assert info["translated"] == 1
    assert info["rendered_pages"] == [0]
    assert info["render_error"] is None
    assert output.is_file()
    page = manifests.load_manifest_raw(CHAPTER)["pages"][0]
    obj = page["text_objects"][0]
    assert obj["region"] == region
    assert obj["style"] == style
    assert obj["translation"] == "Xin chào"
    assert observed[0][0]["style"] == style
    assert observed[0][0]["region"] == region
    assert observed[0][0]["translation"] == "Xin chào"
    assert page["rendered"] is not False


def test_vision_page_rejects_stale_region_without_overwriting(saved_chapter, monkeypatch):
    style, _region, output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")

    def concurrent_edit(self, original_path, cleaned_path, items, **kwargs):
        with manifests.get_manifest_lock(CHAPTER):
            manifest = manifests.load_manifest_raw(CHAPTER)
            manifest["pages"][0]["text_objects"][0]["region"]["x1"] = 45
            manifests.save_manifest_raw(CHAPTER, manifest)
        return VisionTranslationResult({"obj_1": "Xin chào"}, self.model, {}, None)

    monkeypatch.setattr("app.translation.vision.VisionPageTranslator.translate_page", concurrent_edit)
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
    )
    answer = asyncio.run(translation_router.translate_page_with_images(request))
    assert answer["translation_run"]["translated"] == 0
    assert answer["translation_run"]["stale"] == 1
    assert answer["translation_run"]["rendered_pages"] == []
    assert not output.exists()
    page = manifests.load_manifest_raw(CHAPTER)["pages"][0]
    assert page["text_objects"][0]["translation"] == ""
    assert page["text_objects"][0]["style"] == style
    assert page["text_objects"][0]["region"]["x1"] == 45


def test_chapter_memory_carries_characters_address_and_recent_lines_to_the_next_slice(tmp_path, monkeypatch):
    from app.translation.context import ChapterMemory

    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (160, 120), "white").save(original)
    Image.new("RGB", (160, 120), "gray").save(clean)
    answers = iter([
        '{"translations":[{"id":"a","translated_text":"Thầy ơi, em đến rồi."},{"id":"b","translated_text":"Vào đi."}],'
        '"speakers":{"a":"Ian","b":"Baldur"},'
        '"characters":[{"name":"Ian","note":"student, 17"},{"name":"Baldur","note":"Ian\'s teacher"}],'
        '"address":[{"from":"Ian","to":"Baldur","self":"em","other":"thầy"}]}',
        '{"translations":[{"id":"c","translated_text":"Em hiểu\\nrồi.","role":"dialogue"},'
        '{"id":"d","translated_text":"RẦM","role":"sfx","review":true},{"id":"e","translated_text":"x","role":"bogus"}]}',
    ])
    prompts = []

    class Response:
        status_code, ok = 200, True

        def __init__(self, content):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}], "usage": {}}

    def post(url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"][0]["text"])
        return Response(next(answers))

    monkeypatch.setattr("app.translation.vision.requests.post", post)
    memory = ChapterMemory("Academy regression story")
    translator = VisionPageTranslator(PROVIDERS["openai"], "vision-test")
    item = lambda item_id: {"id": item_id, "text": "", "region": [1, 2, 30, 40]}
    translator.translate_page(original, clean, [item("a"), item("b")], api_key="k", source_lang="ko",
                              target_lang="vi", memory=memory, slice_number=1, slice_total=2)
    second = translator.translate_page(original, clean, [item("c"), item("d"), item("f")], api_key="k", source_lang="ko",
                                       target_lang="vi", memory=memory, slice_number=2, slice_total=2)

    assert "Academy regression story" in prompts[0] and "SLICE 1 of 2" in prompts[0]
    carried = prompts[1]
    assert '"self":"em","other":"thầy"' in carried
    assert '"name":"Baldur"' in carried
    assert '"speaker":"Ian","text":"Thầy ơi, em đến rồi."' in carried
    assert second.translations == {"c": "Em hiểu\nrồi.", "d": "RẦM", "f": ""}, "a missing id is left empty"
    assert second.roles == {"c": "dialogue", "d": "sfx"} and second.review_ids == {"d"}


def test_chapter_memory_is_bounded_and_ignores_malformed_entries():
    from app.translation.context import MAX_CHARACTERS, RECENT_LINES, ChapterMemory

    memory = ChapterMemory("x" * 5000)
    memory.update(1, {"characters": [{"name": f"N{i}", "note": "n"} for i in range(MAX_CHARACTERS + 10)] + ["bad", {"note": "no name"}],
                      "address": [{"from": "A", "to": ""}, "bad", {"from": "A", "to": "B", "self": "tôi"}]},
                  {f"t{i}": "line " + "y" * 500 for i in range(RECENT_LINES + 5)}, [f"t{i}" for i in range(RECENT_LINES + 5)])
    sheet = memory.snapshot()
    assert len(sheet["story_notes"]) == 1500
    assert len(sheet["characters"]) == MAX_CHARACTERS
    assert sheet["address"] == [{"from": "A", "to": "B", "self": "tôi"}]
    assert len(sheet["recent_lines"]) == RECENT_LINES and len(sheet["recent_lines"][0]["text"]) == 160
