from __future__ import annotations

import asyncio
import copy
import json

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


def test_vision_client_sends_original_and_clean_with_short_ids(tmp_path, monkeypatch):
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
                    '{"translations":[{"id":"1","translated_text":"Tôi hiểu rồi"}]}'
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
    assert "<task>" in system["content"] and "<vietnamese>" in system["content"]
    assert "font_choices" not in system["content"] and "emphasis.anton" not in system["content"], "fonts are not the model's"
    assert answer.font_choices == {"text_1": {"font_id": "dialogue.mac-dinh-3", "font_mode": "ai"}}, "no role: base font"
    content = payload["messages"][1]["content"]
    images = [item for item in content if item.get("type") == "image_url"]
    assert len(images) == 1, "only the raw slice with its boxes drawn on"
    assert content[0]["text"] == "RAW SLICE WITH TEXT BOXES"
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in images)
    prompt = content[-1]["text"]
    assert '"id":"1"' in prompt and "text_1" not in prompt
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
    answer = asyncio.run(translation_router.translate_page_in_context(request))
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
    answer = asyncio.run(translation_router.translate_page_in_context(request))
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
        # The model sees the objects numbered 1, 2, ... and answers with those numbers.
        '{"translations":[{"id":"1","translated_text":"Thầy ơi, em đến rồi."},{"id":"2","translated_text":"Vào đi."}],'
        '"speakers":{"1":"Ian","2":"Baldur"},'
        '"characters":[{"name":"Ian","note":"student, 17"},{"name":"Baldur","note":"Ian\'s teacher"}],'
        '"address":[{"from":"Ian","to":"Baldur","self":"em","other":"thầy"}]}',
        '{"translations":[{"id":"1","translated_text":"Em hiểu\\nrồi.","role":"dialogue"},'
        '{"id":"2","translated_text":"RẦM","role":"sfx","review":true},{"id":"9","translated_text":"x","role":"bogus"}]}',
    ])
    prompts = []

    class Response:
        status_code, ok = 200, True

        def __init__(self, content):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}], "usage": {}}

    def post(url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"][-1]["text"])
        systems.append(kwargs["json"]["messages"][0]["content"])
        return Response(next(answers))

    systems = []
    monkeypatch.setattr("app.translation.vision.requests.post", post)
    memory = ChapterMemory("Academy regression story")
    translator = VisionPageTranslator(PROVIDERS["openai"], "vision-test")
    item = lambda item_id: {"id": item_id, "text": "", "region": [1, 2, 30, 40]}
    translator.translate_page(original, clean, [item("a"), item("b")], api_key="k", source_lang="ko",
                              target_lang="vi", memory=memory, slice_number=1, slice_total=2)
    second = translator.translate_page(original, clean, [item("c"), item("d"), item("f")], api_key="k", source_lang="ko",
                                       target_lang="vi", memory=memory, slice_number=2, slice_total=2)

    assert "SLICE 1 of 2" in prompts[0] and "Academy regression story" not in prompts[0]
    assert systems[0] == systems[1] and "Academy regression story" in systems[0], "fixed chapter notes sit in the cached system text"
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


def test_vision_parser_accepts_a_translations_map_and_integer_ids():
    assert parse_vision_translation('{"translations":{"1":"A","2":"B"}}', {"1", "2"}) == {"1": "A", "2": "B"}
    assert parse_vision_translation('{"translations":[{"id":1,"translated_text":"A"}]}', {"1"}) == {"1": "A"}


def test_vision_reply_tells_unanswered_from_deliberately_blank_objects(tmp_path, monkeypatch):
    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "gray").save(clean)

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "translations": [{"id": "1", "translated_text": "Chào"}, {"id": "2", "translated_text": ""}],
            })}}], "usage": {}}

    sent = []
    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: sent.append(kwargs) or Response())
    translator = VisionPageTranslator(PROVIDERS["openai"], "vision-test")
    item = lambda item_id: {"id": item_id, "text": "", "region": [1, 2, 30, 40]}
    result = translator.translate_page(original, clean, [item("t1"), item("mark"), item("lost")],
                                       api_key="k", source_lang="en", target_lang="vi")
    assert result.translations == {"t1": "Chào", "mark": "", "lost": ""}
    assert result.missing_ids == {"lost"}, "answered-empty differs from not answered"
    assert '"keep"' not in sent[0]["json"]["messages"][0]["content"], "checkpoint 3 owns kept and missed text"



def test_vietnamese_line_with_cjk_letters_is_retried_and_kept_out_of_memory(tmp_path, monkeypatch):
    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "gray").save(clean)

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "translations": [{"id": "1", "translated_text": "Chào"},
                                 {"id": "2", "translated_text": "Tôi 发现自己 kẹt rồi"}],
                "characters": [{"name": "Frondier", "note": "hoảng loạn khi 发现自己 kẹt"},
                               {"name": "Angper", "note": "cha của Frondier"}],
            })}}], "usage": {}}

    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: Response())
    from app.translation.context import ChapterMemory

    memory = ChapterMemory("")
    item = lambda item_id: {"id": item_id, "text": "", "region": [1, 2, 30, 40]}
    result = VisionPageTranslator(PROVIDERS["openai"], "vision-test").translate_page(
        original, clean, [item("a"), item("b")], api_key="k", source_lang="en", target_lang="vi", memory=memory)
    assert result.translations == {"a": "Chào", "b": ""}
    assert result.missing_ids == {"b"}
    assert [c["name"] for c in memory.snapshot()["characters"]] == ["Angper"]


def test_english_left_as_the_translation_is_treated_as_untranslated():
    from app.translation.vision import _untranslated_english

    assert _untranslated_english("WITH ITS LIFELIKE AI,\nVAST OPEN WORLD,")
    assert _untranslated_english("Were you talking to me?")
    assert not _untranslated_english("GAME OVER") and not _untranslated_english("FRONDIER DE ROAH!")
    assert not _untranslated_english("Level của ngươi là bao nhiêu?")


def test_the_font_follows_role_and_container_not_the_model():
    from app.render.font_guide import font_for

    assert font_for("dialogue", "bubble") == font_for("shout", "bubble") == "dialogue.mac-dinh-3"
    assert font_for("narration", "box") == "dialogue.mac-dinh-3", "Mặc Định 2 is handwriting, not a caption face"
    assert font_for("thought", "free") == "narration.mac-dinh-2"
    assert font_for("narration", "free") == font_for("title", "free") == "emphasis.anton", "bold captions on the art"
    assert font_for("shout", "spiky") == "emphasis.bangers" and font_for("system_ui", "bubble") == "skill.exo-2"
    assert font_for(None, None) == "dialogue.mac-dinh-3"
    assert font_for("dialogue", "bubble", "es") == "thought.comic-neue", "the base font has no ñ, ç or ß"
    assert font_for("narration", "free", "de") == "emphasis.anton"
    assert parse_vision_translation('{"translations":[{"id":"a","translated_text":"Thì…"}]}', {"a"}) == {"a": "Thì..."}


def test_sound_effect_left_as_art_returns_its_region_for_restoring(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": ""}, self.model, {}, None, roles={"obj_1": "sfx"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="en", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["blank_ids"] == ["obj_1"] and info["art_regions"] == [region]


def test_a_translated_page_letters_each_kind_of_text_in_its_one_font(tmp_path, monkeypatch):
    original, clean = tmp_path / "original.png", tmp_path / "clean.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "gray").save(clean)

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "translations": [{"id": "1", "translated_text": "Khi ấy", "role": "narration", "container": "free"},
                                 {"id": "2", "translated_text": "Chạy!", "role": "shout", "container": "spiky"},
                                 {"id": "3", "translated_text": "Ừ", "role": "dialogue", "container": "bubble"}],
                "font_choices": {"1": "narration.lora", "2": "thought.itim", "3": "emphasis.anton"},
            })}}], "usage": {}}

    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: Response())
    items = [{"id": item_id, "text": "", "region": [1, 2, 30, 40]} for item_id in ("a", "b", "c")]
    result = VisionPageTranslator(PROVIDERS["openai"], "vision-test").translate_page(
        original, clean, items, api_key="k", source_lang="en", target_lang="vi")
    assert {key: value["font_id"] for key, value in result.font_choices.items()} == {
        "a": "emphasis.anton", "b": "emphasis.bangers", "c": "dialogue.mac-dinh-3"}, "a model's font pick is ignored"


def test_the_character_budget_is_where_the_source_letters_were():
    page = {"text_objects": [{"id": "t", "region": {"x1": 0, "y1": 0, "x2": 1000, "y2": 800},
                              "letter_bounds": {"x1": 300, "y1": 300, "x2": 700, "y2": 400}},
                             {"id": "u", "region": {"x1": 0, "y1": 900, "x2": 100, "y2": 950}}]}
    fit = {candidate["id"]: candidate["fit_region"] for candidate in translation_router._vision_candidates(page, force=False)}
    assert fit == {"t": [285, 285, 715, 415], "u": [0, 900, 100, 950]}


def test_glossary_vote_drops_a_misread_name_and_fixes_one_form_of_address():
    from app.ai_mode.glossary import merge_glossaries
    from app.translation.context import ChapterMemory

    reads = [{"names": [{"source": "Yanguo", "target": "Yanguo"}],
              "address": [{"from": "disciples", "to": "master", "self": "bọn con", "other": "sư phụ"}]}] * 2
    reads.append({"names": [{"source": "Yanglu"}, {"source": "Lee Jin"}], "terms": [{"source": "Qi Refining", "target": "Luyện Khí"}],
                  "address": [{"from": "Disciples", "to": "Master", "self": "bọn ta", "other": "sư phụ"}]})
    glossary = merge_glossaries(reads + ["junk", {"names": "junk"}])
    assert [name["source"] for name in glossary["names"]] == ["Yanguo", "Lee Jin"]
    assert glossary["address"] == [{"from": "disciples", "to": "master", "self": "bọn con", "other": "sư phụ"}]
    assert glossary["terms"] == [{"source": "qi refining", "target": "Luyện Khí"}]
    clash = merge_glossaries([{"names": [{"source": "Hero", "target": "Dũng Sĩ"}], "terms": [{"source": "hero", "target": "Anh Hùng"}]}])
    assert clash["terms"] == [], "a term that is also a name keeps the name's translation"
    assert ChapterMemory("", glossary).snapshot()["glossary"] == glossary


def test_text_lettered_by_the_next_slice_is_shown_but_kept_out_of_every_translation(tmp_path, monkeypatch):
    import base64

    from app.translation.vision import ELSEWHERE_COLOR

    original, clean = tmp_path / "o.png", tmp_path / "c.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "white").save(clean)
    sent = []

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": '{"translations":[{"id":"1","translated_text":"Chào"}]}'}}],
                    "usage": {}}

    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: sent.append(kwargs["json"]) or Response())
    VisionPageTranslator(PROVIDERS["openai"], "vision-test").translate_page(
        original, clean, [{"id": "top", "text": "", "region": [20, 20, 200, 120]}], api_key="k",
        source_lang="en", target_lang="vi", elsewhere=[[20, 400, 380, 560]])
    parts = sent[0]["messages"][1]["content"]
    prompt = next(part["text"] for part in parts if part["type"] == "text" and "objects" in part["text"])
    assert '"lettered_elsewhere":[[20,400,380,560]]' in prompt and "never fold its words" in prompt
    marked = cv2.imdecode(np.frombuffer(base64.b64decode(parts[-2]["image_url"]["url"].split(",", 1)[1]), np.uint8),
                          cv2.IMREAD_COLOR)
    assert np.abs(marked[480, 16].astype(int) - ELSEWHERE_COLOR).max() < 40, "the grey box is drawn on the slice"
    assert marked[480, 22].min() > 200, "outlined just outside the box, never over its letters"


def test_lines_outside_every_box_come_back_as_pixel_boxes(tmp_path, monkeypatch):
    original, clean = tmp_path / "o.png", tmp_path / "c.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "white").save(clean)

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "translations": [{"id": "1", "translated_text": "Chào"}], "speakers": {"1": "Sunny", "9": "x"},
                "unboxed": [{"x1": 100, "y1": 500, "x2": 900, "y2": 600}, {"x1": 9, "y1": 9, "x2": 1, "y2": 1}]})}}],
                "usage": {}}

    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: Response())
    result = VisionPageTranslator(PROVIDERS["openai"], "vision-test").translate_page(
        original, clean, [{"id": "top", "text": "", "region": [20, 20, 200, 120]}], api_key="k",
        source_lang="en", target_lang="vi")
    assert result.unboxed == ((40, 300, 361, 361),)
    assert result.speakers == {"top": "Sunny"}, "speakers come back under the real ids"
