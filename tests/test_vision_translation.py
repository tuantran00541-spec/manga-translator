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
    VisionPageTranslator, VisionTranslationResult, collapse_repeated_lines,
    parse_vision_translation,
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


def test_repeated_lines_collapse_to_one_and_differing_lines_stand():
    assert collapse_repeated_lines("NGUYÊN TỐ\nNGUYÊN TỐ") == "NGUYÊN TỐ"
    assert collapse_repeated_lines("NGUYÊN TỐ\nNGUYÊN TỐ\nNGUYÊN TỐ") == "NGUYÊN TỐ"
    assert collapse_repeated_lines("NGUYÊN TỐ") == "NGUYÊN TỐ"
    assert collapse_repeated_lines("NGUYÊN TỐ\nKHÁC") == "NGUYÊN TỐ\nKHÁC"
    assert collapse_repeated_lines("NGUYÊN TỐ NGUYÊN TỐ") == "NGUYÊN TỐ NGUYÊN TỐ"


def test_vision_page_commits_a_doubled_translation_collapsed(
    saved_chapter, monkeypatch,
):
    _style, _region, _output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, original_path, cleaned_path, items, **kwargs: VisionTranslationResult(
            {"obj_1": "NGUYÊN TỐ\nNGUYÊN TỐ"}, self.model, {"prompt_tokens": 100}, None
        ),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
        source_lang="en", target_lang="vi",
    )
    answer = asyncio.run(translation_router.translate_page_in_context(request))
    assert answer["translation_run"]["translated"] == 1
    obj = manifests.load_manifest_raw(CHAPTER)["pages"][0]["text_objects"][0]
    assert obj["translation"] == "NGUYÊN TỐ"
    assert obj["auto_translation"] == "NGUYÊN TỐ"


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


def test_rerun3_english_returns_are_caught_even_with_confusable_punctuation():
    from app.translation.vision import _untranslated_english

    assert _untranslated_english("IS THERE A PROBLEM WITH THAT?")
    assert _untranslated_english("OF COURSE NOT HE'S STILL A SKILLED SO THE SCHOOL ENDB SUSPENSION.")
    assert _untranslated_english(
        "SHAANXI MAY BE FAR, BUT IT WOULDN'T TAKE LONG TO GET THERE IF I SET MY MIND TO IT...")
    assert _untranslated_english("SENIOR BROTHER!")
    assert _untranslated_english("IS THERE A PROBLEM WITH THAT?\u00a0")
    assert _untranslated_english("IS THERE A PROBLEM WITH THAT?\u200b")
    assert _untranslated_english("OF COURSE NOT HE\u2019S STILL A SKILLED SO THE SCHOOL ENDB SUSPENSION.")
    assert _untranslated_english(
        "SHAANXI MAY BE FAR, BUT IT WOULDN\u2019T TAKE LONG TO GET THERE IF I SET MY MIND TO IT\u2026")
    assert _untranslated_english("SENIOR BROTHER!\u2009")
    assert _untranslated_english("SENIOR BROTHER!\u2007")
    assert _untranslated_english("SENIOR BROTHER!\u202f")
    assert _untranslated_english("SENIOR BROTHER!\u3000")
    assert _untranslated_english("SENIOR BROTHER\uff01")
    assert _untranslated_english("SENIOR BROTHER!\u00ad")
    assert _untranslated_english("SENIOR BROTHER!\u200e")
    assert _untranslated_english("\uff33\uff25\uff2e\uff29\uff2f\uff32 \uff22\uff32\uff2f\uff34\uff28\uff25\uff32\uff01")
    assert not _untranslated_english("HELMUT...?")
    assert not _untranslated_english("GAME OVER") and not _untranslated_english("FRONDIER DE ROAH!")
    assert not _untranslated_english("HELMUT VỪA RỒI PHÒNG 301...?")
    assert not _untranslated_english("MM!")


def test_vietnamese_written_without_diacritics_is_treated_as_untranslated():
    from app.translation.vision import _missing_diacritics

    assert _missing_diacritics("Hậu quả do năng lực\ndang tro thanh mot van de xoi hoi moi")
    assert _missing_diacritics("Toi khong the lam nhu vay")
    assert _missing_diacritics("Toi khong the lam nhu vay\uff01")
    assert _missing_diacritics("Toi khong the lam nhu vay\u2009")
    assert not _missing_diacritics("Bọn tôi không thể!") and not _missing_diacritics("Level Up! Dungeon Break")


def test_a_vision_misread_of_korean_sfx_as_latin_noise_is_treated_as_untranslated():
    from app.translation.vision import _latin_noise

    assert _latin_noise("1/10")
    assert _latin_noise("N/O/Z/N")
    assert not _latin_noise("KYA AH AH!")
    assert not _latin_noise("HUFF")
    assert not _latin_noise("PHÙ...")
    assert not _latin_noise("*THỞ DÀI*")
    assert not _latin_noise("MM!")
    assert not _latin_noise("...")
    assert not _latin_noise("12")


def test_a_bare_scream_is_treated_as_untranslated():
    from app.translation.vision import _bare_scream

    assert _bare_scream("AAAAA!!!")
    assert _bare_scream("HHHHH...")
    assert _bare_scream("ÁÁÁÁ!!!")
    assert not _bare_scream("HAHA!")
    assert not _bare_scream("A!")
    assert not _bare_scream("AA!")
    assert not _bare_scream("Chuyện đó có vấn đề gì không?")


def test_a_hangul_sfx_misread_as_latin_consonants_is_treated_as_untranslated():
    from app.translation.vision import _hangul_misread

    assert _hangul_misread("푸르륵", "FFR")
    assert not _hangul_misread("푸르륵", "BÙM")
    assert not _hangul_misread("푸르륵", "RẦM")
    assert not _hangul_misread("푸르륵", "PHÙ...")
    assert not _hangul_misread("HELLO", "FFR")
    assert not _hangul_misread("푸르륵", "Chuyện đó có vấn đề gì không?")


def test_mixed_vietnamese_and_english_splits_instead_of_passing():
    from app.translation.vision import _drop_foreign_script, _untranslated_english

    assert _untranslated_english("Anh ơi SENIOR BROTHER")
    assert _untranslated_english("Đúng vậy\nSENIOR BROTHER!")
    assert not _untranslated_english("Tôi to lắm")
    assert not _untranslated_english("Quả me to")

    def filtered(value):
        result = VisionTranslationResult({"a": value}, "m", {}, None)
        data = {"translations": [{"id": "a", "translated_text": value}]}
        kept, kept_data = _drop_foreign_script(result, data)
        entry = kept_data["translations"][0] if kept_data["translations"] else None
        return kept.translations["a"], entry["translated_text"] if entry else None

    assert filtered("Anh ơi SENIOR BROTHER") == ("Anh ơi", "Anh ơi")
    assert filtered("Đúng vậy\nSENIOR BROTHER!") == ("Đúng vậy", "Đúng vậy")
    assert filtered("SENIOR BROTHER!") == ("", None)
    assert filtered("IS THE / A PROBI / WITH TH") == ("", None)
    assert filtered("OF COURSE NOT / … ENDB / SUSPENSION.") == ("", None)
    assert filtered("WITH ITS LIFELIKE AI,\nVAST OPEN WORLD,") == ("", None)
    assert filtered("Tôi to lắm") == ("Tôi to lắm", "Tôi to lắm")
    assert filtered("NGỦ") == ("NGỦ", "NGỦ")
    assert filtered("MM!") == ("MM!", "MM!")
    assert filtered("ENDB") == ("ENDB", "ENDB")
    assert filtered("GAME OVER") == ("GAME OVER", "GAME OVER")
    assert filtered("ÁÁÁÁA!!!") == ("", None)
    assert filtered("Đúng vậy\nGAME OVER") == ("Đúng vậy\nGAME OVER", "Đúng vậy\nGAME OVER")


def test_a_diacritic_scream_beside_a_vietnamese_line_drops_only_the_scream():
    from app.translation.vision import _drop_foreign_script

    def filtered(value):
        result = VisionTranslationResult({"a": value}, "m", {}, None)
        data = {"translations": [{"id": "a", "translated_text": value}]}
        kept, kept_data = _drop_foreign_script(result, data)
        entry = kept_data["translations"][0] if kept_data["translations"] else None
        return kept.translations["a"], entry["translated_text"] if entry else None

    assert filtered("Đúng vậy\nÁÁÁÁA!!!") == ("Đúng vậy", "Đúng vậy")
    assert filtered("ÁÁÁÁA!!!\nĐi thôi") == ("Đi thôi", "Đi thôi")


def test_english_names_are_dropped_from_memory_not_learned(tmp_path, monkeypatch):
    from app.translation.vision import _drop_foreign_script

    result = VisionTranslationResult({"a": "Chào"}, "m", {}, None)
    data = {"translations": [{"id": "a", "translated_text": "Chào"}],
            "characters": [{"name": "Senior Brother", "note": "x"},
                           {"name": "Minh", "note": "y"}],
            "address": [{"from": "Senior Brother", "to": "Little Sister", "self": "I", "other": "You"},
                        {"from": "Minh", "to": "Lan", "self": "em", "other": "anh"}]}
    _kept, kept_data = _drop_foreign_script(result, data)
    assert [entry["name"] for entry in kept_data["characters"]] == ["Minh"]
    assert [(entry["from"], entry["to"]) for entry in kept_data["address"]] == [("Minh", "Lan")]


def test_other_gates_also_judge_per_line_not_per_value():
    from app.translation.vision import _bare_scream, _drop_foreign_script, _hangul_misread, _latin_noise

    assert _latin_noise("Chào\n1/10")
    assert _bare_scream("Chào\nAAAAA!!!")
    assert _hangul_misread("푸르륵", "BÙM\nFFR")
    assert not _latin_noise("Chào\nPHÙ...")
    assert not _bare_scream("Chào\nHAHA!")
    assert not _hangul_misread("푸르륵", "BÙM\nRẦM")

    def filtered(value, source=""):
        result = VisionTranslationResult({"a": value}, "m", {}, None)
        data = {"translations": [{"id": "a", "source": source, "translated_text": value}]}
        kept, _data = _drop_foreign_script(result, data)
        return kept.translations["a"]

    assert filtered("Chào\n1/10") == "Chào"
    assert filtered("Chào\nAAAAA!!!") == "Chào"
    assert filtered("BÙM\nFFR", "푸르륵") == "BÙM"
    assert filtered("FFR", "푸르륵") == ""
    assert filtered("FFR") == "FFR"


def test_translate_page_keeps_the_vietnamese_line_of_a_mixed_return(tmp_path, monkeypatch):
    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (400, 600), "white").save(original)
    Image.new("RGB", (400, 600), "gray").save(clean)

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "translations": [{"id": "1", "translated_text": "Đúng vậy\nSENIOR BROTHER!"},
                                 {"id": "2", "translated_text": "Anh ơi SENIOR BROTHER"}],
            })}}], "usage": {}}

    monkeypatch.setattr("app.translation.vision.requests.post", lambda url, **kwargs: Response())
    item = lambda item_id: {"id": item_id, "text": "", "region": [1, 2, 30, 40]}
    result = VisionPageTranslator(PROVIDERS["openai"], "vision-test").translate_page(
        original, clean, [item("a"), item("b")], api_key="k", source_lang="en", target_lang="vi")
    assert result.translations == {"a": "Đúng vậy", "b": "Anh ơi"}
    assert result.missing_ids == set()


def test_recent_lines_keep_the_source_so_a_sentence_can_run_into_the_next_slice():
    from app.translation.context import ChapterMemory

    memory = ChapterMemory("")
    memory.update(3, {"translations": [{"id": "a", "source": "THEY CONQUERED THE WORLD IN THE"}], "speakers": {"a": "narration"}},
                  {"a": "Họ chinh phục thế giới"}, ["a"])
    assert memory.snapshot()["recent_lines"] == [
        {"slice": 3, "speaker": "narration", "source": "THEY CONQUERED THE WORLD IN THE", "text": "Họ chinh phục thế giới"}]


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


def test_a_sound_effect_drawn_twice_in_one_panel_is_translated_once(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    latest = manifests.load_manifest_raw(CHAPTER)
    latest["pages"][0]["text_objects"].append({
        "id": "obj_2", "shape": "rectangle",
        "region": {"x1": 20, "y1": 110, "x2": 150, "y2": 180},
        "style": _style, "ocr_text": "Hello", "translation": "",
    })
    manifests.save_manifest_raw(CHAPTER, latest)
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult(
            {"obj_1": "CỔ VŨ", "obj_2": "CỔ VŨ"}, self.model, {}, None,
            roles={"obj_1": "free_text", "obj_2": "free_text"},
            sources={"obj_1": "와성", "obj_2": "와성"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 1
    assert info["blank_ids"] == ["obj_2"]
    assert info["story_blank_ids"] == []
    assert info["art_regions"] == [{"x1": 20, "y1": 110, "x2": 150, "y2": 180}]
    assert info["missing_ids"] == []


def test_sound_effect_left_as_art_returns_its_region_for_restoring(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": ""}, self.model, {}, None, roles={"obj_1": "sfx"},
                                                       sources={"obj_1": "쾅"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="en", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["blank_ids"] == ["obj_1"] and info["art_regions"] == [region]


def test_a_watermark_the_model_calls_a_sound_effect_stays_erased(saved_chapter, monkeypatch):
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": ""}, self.model, {}, None, roles={"obj_1": "sfx"},
                                                       sources={"obj_1": "ASURASCANS.COM"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["blank_ids"] == ["obj_1"] and info["art_regions"] == []


def test_a_sound_effect_the_artist_drew_stays_as_art_even_when_the_model_translates_it(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": "Cộp cộp"}, self.model, {}, None,
                                                       roles={"obj_1": "sfx"}, sources={"obj_1": "저벅"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 0 and info["blank_ids"] == ["obj_1"] and info["art_regions"] == [region]


def test_a_blank_on_a_story_box_is_listed_for_retry_not_restored_as_art(saved_chapter, monkeypatch):
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": ""}, self.model, {}, None, roles={"obj_1": "dialogue"},
                                                       sources={"obj_1": "그..."}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["blank_ids"] == ["obj_1"] and info["story_blank_ids"] == ["obj_1"] and info["art_regions"] == []


def test_a_whitespace_only_translation_counts_as_blank(saved_chapter, monkeypatch):
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": "   "}, self.model, {}, None,
                                                       roles={"obj_1": "dialogue"}, sources={"obj_1": "그..."}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 0 and info["blank_ids"] == ["obj_1"] and info["story_blank_ids"] == ["obj_1"]


def test_a_sound_effect_lettered_in_latin_script_is_translated(saved_chapter, monkeypatch):
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": "Rầm!"}, self.model, {}, None,
                                                       roles={"obj_1": "sfx"}, sources={"obj_1": "DUANG!"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test", source_lang="ko", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 1 and info["art_regions"] == []


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


def test_translate_prompt_lists_map_labels_and_place_signs_as_unboxed():
    from app.translation.context import system_prompt

    prompt = system_prompt("Vietnamese", "vi")
    assert "labels on a map or diagram" in prompt
    assert "a legible sign naming a place" in prompt
    assert "A sign is only scenery when it is illegible or purely decorative" in prompt


def test_an_object_ids_request_on_a_101_object_page_is_not_rejected(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    latest = manifests.load_manifest_raw(CHAPTER)
    for i in range(2, 102):
        latest["pages"][0]["text_objects"].append({
            "id": f"obj_{i}", "shape": "rectangle",
            "region": {"x1": 20, "y1": 110, "x2": 150, "y2": 180},
            "style": _style, "ocr_text": "Hello", "translation": "",
        })
    manifests.save_manifest_raw(CHAPTER, latest)
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": "Xin chào"}, self.model, {}, None),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
        source_lang="en", target_lang="vi", object_ids=["obj_1"],
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 1 and info["missing_ids"] == []


def test_an_all_blank_page_still_saves_that_it_was_attempted(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult({"obj_1": ""}, self.model, {}, None,
                                                       roles={"obj_1": "dialogue"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
        source_lang="en", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 0 and info["blank_ids"] == ["obj_1"]
    obj = manifests.load_manifest_raw(CHAPTER)["pages"][0]["text_objects"][0]
    assert obj.get("translation_attempted") is True


def test_a_retry_matching_an_already_committed_line_is_blanked_not_doubled(saved_chapter, monkeypatch):
    _style, region, _output = saved_chapter
    latest = manifests.load_manifest_raw(CHAPTER)
    latest["pages"][0]["text_objects"][0].update({
        "translation": "Chào bạn", "source_read": "Hello",
    })
    latest["pages"][0]["text_objects"].append({
        "id": "obj_2", "shape": "rectangle",
        "region": {"x1": 20, "y1": 110, "x2": 150, "y2": 180},
        "style": _style, "ocr_text": "Hello", "translation": "",
    })
    manifests.save_manifest_raw(CHAPTER, latest)
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, *a, **kw: VisionTranslationResult(
            {"obj_2": "Chào bạn"}, self.model, {}, None,
            roles={"obj_2": "dialogue"}, sources={"obj_2": "Hello"}),
    )
    request = translation_router.TranslateVisionPageRequest(
        chapter_id=CHAPTER, page_index=0, provider="openai", model="vision-test",
        source_lang="en", target_lang="vi",
    )
    info = asyncio.run(translation_router.translate_page_in_context(request))["translation_run"]
    assert info["translated"] == 0
    assert info["blank_ids"] == ["obj_2"] and info["story_blank_ids"] == []
    assert info["art_regions"] == []
    obj = manifests.load_manifest_raw(CHAPTER)["pages"][0]["text_objects"][1]
    assert obj.get("translation_attempted") is True
