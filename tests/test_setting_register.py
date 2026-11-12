from app.translation import context
from app.translation.context import (SETTING_REGISTERS, parse_setting, setting_question,
                                     system_prompt, text_system_prompt)


def test_only_the_chosen_register_is_sent():
    prompt = system_prompt("Vietnamese", "vi", setting="murim")
    assert "@@REGISTER@@" not in prompt
    assert SETTING_REGISTERS["murim"] in prompt
    for key, register in SETTING_REGISTERS.items():
        if key != "murim":
            assert register not in prompt, f"leaked register {key}"


def test_unknown_setting_falls_back_to_the_general_rule():
    prompt = system_prompt("Vietnamese", "vi", setting=None)
    assert "@@REGISTER@@" not in prompt
    assert "match the setting" in prompt
    assert parse_setting("nonsense") is None
    assert parse_setting(" MURIM ") == "murim"


def test_one_register_is_cheaper_than_the_whole_table():
    one = system_prompt("Vietnamese", "vi", setting="court")
    whole = system_prompt("Vietnamese", "vi", setting=None).replace(
        "match the setting", " ".join(SETTING_REGISTERS.values()))
    assert len(one) < len(whole)


def test_text_retranslate_gets_the_same_single_register():
    prompt = text_system_prompt("Vietnamese", "vi", setting="court")
    assert SETTING_REGISTERS["court"] in prompt
    assert SETTING_REGISTERS["modern"] not in prompt


def test_murim_and_cultivation_letter_names_in_han_viet():
    for key in ("murim", "cultivation"):
        assert "Hán Việt" in SETTING_REGISTERS[key]
        assert "person names AND technique/skill names" in SETTING_REGISTERS[key]
    assert "Hán Việt" not in SETTING_REGISTERS["modern"]


def test_murim_cultivation_and_general_ban_modern_pronouns():
    # murim / xianxia worlds never use bạn/tôi/cậu/tớ, not even to an AI/system
    for key in ("murim", "cultivation"):
        register = SETTING_REGISTERS[key]
        for word in ("bạn", "tôi", "cậu", "tớ"):
            assert word in register, f"{key} register must ban {word}"
    assert "bạn" in context._GENERAL_REGISTER
    assert "tôi" in context._GENERAL_REGISTER
    prompt = system_prompt("Vietnamese", "vi", setting="murim")
    assert "not even when speaking to an AI" in prompt


def test_setting_question_lists_every_world():
    question = setting_question()
    for key in SETTING_REGISTERS:
        assert f"- {key}:" in question
    assert '{"setting"' in question


def test_detect_setting_parses_the_model_answer(monkeypatch):
    from app.translation import vision

    class Result:
        data = {"setting": "murim"}

    seen = {}

    def fake(provider, model, api_key, prompt, images, **kw):
        seen["prompt"] = prompt
        seen["images"] = images
        seen["max_tokens"] = kw.get("max_tokens")
        return Result()

    monkeypatch.setattr(vision, "request_vision_json", fake)

    class Provider:
        label = "x"
        supports_visual_qc = True

    translator = vision.VisionPageTranslator(Provider(), "m")
    import numpy as np
    monkeypatch.setattr(vision, "_read_image", lambda p: np.zeros((8, 8, 3), dtype=np.uint8))
    assert translator.detect_setting("page.png", "key") == "murim"
    assert seen["prompt"] == setting_question()
    assert len(seen["images"]) == 1
    assert seen["max_tokens"] <= 2048