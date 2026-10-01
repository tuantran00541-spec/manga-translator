import asyncio
import json

from app.ai_mode import polish
from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings
from app.ai_mode.vision_json import VisionJSONResult
from app.ai_providers import PROVIDERS


def _obj(obj_id, translation, source="Get out of here now!", role="dialogue"):
    return {"id": obj_id, "translation": translation, "source_read": source, "typography_role": role}


def test_only_translated_speech_with_a_source_and_three_words_is_graded():
    pages = [{"text_objects": [
        _obj("a", "Cút khỏi đây ngay!"),
        _obj("b", "Dám mơ ước", role="title"),
        _obj("c", "Đi thôi"),  # two words
        _obj("d", "Không có gì đâu", source=""),
        _obj("e", ""),
    ]}, {"text_objects": [_obj("f", "Tôi sẽ không đi đâu cả.")]}]
    lines = polish.collect_lines(pages, [0, 1])
    assert [(line.page_index, line.id) for line in lines] == [(0, "a"), (1, "f")]
    assert lines[1].before == "Không có gì đâu", "neighbours are the translated lines around it"
    assert "SOURCE: Get out of here now!" in polish.line_state(lines[0])


def test_low_scores_name_the_problem_and_quality_is_one_scale():
    assert polish.problems({"natural": 3.0, "clear": 2.0, "faithful": 2.0}) == []
    assert polish.problems({"natural": 1.0, "clear": 2.0, "faithful": 0.5}) == [
        polish.PROBLEMS["natural"], polish.PROBLEMS["faithful"]]
    assert polish.quality({"natural": 3.0, "clear": 2.0, "faithful": 2.0}) == 1.0


def test_rewrites_keep_only_known_ids_and_text(monkeypatch):
    monkeypatch.setattr(polish, "request_vision_json", lambda *a, **k: VisionJSONResult(
        {"rewrites": [{"id": "a", "text": "Biến khỏi đây ngay…"}, {"id": "zz", "text": "x"}, {"id": "f", "text": " "}]},
        {}, None))
    lines = [polish.Line(0, "a", "s", "t"), polish.Line(0, "f", "s", "t")]
    assert polish.rewrite(PROVIDERS["openai"], "m", "k", lines, {}) == ({"a": "Biến khỏi đây ngay..."}, None)


def test_the_stage_keeps_a_rewrite_only_when_the_judge_grades_it_better(monkeypatch):
    pages = [{"text_objects": [_obj("a", "Ngươi hãy ra khỏi nơi này lập tức!"), _obj("b", "Tôi đi ngay bây giờ.")]}]
    graded = {"Ngươi hãy ra khỏi nơi này lập tức!": {"natural": 1.0, "clear": 2.0, "faithful": 2.0},
              "Tôi đi ngay bây giờ.": {"natural": 1.2, "clear": 2.0, "faithful": 2.0},
              "Cút khỏi đây ngay!": {"natural": 2.8, "clear": 2.0, "faithful": 2.0},
              "Tôi đi luôn đây.": {"natural": 0.5, "clear": 1.0, "faithful": 1.0}}
    monkeypatch.setattr(polish, "judge", lambda provider, key, state, asked: graded[state.split("\n")[1].split(": ", 1)[1]])
    monkeypatch.setattr(polish, "rewrite", lambda *a: ({"a": "Cút khỏi đây ngay!", "b": "Tôi đi luôn đây."}, None))
    job = AIModeJob("j", AIModeSettings(url="u", provider="openai", model="m", polish=True))
    job.stages = {"polish": {}}
    job.stage = "polish"
    runner = AIModeRunner(job, PROVIDERS["openai"], "k")
    saved = []
    monkeypatch.setattr(runner, "_manifest", lambda: {"pages": pages})
    monkeypatch.setattr(runner, "_active_pages", lambda: [0])
    monkeypatch.setattr(runner, "_save_polished", lambda lines: saved.extend((line.id, line.text) for line in lines))

    async def no_seams():
        return []

    monkeypatch.setattr(runner, "_sync_seams", no_seams)
    asyncio.run(runner.polish())
    assert saved == [("a", "Cút khỏi đây ngay!")], "the worse rewrite of b is dropped"
    stats = runner.report["polish"]
    assert (stats["judged"], stats["flagged"], stats["rewritten"], stats["still_flagged"]) == (2, 2, 1, 1)


def test_the_stage_does_nothing_when_off():
    job = AIModeJob("j", AIModeSettings(url="u", provider="openai", model="m"))
    job.stages = {"polish": {}}
    job.stage = "polish"
    runner = AIModeRunner(job, PROVIDERS["openai"], "k")
    asyncio.run(runner.polish())
    assert runner.report["polish"]["judged"] == 0 and job.stages["polish"]["detail"] == "Tắt"


def test_the_judge_reads_each_line_flat_and_ignores_noise_level_gains():
    state = polish.line_state(polish.Line(0, "a", "I HAVE BEEN\nMARKED...", "Mình đã bị\nđánh dấu..."))
    assert "TRANSLATION (Vietnamese): Mình đã bị đánh dấu...\n" in state and "SOURCE: I HAVE BEEN MARKED..." in state
    old = {"natural": 2.51, "clear": 1.78, "faithful": 1.91}
    assert not polish.better({"natural": 2.57, "clear": 1.79, "faithful": 1.93}, old), "adding 'các' is not a rewrite"
    assert polish.better({"natural": 2.83, "clear": 1.9, "faithful": 1.97}, old)


def test_the_judge_sees_the_speakers_settled_address_and_the_glossary_terms_in_the_line():
    memory = {"glossary": {"names": [{"source": "Sunny", "target": "Sunny"}, {"source": "Nephis", "target": "Nephis"}],
                           "terms": [{"source": "Nightmare Spell", "target": "Bùa Chú Ác Mộng"}],
                           "address": [{"from": "Sunny", "to": "Nephis", "self": "tôi", "other": "cô"}]},
              "address": [{"from": "Sunny", "to": "Nephis", "self": "tôi", "other": "cô"},
                          {"from": "Nephis", "to": "Sunny", "self": "tôi", "other": "cậu"}]}
    line = polish.Line(0, "a", "The Nightmare Spell took me, Nephis.", "Bùa chú bắt tôi rồi, Nephis.", speaker="Sunny")
    state = polish.line_state(line, memory=memory)
    assert "SPEAKER: Sunny" in state and 'self "tôi"; to Nephis: "cô"' in state and "cậu" not in state
    assert "Nightmare Spell = Bùa Chú Ác Mộng" in state and "Sunny = Sunny" not in state
    assert "settled" in polish.questions(line, memory)
    assert "settled" not in polish.questions(polish.Line(0, "b", "Run!", "Chạy!"), memory)
    assert polish.problems({"natural": 3.0, "clear": 2.0, "faithful": 2.0, "settled": 0.4}) == [polish.PROBLEMS["settled"]]


def test_address_shows_only_what_holds_for_any_listener_and_a_rewrite_keeps_its_voice():
    memory = {"address": [{"from": "Hero", "to": "Goddess", "self": "ta", "other": "ngươi"},
                          {"from": "Hero", "to": "Maid", "self": "tôi", "other": "cô"}]}
    line = polish.Line(0, "a", "How did I pull that off?", "Sao mình làm được vậy?", speaker="Hero")
    assert polish.settled(line, memory)[0] == [], "two self forms: the listener decides, so nothing is settled"
    old = {"natural": 2.6, "clear": 1.87, "faithful": 1.85, "settled": 0.15}
    assert not polish.better({"natural": 2.35, "clear": 1.87, "faithful": 1.86, "settled": 0.91}, old)
    assert polish.better({"natural": 2.55, "clear": 1.87, "faithful": 1.86, "settled": 0.91}, old)


def test_the_judge_and_the_rewrite_follow_the_target_language():
    line = polish.Line(0, "a", "Run!", "¡Corre ya!")
    assert "TRANSLATION (Spanish): ¡Corre ya!" in polish.line_state(line, language="Spanish")
    asked = polish.questions(line, {}, "Spanish")
    assert "as Spanish" in asked["natural"]["instructions"] and "Vietnamese" not in json.dumps(asked)
    assert polish.questions(line, {}) == polish.QUESTIONS, "Vietnamese keeps the calibrated wording"
