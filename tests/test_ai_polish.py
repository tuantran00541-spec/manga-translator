import asyncio

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
    monkeypatch.setattr(polish, "judge", lambda provider, key, state: graded[state.split("\n")[1].split(": ", 1)[1]])
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
