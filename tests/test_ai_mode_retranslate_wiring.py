"""Family F: the text re-translate must actually run on the Manga Cloud path.

Run 42492185657 (Luna re-run 3) shows the break: every one of its 98
``translate`` gateway rows carries one image, and a re-translate call carries
none, so the re-translate never ran and Luna's vision wording stood everywhere.
Three gaps chained to cause it: the cost-run script never sent
translate_provider/translate_model, the cloud branch of /api/ai_mode/start
dropped them even when sent, and the gateway routed the re-translate under the
same "translate" stage as the vision read, so no second model could answer it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.routers.ai_mode as ai_mode_router
import app.translation.vision as vision
from app.ai_mode.vision_json import VisionJSONResult, stage_headers
from app.cloud import cloud_provider
from app.translation.context import ChapterMemory
from app.translation.vision import VisionPageTranslator, VisionTranslationResult, retranslate_text

CHAPTER_URL = "https://asurascans.com/comics/the-former-supreme-ab39de10/chapter/1"
TRANSLATE_MODEL = "xiaomi/mimo-v2.6-flash"


def _load_ai_cost_run():
    spec = importlib.util.spec_from_file_location("ai_cost_run", ROOT / "scripts" / "ai_cost_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retranslate_is_its_own_gateway_stage(monkeypatch):
    from gateway.app import STAGES, upstream_from_env

    assert "retranslate" in STAGES
    monkeypatch.setenv("GATEWAY_STAGE_MODELS",
                       '[{"stage": "translate", "model": "deepseek/deepseek-v4.1-flash-1017",'
                       ' "price": [0.07875, 0.315, 0.001575]},'
                       ' {"stage": "retranslate", "model": "xiaomi/mimo-v2.6-flash",'
                       ' "price": [0.147, 0.294, 0.00294]}]')
    upstream = upstream_from_env()
    assert [route.model for route in upstream.chain("translate")] == ["deepseek/deepseek-v4.1-flash-1017"]
    assert [route.model for route in upstream.chain("retranslate")] == ["xiaomi/mimo-v2.6-flash"]
    monkeypatch.setenv("GATEWAY_STAGE_MODELS", '[{"stage": "read", "model": "x", "price": [0, 0]}]')
    with pytest.raises(ValueError):
        upstream_from_env()


def test_retranslate_text_names_the_retranslate_stage(monkeypatch):
    seen = {}

    def fake_request(provider, model, api_key, prompt, images, **kwargs):
        seen.update(provider=provider, model=model, images=images, stage=kwargs.get("stage"), prompt=prompt)
        return VisionJSONResult({"translations": [{"id": "a", "translated_text": "Xin chào"}]},
                                {"prompt_tokens": 5}, 0.01)

    monkeypatch.setattr(vision, "request_vision_json", fake_request)
    out, cost, usage = retranslate_text(cloud_provider(), TRANSLATE_MODEL, "job-token", [("a", "Hello")],
                                      source_lang="en", target_lang="vi")
    assert out == {"a": "Xin chào"}
    assert cost == 0.01 and usage == {"prompt_tokens": 5}, "the second pass returns its billed cost and usage"
    assert seen["stage"] == "retranslate", "the gateway must not route this to the vision model"
    assert seen["images"] == [], "a re-translate carries no image, unlike the vision read"
    assert seen["model"] == TRANSLATE_MODEL
    assert '"source":"Hello"' in seen["prompt"]
    assert stage_headers(cloud_provider(), "retranslate") == {"X-MT-Stage": "retranslate"}


def _slice_images(tmp_path):
    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    Image.new("RGB", (160, 120), "white").save(original)
    Image.new("RGB", (160, 120), "white").save(clean)
    return original, clean


def _vision_answer(monkeypatch):
    """The vision read answers with its own wording, role and container."""

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        result = VisionTranslationResult({"1": "vision wording"}, self.model, {}, None)
        data = {"translations": [{"id": "1", "translated_text": "vision wording",
                                  "source": "Hello", "role": "dialogue", "container": "bubble"}]}
        return result, data

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)


def test_translate_batch_retranslates_each_batch(tmp_path, monkeypatch):
    original, clean = _slice_images(tmp_path)
    _vision_answer(monkeypatch)
    retold = {}

    def fake_retranslate(provider, model, api_key, pairs, **kwargs):
        retold.update(provider=provider, model=model, pairs=pairs, stage=kwargs.get("stage"))
        return {"text_1": "mimo wording"}, None, {}

    monkeypatch.setattr(vision, "retranslate_text", fake_retranslate)
    translator = VisionPageTranslator(
        cloud_provider(), "deepseek/deepseek-v4.1-flash-1017",
        translate_provider=cloud_provider(), translate_model=TRANSLATE_MODEL,
        translate_api_key="job-token",
    )
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "Hello", "region": [10, 20, 50, 40]}],
        api_key="job-token", source_lang="en", target_lang="vi",
    )
    assert retold["pairs"] == [("text_1", "Hello")]
    assert retold["model"] == TRANSLATE_MODEL
    assert answer.translations == {"text_1": "mimo wording"}, "the text model's wording replaces the vision wording"
    # Role/container still come from the vision read, so the font does not move.
    assert answer.roles == {"text_1": "dialogue"} and answer.containers == {"text_1": "bubble"}


def test_translate_batch_adds_the_retranslate_cost_and_usage(tmp_path, monkeypatch):
    """The text re-translate is a second billed pass; its cost and usage must reach the budget."""
    original, clean = _slice_images(tmp_path)

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        result = VisionTranslationResult({"1": "vision wording"}, self.model, {"prompt_tokens": 100}, 0.05)
        data = {"translations": [{"id": "1", "translated_text": "vision wording",
                                  "source": "Hello", "role": "dialogue", "container": "bubble"}]}
        return result, data

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)
    monkeypatch.setattr(vision, "request_vision_json", lambda *a, **k: VisionJSONResult(
        {"translations": [{"id": "text_1", "translated_text": "mimo wording"}]},
        {"prompt_tokens": 7, "completion_tokens": 3}, 0.02))
    translator = VisionPageTranslator(
        cloud_provider(), "deepseek/deepseek-v4.1-flash-1017",
        translate_provider=cloud_provider(), translate_model=TRANSLATE_MODEL, translate_api_key="job-token",
    )
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "Hello", "region": [10, 20, 50, 40]}],
        api_key="job-token", source_lang="en", target_lang="vi",
    )
    assert answer.translations == {"text_1": "mimo wording"}
    assert answer.estimated_cost_usd == pytest.approx(0.07), "the second pass joins the vision read's cost"
    assert answer.usage == {"prompt_tokens": 107, "completion_tokens": 3}


def test_translate_batch_without_a_text_model_keeps_vision_wording(tmp_path, monkeypatch):
    original, clean = _slice_images(tmp_path)
    _vision_answer(monkeypatch)
    called = []
    monkeypatch.setattr(vision, "retranslate_text", lambda *a, **k: called.append(True) or ({}, None, {}))
    translator = VisionPageTranslator(cloud_provider(), "deepseek/deepseek-v4.1-flash-1017")
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "Hello", "region": [10, 20, 50, 40]}],
        api_key="job-token", source_lang="en", target_lang="vi",
    )
    assert not called
    assert answer.translations == {"text_1": "vision wording"}


def test_translate_batch_retranslates_the_vision_read_source_without_ocr_text(tmp_path, monkeypatch):
    """Run 42505427459: A.I mode candidates carry no OCR text, so the pairs
    must come from the ``source`` the vision reply echoes, not from the item."""
    original, clean = _slice_images(tmp_path)
    _vision_answer(monkeypatch)
    retold = {}

    def fake_retranslate(provider, model, api_key, pairs, **kwargs):
        retold.update(provider=provider, model=model, pairs=pairs)
        return {"text_1": "mimo wording"}, None, {}

    monkeypatch.setattr(vision, "retranslate_text", fake_retranslate)
    translator = VisionPageTranslator(
        cloud_provider(), "deepseek/deepseek-v4.1-flash-1017",
        translate_provider=cloud_provider(), translate_model=TRANSLATE_MODEL,
        translate_api_key="job-token",
    )
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "", "region": [10, 20, 50, 40]}],
        api_key="job-token", source_lang="en", target_lang="vi",
    )
    assert retold["pairs"] == [("text_1", "Hello")], "the vision-read source is the pair, not the empty OCR text"
    assert retold["model"] == TRANSLATE_MODEL
    assert answer.translations == {"text_1": "mimo wording"}
    assert answer.sources == {"text_1": "Hello"}


def test_translate_batch_without_a_vision_source_or_ocr_text_cannot_retranslate(tmp_path, monkeypatch):
    """No source anywhere means nothing to re-translate: no request leaves,
    but only when the vision reply really echoed nothing."""
    original, clean = _slice_images(tmp_path)

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        result = VisionTranslationResult({"1": "vision wording"}, self.model, {}, None)
        data = {"translations": [{"id": "1", "translated_text": "vision wording",
                                  "role": "dialogue", "container": "bubble"}]}
        return result, data

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)
    calls = []
    monkeypatch.setattr(vision, "request_vision_json",
                        lambda *a, **k: calls.append(True) or VisionJSONResult({"translations": []}, {}, None))
    translator = VisionPageTranslator(
        cloud_provider(), "deepseek/deepseek-v4.1-flash-1017",
        translate_provider=cloud_provider(), translate_model=TRANSLATE_MODEL,
        translate_api_key="job-token",
    )
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "", "region": [10, 20, 50, 40]}],
        api_key="job-token", source_lang="en", target_lang="vi",
    )
    assert not calls
    assert answer.translations == {"text_1": "vision wording"}


def test_retry_passes_the_retranslate_into_the_request(monkeypatch):
    """A retry re-translates too: it goes through the same request as the first pass."""
    from types import SimpleNamespace

    import app.routers.translation as translation_router
    from app.ai_mode.job import AIModeJob, AIModeRunner, AIModeSettings

    seen = {}

    async def fake_translate(req, **kwargs):
        seen.update(translate_provider=req.translate_provider, translate_model=req.translate_model)
        return {"translation_run": {"translated": 0}}

    monkeypatch.setattr(translation_router, "translate_page_in_context", fake_translate)
    job = AIModeJob(job_id="job-1", chapter_id="chapter-1", settings=AIModeSettings(
        url=CHAPTER_URL, provider="manga-cloud", model="deepseek/deepseek-v4.1-flash-1017",
        translate_provider="manga-cloud", translate_model=TRANSLATE_MODEL))
    runner = AIModeRunner(job, SimpleNamespace(id="manga-cloud", tracks_cost=False), "tok")
    asyncio.run(runner._translate_retry(0, "en", None))
    assert seen == {"translate_provider": "manga-cloud", "translate_model": TRANSLATE_MODEL}


class _Jobs:
    def __init__(self):
        self.settings = None

    def active_job(self):
        return None

    def start(self, settings, *, provider, api_key, on_finish=None):
        self.settings = settings
        return {"job_id": "job-1"}


def _start_cloud(monkeypatch, **overrides):
    jobs = _Jobs()
    monkeypatch.setattr(ai_mode_router, "ai_mode_jobs", jobs)
    # URL validation does a live DNS lookup; the wiring under test starts after it.
    monkeypatch.setattr(ai_mode_router, "validate_url", lambda url: None)
    monkeypatch.setattr(ai_mode_router.cloud, "reserve_job",
                        lambda: {"job_token": "tok", "cost_cap_usd": 0.30,
                                 "model": "deepseek/deepseek-v4.1-flash-1017"})
    monkeypatch.setattr(ai_mode_router, "bind_cloud_job", lambda *a: "bound")
    monkeypatch.setattr(ai_mode_router, "unbind_cloud_job", lambda token: None)
    request = ai_mode_router.AIModeStartRequest(
        url=CHAPTER_URL, provider="manga-cloud", **overrides)
    asyncio.run(ai_mode_router.start_ai_mode(request))
    return jobs.settings


def test_cloud_start_carries_the_retranslate_into_job_settings(monkeypatch):
    settings = _start_cloud(monkeypatch, translate_provider="manga-cloud",
                            translate_model=TRANSLATE_MODEL)
    assert settings.translate_provider == "manga-cloud"
    assert settings.translate_model == TRANSLATE_MODEL
    assert settings.model == "deepseek/deepseek-v4.1-flash-1017"


def test_cloud_start_refuses_a_direct_retranslate_provider(monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _start_cloud(monkeypatch, translate_provider="deepseek",
                     translate_model="deepseek/deepseek-v4.1-flash-1017")
    assert exc.value.status_code == 400


def test_cloud_start_without_retranslate_fields_leaves_them_unset(monkeypatch):
    settings = _start_cloud(monkeypatch)
    assert settings.translate_provider is None and settings.translate_model is None


def test_ai_cost_run_posts_the_retranslate_by_default():
    module = _load_ai_cost_run()
    args = module._parser().parse_args(["https://example.com/chapter/1", "--out", "out"])
    assert args.translate_provider == "manga-cloud"
    assert args.translate_model == TRANSLATE_MODEL
    payload = module._start_payload(args)
    assert payload == {"url": "https://example.com/chapter/1", "provider": "manga-cloud",
                       "polish": False, "translate_provider": "manga-cloud",
                       "translate_model": TRANSLATE_MODEL}
    assert module._start_payload(SimpleNamespace(
        url="https://example.com/chapter/1", polish=True,
        translate_provider="", translate_model="")) == {
        "url": "https://example.com/chapter/1", "provider": "manga-cloud", "polish": True}


def test_translate_batch_keeps_vision_wording_when_retranslate_fails(tmp_path, monkeypatch):
    """P1-4: a re-translate crash must not lose the slice; the vision wording stands."""
    original, clean = _slice_images(tmp_path)
    _vision_answer(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(vision, "retranslate_text", boom)
    translator = VisionPageTranslator(
        cloud_provider(), "deepseek/deepseek-v4.1-flash-1017",
        translate_provider=cloud_provider(), translate_model=TRANSLATE_MODEL,
translate_api_key="k"
    )
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "Hello", "region": [10, 20, 50, 40]}],
        api_key="k", source_lang="en", target_lang="vi")
    assert answer.translations == {"text_1": "vision wording"}


def test_translate_batch_backfills_a_missing_source_so_a_misread_drops(tmp_path, monkeypatch):
    """P1-2: a salvaged entry has no source, so the gate cannot see the
    Hangul misread; the candidate OCR text is the source to check."""
    original, clean = _slice_images(tmp_path)

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        result = VisionTranslationResult({"1": "FFR"}, self.model, {}, None)
        data = {"translations": [{"id": "1", "translated_text": "FFR",
                                  "role": "dialogue", "container": "bubble"}]}
        return result, data

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)
    translator = VisionPageTranslator(cloud_provider(), "deepseek/deepseek-v4.1-flash-1017")
    answer = translator.translate_page(
        original, clean, [{"id": "text_1", "text": "\ud478\ub974\ub985", "region": [10, 20, 50, 40]}],
        api_key="k", source_lang="ko", target_lang="vi")
    assert answer.translations == {"text_1": ""}, "FFR is the model reading Hangul strokes as Latin"


def test_translate_batch_collapses_doubled_lines_before_memory(tmp_path, monkeypatch):
    """P2-3: a doubled line must not teach the next slice to double it too."""
    original, clean = _slice_images(tmp_path)

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        result = VisionTranslationResult({"1": "Cút đi!\nCút đi!"}, self.model, {}, None)
        data = {"translations": [{"id": "1", "translated_text": "Cút đi!\nCút đi!",
                                  "source": "Hello", "role": "dialogue", "container": "bubble"}]}
        return result, data

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)
    translator = VisionPageTranslator(cloud_provider(), "deepseek/deepseek-v4.1-flash-1017")
    memory = ChapterMemory()
    translator.translate_page(
        original, clean, [{"id": "text_1", "text": "Hello", "region": [10, 20, 50, 40]}],
        api_key="k", source_lang="en", target_lang="vi", memory=memory, slice_number=0)
    recent = memory.snapshot(settled=False)["recent_lines"]
    assert [entry["text"] for entry in recent] == ["Cút đi!"]


def test_retranslate_text_accepts_a_dict_shaped_translation_list(monkeypatch):
    """P2-2: the re-translate answers with the same shapes the vision read does."""
    def fake_request(provider, model, api_key, prompt, images, **kwargs):
        return VisionJSONResult({"translations": {"a": "Xin chào"}}, {"prompt_tokens": 5}, 0.01)

    monkeypatch.setattr(vision, "request_vision_json", fake_request)
    out, cost, usage = retranslate_text(cloud_provider(), TRANSLATE_MODEL, "job-token", [("a", "Hello")],
                                        source_lang="en", target_lang="vi")
    assert out == {"a": "Xin chào"} and cost == 0.01 and usage == {"prompt_tokens": 5}


def test_retranslate_text_with_no_translations_keeps_the_vision_wording(monkeypatch):
    """P2-2: an empty re-translate is a warning, not a crash; the caller keeps the vision wording."""
    def fake_request(provider, model, api_key, prompt, images, **kwargs):
        return VisionJSONResult({"translations": []}, {}, 0.01)

    monkeypatch.setattr(vision, "request_vision_json", fake_request)
    out, _, _ = retranslate_text(cloud_provider(), TRANSLATE_MODEL, "job-token", [("a", "Hello")],
                                 source_lang="en", target_lang="vi")
    assert out == {}
