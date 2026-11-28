import asyncio
from types import SimpleNamespace

import pytest

import cv2
import numpy as np

from app.ai_mode import decider
from app.ai_providers import PROVIDERS, resolve_provider
from app.detector.boxes import BubbleBox
from app.page_processing_jobs import ChapterProcessingJobManager
from app.pipeline import ChapterPipeline


def _box(x1, y1, x2, y2):
    return BubbleBox(x1, y1, x2, y2, 0.9, np.full((y2 - y1, x2 - x1), 255, np.uint8),
                     source_role="text_segmenter", semantic_type="free_text", safe_to_inpaint=True)


@pytest.mark.parametrize("provider", [
    *PROVIDERS.values(),
    resolve_provider("custom", protocol="openai", api_base="https://example.com/v1"),
])
def test_screen_uses_selected_vision_model(monkeypatch, provider):
    calls = []
    def vision(p, model, key, prompt, images, **kwargs):
        calls.append((p, model, key, images[0][1].shape))
        return SimpleNamespace(data={"kind": "art"}, estimated_cost_usd=None)

    monkeypatch.setattr(decider, "request_vision_json", vision)
    assert decider.screen_for(provider, "", "chosen") is None
    screen = decider.screen_for(provider, "key", "chosen")
    image = np.zeros((1600, 1600, 3), np.uint8)
    assert screen(image, [_box(100, 100, 1500, 1500)]) == {0}
    assert calls == [(provider, "chosen", "key", (768, 768, 3))]


class _Detector:
    def detect(self, image, **_kwargs):
        return [_box(10, 10, 40, 30), _box(50, 10, 75, 30)]

    @staticmethod
    def last_metrics():
        return {}

    @staticmethod
    def leftover_boxes(_image, _first_pass):
        return []


class _Inpainter:
    lama_model_path = None
    session_loaded = False
    _prefer_dynamic = False
    serialized_inference = False

    def __init__(self):
        self.erased = []

    def inpaint(self, image, boxes, **_kwargs):
        self.erased.extend((b.x1, b.y1) for b in boxes)
        return image.copy()

    @staticmethod
    def last_metrics():
        return {}


def test_a_screened_box_is_kept_as_drawn_and_left_out_of_translation(tmp_path):
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._detector, pipeline._inpainter = _Detector(), _Inpainter()
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), np.full((60, 80, 3), 200, np.uint8))
    result = pipeline._process_page(image_path, tmp_path, screen=lambda _image, boxes: {1})
    assert pipeline._inpainter.erased == [(10, 10)]
    art = [b for b in result["boxes"] if b.get("screened_art")]
    assert [(b["x1"], b["removed"]) for b in art] == [(50, True)]
    assert result["processing_metrics"]["detector"]["screened_art"] == 1


def test_the_processing_job_hands_the_screen_to_the_plan():
    seen = {}

    def plan(chapter_id, indices, workers, done, screen=None):
        seen["screen"] = screen
        for index in indices:
            done(index)

    async def run():
        manager = ChapterProcessingJobManager(plan)
        screen = object()
        snapshot = manager.start("c1", page_indices=[0], workers=1, screen=screen)
        for _ in range(200):
            if manager.snapshot(snapshot["job_id"])["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        return screen

    screen = asyncio.run(run())
    assert seen["screen"] is screen


@pytest.mark.parametrize("answer", ["uncertain", "invalid", None, "error"])
def test_unconfirmed_box_is_deferred_in_pipeline(monkeypatch, tmp_path, answer):
    def vision(*args, **kwargs):
        if answer == "error":
            raise RuntimeError("provider offline")
        return SimpleNamespace(data={"kind": answer}, estimated_cost_usd=None)

    monkeypatch.setattr(decider, "request_vision_json", vision)
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._detector, pipeline._inpainter = _Detector(), _Inpainter()
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.zeros((60, 80, 3), np.uint8))
    result = pipeline._process_page(image_path, tmp_path, screen=decider.screen_for(PROVIDERS["openai"], "k", "m"))
    assert pipeline._inpainter.erased == []
    assert len(result["boxes"]) == 2
    assert all(b["needs_review"] and b["deferred_reason"] == "lettering_unconfirmed" for b in result["boxes"])


def test_story_tiles_keep_unanswered_slice_and_stop_on_story(monkeypatch):
    answers = iter(["none", "story", "none", None, "credit", "none", "none"])
    monkeypatch.setattr(decider, "WORKERS", 1)
    monkeypatch.setattr(decider, "request_vision_json", lambda *a, **kw:
                        SimpleNamespace(data={"kind": next(answers)}, estimated_cost_usd=None))
    image = np.zeros((2700, 900, 3), np.uint8)
    skips, chances = decider.storyless_slices(PROVIDERS["gemini"], "k", [0, 1, 2], lambda _: image, "model")
    assert skips == [2] and chances == {0: 1.0, 1: None, 2: 0.0}


def test_polish_uses_selected_model_and_requires_valid_scores(monkeypatch):
    from app.ai_mode import polish
    provider = PROVIDERS["openai"]
    replies = iter([{"natural": 2.5}, {"natural": 7}])
    def vision(p, model, key, prompt, images, **kwargs):
        assert (p, model, key, images) == (provider, "selected", "k", [])
        return SimpleNamespace(data=next(replies), estimated_cost_usd=None)
    monkeypatch.setattr(polish, "request_vision_json", vision)
    asked = {"natural": polish.QUESTIONS["natural"]}
    assert polish.judge(provider, "k", "passage", asked, "selected") == {"natural": 2.5}
    with pytest.raises(ValueError):
        polish.judge(provider, "k", "passage", asked, "selected")


@pytest.mark.parametrize("native_scores", [None, {"clear": 1.5}])
def test_cloud_polish_uses_gateway_judge_or_selected_model(monkeypatch, native_scores):
    from app import cloud
    from app.ai_mode import polish

    provider = cloud.cloud_provider("cloud-model")
    calls = []
    monkeypatch.setattr(polish, "_cloud_judge", lambda *args: native_scores)

    def fallback(p, model, key, prompt, images, **kwargs):
        calls.append((p.id, model, key))
        return SimpleNamespace(data={"clear": 2}, estimated_cost_usd=None)

    monkeypatch.setattr(polish, "request_vision_json", fallback)
    result = polish.judge(provider, "job-token", "passage", {"clear": polish.QUESTIONS["clear"]}, "cloud-model")
    assert result == (native_scores or {"clear": 2})
    assert calls == ([] if native_scores else [("manga-cloud", "cloud-model", "job-token")])


def test_art_screen_stops_calls_after_unknown_cost(monkeypatch):
    from app.detector.boxes import BubbleBox

    calls = []
    remaining = [0.01]

    def vision(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(data={'kind': 'text'}, estimated_cost_usd=None)

    monkeypatch.setattr(decider, 'request_vision_json', vision)
    boxes = [BubbleBox(10, 10, 30, 30, 0.9, None), BubbleBox(40, 10, 60, 30, 0.9, None)]
    def cost(value):
        remaining[0] = 0 if value is None else remaining[0] - value

    screen = decider.screen_for(PROVIDERS['openai'], 'k', 'm', cost, can_call=lambda: remaining[0] >= 0.001)
    assert screen(np.zeros((50, 80, 3), np.uint8), boxes) == set()
    assert len(calls) == 1
    assert boxes[1].needs_review and not boxes[1].ocr_eligible
