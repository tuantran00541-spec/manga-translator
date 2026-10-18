"""The decision model keeps art from being erased and decides which slices hold no story text."""
import asyncio
import base64

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


def test_only_minirouter_with_a_key_gets_a_screen():
    minirouter = resolve_provider("mr", label="MiniRouter", protocol="openai", api_base="https://api.minirouter.sh/v1")
    assert decider.screen_for(minirouter, "k") is not None
    assert decider.screen_for(minirouter, "") is None
    assert decider.screen_for(PROVIDERS["openai"], "k") is None


def test_a_crop_of_a_huge_noisy_box_still_fits_one_request():
    image = np.random.default_rng(0).integers(0, 255, (3000, 1600, 3), dtype=np.uint8)
    url = decider.crop_url(image, _box(0, 0, 1600, 3000))
    assert len(base64.b64decode(url.split(",", 1)[1])) < decider.MAX_JPEG_BYTES


def test_only_boxes_the_model_is_sure_about_are_art(monkeypatch):
    chances = iter([0.05, 0.3, None, 0.99])
    monkeypatch.setattr(decider, "WORKERS", 1)
    monkeypatch.setattr(decider, "ask", lambda *_a: None if (c := next(chances)) is None else {"any": {"probability": c}})
    image = np.full((100, 100, 3), 200, np.uint8)
    assert decider.art_boxes("https://api.minirouter.sh/v1", "k", image, [_box(10, 10, 30, 30)] * 4) == {0}


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


def _kind(story):
    return {"kind": {"type": "choice", "choice": "story" if story >= 0.5 else "none",
                     "probabilities": {"story": story, "credit": 0.0, "none": 1 - story}}}


def test_a_slice_is_cut_into_square_tiles_and_stops_at_the_first_with_story(monkeypatch):
    asked = []
    replies = iter([_kind(0.01), _kind(0.9), _kind(0.01)])
    monkeypatch.setattr(decider, "ask", lambda *_a: asked.append(1) or next(replies))
    image = np.full((2700, 900, 3), 200, np.uint8)
    assert len(decider.tile_urls(image)) == 3
    assert decider.slice_story("b", "k", image) == 0.9 and len(asked) == 2


def test_a_slice_with_an_unanswered_tile_is_never_skipped(monkeypatch):
    replies = {0: [_kind(0.01), _kind(0.01)], 1: [_kind(0.01), None], 2: [_kind(0.5)]}
    images = {i: np.full((1800 if i < 2 else 900, 900, 3), i, np.uint8) for i in replies}
    monkeypatch.setattr(decider, "WORKERS", 1)
    queue = [r for i in sorted(replies) for r in replies[i]]
    monkeypatch.setattr(decider, "ask", lambda *_a: queue.pop(0))
    minirouter = resolve_provider("mr", label="MiniRouter", protocol="openai", api_base="https://api.minirouter.sh/v1")
    skip, chances = decider.storyless_slices(minirouter, "k", [0, 1, 2], images.__getitem__)
    assert skip == [0] and chances[1] is None and chances[2] == 0.5


def test_story_chance_reads_list_and_choice_answers():
    assert decider.story_chance({"kind": {"probabilities": [{"choice": "story", "probability": 0.3}]}}) == 0.3
    assert decider.story_chance({"kind": {"choice": "credit"}}) == 0.0
    assert decider.story_chance({"kind": {}}) is None and decider.story_chance(None) is None
