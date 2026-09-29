"""A.I mode: credit/logo scan parsing, job orchestration and the gate-free ZIP."""
from __future__ import annotations

import asyncio
import zipfile

import cv2
import numpy as np
import pytest

import app.ai_mode.job as ai_job
import app.config as config
import app.manifest_utils as manifests
import app.routers.export as export_router
import app.routers.image as image_router
import app.routers.render_commit as render_commit
import app.routers.translation as translation_router
from app.ai_mode.job import AIModeJobManager, AIModeRunner, AIModeSettings
from app.ai_mode.page_scan import SliceScan, parse_scan, scan_slices
from app.ai_providers import PROVIDERS
from app.translation.vision import VisionTranslationResult

CHAPTER = "a1b2c3d5"
SETTINGS = AIModeSettings(url="https://example.com/c/1", provider="openai", model="vision-test")


# -- scan parsing -------------------------------------------------------------

def test_scan_marks_only_confident_credit_slices_and_maps_logo_boxes():
    sizes = {0: (800, 2000), 1: (800, 2000), 2: (800, 2000), 3: (800, 1000)}
    data = {"slices": [
        {"slice": 0, "is_credit": False, "credit_confidence": 0.1,
         "logos": [{"box_2d": [100, 250, 200, 750], "confidence": 0.9}]},
        {"slice": 1, "is_credit": True, "credit_confidence": 0.5, "logos": []},
        {"slice": 2, "is_credit": True, "credit_confidence": 0.95,
         "logos": [{"box_2d": [0, 0, 100, 100], "confidence": 0.9}]},
        {"slice": 99, "is_credit": True, "credit_confidence": 1.0, "logos": []},
    ]}
    scans = {scan.page_index: scan for scan in parse_scan(data, sizes)}

    assert set(scans) == {0, 1, 2, 3}
    assert scans[0].logos == ((192, 192, 608, 408),)
    assert not scans[1].is_credit, "a credit guess below the confidence floor is ignored"
    assert scans[2].is_credit and scans[2].logos == (), "a skipped slice needs no logo regions"
    assert scans[3] == SliceScan(3, False, 0.0, ())


def test_scan_rejects_low_confidence_huge_and_malformed_logo_boxes():
    data = {"slices": [{"slice": 0, "is_credit": False, "credit_confidence": 0, "logos": [
        {"box_2d": [100, 100, 300, 300], "confidence": 0.3},
        {"box_2d": [0, 0, 900, 1000], "confidence": 0.99},
        {"box_2d": [300, 300, 100, 100], "confidence": 0.99},
        {"box_2d": ["a", 0, 1, 1], "confidence": 0.99},
        {"box_2d": [500, 500, 501, 501], "confidence": 0.99},
    ]}]}
    assert parse_scan(data, {0: (800, 1600)})[0].logos == ()


def test_scan_sends_every_slice_as_a_labelled_image(monkeypatch):
    sent = []

    class Response:
        status_code, ok = 200, True

        def json(self):
            return {"choices": [{"message": {"content": (
                '{"slices":[{"slice":4,"is_credit":true,"credit_confidence":0.9,"logos":[]},'
                '{"slice":5,"is_credit":false,"credit_confidence":0.0,"logos":[]}]}'
            )}}], "usage": {"prompt_tokens": 10}}

    monkeypatch.setattr("app.ai_mode.vision_json.requests.post", lambda url, **kw: sent.append(kw) or Response())
    images = [(4, np.full((3000, 800, 3), 255, np.uint8)), (5, np.zeros((900, 800, 3), np.uint8))]
    scans, cost = scan_slices(PROVIDERS["openai"], "vision-test", "key", images)

    content = sent[0]["json"]["messages"][0]["content"]
    assert [item["text"] for item in content if item["type"] == "text"][1:] == ["SLICE 4", "SLICE 5"]
    assert sum(item["type"] == "image_url" for item in content) == 2
    assert [scan.is_credit for scan in scans] == [True, False]
    assert cost is None, "OpenAI does not report priced usage"


@pytest.mark.parametrize("provider_id, label", [
    ("openai", "OpenAI"), ("openrouter", "OpenRouter"), ("my-proxy", "my-proxy"),
])
def test_chapter_qc_accepts_every_vision_provider(provider_id, label):
    # The QC step of A.I mode (and the QC button) used to reject everything
    # except Gemini and DeepSeek before even checking the key.
    import contextlib
    from app.visual_qc.service import ChapterQCService

    service = ChapterQCService(
        object(), provider=provider_id, api_key_provider=lambda: None,
        manifest_loader=lambda chapter_id: {"pages": []}, manifest_saver=lambda *args: None,
        manifest_lock=lambda chapter_id: contextlib.nullcontext(),
    )
    with pytest.raises(ValueError, match=f"^{label} API key is not configured$"):
        asyncio.run(service.start("abcd1234"))


# -- orchestration ------------------------------------------------------------

class RecordingRunner:
    calls: list[str] = []
    fail_at: str | None = None

    def __init__(self, job, provider, api_key):
        self.job = job

    def __getattr__(self, name):
        async def stage():
            RecordingRunner.calls.append(name)
            if name == RecordingRunner.fail_at:
                raise RuntimeError("boom")
            if name == "download":
                self.job.chapter_id = CHAPTER
        return stage


def _run_manager(fail_at=None, cancel=False):
    RecordingRunner.calls, RecordingRunner.fail_at = [], fail_at
    manager = AIModeJobManager(runner_factory=RecordingRunner)

    async def scenario():
        snapshot = manager.start(SETTINGS, provider=PROVIDERS["openai"], api_key="key")
        if cancel:
            manager.cancel(snapshot["job_id"])
        await manager._jobs[snapshot["job_id"]].task
        return manager.snapshot(snapshot["job_id"])

    return asyncio.run(scenario())


def test_manager_runs_every_stage_in_order():
    snapshot = _run_manager()
    assert RecordingRunner.calls == ["download", "scan", "clean", "review", "translate", "render", "finish"]
    assert snapshot["status"] == "completed"
    assert all(stage["status"] == "done" for stage in snapshot["stages"])
    assert snapshot["chapter_id"] == CHAPTER
    assert snapshot["cost_usd"] is None, "OpenAI cost is unknown, not zero"


def test_manager_stops_at_the_failing_stage():
    snapshot = _run_manager(fail_at="review")
    assert RecordingRunner.calls == ["download", "scan", "clean", "review"]
    assert snapshot["status"] == "failed" and snapshot["error"] == "boom"
    states = {stage["key"]: stage["status"] for stage in snapshot["stages"]}
    assert states["review"] == "failed" and states["translate"] == "pending"


def test_manager_cancel_before_first_stage():
    snapshot = _run_manager(cancel=True)
    assert RecordingRunner.calls == []
    assert snapshot["status"] == "cancelled"


def test_manager_allows_one_active_job():
    manager = AIModeJobManager(runner_factory=RecordingRunner)

    async def scenario():
        manager.start(SETTINGS, provider=PROVIDERS["openai"], api_key="key")
        with pytest.raises(RuntimeError):
            manager.start(SETTINGS, provider=PROVIDERS["openai"], api_key="key")
        await asyncio.gather(*(job.task for job in manager._jobs.values()))

    asyncio.run(scenario())


# -- scan stage ---------------------------------------------------------------

def _scan_stage(monkeypatch, scans, pages=8):
    from app.dependencies import pipeline
    import app.routers.chapters as chapters

    manifest = {"pages": [{"original": f"p{i}.png", "preserve_regions": []} for i in range(pages)]}
    skipped, preserved = [], {}
    monkeypatch.setattr(ai_job, "validate_managed_path", lambda value, root: value)
    monkeypatch.setattr(ai_job, "read_image", lambda path: np.zeros((100, 80, 3), np.uint8))
    monkeypatch.setattr(ai_job, "scan_slices", lambda provider, model, key, images: (
        [scan for scan in scans if scan.page_index in {index for index, _ in images}], 0.001,
    ))
    monkeypatch.setattr(pipeline, "mark_skipped", lambda chapter_id, indices, value: skipped.extend(indices))
    monkeypatch.setattr(chapters, "_set_page_preserve_regions",
                        lambda chapter_id, index, regions: preserved.__setitem__(index, regions))

    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="scan",
                           stages={"scan": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["deepseek"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    asyncio.run(runner.scan())
    return runner, skipped, preserved


def test_scan_stage_skips_credit_slices_and_preserves_logos(monkeypatch):
    runner, skipped, preserved = _scan_stage(monkeypatch, [
        SliceScan(0, False, 0.0, ((10, 10, 60, 40),)),
        SliceScan(7, True, 0.9, ()),
    ])
    assert skipped == [7]
    assert list(preserved) == [0]
    assert preserved[0][0].model_dump() == {"x1": 10, "y1": 10, "x2": 60, "y2": 40}
    assert runner.report["credit_pages"] == [7] and runner.report["logo_regions"] == 1
    assert runner.job.cost_usd == pytest.approx(0.002), "two scan batches of four slices"


def test_scan_stage_retries_a_rejected_batch_one_slice_at_a_time(monkeypatch):
    calls = []

    def one_image_only(provider, model, key, images):
        calls.append([index for index, _ in images])
        if len(images) > 1:
            raise RuntimeError("Manga Cloud HTTP 502: Upstream HTTP 422")
        if images[0][0] == 6:
            raise RuntimeError("still rejected")
        return [SliceScan(images[0][0], images[0][0] == 7, 0.9, ())], 0.001

    runner, skipped, _ = _scan_stage(monkeypatch, [])
    monkeypatch.setattr(ai_job, "scan_slices", one_image_only)
    asyncio.run(runner.scan())
    assert calls[0] == [0, 1, 2, 3], "the first batch probes whether the provider takes several images"
    assert sorted(calls[1:]) == [[0], [1], [2], [3], [4], [5], [6], [7]], "after one rejected batch, scan slice by slice"
    assert runner.report["scan_errors"][-1] == {"pages": [6], "error": "still rejected"}
    assert skipped[-1:] == [7]


def test_scan_stage_retries_only_the_batch_that_failed(monkeypatch):
    calls = []

    def flaky_second_batch(provider, model, key, images):
        indices = [index for index, _ in images]
        calls.append(indices)
        if indices == [4, 5, 6, 7]:
            raise RuntimeError("Manga Cloud HTTP 502")
        return [SliceScan(index, index == 7, 0.9, ()) for index in indices], 0.001

    runner, skipped, _ = _scan_stage(monkeypatch, [])
    monkeypatch.setattr(ai_job, "scan_slices", flaky_second_batch)
    asyncio.run(runner.scan())
    assert sorted(calls) == [[0, 1, 2, 3], [4], [4, 5, 6, 7], [5], [6], [7]]
    assert runner.report["scan_errors"] == [] and skipped[-1:] == [7]


def test_scan_stage_refuses_to_skip_most_of_a_chapter(monkeypatch):
    runner, skipped, _ = _scan_stage(monkeypatch, [SliceScan(i, True, 0.9, ()) for i in range(5)])
    assert skipped == []
    assert runner.report["credit_rejected"] == [0, 1, 2, 3, 4]


def test_scan_stage_skips_textless_slices_but_not_most_of_a_chapter(monkeypatch):
    runner, skipped, _ = _scan_stage(monkeypatch, [
        SliceScan(2, False, 0.0, (), no_text=True), SliceScan(5, False, 0.0, (), no_text=True),
    ])
    assert skipped == [2, 5] and runner.report["textless_pages"] == [2, 5]

    runner, skipped, _ = _scan_stage(monkeypatch, [SliceScan(i, False, 0.0, (), no_text=True) for i in range(5)])
    assert skipped == [] and runner.report["textless_rejected"] == [0, 1, 2, 3, 4]


def test_scan_parse_requires_a_confident_textless_answer():
    sizes = {0: (800, 1000), 1: (800, 1000), 2: (800, 1000)}
    scans = parse_scan({"slices": [
        {"slice": 0, "is_credit": False, "credit_confidence": 0, "no_text": True, "no_text_confidence": 0.9, "logos": []},
        {"slice": 1, "is_credit": False, "credit_confidence": 0, "no_text": True, "no_text_confidence": 0.7, "logos": []},
        {"slice": 2, "is_credit": False, "credit_confidence": 0, "no_text": True, "no_text_confidence": 0.95,
         "logos": [{"box_2d": [100, 100, 300, 500], "confidence": 0.9}]},
    ]}, sizes)
    assert [scan.no_text for scan in scans] == [True, False, False], "a slice with a logo has lettering"


def _runner(monkeypatch, stage, manifest):
    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage=stage, cost_usd=None,
                           stages={stage: {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)

    async def no_seams():
        return []

    monkeypatch.setattr(runner, "_sync_seams", no_seams)
    monkeypatch.setattr(runner, "_join_stacked_lines", lambda indices: None)
    return runner


def test_review_stage_applies_each_slice_in_one_pass_without_duplicating_boxes(monkeypatch):
    from app.ai_mode.checkpoints import CleanReview
    from app.dependencies import pipeline

    existing = {"x1": 100, "y1": 100, "x2": 200, "y2": 160}
    runner = _runner(monkeypatch, "review", {"pages": [{"boxes": [existing]}, {}]})
    monkeypatch.setattr(runner, "_active_pages", lambda: [0, 1])
    monkeypatch.setattr(runner, "_images", lambda index, key, root: (np.zeros((10, 10, 3)), np.zeros((10, 10, 3))))
    reviews = {0: CleanReview(0, missed=((1, 2, 30, 40), (105, 105, 195, 150)), residue=((5, 5, 20, 20),),
                              restore=((300, 300, 400, 400), (0, 0, 50, 50))),
               1: CleanReview(1)}
    monkeypatch.setattr(ai_job, "review_clean", lambda provider, model, key, index, a, b: (reviews[index], None))
    calls = []
    monkeypatch.setattr(pipeline, "apply_review_fixes",
                        lambda chapter, index, **fixes: calls.append((index, fixes)), raising=False)
    asyncio.run(runner.review())
    assert calls == [(0, {
        "preserve": [(300, 300, 400, 400)],
        "boxes": [(1, 2, 30, 40)],
        "repaint": [(5, 5, 20, 20), (105, 105, 195, 150)],
    })], "missed text over an existing box is re-erased, and a restore over an erase is dropped"
    assert (runner.report["kept_regions"], runner.report["missed_added"], runner.report["repainted_regions"]) == (1, 1, 2)


def test_leftover_text_on_a_scan_logo_frees_it_so_the_repaint_reaches_it(monkeypatch):
    import contextlib

    from app.ai_mode.checkpoints import CleanReview
    from app.dependencies import pipeline

    # Shadow Slave 1, slice 135: the scan boxed the "Read at ASURASCANS.COM" watermark as a logo.
    watermark, kept = (323, 2825, 752, 3107), {"x1": 10, "y1": 10, "x2": 200, "y2": 100}
    stored = {"pages": [{"preserve_regions": [dict(zip(("x1", "y1", "x2", "y2"), watermark)), kept]}]}
    runner = _runner(monkeypatch, "review", stored)
    runner._scan_logos[0] = [watermark]
    monkeypatch.setattr(runner, "_active_pages", lambda: [0])
    monkeypatch.setattr(runner, "_images", lambda index, key, root: (np.zeros((10, 10, 3)), np.zeros((10, 10, 3))))
    monkeypatch.setattr(ai_job, "get_manifest_lock", lambda chapter_id: contextlib.nullcontext())
    monkeypatch.setattr(ai_job, "load_manifest_raw", lambda chapter_id: stored)
    monkeypatch.setattr(ai_job, "save_manifest_raw", lambda chapter_id, manifest: None)
    monkeypatch.setattr(ai_job, "review_clean", lambda provider, model, key, index, a, b: (
        CleanReview(0, residue=((330, 2830, 740, 3100),)), None))
    seen = []
    monkeypatch.setattr(pipeline, "apply_review_fixes",
                        lambda chapter, index, **fixes: seen.append(list(stored["pages"][0]["preserve_regions"])),
                        raising=False)
    asyncio.run(runner.review())
    assert seen == [[kept]], "the watermark is no longer protected when the repaint runs; other regions stay"
    assert runner.report["logos_freed"] == 1


def test_each_slice_is_reviewed_as_soon_as_it_is_clean(monkeypatch):
    import app.routers.chapters as chapters_router
    from app.ai_mode.checkpoints import CleanReview

    runner = _runner(monkeypatch, "clean", {"pages": [{}, {}, {}]})
    runner.job.stages["review"] = {"done": 0, "total": 0, "detail": ""}
    monkeypatch.setattr(runner, "_active_pages", lambda: [0, 1, 2])
    monkeypatch.setattr(runner, "_images", lambda index, key, root: (np.zeros((10, 10, 3)), np.zeros((10, 10, 3))))

    async def no_glossary(indices):
        return {}

    monkeypatch.setattr(runner, "_read_glossary", no_glossary)
    monkeypatch.setattr(ai_job, "POLL_SECONDS", 0)
    # Slice 2 is clean first; the cleanup of 0 and 1 only goes on once slice 2 has been reviewed.
    state = {"done": [2], "status": "running", "polls": 0}
    reviewed_while_cleaning = []

    class Jobs:
        def start(self, chapter_id, page_indices, workers):
            return {"job_id": "clean"}

        def snapshot(self, job_id):
            state["polls"] += 1
            assert state["polls"] < 100_000, "slice 2 was never reviewed during the cleanup"
            if any(index == 2 for index, _ in reviewed_while_cleaning):
                state["done"], state["status"] = [2, 0, 1], "completed"
            return {"status": state["status"], "completed": len(state["done"]), "total": 3,
                    "done_indices": list(state["done"]), "errors": []}

    def review_clean(provider, model, key, index, a, b):
        reviewed_while_cleaning.append((index, state["status"] == "running"))
        return CleanReview(index), None

    monkeypatch.setattr(chapters_router, "chapter_processing_jobs", Jobs())
    monkeypatch.setattr(ai_job, "review_clean", review_clean)

    async def scenario():
        await runner.clean()
        runner.job.stage = "review"
        await runner.review()

    asyncio.run(scenario())
    assert sorted(index for index, _ in reviewed_while_cleaning) == [0, 1, 2], "every slice is reviewed once"
    assert (2, True) in reviewed_while_cleaning, "review overlaps the cleanup"
    assert runner.job.stages["review"]["done"] == 3


def test_retry_uses_small_batches_and_restores_what_never_translates(monkeypatch):
    from app.dependencies import pipeline

    def obj(obj_id, translation=""):
        return {"id": obj_id, "region": {"x1": 10, "y1": 10, "x2": 60, "y2": 40}, "translation": translation}

    manifest = {"pages": [
        {"width": 400, "height": 600, "text_objects": [obj("b1"), obj("b2"), obj("done", "Xong")]},
        {"width": 400, "height": 600, "text_objects": [obj("a"), obj("other")]},
    ]}
    preserved, retries = [], []
    monkeypatch.setattr(pipeline, "preserve_and_reinpaint",
                        lambda chapter, index, regions: preserved.append((index, regions)), raising=False)
    runner = _runner(monkeypatch, "translate", manifest)
    monkeypatch.setattr(runner, "_ensure_objects", lambda index: None)

    async def retry(page_index, source_lang, object_ids, force=False):
        retries.append((page_index, object_ids))
        if page_index == 0:  # b1 translated, b2 deliberately left blank (a watermark)
            manifest["pages"][0]["text_objects"][0]["translation"] = "Chào"
            return {"blank_ids": ["b2"]}
        return {"blank_ids": []}  # page 1: "a" is never answered

    monkeypatch.setattr(runner, "_translate_retry", retry)
    asyncio.run(runner._retry_untranslated(0, "en", None))
    asyncio.run(runner._retry_untranslated(1, "en", {"a"}))
    assert retries == [(0, ["b1", "b2"]), (1, ["a"]), (1, ["a"]), (1, ["a"])], (
        "a failed slice retries every untranslated object, otherwise only the unanswered ids, "
        "until three batches make no progress"
    )
    assert preserved == [(1, [{"x1": 10, "y1": 10, "x2": 60, "y2": 40}])], "a deliberate blank is not restored"
    assert [item["id"] for item in runner.report["review_list"]] == ["a"]


def test_render_stage_gives_an_unletterable_object_its_original_pixels(monkeypatch):
    from app.dependencies import pipeline
    import app.routers.export as export_router
    import app.routers.image as image_router
    import app.routers.render_commit as render_commit

    region = {"x1": 10, "y1": 20, "x2": 60, "y2": 40}
    manifest = {"pages": [{"text_objects": [{"id": "text_ok", "region": region}, {"id": "text_long", "region": region}]}]}
    calls, preserved = [], []

    def render(req):
        calls.append(req)
        if len(calls) == 1:
            raise export_router.HTTPException(500, "Chèn chữ thất bại (vùng text_long)")

    monkeypatch.setattr(render_commit, "render_page", render)
    monkeypatch.setattr(export_router, "_render_request_from_page", lambda chapter_id, index, page: index)
    monkeypatch.setattr(image_router, "_current_rendered_path", lambda chapter_id, index, manifest: None)
    monkeypatch.setattr(pipeline, "preserve_and_reinpaint",
                        lambda chapter_id, index, regions: preserved.append((index, regions)), raising=False)

    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="render",
                           stages={"render": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    monkeypatch.setattr(runner, "_active_pages", lambda: [0])
    monkeypatch.setattr(runner, "_ensure_objects", lambda index: None)
    asyncio.run(runner.render())

    assert len(calls) == 2, "the slice is rendered again once the object is restored"
    assert preserved == [(0, [region])]
    assert runner.report["restored_regions"] == 1
    assert runner.report["review_list"][0]["id"] == "text_long"


def test_translate_stage_keeps_a_few_slices_in_flight_and_reports_failures(monkeypatch):
    import app.routers.ocr as ocr_router

    active = {"now": 0, "peak": 0}

    async def detected(chapter_id, req=None):
        return {"source_lang": None}

    started, memories = [], set()

    async def fake_translate(req, memory=None, slice_total=None, skip_seam_mirrors=False):
        assert skip_seam_mirrors, "A.I mode leaves seam copies to their owning slice"
        started.append(req.page_index)
        memories.add(id(memory))
        assert slice_total == 10
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.01)
        active["now"] -= 1
        if req.page_index == 3:
            raise translation_router.HTTPException(502, "provider down")
        assert req.source_lang == "auto"
        return {"translation_run": {"translated": 2, "unreadable": 0, "estimated_cost_usd": None}}

    monkeypatch.setattr(ocr_router, "detect_chapter_language", detected)
    monkeypatch.setattr(translation_router, "translate_page_in_context", fake_translate)
    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="translate", cost_usd=None,
                           stages={"translate": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_active_pages", lambda: list(range(10)))
    monkeypatch.setattr(runner, "_manifest", lambda: {"pages": [{} for _ in range(10)]})
    synced = []

    async def sync():
        synced.append(True)
        return []

    monkeypatch.setattr(runner, "_sync_seams", sync)
    monkeypatch.setattr(runner, "_join_stacked_lines", lambda indices: None)
    retried = []

    async def retry(page_index, source_lang, only):
        retried.append((page_index, only))

    monkeypatch.setattr(runner, "_retry_untranslated", retry)
    asyncio.run(runner.translate())
    assert retried == [(3, None)], "a failed slice retries every object in the same stage"

    assert started == list(range(10)), "slices start in reading order so earlier memory reaches later slices"
    assert len(memories) == 1 and None not in memories, "one chapter memory shared by every slice"
    assert runner.report["characters"] == [] and runner.report["address"] == []

    assert 1 < active["peak"] <= ai_job.TRANSLATE_CONCURRENCY
    assert runner.report["translated"] == 18
    assert runner.report["translate_errors"] == ["Lát 4: provider down"]
    assert job.stages["translate"]["done"] == 10


# -- translate -> render -> gate-free export ---------------------------------

@pytest.fixture
def cleaned_chapter(tmp_path, monkeypatch):
    raw, processed, output = tmp_path / "raw", tmp_path / "processed", tmp_path / "output"
    for root in (raw, processed, output):
        (root / CHAPTER).mkdir(parents=True)
    image = np.full((150, 220, 3), 245, dtype=np.uint8)
    pages = []
    for index in range(2):
        original, clean = raw / CHAPTER / f"p{index}.png", processed / CHAPTER / f"c{index}.png"
        cv2.putText(image, "HELLO", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        assert cv2.imwrite(str(original), image)
        assert cv2.imwrite(str(clean), np.full_like(image, 245))
        pages.append({
            "original": original.as_posix(), "clean": clean.as_posix(), "boxes": [],
            "text_objects": [{
                "id": f"obj_{index}", "shape": "rectangle", "region": {"x1": 20, "y1": 23, "x2": 180, "y2": 100},
                "style": {"font": "default"}, "ocr_text": "Hello", "translation": "",
                "semantic_type": "speech_bubble",
            }] if index == 0 else [],
            "skipped": False, "process_required": False, "source_revision": 1, "clean_revision": 1,
            "render_revision": 0, "rendered": False,
        })
    for module, names in (
        (config, ("RAW_DIR", "PROCESSED_DIR", "OUTPUT_DIR")), (translation_router, ("RAW_DIR", "PROCESSED_DIR")),
        (image_router, ("RAW_DIR", "PROCESSED_DIR", "OUTPUT_DIR")), (ai_job, ("RAW_DIR", "OUTPUT_DIR")),
        (render_commit, ("OUTPUT_DIR",)), (export_router, ("OUTPUT_DIR",)), (manifests, ("PROCESSED_DIR",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, {"RAW_DIR": raw, "PROCESSED_DIR": processed, "OUTPUT_DIR": output}[name])
    manifests.save_manifest_raw(CHAPTER, {
        "chapter_id": CHAPTER, "pages": pages, "script_review_required": True,
        "workflow": {"stage": "review", "page_index": 0},
    })
    return output


def test_translate_render_and_leave_the_chapter_open_in_the_editor(cleaned_chapter, monkeypatch):
    import app.routers.ocr as ocr_router

    async def detected(chapter_id, req=None):
        raise AssertionError("A.I mode must not run the OCR language probe")

    monkeypatch.setattr(ocr_router, "detect_chapter_language", detected)
    monkeypatch.setattr(translation_router, "get_provider_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        "app.translation.vision.VisionPageTranslator.translate_page",
        lambda self, original_path, cleaned_path, items, **kwargs: VisionTranslationResult(
            {item["id"]: "Xin chào" for item in items}, self.model, {}, None,
        ),
    )
    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, cost_usd=None,
                           stages={key: {"done": 0, "total": 0, "detail": ""} for key, _ in ai_job.STAGES})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")

    async def run():
        for stage in ("translate", "render", "finish"):
            job.stage = stage
            await getattr(runner, stage)()

    asyncio.run(run())

    assert runner.report["translated"] == 1 and runner.report["source_lang"] == "auto"
    assert runner.report["editorial_blockers"] >= 1, "unreviewed script is reported, not hidden"
    assert (cleaned_chapter / CHAPTER / "ai_mode_report.json").is_file()
    assert not list(cleaned_chapter.glob("**/*.zip")), "A.I mode leaves exporting to the editor"
    manifest = manifests.load_manifest_raw(CHAPTER)
    assert manifest["workflow"] == {"stage": "review", "page_index": 0}
    # The editor's export refuses until a human reviews the script, then exports the lettered pages.
    with pytest.raises(export_router.HTTPException):
        export_router.export_chapter(CHAPTER)
    manifests.save_manifest_raw(CHAPTER, {**manifest, "script_review_required": False})
    with zipfile.ZipFile(export_router.write_chapter_archive(CHAPTER)) as archive:
        assert archive.namelist() == ["page_001.png", "page_002.png"]


def test_retry_keeps_going_on_a_long_slice_while_batches_make_progress(monkeypatch):
    manifest = {"pages": [{"width": 400, "height": 600, "text_objects": [
        {"id": f"o{i}", "region": {"x1": 10, "y1": 10 + i, "x2": 60, "y2": 40 + i}, "translation": ""} for i in range(10)
    ]}]}
    runner = _runner(monkeypatch, "translate", manifest)
    monkeypatch.setattr(runner, "_ensure_objects", lambda index: None)
    batches = []

    async def retry(page_index, source_lang, object_ids, force=False):
        batches.append(object_ids)
        for obj in manifest["pages"][0]["text_objects"]:
            if obj["id"] in object_ids:
                obj["translation"] = "ok"
        return {"blank_ids": []}

    monkeypatch.setattr(runner, "_translate_retry", retry)
    asyncio.run(runner._retry_untranslated(0, "en", None))
    assert [len(batch) for batch in batches] == [4, 4, 2], "ten objects in batches of four, none given up"
    assert runner.report["review_list"] == [] and runner.report["restored_regions"] == 0


def test_checkpoint_three_never_restores_over_a_detected_text_box():
    from app.ai_mode.checkpoints import CleanReview, settle_clean_review

    caption = (0, 170, 1500, 550)
    art = (100, 900, 400, 1200)
    settled = settle_clean_review(CleanReview(3, (), (), ((100, 150, 1500, 560), art)), [caption])
    assert settled.restore == (art,), "the caption stays erased; damaged art away from text is restored"
