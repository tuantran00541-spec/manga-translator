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
    assert RecordingRunner.calls == ["download", "scan", "clean", "qc", "translate", "repair", "render", "export"]
    assert snapshot["status"] == "completed"
    assert all(stage["status"] == "done" for stage in snapshot["stages"])
    assert snapshot["chapter_id"] == CHAPTER
    assert snapshot["cost_usd"] is None, "OpenAI cost is unknown, not zero"


def test_manager_stops_at_the_failing_stage():
    snapshot = _run_manager(fail_at="qc")
    assert RecordingRunner.calls == ["download", "scan", "clean", "qc"]
    assert snapshot["status"] == "failed" and snapshot["error"] == "boom"
    states = {stage["key"]: stage["status"] for stage in snapshot["stages"]}
    assert states["qc"] == "failed" and states["translate"] == "pending"
    assert snapshot["download_url"] is None


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


def test_repair_stage_uses_preserve_add_box_and_retries_before_restoring(monkeypatch):
    from app.dependencies import pipeline

    def obj(obj_id, translation=""):
        return {"id": obj_id, "region": {"x1": 10, "y1": 10, "x2": 60, "y2": 40}, "translation": translation}

    manifest = {"pages": [
        {"width": 400, "height": 600, "text_objects": [obj("k", "")]},
        {"width": 400, "height": 600, "text_objects": [obj("b1"), obj("b2"), obj("done", "Xong")]},
        {"width": 400, "height": 600, "text_objects": [obj("a"), obj("other")]},
    ]}
    preserved, added, retries = [], [], []
    def preserve(chapter_id, index, regions):
        preserved.append((index, regions))
        manifest["pages"][index].setdefault("preserve_regions", []).extend(regions)

    monkeypatch.setattr(pipeline, "preserve_and_reinpaint", preserve, raising=False)
    monkeypatch.setattr(pipeline, "add_manual_box",
                        lambda chapter_id, index, *box: added.append((index, box)), raising=False)

    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="repair", cost_usd=None,
                           stages={"repair": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    ensured = []
    monkeypatch.setattr(runner, "_ensure_objects", ensured.append)

    async def retry(page_index, source_lang, object_ids):
        retries.append((page_index, object_ids))
        if page_index == 1:  # b1 translated, b2 deliberately left blank (a watermark)
            manifest["pages"][1]["text_objects"][0]["translation"] = "Chào"
            return {"blank_ids": ["b2"]}
        return {"blank_ids": [], "missing_ids": object_ids}  # page 2: "a" is never answered

    monkeypatch.setattr(runner, "_translate_retry", retry)
    runner._repair = {
        0: {"keep": [[10, 10, 60, 40]], "missed": [[100, 100, 200, 160, "HI"], [0, 0, 5, 5, "tiny"]], "missing": []},
        1: {"failed": True},
        2: {"missing": ["a"]},
    }
    asyncio.run(runner.repair())

    assert preserved[0] == (0, [{"x1": 6, "y1": 6, "x2": 64, "y2": 44}]), "keep -> padded preserve region"
    assert added == [(0, (100, 100, 200, 160))], "missed text -> add_box; boxes under 10px are dropped"
    assert ensured == [0], "the new box gets its text object before the retry"
    assert retries == [(1, ["b1", "b2"]), (2, ["a"]), (2, ["a"]), (2, ["a"])], (
        "a kept object is not retried; a failed slice retries every untranslated object, "
        "otherwise only the unanswered ids, until three batches make no progress"
    )
    assert not any(index == 1 for index, _ in preserved), "a deliberate blank is not restored"
    assert preserved[-1] == (2, [{"x1": 10, "y1": 10, "x2": 60, "y2": 40}]), "never answered -> original restored"
    assert [item["id"] for item in runner.report["review_list"]] == ["a"]
    assert runner.report["missed_added"] == 1 and runner.report["kept_regions"] == 1


def test_translate_stage_keeps_a_few_slices_in_flight_and_reports_failures(monkeypatch):
    import app.routers.ocr as ocr_router

    active = {"now": 0, "peak": 0}

    async def detected(chapter_id, req=None):
        return {"source_lang": None}

    started, memories = [], set()

    async def fake_translate(req, memory=None, slice_total=None):
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
    asyncio.run(runner.translate())

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


def test_translate_render_and_export_without_human_review(cleaned_chapter, monkeypatch):
    import app.routers.ocr as ocr_router

    async def detected(chapter_id, req=None):
        return {"source_lang": "en"}

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
        for stage in ("translate", "render", "export"):
            job.stage = stage
            await getattr(runner, stage)()

    asyncio.run(run())

    assert runner.report["translated"] == 1 and runner.report["source_lang"] == "en"
    assert runner.report["editorial_blockers"] >= 1, "unreviewed script is reported, not hidden"
    assert job.archive_path and job.archive_path.endswith(f"ai_mode_{CHAPTER}.zip")
    with zipfile.ZipFile(job.archive_path) as archive:
        assert archive.namelist() == ["page_001.png", "page_002.png"]
    assert (cleaned_chapter / CHAPTER / "ai_mode_report.json").is_file()
    # The normal export keeps refusing until a human reviews the script.
    with pytest.raises(export_router.HTTPException):
        export_router.export_chapter(CHAPTER)


def test_repair_keeps_retrying_a_long_slice_while_batches_make_progress(monkeypatch):
    manifest = {"pages": [{"width": 400, "height": 600, "text_objects": [
        {"id": f"o{i}", "region": {"x1": 10, "y1": 10 + i, "x2": 60, "y2": 40 + i}, "translation": ""} for i in range(10)
    ]}]}
    job = ai_job.AIModeJob(job_id="j", settings=SETTINGS, chapter_id=CHAPTER, stage="repair", cost_usd=None,
                           stages={"repair": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    batches = []

    async def retry(page_index, source_lang, object_ids):
        batches.append(object_ids)
        for obj in manifest["pages"][0]["text_objects"]:
            if obj["id"] in object_ids:
                obj["translation"] = "ok"
        return {"blank_ids": []}

    monkeypatch.setattr(runner, "_translate_retry", retry)
    runner._repair = {0: {"failed": True}}
    asyncio.run(runner.repair())
    assert [len(batch) for batch in batches] == [4, 4, 2], "ten objects in batches of four, none given up"
    assert runner.report["review_list"] == [] and runner.report["restored_regions"] == 0

