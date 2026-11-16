from __future__ import annotations

import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.ai_mode.job import AIModeCancelled, AIModeJobManager, AIModeSettings
from app.ai_mode.queue import AIModeQueueManager, MAX_QUEUE_URLS
from app.ai_providers import PROVIDERS
from app.downloader.series_toc import discover_chapter_urls, extract_chapter_links
from app.main import app
import app.routers.ai_mode as ai_mode_router


def test_extract_chapter_links_dedupes_filters_and_sorts_ascending():
    html = """
    <html><body>
      <nav>
        <a href="/about">About</a>
        <a href="https://other.example.com/series/foo/chapter-1">External</a>
        <a href="/series/other-manga/chapter-3860">Chapter 3860</a>
        <a href="/series/foo/chapter-10.5#top">Chapter 10.5</a>
        <a href="/series/foo/chapter-10">Chapter 10</a>
        <a href="/series/foo/chapter-2/">Chapter 2</a>
        <a href="/series/foo/chapter-1">Chapter 1</a>
        <a href="/series/foo/chapter-1#comments">Chapter 1 comments</a>
        <a href="/assets/banner.jpg">Banner</a>
      </nav>
    </body></html>
    """
    links = extract_chapter_links(html, "https://example.com/series/foo")
    urls = [row["url"] for row in links]
    assert urls == [
        "https://example.com/series/foo/chapter-1",
        "https://example.com/series/foo/chapter-2/",
        "https://example.com/series/foo/chapter-10",
        "https://example.com/series/foo/chapter-10.5",
    ]


def test_discover_chapter_urls_blocks_private_targets():
    import pytest

    with pytest.raises(HTTPException):
        discover_chapter_urls("http://127.0.0.1:8000/series/local", allow_js=False)


def test_queue_manager_runs_sequentially_and_handles_failures():
    seen_order: list[str] = []
    running_count = 0
    max_parallel = 0

    class FakeRunner:
        def __init__(self, job, provider, api_key):
            self.job = job
            self.report = {}

        async def _step(self):
            nonlocal running_count, max_parallel
            running_count += 1
            max_parallel = max(max_parallel, running_count)
            seen_order.append(self.job.settings.url)
            await asyncio.sleep(0.01)
            running_count -= 1
            if "ch-2" in self.job.settings.url:
                raise RuntimeError("boom on ch-2")
            self.job.chapter_id = "ch_" + self.job.settings.url.rsplit("/", 1)[-1]
            self.job.cost_usd = 0.02

        async def download(self):
            await self._step()

        async def scan(self):
            pass

        async def clean(self):
            pass

        async def review(self):
            pass

        async def translate(self):
            pass

        async def polish(self):
            pass

        async def render(self):
            pass

        async def finish(self):
            pass

    async def exercise():
        jobs = AIModeJobManager(runner_factory=FakeRunner)
        queues = AIModeQueueManager(jobs)
        base = AIModeSettings(url="", provider="deepseek", model="deepseek-chat")

        async def starter(item_settings: AIModeSettings) -> dict:
            return jobs.start(item_settings, provider=PROVIDERS["deepseek"], api_key="k")

        snap = queues.start(
            ["https://example.com/ch-1", "https://example.com/ch-2", "https://example.com/ch-3"],
            base,
            item_starter=starter,
            continue_on_error=True,
        )
        qid = snap["queue_id"]
        while snap["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
            snap = queues.snapshot(qid)
        assert max_parallel == 1
        assert seen_order == ["https://example.com/ch-1", "https://example.com/ch-2", "https://example.com/ch-3"]
        assert snap["status"] == "completed_with_errors"
        assert snap["completed_count"] == 2
        assert snap["failed_count"] == 1
        assert snap["total_cost_usd"] == 0.04
        assert [i["status"] for i in snap["items"]] == ["completed", "failed", "completed"]
        assert snap["items"][0]["chapter_id"] == "ch_ch-1"

        stopped = queues.start(
            ["https://example.com/ch-2", "https://example.com/ch-4"],
            base,
            item_starter=starter,
            continue_on_error=False,
        )
        sid = stopped["queue_id"]
        while stopped["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
            stopped = queues.snapshot(sid)
        assert stopped["status"] == "failed"
        assert [i["status"] for i in stopped["items"]] == ["failed", "skipped"]

    asyncio.run(exercise())


def test_queue_manager_cancel_stops_pending_items():
    class BlockingRunner:
        def __init__(self, job, provider, api_key):
            self.job = job
            self.report = {}

        async def download(self):
            for _ in range(40):
                if self.job.cancel_requested:
                    raise AIModeCancelled()
                await asyncio.sleep(0.01)

        async def scan(self):
            pass

        async def clean(self):
            pass

        async def review(self):
            pass

        async def translate(self):
            pass

        async def polish(self):
            pass

        async def render(self):
            pass

        async def finish(self):
            pass

    async def exercise():
        jobs = AIModeJobManager(runner_factory=BlockingRunner)
        queues = AIModeQueueManager(jobs)
        base = AIModeSettings(url="", provider="openai", model="gpt-5-mini")

        async def starter(item_settings: AIModeSettings) -> dict:
            return jobs.start(item_settings, provider=PROVIDERS["openai"], api_key="k")

        snap = queues.start(
            ["https://example.com/ch-1", "https://example.com/ch-2"],
            base,
            item_starter=starter,
        )
        await asyncio.sleep(0.02)
        queues.cancel(snap["queue_id"])
        while snap["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
            snap = queues.snapshot(snap["queue_id"])
        assert snap["status"] == "cancelled"
        assert [i["status"] for i in snap["items"]] == ["cancelled", "cancelled"]

    asyncio.run(exercise())


def test_ai_mode_queue_router_endpoints(monkeypatch):
    monkeypatch.setattr(ai_mode_router, "validate_url", lambda url: url)
    monkeypatch.setattr(ai_mode_router, "get_provider_api_key", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(
        ai_mode_router,
        "discover_chapter_urls",
        lambda url: [{"url": f"{url}/chapter-1", "label": "Chapter 1"}, {"url": f"{url}/chapter-2", "label": "Chapter 2"}],
    )

    class QuickRunner:
        def __init__(self, job, provider, api_key):
            self.job = job
            self.report = {}

        async def download(self):
            self.job.chapter_id = "abcd1234"

        async def scan(self):
            pass

        async def clean(self):
            pass

        async def review(self):
            pass

        async def translate(self):
            pass

        async def polish(self):
            pass

        async def render(self):
            pass

        async def finish(self):
            pass

    jobs = AIModeJobManager(runner_factory=QuickRunner)
    queues = AIModeQueueManager(jobs)
    monkeypatch.setattr(ai_mode_router, "ai_mode_jobs", jobs)
    monkeypatch.setattr(ai_mode_router, "ai_mode_queues", queues)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        disc = client.post("/api/ai_mode/queue/discover", json={"url": "https://example.com/manga/solo"})
        assert disc.status_code == 200
        assert len(disc.json()["chapters"]) == 2

        too_many = [f"https://example.com/ch-{i}" for i in range(MAX_QUEUE_URLS + 1)]
        bad = client.post("/api/ai_mode/queue/start", json={"urls": too_many, "provider": "openai"})
        assert bad.status_code == 422

        started = client.post(
            "/api/ai_mode/queue/start",
            json={
                "urls": ["https://example.com/ch-1", "https://example.com/ch-1/", "https://example.com/ch-2"],
                "provider": "openai",
                "model": "gpt-5-mini",
            },
        )
        assert started.status_code == 200
        body = started.json()
        assert body["total"] == 2
        qid = body["queue_id"]

        snap = body
        for _ in range(50):
            res = client.get(f"/api/ai_mode/queue/{qid}")
            assert res.status_code == 200
            snap = res.json()
            if snap["status"] not in {"pending", "running"}:
                break
        assert snap["status"] == "completed"
        assert snap["completed_count"] == 2
        assert all(i["chapter_id"] == "abcd1234" for i in snap["items"])

        active = client.get("/api/ai_mode/active")
        assert active.status_code == 200
        assert "queue" in active.json()


def test_queue_stops_on_fatal_billing_or_auth_error_even_if_continue_enabled():
    async def exercise():
        jobs = AIModeJobManager()
        queues = AIModeQueueManager(jobs)
        base = AIModeSettings(url="", provider="manga-cloud", model="")
        calls: list[str] = []

        async def starter(item_settings: AIModeSettings) -> dict:
            calls.append(item_settings.url)
            raise HTTPException(402, "Số dư Manga Cloud không đủ.")

        snap = queues.start(
            ["https://example.com/ch-1", "https://example.com/ch-2", "https://example.com/ch-3"],
            base,
            item_starter=starter,
            continue_on_error=True,
        )
        while snap["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
            snap = queues.snapshot(snap["queue_id"])
        assert calls == ["https://example.com/ch-1"]
        assert snap["status"] == "failed"
        assert [i["status"] for i in snap["items"]] == ["failed", "skipped", "skipped"]

    asyncio.run(exercise())
