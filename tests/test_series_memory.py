"""Series memory: what earlier chapters settled reaches the next one, and the user can edit it."""
import asyncio

from app import series_memory
from app.ai_mode import job as ai_job
from app.ai_mode.job import AIModeRunner, AIModeSettings
from app.ai_providers import PROVIDERS
from app.translation.context import ChapterMemory


def test_chapters_of_one_series_share_a_key_and_a_bare_chapter_url_has_none():
    key = series_memory.series_key("https://asurascans.com/comics/solo-swordmaster-3ec3b16f/chapter/12#resume")
    assert key == "asurascans-solo-swordmaster"
    assert series_memory.series_key("https://asurascans.com/comics/solo-swordmaster-99aa11bb/chapter/1") == key
    assert series_memory.series_key("https://example.com/manga/one-piece/chapter-1100") == "example-one-piece"
    assert series_memory.series_key("https://mangadex.org/chapter/0f3a") is None
    assert series_memory.series_key("not a url") is None


def test_what_the_series_settled_wins_and_the_chapter_adds_what_is_new():
    series = {"names": [{"source": "Raon", "target": "Raon"}], "terms": [{"source": "Seven Dragons", "target": "Thất Long"}]}
    chapter = {"names": [{"source": "raon", "target": "Ra-ôn"}, {"source": "Terna", "target": "Terna"}],
               "terms": [{"source": "seven dragons", "target": "Bảy Con Rồng"}]}
    merged = series_memory.settled_glossary(series, chapter)
    assert [n["target"] for n in merged["names"]] == ["Raon", "Terna"]
    assert [t["target"] for t in merged["terms"]] == ["Thất Long"]


def test_the_text_form_round_trips_and_keeps_notes_and_chapters():
    data = series_memory.clean({
        "names": [{"source": "Raon", "target": "Raon", "note": "hero"}], "terms": [{"source": "Avatar", "target": "Hóa Thân"}],
        "address": [{"from": "Raon", "to": "Terna", "self": "tôi", "other": "ngài"}],
        "characters": [{"name": "Raon", "note": "tired hero"}], "notes": "fantasy", "chapters": ["u1"]})
    text = series_memory.to_text(data)
    assert "Raon -> Terna: tôi / ngài" in text and "term: Avatar = Hóa Thân" in text
    back = series_memory.from_text(text + "\n# a comment\nnonsense line", data)
    assert back == data


def test_a_finished_chapter_teaches_the_series_and_the_next_chapter_starts_from_it(monkeypatch):
    url = "https://asurascans.com/comics/solo-swordmaster-3ec3b16f/chapter/1"
    key = series_memory.series_key(url)
    series_memory.save(key, {"names": [{"source": "Raon", "target": "Raon"}], "notes": "fantasy"})
    job = ai_job.AIModeJob(job_id="j", settings=AIModeSettings(url=url, provider="openai", model="m"), chapter_id="c1",
                           stage="finish", stages={"finish": {"done": 0, "total": 0, "detail": ""}})
    runner = AIModeRunner(job, PROVIDERS["openai"], "key")
    runner._series = series_memory.load(key)
    runner._memory = ChapterMemory("", {})
    runner._memory.update(1, {"characters": [{"name": "Terna", "note": "goddess"}],
                              "address": [{"from": "Terna", "to": "Raon", "self": "ta", "other": "ngươi"}]}, {}, [])
    runner.report["glossary"] = {"terms": [{"source": "Avatar", "target": "Hóa Thân"}]}
    monkeypatch.setattr("app.routers.export._preflight_copy", lambda manifest, **_: {"blocker_count": 0, "blockers": []})
    monkeypatch.setattr(ai_job, "load_manifest_raw", lambda chapter_id: {})
    monkeypatch.setattr(ai_job, "save_manifest_raw", lambda chapter_id, manifest: None)
    monkeypatch.setattr(ai_job, "OUTPUT_DIR", series_memory.SERIES_DIR / "out")
    asyncio.run(runner.finish())

    saved = series_memory.load(key)
    assert [t["target"] for t in saved["terms"]] == ["Hóa Thân"] and saved["chapters"] == [url]
    assert saved["address"] == [{"from": "Terna", "to": "Raon", "self": "ta", "other": "ngươi"}]
    assert saved["characters"] == [{"name": "Terna", "note": "goddess"}] and saved["notes"] == "fantasy"


def test_the_user_reads_and_edits_a_series_memory_by_its_chapter_url():
    import pytest
    from fastapi import HTTPException

    from app.routers.ai_mode import SeriesMemoryRequest, read_series_memory, save_series_memory

    url = "https://asurascans.com/comics/the-newbie-civil-servant-hunts-villains-before-goi-11fe7da3/chapter/1"
    assert read_series_memory(SeriesMemoryRequest(url=url))["text"] == ""
    saved = save_series_memory(SeriesMemoryRequest(url=url, text="Han Seo-jun = Han Seo-jun\nSeo-jun -> boss: tôi / anh"))
    assert saved["text"] == "Han Seo-jun = Han Seo-jun\nSeo-jun -> boss: tôi / anh"
    assert read_series_memory(SeriesMemoryRequest(url=url.replace("/1", "/2")))["text"] == saved["text"]
    with pytest.raises(HTTPException):
        read_series_memory(SeriesMemoryRequest(url="https://mangadex.org/chapter/0f3a"))
