from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.media_ai_studio import (
    AICanceledError,
    StoryBibleSchema,
    TRANSLATION_BATCH_SIZE,
    TranslatedCueItemSchema,
    TranslatedCueBatchSchema,
    is_piper_tts_available,
)
from app.media_downloader import find_ffmpeg_binary
from app.routers import studio as studio_router
from app import video_studio


def _make_sample_mp4(dest: Path, duration: float = 2.0) -> Path:
    ffmpeg_bin = find_ffmpeg_binary()
    assert ffmpeg_bin, "FFmpeg is required for Video Studio tests"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x1e2430:s=320x180:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-shortest",
        str(dest),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"FFmpeg sample generation failed: {res.stderr}"
    return dest


@pytest.fixture()
def isolated_studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    studio_dir = tmp_path / "video_studio"
    studio_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(video_studio, "STUDIO_DIR", studio_dir)
    monkeypatch.setattr(studio_router, "STUDIO_DIR", studio_dir)
    # Reset queue runtime state between tests
    with video_studio._RUNTIME_LOCK:
        video_studio._ACTIVE_WORKERS.clear()
        video_studio._QUEUE_STATE.update(
            {
                "running": False,
                "project_ids": [],
                "current_project_id": None,
                "completed_ids": [],
                "failed_ids": [],
                "cancel_event": None,
            }
        )
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    return client, studio_dir


def test_parse_ass_and_srt_subtitles(tmp_path: Path):
    ass_file = tmp_path / "sample.ass"
    ass_file.write_text(
        "[Script Info]\nTitle: Test\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.20,0:00:03.50,Default,,0,0,0,,{\\an2}Hello\\NWorld\n"
        "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Second line\n",
        encoding="utf-8",
    )
    cues = video_studio.parse_studio_subtitle_file(ass_file)
    assert len(cues) == 2
    assert cues[0]["start_sec"] == 1.2
    assert cues[0]["end_sec"] == 3.5
    assert cues[0]["original_text"] == "Hello World"
    assert cues[1]["original_text"] == "Second line"

    srt_file = tmp_path / "sample.srt"
    srt_file.write_text(
        "1\n00:00:00,200 --> 00:00:01,500\nXin chào mọi người.\n\n"
        "2\n00:00:01,600 --> 00:00:02,900\nChào mừng tới Xưởng Video AI.\n",
        encoding="utf-8",
    )
    srt_cues = video_studio.parse_studio_subtitle_file(srt_file)
    assert len(srt_cues) == 2
    assert srt_cues[0]["original_text"] == "Xin chào mọi người."


def test_upload_project_crud_and_subtitle_attach(isolated_studio, tmp_path: Path):
    client, _ = isolated_studio
    sample_mp4 = _make_sample_mp4(tmp_path / "clip.mp4", duration=2.0)
    srt_bytes = (
        "1\n00:00:00,100 --> 00:00:00,900\nHello world\n\n"
        "2\n00:00:01,000 --> 00:00:01,850\nWelcome back\n"
    ).encode("utf-8")

    with sample_mp4.open("rb") as vf:
        resp = client.post(
            "/api/studio/projects/upload",
            files={
                "video": ("clip.mp4", vf, "video/mp4"),
                "subtitle": ("clip.srt", srt_bytes, "text/plain"),
            },
            data={
                "title": "Demo Review Clip",
                "settings_json": json.dumps({"output_mode": "dub_only", "default_voice": "Puck"}),
            },
        )
    assert resp.status_code == 200, resp.text
    proj = resp.json()
    pid = proj["project_id"]
    assert proj["title"] == "Demo Review Clip"
    assert proj["has_source_video"] is True
    assert proj["segment_count"] == 2
    assert proj["checkpoints"]["asr_done"] is True
    assert proj["checkpoints"]["translation_done"] is False
    assert proj["settings"]["output_mode"] == "dub_only"

    # List & Get endpoints
    listed = client.get("/api/studio/projects")
    assert listed.status_code == 200
    assert len(listed.json()["projects"]) == 1

    fetched = client.get(f"/api/studio/projects/{pid}")
    assert fetched.status_code == 200
    assert len(fetched.json()["segments"]) == 2

    # Source video & thumbnail serving
    vid_resp = client.get(f"/api/studio/projects/{pid}/source-video")
    assert vid_resp.status_code == 200
    assert len(vid_resp.content) > 1000
    thumb_resp = client.get(f"/api/studio/projects/{pid}/thumbnail")
    assert thumb_resp.status_code == 200
    assert thumb_resp.content[:2] == b"\xff\xd8"

    # PATCH update segments, speaker voices, story bible & title
    segs = fetched.json()["segments"]
    segs[0]["translated_text"] = "Xin chào thế giới"
    segs[0]["speaker"] = "MC"
    segs[0]["voice_override"] = "Kore"
    patch_resp = client.patch(
        f"/api/studio/projects/{pid}",
        json={
            "title": "Demo Updated",
            "speaker_voices": {"MC": "Puck", "S1": "Kore"},
            "story_bible": {
                "source_language": "en",
                "genre": "review",
                "synopsis": "Clip thử nghiệm",
                "glossary": ["AI = Trí tuệ nhân tạo"],
                "pronoun_rules": ["MC xưng tôi"],
                "style_notes": "Ngắn gọn",
            },
            "segments": segs,
        },
    )
    assert patch_resp.status_code == 200, patch_resp.text
    updated = patch_resp.json()
    assert updated["title"] == "Demo Updated"
    assert updated["speaker_voices"]["MC"] == "Puck"
    assert updated["story_bible"]["synopsis"] == "Clip thử nghiệm"
    assert updated["segments"][0]["translated_text"] == "Xin chào thế giới"
    assert updated["segments"][0]["voice_override"] == "Kore"
    # Editing untranslated SRT must NOT falsely mark translation_done=True
    assert updated["checkpoints"]["translation_done"] is False

    # Attach .ass subtitle replaces segments cleanly
    ass_bytes = (
        "[Events]\n"
        "Dialogue: 0,0:00:00.20,0:00:01.20,Default,,0,0,0,,Một câu từ tệp ASS\n"
    ).encode("utf-8")
    sub_resp = client.post(
        f"/api/studio/projects/{pid}/subtitle",
        files={"subtitle": ("custom.ass", ass_bytes, "text/plain")},
    )
    assert sub_resp.status_code == 200, sub_resp.text
    assert sub_resp.json()["segment_count"] == 1
    assert sub_resp.json()["segments"][0]["original_text"] == "Một câu từ tệp ASS"

    # Delete project
    del_resp = client.delete(f"/api/studio/projects/{pid}")
    assert del_resp.status_code == 200
    assert client.get(f"/api/studio/projects/{pid}").status_code == 404


def test_resumable_translation_checkpoint_resume(isolated_studio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _ = isolated_studio
    sample_mp4 = _make_sample_mp4(tmp_path / "long_mock.mp4", duration=2.0)

    with sample_mp4.open("rb") as vf:
        created = client.post("/api/studio/projects/upload", files={"video": ("long_mock.mp4", vf, "video/mp4")}).json()
    pid = created["project_id"]

    # Seed 55 cues (> TRANSLATION_BATCH_SIZE=45 => 2 batches + Story Bible)
    total_cues = TRANSLATION_BATCH_SIZE + 10
    proj = video_studio._load_project(pid)
    proj["segments"] = [
        {
            "id": i,
            "start_sec": round(i * 1.0, 2),
            "end_sec": round(i * 1.0 + 0.85, 2),
            "speaker": "S1",
            "gender": "male",
            "tone": "natural and clear",
            "original_text": f"Original sentence number {i}",
            "translated_text": f"Original sentence number {i}",
            "voice_override": "",
            "words": [],
        }
        for i in range(total_cues)
    ]
    proj["checkpoints"]["asr_done"] = True
    video_studio._save_project(proj)

    class _DummyProvider:
        id = "dummy"
        label = "Dummy AI"
        protocol = "openai"

    monkeypatch.setattr(video_studio, "resolve_configured_ai_provider", lambda _pid: (_DummyProvider(), "test-key"))
    monkeypatch.setattr(video_studio, "default_transcribe_model_for", lambda _p: "dummy-flash")

    bible_calls = []
    batch_calls = []
    cancel_ev = threading.Event()

    def _fake_bible(*args, **kwargs):
        bible_calls.append(1)
        return StoryBibleSchema(source_language="en", genre="drama", synopsis="Test movie")

    def _fake_batch(cues, **kwargs):
        batch_calls.append([c["id"] for c in cues])
        # Simulate interruption right as Batch 2 starts (after Batch 1 checkpoint is saved on disk!)
        if len(batch_calls) == 2 and not cancel_ev.is_set():
            cancel_ev.set()
            raise AICanceledError("Simulated pause on Batch 2")
        items = [
            TranslatedCueItemSchema(
                id=c["id"],
                corrected_original=c["original_text"],
                draft_translation=f"Bản dịch {c['id']}",
                translated_text=f"Câu dịch chuẩn {c['id']}",
                gender="male",
                tone="natural",
            )
            for c in cues
        ]
        return TranslatedCueBatchSchema(source_language="en", items=items)

    monkeypatch.setattr(video_studio, "_build_story_bible", _fake_bible)
    monkeypatch.setattr(video_studio, "_translate_cue_batch", _fake_batch)

    # First run: Batch 1 saves checkpoint, then pauses on Batch 2 via AICanceledError
    with pytest.raises(AICanceledError):
        video_studio._run_resumable_translation(pid, video_studio._load_project(pid), cancel_ev)

    mid = video_studio._load_project(pid)
    assert mid["checkpoints"]["bible_done"] is True
    assert mid["checkpoints"]["translated_batches"] == 1
    assert len(mid["checkpoints"]["translated_ids"]) == TRANSLATION_BATCH_SIZE
    assert mid["checkpoints"]["translation_done"] is False
    assert mid["segments"][0]["translated_text"] == "Câu dịch chuẩn 0"
    assert mid["segments"][TRANSLATION_BATCH_SIZE]["translated_text"] == f"Original sentence number {TRANSLATION_BATCH_SIZE}"

    # Second run (Resume!): must skip Story Bible AND skip Batch 1!
    resume_ev = threading.Event()
    video_studio._run_resumable_translation(pid, video_studio._load_project(pid), resume_ev)

    done = video_studio._load_project(pid)
    assert len(bible_calls) == 1, "Story Bible must be reused from checkpoint, not re-called"
    assert len(batch_calls) == 3, "Batch 1 ran once, Batch 2 aborted once and resumed once"
    assert batch_calls[-1] == list(range(TRANSLATION_BATCH_SIZE, total_cues))
    assert done["checkpoints"]["translation_done"] is True
    assert done["segments"][-1]["translated_text"] == f"Câu dịch chuẩn {total_cues - 1}"


@pytest.mark.skipif(not is_piper_tts_available(), reason="piper-tts not installed")
def test_preview_tts_and_render_exports_with_piper(isolated_studio, tmp_path: Path):
    client, _ = isolated_studio
    sample_mp4 = _make_sample_mp4(tmp_path / "dub_test.mp4", duration=2.2)
    srt_bytes = (
        "1\n00:00:00,100 --> 00:00:01,000\nXin chào các bạn.\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\nChào mừng đến Xưởng Video.\n"
    ).encode("utf-8")

    with sample_mp4.open("rb") as vf:
        proj = client.post(
            "/api/studio/projects/upload",
            files={
                "video": ("dub_test.mp4", vf, "video/mp4"),
                "subtitle": ("dub_test.srt", srt_bytes, "text/plain"),
            },
            data={"settings_json": json.dumps({"output_mode": "dub_only", "tts_provider_id": "piper_tts", "default_voice": "Puck"})},
        ).json()
    pid = proj["project_id"]

    # 1. Instant ~50ms preview TTS endpoint
    prev_resp = client.post(
        f"/api/studio/projects/{pid}/preview-tts",
        json={"text": "Xin chào, đây là giọng Quang Huy.", "voice": "Puck", "tts_provider_id": "piper_tts", "target_lang": "vi"},
    )
    assert prev_resp.status_code == 200, prev_resp.text
    assert "audio/wav" in prev_resp.headers.get("content-type", "")
    assert prev_resp.content[:4] == b"RIFF"
    assert len(prev_resp.content) > 2000

    # 2. Full Phase-2 dub_only render (-c:v copy) & verify all 4 export downloads!
    ffmpeg_bin = find_ffmpeg_binary()
    assert ffmpeg_bin
    video_studio._run_studio_render(
        pid,
        video_studio._load_project(pid),
        video_studio._project_dir(pid),
        ffmpeg_bin,
        threading.Event(),
    )
    rendered = client.get(f"/api/studio/projects/{pid}").json()
    assert rendered["status"] == "completed"
    assert rendered["checkpoints"]["render_done"] is True

    for kind in ("video", "srt", "ass", "audio"):
        exp_resp = client.get(f"/api/studio/projects/{pid}/export/{kind}")
        assert exp_resp.status_code == 200, f"Export {kind} failed"
        assert len(exp_resp.content) > 50


def test_batch_queue_runs_multiple_projects(isolated_studio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _ = isolated_studio
    sample_mp4 = _make_sample_mp4(tmp_path / "ep.mp4", duration=1.5)
    pids = []
    for idx in range(2):
        with sample_mp4.open("rb") as vf:
            p = client.post(
                "/api/studio/projects/upload",
                files={"video": (f"ep_{idx}.mp4", vf, "video/mp4")},
                data={"title": f"Tập {idx + 1}"},
            ).json()
            pids.append(p["project_id"])

    executed = []

    def _fake_exec(project_id, cancel_event, *, auto_render=False, force_retranslate=False):
        executed.append((project_id, auto_render))
        video_studio._update_project_fields(project_id, status="completed", stage="completed", percent=100.0)

    monkeypatch.setattr(video_studio, "_execute_prepare_sync", _fake_exec)

    start_resp = client.post("/api/studio/queue/start", json={"project_ids": pids, "auto_render": True})
    assert start_resp.status_code == 200

    deadline = time.time() + 5.0
    st = start_resp.json()
    while st["running"] and time.time() < deadline:
        time.sleep(0.05)
        st = client.get("/api/studio/queue").json()

    assert st["running"] is False
    assert st["completed_ids"] == pids
    assert st["failed_ids"] == []
    assert executed == [(pids[0], True), (pids[1], True)]


def test_security_rejects_bad_inputs(isolated_studio):
    client, _ = isolated_studio

    assert client.get("/api/studio/projects/not-valid-id").status_code == 400
    assert client.post(
        "/api/studio/projects/upload",
        files={"video": ("malware.exe", b"MZ1234", "application/octet-stream")},
    ).status_code == 400
    assert client.post(
        "/api/studio/projects/from-url",
        json={"url": "http://127.0.0.1:8000/secret.mp4"},
    ).status_code == 400


def test_cancel_project_is_instant_even_during_blocking_llm(
    isolated_studio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client, _ = isolated_studio
    sample_mp4 = _make_sample_mp4(tmp_path / "slow.mp4", duration=1.5)
    srt_bytes = (
        "1\n00:00:00,100 --> 00:00:00,700\nHello slow server\n\n"
        "2\n00:00:00,800 --> 00:00:01,400\nStill waiting here\n\n"
    ).encode("utf-8")
    with sample_mp4.open("rb") as vf:
        proj = client.post(
            "/api/studio/projects/upload",
            files={"video": ("slow.mp4", vf, "video/mp4"), "subtitle": ("slow.srt", srt_bytes, "text/plain")},
        ).json()
    pid = proj["project_id"]
    assert proj["segment_count"] == 2

    class _DummyProvider:
        id = "dummy"
        label = "Dummy AI"
        protocol = "openai"

    started = threading.Event()
    monkeypatch.setattr(video_studio, "resolve_configured_ai_provider", lambda _pid: (_DummyProvider(), "k"))
    monkeypatch.setattr(video_studio, "default_transcribe_model_for", lambda _p: "dummy")

    def _hanging_batch(*args, cancel_event=None, **kwargs):
        started.set()
        from app.media_ai_studio import _run_cancellable

        return _run_cancellable(lambda: time.sleep(10.0), cancel_event=cancel_event)

    monkeypatch.setattr(video_studio, "_translate_cue_batch", _hanging_batch)

    prep = client.post(f"/api/studio/projects/{pid}/prepare", json={"auto_render": False})
    assert prep.status_code == 200
    assert started.wait(timeout=3.0), f"Worker did not enter LLM call: {client.get(f'/api/studio/projects/{pid}').json()}"

    t0 = time.perf_counter()
    cancel_resp = client.post(f"/api/studio/projects/{pid}/cancel")
    elapsed = time.perf_counter() - t0
    assert cancel_resp.status_code == 200
    assert elapsed < 0.5, f"Cancel took too long: {elapsed:.2f}s"
    body = cancel_resp.json()
    assert body["status"] == "paused"
    assert body["active"] is False

    # Wait briefly to prove the cancelled thread cannot overwrite paused status back to failed/running!
    time.sleep(0.25)
    after = client.get(f"/api/studio/projects/{pid}").json()
    assert after["status"] == "paused"
    assert after["active"] is False

