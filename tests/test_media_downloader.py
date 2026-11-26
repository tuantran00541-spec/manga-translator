from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from app import media_downloader as md
from app.main import app


def _patch_public_dns(monkeypatch):
    monkeypatch.setattr(md, "validate_url", lambda url: url)


def test_safety_rejects_non_https_and_malformed_urls():
    for bad in (
        "",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:pass@www.youtube.com/watch?v=abc",
        "https://www.youtube.com:8443/watch?v=abc",
        "https://www.youtube.com/watch?v=abc\r\nX-Inject: 1",
    ):
        res = md.inspect_link_safety(bad, resolve_redirects=False)
        assert res["safe"] is False
        assert res["checks"][0]["id"] == "https_protocol"
        assert res["checks"][0]["passed"] is False


def test_safety_rejects_phishing_punycode_ip_and_internal_domains():
    for bad in (
        "https://youtube.com.evil.example/watch?v=abc",
        "https://tiktok.com.phish.net/@user/video/1",
        "https://xn--youtbe-9ta.com/watch?v=abc",
        "https://142.250.190.46/watch?v=abc",
        "https://intranet.local/video.mp4",
        "https://service.internal/clip.mp4",
        "https://www.youtube.com/",
    ):
        res = md.inspect_link_safety(bad, resolve_redirects=False)
        assert res["safe"] is False
        domain_chk = next(c for c in res["checks"] if c["id"] == "official_domain")
        assert domain_chk["passed"] is False


def test_safety_accepts_universal_public_video_platforms(monkeypatch):
    _patch_public_dns(monkeypatch)
    cases = [
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "youtube"),
        ("https://youtu.be/jNQXAC9IVRw", "youtube"),
        ("https://www.tiktok.com/@creator/video/7123456789012345678", "tiktok"),
        ("https://vm.tiktok.com/ZM8abcdef/", "tiktok"),
        ("https://www.facebook.com/reel/1234567890", "facebook"),
        ("https://fb.watch/abc123XYZ/", "facebook"),
        ("https://www.instagram.com/reel/Cx123456/", "instagram"),
        ("https://x.com/user/status/1234567890", "twitter"),
        ("https://www.bilibili.com/video/BV1xx411c7mD", "bilibili"),
        ("https://vimeo.com/123456789", "vimeo"),
        ("https://cdn.example-video-site.org/clips/demo.mp4", "web"),
    ]
    for url, expected_platform in cases:
        res = md.inspect_link_safety(url, resolve_redirects=False)
        assert res["safe"] is True, f"Failed for {url}: {res}"
        assert res["platform"] == expected_platform
        assert all(c["passed"] for c in res["checks"])


def test_safety_blocks_ssrf_private_ip(monkeypatch):
    def fake_validate_url(url: str):
        raise HTTPException(400, "Hostname resolves to non-public IP 127.0.0.1")

    monkeypatch.setattr(md, "validate_url", fake_validate_url)
    res = md.inspect_link_safety(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        resolve_redirects=False,
    )
    assert res["safe"] is False
    ssrf_chk = next(c for c in res["checks"] if c["id"] == "network_ssrf")
    assert ssrf_chk["passed"] is False


def test_safety_blocks_redirect_to_phishing_or_http(monkeypatch):
    _patch_public_dns(monkeypatch)

    fake_resp = MagicMock()
    fake_resp.status_code = 302
    fake_resp.headers = {"location": "https://tiktok.com.evil-phish.example/steal"}
    monkeypatch.setattr(md, "public_request", lambda *args, **kwargs: fake_resp)

    res = md.inspect_link_safety("https://vm.tiktok.com/ZM8abcdef/", resolve_redirects=True)
    assert res["safe"] is False
    redir_chk = next(c for c in res["checks"] if c["id"] == "redirect_chain")
    assert redir_chk["passed"] is False

    fake_http = MagicMock()
    fake_http.status_code = 302
    fake_http.headers = {"location": "http://example.com/downgrade.mp4"}
    monkeypatch.setattr(md, "public_request", lambda *args, **kwargs: fake_http)
    res2 = md.inspect_link_safety("https://vm.tiktok.com/ZM8abcdef/", resolve_redirects=True)
    assert res2["safe"] is False


def test_safety_allows_same_platform_https_redirect(monkeypatch):
    _patch_public_dns(monkeypatch)

    hop1 = MagicMock()
    hop1.status_code = 301
    hop1.headers = {"location": "https://www.tiktok.com/@creator/video/71234567890"}
    hop2 = MagicMock()
    hop2.status_code = 200
    hop2.headers = {}

    responses = iter([hop1, hop2])
    monkeypatch.setattr(md, "public_request", lambda *args, **kwargs: next(responses))

    res = md.inspect_link_safety("https://vm.tiktok.com/ZM8abcdef/", resolve_redirects=True)
    assert res["safe"] is True
    assert res["resolved_url"] == "https://www.tiktok.com/@creator/video/71234567890"


def test_build_quality_options_handles_vertical_1080p_and_audio():
    info = {
        "duration": 30,
        "formats": [
            {"vcodec": "avc1", "acodec": "none", "width": 1080, "height": 1920, "ext": "mp4", "fps": 30, "filesize": 12_000_000},
            {"vcodec": "avc1", "acodec": "none", "width": 720, "height": 1280, "ext": "mp4", "fps": 30, "filesize": 6_000_000},
            {"vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128, "filesize": 500_000},
        ],
    }
    options, default_preset, is_vertical = md.build_quality_options(info, has_ffmpeg=True)
    assert is_vertical is True
    assert default_preset == "video_1080"
    by_id = {o["id"]: o for o in options}
    assert by_id["video_1080"]["available"] is True
    assert by_id["video_1080"]["resolution"] == "1080×1920"
    assert by_id["video_720"]["available"] is True
    assert by_id["audio_only"]["container"] == "MP3"


def test_build_quality_options_disables_1080p_for_sd_source():
    info = {
        "duration": 20,
        "formats": [
            {"vcodec": "avc1", "acodec": "mp4a", "width": 540, "height": 960, "ext": "mp4", "filesize": 3_000_000},
        ],
    }
    options, default_preset, _ = md.build_quality_options(info, has_ffmpeg=False)
    by_id = {o["id"]: o for o in options}
    assert by_id["video_1080"]["available"] is False
    assert by_id["video_720"]["available"] is True
    assert default_preset == "video_720"
    assert by_id["audio_only"]["container"] == "M4A"


def test_inspect_media_url_rejects_drm_and_live(monkeypatch):
    _patch_public_dns(monkeypatch)
    monkeypatch.setattr(
        md,
        "inspect_link_safety",
        lambda url, resolve_redirects=True: {
            "safe": True,
            "status": "safe",
            "risk_level": "low",
            "platform": "youtube",
            "platform_label": "YouTube",
            "normalized_url": url,
            "resolved_url": url,
            "hostname": "www.youtube.com",
            "summary": "OK",
            "checks": [],
        },
    )

    monkeypatch.setattr(md, "_extract_info", lambda url, platform: {"title": "DRM", "_has_drm": True, "formats": []})
    res_drm = md.inspect_media_url("https://www.youtube.com/watch?v=abc")
    assert res_drm["safe"] is False
    assert any(c["id"] == "content_policy" and not c["passed"] for c in res_drm["safety"]["checks"])

    monkeypatch.setattr(md, "_extract_info", lambda url, platform: {"title": "Live", "is_live": True, "formats": []})
    res_live = md.inspect_media_url("https://www.youtube.com/watch?v=abc")
    assert res_live["safe"] is False


def test_sanitize_download_filename():
    assert md.sanitize_download_filename('Hello: <World> / "Test"?*', "1080p", "mp4") == "Hello World Test [1080p].mp4"
    assert md.sanitize_download_filename("CON", "audio", "mp3") == "video [audio].mp3"


def test_media_api_inspect_download_and_file_serving(monkeypatch, tmp_path):
    monkeypatch.setattr(md, "MEDIA_DIR", tmp_path)
    with md._JOBS_LOCK:
        md._JOBS.clear()

    client = TestClient(app, base_url="http://127.0.0.1:8000")

    # 1. Unsafe link returns structured safety report (safe=False)
    r_bad = client.post("/api/media/inspect", json={"url": "http://evil.example/video"})
    assert r_bad.status_code == 200
    body_bad = r_bad.json()
    assert body_bad["safe"] is False
    assert body_bad["safety"]["checks"][0]["passed"] is False

    # 2. Safe inspect returns 1080p / 720p / audio options
    monkeypatch.setattr(
        md,
        "inspect_link_safety",
        lambda url, resolve_redirects=True: {
            "safe": True,
            "status": "safe",
            "risk_level": "low",
            "platform": "youtube",
            "platform_label": "YouTube",
            "normalized_url": url,
            "resolved_url": url,
            "hostname": "www.youtube.com",
            "summary": "An toàn",
            "checks": [{"id": "https_protocol", "label": "HTTPS", "passed": True, "detail": "OK"}],
        },
    )
    monkeypatch.setattr(
        md,
        "_extract_info",
        lambda url, platform: {
            "title": "Sample Clip",
            "uploader": "Demo Channel",
            "duration": 45,
            "thumbnail": "https://i.ytimg.com/vi/abc/hqdefault.jpg",
            "formats": [
                {"vcodec": "avc1", "acodec": "none", "width": 1920, "height": 1080, "ext": "mp4", "fps": 30, "filesize": 15_000_000},
                {"vcodec": "avc1", "acodec": "none", "width": 1280, "height": 720, "ext": "mp4", "fps": 30, "filesize": 8_000_000},
                {"vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128, "filesize": 700_000},
            ],
        },
    )
    r_ok = client.post("/api/media/inspect", json={"url": "https://www.youtube.com/watch?v=abc"})
    assert r_ok.status_code == 200
    body_ok = r_ok.json()
    assert body_ok["safe"] is True
    assert [o["id"] for o in body_ok["options"]] == ["video_1080", "video_720", "audio_only"]
    assert body_ok["media"]["title"] == "Sample Clip"

    # 3. Simulated completed download job can be polled, downloaded, and deleted
    def fake_worker(job_id, target_url, platform_label, preset, is_vertical, ai_options=None):
        job_dir = tmp_path / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        out_file = job_dir / "media.mp4"
        out_file.write_bytes(b"FAKE_MP4_DATA")
        with md._JOBS_LOCK:
            job = md._JOBS[job_id]
            job["file_path"] = str(out_file)
            job["filename"] = "Sample Clip [1080p].mp4"
            job["file_size"] = out_file.stat().st_size
            job["percent"] = 100.0
            job["stage"] = "completed"
            job["status"] = "completed"

    monkeypatch.setattr(md, "_run_download_worker", fake_worker)

    r_dl = client.post(
        "/api/media/download",
        json={"url": "https://www.youtube.com/watch?v=abc", "preset": "video_1080", "title": "Sample Clip"},
    )
    assert r_dl.status_code == 200
    job_id = r_dl.json()["job_id"]

    # Wait briefly for daemon thread to finish
    import time
    for _ in range(50):
        st = client.get(f"/api/media/jobs/{job_id}").json()
        if st["status"] == "completed":
            break
        time.sleep(0.02)
    assert st["status"] == "completed"
    assert st["download_url"] == f"/api/media/files/{job_id}"

    r_file = client.get(st["download_url"])
    assert r_file.status_code == 200
    assert r_file.content == b"FAKE_MP4_DATA"

    # Invalid job_id rejected
    assert client.get("/api/media/files/../../etc/passwd").status_code in (400, 404)
    assert client.get("/api/media/jobs/not-a-valid-id!").status_code == 400

    # Delete removes job and cleans up folder
    r_del = client.delete(f"/api/media/jobs/{job_id}")
    assert r_del.status_code == 200
    assert not (tmp_path / job_id).exists()


def test_ai_studio_srt_ass_and_voice_assignment(tmp_path):
    from app import media_ai_studio as studio

    raw = [
        studio.TranscriptSegmentSchema(
            start_sec=1.5,
            end_sec=4.2,
            speaker="S1",
            gender="male",
            tone="casual",
            original_text="Alright, so here we are in front of the elephants.",
            translated_text="Được rồi, chúng ta đang đứng trước đàn voi đây.",
        ),
        studio.TranscriptSegmentSchema(
            start_sec=4.0,  # slightly overlapping on purpose
            end_sec=7.0,
            speaker="S2",
            gender="female",
            tone="excited",
            original_text="The cool thing about these guys is that they have really long trunks.",
            translated_text="Điểm thú vị của mấy chú voi này là chiếc vòi siêu dài.",
        ),
    ]
    segs = studio._normalize_segments(raw, duration_sec=19.0)
    assert len(segs) == 2
    assert segs[0]["end_sec"] <= segs[1]["start_sec"]

    voices = studio._assign_speaker_voices(segs, "auto")
    assert voices["S1"] in studio._MALE_VOICES
    assert voices["S2"] in studio._FEMALE_VOICES

    srt_path = studio.write_srt_file(segs, tmp_path / "test.srt", bilingual=True)
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "00:00:01,500 -->" in srt_text
    assert "Được rồi, chúng ta đang đứng trước đàn voi đây." in srt_text

    ass_path = studio.write_ass_file(
        segs,
        tmp_path / "test.ass",
        width=1080,
        height=1920,
        style_preset="cinema",
        animation="reveal",
        bilingual=True,
    )
    ass_text = ass_path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "&H004CE6FF" in ass_text
    assert "\\1a&HFF&" in ass_text  # word-by-word reveal tags present

    karaoke_path = studio.write_ass_file(
        segs,
        tmp_path / "test_k.ass",
        width=1280,
        height=720,
        style_preset="classic",
        animation="karaoke",
    )
    assert "\\kf" in karaoke_path.read_text(encoding="utf-8")

    # Word grouping splits cleanly on pauses and preserves word timestamps
    words = [
        {"start_sec": 2.8, "end_sec": 3.2, "text": "Terra", "speaker": "spk_0"},
        {"start_sec": 3.2, "end_sec": 4.1, "text": "Nova.", "speaker": "spk_0"},
        {"start_sec": 11.6, "end_sec": 12.0, "text": "Terra", "speaker": "spk_0"},
        {"start_sec": 12.0, "end_sec": 12.9, "text": "Nova.", "speaker": "spk_0"},
    ]
    cues = studio.group_words_into_cues(words)
    assert len(cues) == 2
    assert cues[0]["start_sec"] == 2.8 and cues[0]["end_sec"] == 4.1
    assert cues[1]["start_sec"] == 11.6 and len(cues[1]["words"]) == 2


def test_ai_studio_dub_track_assembly(tmp_path):
    import wave
    from array import array
    from app import media_ai_studio as studio

    ffmpeg_bin = md.find_ffmpeg_binary()
    assert ffmpeg_bin, "Bundled imageio-ffmpeg should be available"

    clip1 = tmp_path / "c1.wav"
    with wave.open(str(clip1), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(studio.DUB_SAMPLE_RATE)
        # 0.5s tone burst
        samples = array("h", [4000 if (i % 40) < 20 else -4000 for i in range(studio.DUB_SAMPLE_RATE // 2)])
        wf.writeframes(samples.tobytes())

    segs = [
        {"start_sec": 0.5, "end_sec": 1.5, "speaker": "S1", "gender": "male", "tone": "clear", "original_text": "Hi", "translated_text": "Chào"},
    ]
    out_wav = studio.build_dub_track_wav(ffmpeg_bin, segs, [clip1], 3.0, tmp_path / "dub.wav")
    assert out_wav.is_file() and out_wav.stat().st_size > 44
    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == studio.DUB_SAMPLE_RATE
        assert wf.getnframes() >= int(3.0 * studio.DUB_SAMPLE_RATE)


def test_media_api_ai_validation_and_subtitle_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(md, "MEDIA_DIR", tmp_path)
    with md._JOBS_LOCK:
        md._JOBS.clear()

    client = TestClient(app, base_url="http://127.0.0.1:8000")

    caps = client.get("/api/media/capabilities")
    assert caps.status_code == 200
    assert "ai_voices" in caps.json()

    # Audio-only + ai_mode=dub must be rejected cleanly
    r_bad = client.post(
        "/api/media/download",
        json={
            "url": "https://www.youtube.com/watch?v=abc",
            "preset": "audio_only",
            "ai_mode": "dub",
        },
    )
    assert r_bad.status_code == 400

    # Subtitle download endpoint serves completed SRT cleanly
    job_id = "abcdef123456"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    srt_file = job_dir / "subtitles.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8")
    with md._JOBS_LOCK:
        md._JOBS[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "stage": "completed",
            "preset": "video_720",
            "ai_mode": "hardsub",
            "target_lang": "vi",
            "platform": "youtube",
            "platform_label": "YouTube",
            "title": "Demo",
            "percent": 100.0,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "speed": None,
            "eta": 0,
            "filename": "Demo [720p Sub-VI].mp4",
            "file_path": None,
            "file_size": None,
            "subtitle_path": str(srt_file),
            "subtitle_filename": "Demo [Sub-VI].srt",
            "job_dir": str(job_dir),
            "error": None,
            "created_at": 9999999999.0,
            "cancel_event": MagicMock(),
        }

    r_srt = client.get(f"/api/media/subtitles/{job_id}")
    assert r_srt.status_code == 200
    assert "Xin chào" in r_srt.text


def test_ai_provider_resolution_and_custom_openai_support(monkeypatch):
    from app import media_ai_studio as studio
    from app.ai_providers import AIProvider

    assert studio.normalize_model_id("models/gemini-4-neon", "gemini-4-hydrogen") == "gemini-4-neon"
    assert studio.normalize_model_id("openai/gpt-4o-audio-preview", "gpt-4o-mini-audio-preview") == "openai/gpt-4o-audio-preview"
    with pytest.raises(ValueError):
        studio.normalize_model_id("bad model; rm -rf /", "gemini-4-hydrogen")

    # Custom OpenAI-compatible provider configured in Settings resolves cleanly
    monkeypatch.setattr(
        studio,
        "get_provider_config",
        lambda pid: {
            "id": "my-relay",
            "label": "My Relay",
            "protocol": "openai",
            "api_base": "https://relay.example.com/v1",
        } if pid == "my-relay" else None,
    )
    monkeypatch.setattr(studio, "validate_url", lambda url: url)
    monkeypatch.setattr(studio, "get_provider_api_key", lambda pid, provider_label=None: "sk-custom-key-123" if pid == "my-relay" else None)

    prov, key = studio.resolve_configured_ai_provider("my-relay")
    assert isinstance(prov, AIProvider)
    assert prov.id == "my-relay"
    assert prov.protocol == "openai"
    assert key == "sk-custom-key-123"

    # Capabilities endpoint exposes providers from Settings
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    caps = client.get("/api/media/capabilities")
    assert caps.status_code == 200
    body = caps.json()
    assert isinstance(body["ai_providers"], list)
    assert any(p["id"] == "gemini" for p in body["ai_providers"])


def test_silence_split_and_semantic_word_grouping():
    from app import media_ai_studio as studio

    # 1. Silence-aware splitting snaps to natural pauses near ~40s instead of hard 40.0s cuts
    silences = [(12.0, 12.6), (36.8, 37.8), (78.2, 79.4), (118.0, 118.9)]
    ranges = studio.choose_silence_split_points(126.0, silences)
    assert len(ranges) == 3
    assert ranges[0] == (0.0, 37.3)
    assert ranges[1] == (37.3, 78.8)
    assert ranges[2] == (78.8, 126.0)

    # 2. Semantic lookahead keeps a 9-word sentence together instead of stranding the last word ("mankind.")
    sentence = ["It's", "a", "brand", "new", "world", "of", "possibilities", "for", "mankind."]
    words = [
        {"start_sec": round(1.0 + i * 0.25, 2), "end_sec": round(1.22 + i * 0.25, 2), "text": tok, "speaker": "S1"}
        for i, tok in enumerate(sentence)
    ]
    cues = studio.group_words_into_cues(words, max_words_per_cue=8, max_cue_duration=3.6)
    assert len(cues) == 1
    assert cues[0]["original_text"].endswith("mankind.")

    # 3. Long continuous clause splits cleanly before conjunction ("and") without stranding prepositions
    long_tokens = [
        "We", "walked", "through", "the", "quiet", "forest",
        "and", "we", "found", "a", "hidden", "golden", "temple", "today.",
    ]
    long_words = [
        {"start_sec": round(i * 0.28, 2), "end_sec": round(i * 0.28 + 0.24, 2), "text": tok, "speaker": "S1"}
        for i, tok in enumerate(long_tokens)
    ]
    split_cues = studio.group_words_into_cues(long_words, max_words_per_cue=8, max_cue_duration=3.6)
    assert len(split_cues) == 2
    assert split_cues[0]["original_text"].endswith("forest")
    assert split_cues[1]["original_text"].startswith("and")


def test_story_bible_sliding_window_and_missing_id_retry(monkeypatch):
    from app import media_ai_studio as studio
    from app.ai_providers import AIProvider

    monkeypatch.setattr(studio, "TRANSLATION_BATCH_SIZE", 2)
    monkeypatch.setattr(studio, "SLIDING_CONTEXT_WINDOW", 2)

    calls: list[tuple[str, str]] = []
    batch_call_num = 0

    def fake_llm(*, provider, api_key, model_id, prompt, schema_cls):
        nonlocal batch_call_num
        calls.append((schema_cls.__name__, prompt))
        if schema_cls is studio.StoryBibleSchema:
            return studio.StoryBibleSchema(
                source_language="English",
                genre="movie",
                synopsis="Two friends journey across a desert.",
                glossary=["Arrakis -> Arrakis"],
                pronoun_rules=["S1 -> S2: anh / em"],
                style_notes="Cinematic natural Vietnamese.",
            )
        batch_call_num += 1
        if batch_call_num == 1:
            # Deliberately omit id=1 to exercise missing-ID retry!
            return studio.TranslatedCueBatchSchema(
                source_language="English",
                items=[
                    studio.TranslatedCueItemSchema(
                        id=0,
                        corrected_original="Welcome to Arrakis.",
                        draft_translation="Chào mừng tới Arrakis.",
                        translated_text="Chào mừng em tới Arrakis.",
                        gender="male",
                        tone="warm",
                    )
                ],
            )
        if batch_call_num == 2:
            return studio.TranslatedCueBatchSchema(
                source_language="English",
                items=[
                    studio.TranslatedCueItemSchema(
                        id=1,
                        corrected_original="Stay close to me.",
                        draft_translation="Hãy ở gần tôi.",
                        translated_text="Em cứ đi sát anh nhé.",
                        gender="male",
                        tone="protective",
                    )
                ],
            )
        return studio.TranslatedCueBatchSchema(
            source_language="English",
            items=[
                studio.TranslatedCueItemSchema(
                    id=2,
                    corrected_original="I trust you.",
                    draft_translation="Tôi tin bạn.",
                    translated_text="Em tin anh.",
                    gender="female",
                    tone="sincere",
                )
            ],
        )

    monkeypatch.setattr(studio, "_call_structured_llm", fake_llm)
    prov = studio.PROVIDERS["gemini"]
    cues = [
        {"id": 0, "start_sec": 0.0, "end_sec": 2.0, "speaker": "S1", "gender": "male", "tone": "clear", "original_text": "Welcome to Arakis.", "translated_text": "Welcome to Arakis.", "words": []},
        {"id": 1, "start_sec": 2.2, "end_sec": 4.0, "speaker": "S1", "gender": "male", "tone": "clear", "original_text": "Stay close to me.", "translated_text": "Stay close to me.", "words": []},
        {"id": 2, "start_sec": 4.2, "end_sec": 6.0, "speaker": "S2", "gender": "female", "tone": "clear", "original_text": "I trust you.", "translated_text": "I trust you.", "words": []},
    ]
    lang, out = studio._translate_cues_with_model(
        cues,
        provider=prov,
        api_key="fake-key",
        model_id="gemini-3.5-flash-lite",
        target_lang="vi",
        for_dubbing=False,
        context_hint="S1 và S2 là người yêu",
    )
    assert lang == "English"
    assert [c["translated_text"] for c in out] == [
        "Chào mừng em tới Arrakis.",
        "Em cứ đi sát anh nhé.",
        "Em tin anh.",
    ]
    # Verify Pass 0 Story Bible ran, user context_hint injected, and batch 2 received sliding context!
    assert calls[0][0] == "StoryBibleSchema"
    assert "S1 và S2 là người yêu" in calls[0][1]
    assert "Pronoun & Register Rules: S1 -> S2: anh / em" in calls[1][1]
    assert "Chào mừng em tới Arrakis." in calls[3][1]


def test_platform_subtitle_parser_and_edge_tts_mapping(tmp_path):
    from app import media_ai_studio as studio

    # 1. YouTube json3 with word-level tOffsetMs timings
    j3 = tmp_path / "cap.en.json3"
    j3.write_text(
        '{"events":[{"tStartMs":1000,"dDurationMs":2000,"segs":[{"utf8":"Hello","tOffsetMs":0},{"utf8":" world.","tOffsetMs":500}]},'
        '{"tStartMs":4000,"dDurationMs":2000,"segs":[{"utf8":"Good","tOffsetMs":0},{"utf8":" day.","tOffsetMs":400}]}]}',
        encoding="utf-8",
    )
    cues_j3 = studio.parse_platform_subtitle_file(j3)
    assert len(cues_j3) == 2
    assert cues_j3[0]["original_text"] == "Hello world."
    assert len(cues_j3[0]["words"]) == 2
    assert cues_j3[1]["start_sec"] == 4.0

    # 2. WebVTT parser cleans tags and ignores [Music] noise
    vtt = tmp_path / "cap.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n[Music]\n\n"
        "00:00:03.500 --> 00:00:06.000\n<i>First</i> line\n\n"
        "00:00:06.200 --> 00:00:08.500\nSecond line\n",
        encoding="utf-8",
    )
    cues_vtt = studio.parse_platform_subtitle_file(vtt)
    assert [c["original_text"] for c in cues_vtt] == ["First line", "Second line"]

    # 3. Free Edge-TTS & Vietnamese Piper ONNX multi-speaker mapping
    assert studio._resolve_edge_tts_voice("vi", "Kore") == "vi-VN-HoaiMyNeural"
    assert studio._resolve_edge_tts_voice("vi", "Puck") == "vi-VN-NamMinhNeural"
    assert studio._resolve_edge_tts_voice("ja", "Charon") == "ja-JP-KeitaNeural"
    assert studio._resolve_piper_voice_spec("vi", "Puck")[1:] == ("vi_VN-csa-voice-piper-v3-medium", 2)
    assert studio._resolve_piper_voice_spec("vi", "Kore")[1:] == ("vi_VN-csa-voice-piper-v3-medium", 0)
    assert studio._resolve_piper_voice_spec("vi", "Sulafat")[1:] == ("vi_VN-csa-voice-piper-v3-medium", 4)
    assert studio._resolve_piper_voice_spec("vi", "Charon")[1:] == ("vi_VN-tramvang-medium", None)


def test_sub_10ms_dsp_vocal_onset_alignment_and_dub_locking(tmp_path):
    import wave
    import numpy as np
    from app import media_ai_studio as studio

    sr = 16000
    dur = 3.0
    t = np.arange(int(sr * dur), dtype=np.float32) / sr

    # Synthetic audio: low-freq 80Hz EDM bass throughout + vocal formant chord (500Hz + 1200Hz + 2400Hz)
    # starting at EXACTLY t_true = 1.372s!
    t_true = 1.372
    audio = 0.08 * np.sin(2 * np.pi * 80.0 * t)
    vocal_mask = (t >= t_true) & (t <= 2.150)
    vocal = (
        0.55 * np.sin(2 * np.pi * 520.0 * t)
        + 0.35 * np.sin(2 * np.pi * 1250.0 * t)
        + 0.25 * np.sin(2 * np.pi * 2400.0 * t)
    )
    audio = (audio + vocal * vocal_mask).astype(np.float32)

    # Whisper estimate intentionally off by -85ms (1.287s instead of 1.372s)!
    noisy_words = [{"start_sec": 1.287, "end_sec": 2.220, "text": "Terranova", "speaker": "S1"}]
    snapped = studio.snap_words_to_vocal_onsets(noisy_words, audio, sample_rate=sr)
    err_ms = abs(snapped[0]["start_sec"] - t_true) * 1000.0
    assert err_ms < 10.0, f"Expected <10ms DSP vocal onset alignment, got {err_ms:.2f}ms (snapped={snapped[0]['start_sec']})"

    # Also verify a loud pre-vocal synth hit at -95ms does NOT drag the vocal early!
    synth_hit = ((t >= t_true - 0.095) & (t <= t_true - 0.055)).astype(np.float32) * 0.65 * np.sin(2 * np.pi * 1500.0 * t)
    snapped_with_bgm = studio.snap_words_to_vocal_onsets(
        [{"start_sec": t_true - 0.010, "end_sec": 2.220, "text": "Terranova", "speaker": "S1"}],
        (audio + synth_hit).astype(np.float32),
        side_f32=synth_hit.astype(np.float32),
        sample_rate=sr,
    )
    assert snapped_with_bgm[0]["start_sec"] >= t_true - 0.012, "Must not jump early onto stereo BGM beat!"

    # Now verify build_dub_track_wav locks a TTS clip (with 180ms leading silence!) cleanly at P-center (never early!)
    ffmpeg_bin = md.find_ffmpeg_binary()
    assert ffmpeg_bin
    sr_dub = studio.DUB_SAMPLE_RATE
    lead_sil = np.zeros(int(sr_dub * 0.18), dtype=np.int16)
    t_clip = np.arange(int(sr_dub * 0.45), dtype=np.float32) / sr_dub
    speech_burst = (12000.0 * np.sin(2 * np.pi * 440.0 * t_clip)).astype(np.int16)
    tts_wav = tmp_path / "tts.wav"
    with wave.open(str(tts_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr_dub)
        wf.writeframes(np.concatenate([lead_sil, speech_burst]).tobytes())

    segs = [
        {
            "start_sec": snapped[0]["start_sec"],
            "end_sec": snapped[0]["end_sec"],
            "speaker": "S1",
            "gender": "female",
            "tone": "clear",
            "original_text": "Terranova",
            "translated_text": "Terranova",
        }
    ]
    dub_wav = studio.build_dub_track_wav(ffmpeg_bin, segs, [tts_wav], dur, tmp_path / "dub.wav")
    with wave.open(str(dub_wav), "rb") as wf:
        dub_pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    active = np.where(np.abs(dub_pcm) > 1200)[0]
    assert len(active) > 0
    dub_onset_sec = float(active[0]) / float(sr_dub)
    # P-center alignment: never earlier than vocal onset (-5ms) and within <=50ms perceptual lock
    signed_ms = (dub_onset_sec - t_true) * 1000.0
    assert -5.0 <= signed_ms <= 50.0, f"Expected non-early P-center dub lock, got {signed_ms:+.2f}ms"





