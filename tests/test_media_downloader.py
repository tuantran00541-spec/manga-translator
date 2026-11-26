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


def test_safety_rejects_phishing_punycode_ip_and_unsupported_domains():
    for bad in (
        "https://youtube.com.evil.example/watch?v=abc",
        "https://xn--youtbe-9ta.com/watch?v=abc",
        "https://142.250.190.46/watch?v=abc",
        "https://vimeo.com/123456",
        "https://www.youtube.com/",
    ):
        res = md.inspect_link_safety(bad, resolve_redirects=False)
        assert res["safe"] is False
        domain_chk = next(c for c in res["checks"] if c["id"] == "official_domain")
        assert domain_chk["passed"] is False


def test_safety_accepts_official_platforms_and_short_links(monkeypatch):
    _patch_public_dns(monkeypatch)
    cases = [
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "youtube"),
        ("https://youtu.be/jNQXAC9IVRw", "youtube"),
        ("https://www.tiktok.com/@creator/video/7123456789012345678", "tiktok"),
        ("https://vm.tiktok.com/ZM8abcdef/", "tiktok"),
        ("https://www.facebook.com/reel/1234567890", "facebook"),
        ("https://fb.watch/abc123XYZ/", "facebook"),
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


def test_safety_blocks_redirect_to_untrusted_domain(monkeypatch):
    _patch_public_dns(monkeypatch)

    fake_resp = MagicMock()
    fake_resp.status_code = 302
    fake_resp.headers = {"location": "https://evil-phish.example/steal"}
    monkeypatch.setattr(md, "public_request", lambda *args, **kwargs: fake_resp)

    res = md.inspect_link_safety("https://vm.tiktok.com/ZM8abcdef/", resolve_redirects=True)
    assert res["safe"] is False
    redir_chk = next(c for c in res["checks"] if c["id"] == "redirect_chain")
    assert redir_chk["passed"] is False
    assert "evil-phish.example" in redir_chk["detail"]


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
    def fake_worker(job_id, target_url, platform, preset, is_vertical):
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
