import hashlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import extension

# Extension code per manifest version; changing the code means bumping the version so installed copies are told to reload.
CODE_SHA256 = {"1.1": "6f70f7a92b8ed236c150ba0c3baeaa3bfccd0b8d8dad25de3e337939b4d73ab1"}


def client(monkeypatch):
    monkeypatch.setitem(extension._seen, "version", None)
    app = FastAPI()
    app.include_router(extension.router)
    return TestClient(app)


def test_the_app_tells_an_installed_extension_when_it_holds_a_newer_version(monkeypatch):
    http = client(monkeypatch)
    status = http.get("/api/extension").json()
    assert status["path"].endswith("extension") and not status["seen"] and not status["outdated"]
    assert http.post("/api/extension/hello", json={"version": extension.version()}).json()["outdated"] is False
    assert http.post("/api/extension/hello", json={"version": "1.0"}).json()["outdated"] is True
    assert http.get("/api/extension").json()["outdated"] is True


def test_changing_the_extension_code_needs_a_new_manifest_version():
    code = b"".join((extension.EXTENSION_DIR / name).read_bytes().replace(b"\r\n", b"\n") for name in ("background.js", "popup.html", "popup.js"))
    assert CODE_SHA256.get(extension.version()) == hashlib.sha256(code).hexdigest(), \
        "extension code changed: bump the version in extension/manifest.json and pin the new hash here"


def test_only_the_app_page_can_open_the_extension_folder(monkeypatch):
    opened = []
    monkeypatch.setattr(extension.subprocess, "Popen", lambda args, **_: opened.append(args))
    monkeypatch.setattr(extension.sys, "platform", "linux")
    http = client(monkeypatch)
    assert http.post("/api/extension/open", headers={"Origin": "https://evil.example"}).status_code == 403
    assert opened == []
    assert http.post("/api/extension/open", headers={"Origin": "http://testserver"}).status_code == 200
    assert opened == [["xdg-open", str(extension.EXTENSION_DIR)]]
