import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import install  # noqa: E402
import run  # noqa: E402


def test_the_command_folder_is_added_to_path_once():
    sep = os.pathsep
    assert install.with_dir(f"/a{sep}/b", "/c") == f"/a{sep}/b{sep}/c"
    assert install.with_dir(f"/a{sep}/c/", "/c") == f"/a{sep}/c/", "already there, even with a trailing slash"
    assert install.with_dir("", "/c") == "/c"


def test_the_launcher_calls_the_private_python_with_the_open_flag():
    python = Path("/opt/mt/.venv/bin/python")
    unix = install.launcher_text(python, windows=False)
    assert unix.startswith("#!/bin/sh\n") and f'exec "{python}" "{install.ROOT / "run.py"}" --open "$@"' in unix
    win = install.launcher_text(Path(r"C:\mt\.venv\Scripts\python.exe"), windows=True)
    assert win.startswith("@echo off\r\n") and win.rstrip().endswith("--open %*")


def test_the_command_is_written_to_the_user_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "WINDOWS", False)
    monkeypatch.setattr(install.Path, "home", lambda: tmp_path)
    shim = install.install_command(Path("/x/python"))
    assert shim == tmp_path / ".local" / "bin" / "manga" and os.access(shim, os.X_OK)


def test_an_existing_model_is_kept_and_a_bad_download_is_refused(tmp_path, monkeypatch):
    kept = tmp_path / "mine.onnx"
    kept.write_bytes(b"my own model")
    install.download("https://unused.invalid/x", kept, "0" * 64)
    assert kept.read_bytes() == b"my own model"

    class Response:
        def __init__(self):
            self.chunks = [b"tampered", b""]

        def read(self, _size=-1):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(install.urllib.request, "urlopen", lambda *_a, **_k: Response())
    with pytest.raises(SystemExit):
        install.download("https://example.invalid/model.onnx", tmp_path / "new.onnx", "0" * 64)
    assert not (tmp_path / "new.onnx").exists() and not (tmp_path / "new.onnx.part").exists()


def test_manga_opens_the_running_app_instead_of_starting_a_second_one(monkeypatch):
    opened = []
    monkeypatch.setattr(run, "_is_up", lambda url: True)
    monkeypatch.setattr(run.webbrowser, "open", opened.append)
    monkeypatch.setattr(run.uvicorn, "run", lambda *a, **k: pytest.fail("a second server was started"))
    run.main(["--open"])
    assert opened == [run._local_url()]
