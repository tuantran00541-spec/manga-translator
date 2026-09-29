import json
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import install  # noqa: E402
import run  # noqa: E402


def test_the_command_folder_goes_in_front_of_path_once():
    sep = os.pathsep
    assert install.with_dir(f"/a{sep}/b", "/c") == f"/c{sep}/a{sep}/b"
    assert install.with_dir(f"/a{sep}/c/", "/c") == f"/a{sep}/c/", "already there, even with a trailing slash"
    assert install.with_dir("", "/c") == "/c"


def test_the_unix_launcher_runs_the_private_python_with_the_open_flag():
    text = install.launcher_text(Path("/opt/mt/.venv/bin/python"), Path("/opt/mt/run.py"), windows=False)
    assert text == '#!/bin/sh\nexec "/opt/mt/.venv/bin/python" "/opt/mt/run.py" --open "$@"\n'


def test_the_windows_launcher_never_writes_an_accented_user_name():
    local = "C:\\Users\\Nguyễn\\AppData\\Local"
    text = install.launcher_text(Path(local + r"\manga-translator\app\.venv\Scripts\python.exe"),
                                 Path(local + r"\manga-translator\app\run.py"), windows=True, local_appdata=local)
    assert text.isascii() and "%LOCALAPPDATA%\\manga-translator\\app\\run.py" in text and text.endswith("--open %*\r\n")


def test_an_accented_clone_path_switches_cmd_to_utf8_and_back():
    text = install.launcher_text(Path(r"D:\Truyện\mt\.venv\Scripts\python.exe"), Path(r"D:\Truyện\mt\run.py"),
                                 windows=True, local_appdata=r"C:\Users\a\AppData\Local")
    lines = text.split("\r\n")
    assert lines[2] == "chcp 65001 >nul" and "Truyện" in lines[3] and lines[4] == "chcp %_mt_cp% >nul"


def test_an_update_replaces_the_code_but_keeps_models_chapters_and_the_environment(tmp_path):
    source, target = tmp_path / "new", tmp_path / "app"
    (source / "app").mkdir(parents=True)
    (source / "app" / "main.py").write_text("new")
    (source / "models").mkdir()
    (source / "models" / "kiuyha_text_1280.onnx").write_text("shipped v2")
    (source / "run.py").write_text("run")
    for folder in ("app", "models", "data", ".venv"):
        (target / folder).mkdir(parents=True)
    (target / install.MARKER).write_text("{}")
    (target / "app" / "old_module.py").write_text("stale")
    (target / "gone.py").write_text("stale")
    (target / "models" / "lama-manga-dynamic.onnx").write_text("user download")
    (target / "models" / "kiuyha_text_1280.onnx").write_text("shipped v1")
    (target / "data" / "chapter.json").write_text("mine")
    (target / ".venv" / "pyvenv.cfg").write_text("env")

    install.sync_code(source, target)

    assert (target / "app" / "main.py").read_text() == "new" and not (target / "app" / "old_module.py").exists()
    assert not (target / "gone.py").exists() and (target / "run.py").is_file()
    assert (target / "models" / "lama-manga-dynamic.onnx").read_text() == "user download"
    assert (target / "models" / "kiuyha_text_1280.onnx").read_text() == "shipped v2"
    assert (target / "data" / "chapter.json").read_text() == "mine" and (target / ".venv" / "pyvenv.cfg").is_file()


def test_the_installer_refuses_to_take_over_a_folder_it_did_not_make(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "thesis.docx").write_text("precious")
    with pytest.raises(SystemExit):
        install.sync_code(tmp_path, tmp_path / "app")
    assert (tmp_path / "app" / "thesis.docx").read_text() == "precious"


class _Response:
    def __init__(self, data: bytes, status: int = 200):
        self.data, self.status = data, status

    def read(self, size=-1):
        chunk, self.data = self.data, b""
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_an_existing_model_is_kept_and_a_bad_download_is_refused(tmp_path, monkeypatch):
    kept = tmp_path / "mine.onnx"
    kept.write_bytes(b"my own model")
    install.download("https://unused.invalid/x", kept, "0" * 64)
    assert kept.read_bytes() == b"my own model"
    monkeypatch.setattr(install.urllib.request, "urlopen", lambda *_a, **_k: _Response(b"tampered"))
    with pytest.raises(SystemExit):
        install.download("https://example.invalid/model.onnx", tmp_path / "new.onnx", "0" * 64)
    assert not (tmp_path / "new.onnx").exists() and not (tmp_path / "new.onnx.part").exists()


def test_a_broken_download_resumes_where_it_stopped(tmp_path, monkeypatch):
    whole = b"0123456789"
    (tmp_path / "m.onnx.part").write_bytes(whole[:4])
    asked = []

    def urlopen(request, timeout):
        asked.append(request.get_header("Range"))
        if len(asked) == 1:
            raise OSError("connection reset")
        return _Response(whole[4:], status=206)

    monkeypatch.setattr(install.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(install.time, "sleep", lambda _s: None)
    install.download("https://example.invalid/m.onnx", tmp_path / "m.onnx", install.hashlib.sha256(whole).hexdigest())
    assert (tmp_path / "m.onnx").read_bytes() == whole and asked == ["bytes=4-", "bytes=4-"]


def test_missing_visual_cpp_runtime_files_are_found(tmp_path):
    (tmp_path / "msvcp140.dll").write_bytes(b"")
    assert install.missing_vc_runtime(tmp_path) == ["vcruntime140_1.dll", "vcomp140.dll"]


def test_a_new_mac_account_without_zshrc_gets_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    (tmp_path / ".profile").write_text("umask 022")
    folder = tmp_path / ".local" / "bin"
    assert install.add_to_shell_path(folder, tmp_path) == [tmp_path / ".zshrc", tmp_path / ".profile"]
    assert install.add_to_shell_path(folder, tmp_path) == [], "added once"
    assert (tmp_path / ".zshrc").read_text() == f'export PATH="{folder}:$PATH"  # manga-translator\n'
    assert (tmp_path / ".profile").read_text().splitlines() == ["umask 022", f'export PATH="{folder}:$PATH"  # manga-translator']


def test_fish_gets_its_own_syntax(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    folder = tmp_path / ".local" / "bin"
    install.add_to_shell_path(folder, tmp_path)
    assert (tmp_path / ".config" / "fish" / "config.fish").read_text() == f"fish_add_path {folder}  # manga-translator\n"


def test_a_command_is_written_to_the_user_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "WINDOWS", False)
    monkeypatch.setattr(install.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / ".local" / "bin"))
    shim = install.install_command(Path("/x/python"), Path("/x/app"), tmp_path / "home")
    assert shim == tmp_path / ".local" / "bin" / "manga" and os.access(shim, os.X_OK)


def test_manga_opens_the_running_app_instead_of_starting_a_second_one(monkeypatch):
    opened = []
    monkeypatch.setattr(run, "_is_ours", lambda url: url.endswith(f":{run.PORT}"))
    monkeypatch.setattr(run.webbrowser, "open", opened.append)
    monkeypatch.setattr(run, "_serve", lambda *a, **k: pytest.fail("a second server was started"))
    run.main(["--open"])
    assert opened == [run._url(run.PORT)]


def test_another_program_on_the_port_moves_the_app_to_the_next_free_one(monkeypatch):
    monkeypatch.setattr(run, "_is_ours", lambda url: False)
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        taken = squatter.getsockname()[1]
        monkeypatch.setattr(run, "PORT", taken)
        assert run._pick_port() not in (None, taken)


def test_update_waits_until_the_app_is_closed(monkeypatch):
    monkeypatch.setattr(run, "_running_port", lambda: 8000)
    with pytest.raises(SystemExit, match="đang chạy"):
        run._update()


def test_update_of_a_downloaded_install_reruns_the_new_installer_on_this_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_running_port", lambda: None)
    monkeypatch.setattr(run, "BASE_DIR", tmp_path)
    monkeypatch.setattr(run, "MARKER", tmp_path / ".manga-install.json")
    (tmp_path / ".manga-install.json").write_text(json.dumps({"ref": "main", "home": str(tmp_path / "h")}))

    def fetch(url, path):
        assert url.endswith("/zip/refs/heads/main")
        import zipfile
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manga-translator-main/scripts/install.py", "")

    ran = []
    monkeypatch.setattr(run.urllib.request, "urlretrieve", fetch)
    monkeypatch.setattr(run.shutil, "which", lambda _name: None)
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kw: ran.append((command, kw["env"])))
    run._update()
    command, env = ran[0]
    assert command[-2:] == ["--target", str(tmp_path)] and command[1].endswith(os.path.join("scripts", "install.py"))
    assert env["MANGA_HOME"] == str(tmp_path / "h")
