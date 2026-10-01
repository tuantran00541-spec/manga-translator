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


def test_missing_or_old_visual_cpp_runtime_is_found(tmp_path, monkeypatch):
    from app import vc_runtime

    monkeypatch.delenv("MANGA_VC_ASSUME_MISSING", raising=False)
    for name in ("msvcp140.dll", "vcomp140.dll"):
        (tmp_path / name).write_bytes(b"")
    versions = {"msvcp140.dll": (14, 29, 30133, 0), "vcomp140.dll": None}
    found = vc_runtime.problems(tmp_path, version_of=lambda path: versions[path.name])
    assert found == ["msvcp140.dll (bản 14.29.30133.0 quá cũ)", "vcruntime140_1.dll (thiếu)"]
    (tmp_path / "vcruntime140_1.dll").write_bytes(b"")
    assert vc_runtime.problems(tmp_path, version_of=lambda path: (14, 44, 35211, 0)) == []


class _Ran:
    def __init__(self, code):
        self.returncode = code


@pytest.mark.parametrize("code, outcome", [
    (0, ""), (1638, ""), (3010, "khởi động lại"), (1602, "từ chối quyền admin"), (1603, "mã 1603"),
])
def test_every_runtime_installer_outcome_gets_a_clear_answer(code, outcome, monkeypatch):
    from app import vc_runtime

    monkeypatch.setattr(vc_runtime.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"x"))
    monkeypatch.setattr(vc_runtime.subprocess, "run", lambda args: _Ran(code))
    monkeypatch.setattr(vc_runtime, "signed_by_microsoft", lambda path: True)
    monkeypatch.setattr(vc_runtime, "problems", lambda: [])
    if code in vc_runtime.OK_CODES:
        assert outcome in vc_runtime.install()
    else:
        with pytest.raises(RuntimeError, match=outcome):
            vc_runtime.install()


def test_a_runtime_that_is_still_missing_after_installing_is_reported(monkeypatch):
    from app import vc_runtime

    monkeypatch.setattr(vc_runtime.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"x"))
    monkeypatch.setattr(vc_runtime.subprocess, "run", lambda args: _Ran(0))
    monkeypatch.setattr(vc_runtime, "signed_by_microsoft", lambda path: True)
    monkeypatch.setattr(vc_runtime, "problems", lambda: ["msvcp140.dll (thiếu)"])
    with pytest.raises(RuntimeError, match="Khởi động lại máy"):
        vc_runtime.install()


def test_no_network_for_the_runtime_points_to_the_manual_download(monkeypatch):
    from app import vc_runtime

    def offline(url, path):
        raise OSError("no route")

    monkeypatch.setattr(vc_runtime.urllib.request, "urlretrieve", offline)
    with pytest.raises(RuntimeError, match="aka.ms/vs/17/release/vc_redist.x64.exe"):
        vc_runtime.install()


def test_the_app_repairs_the_runtime_before_loading_torch(monkeypatch, capsys):
    from app import vc_runtime

    calls = []
    monkeypatch.setattr(run.os, "name", "nt")
    monkeypatch.setattr(vc_runtime, "problems", lambda: ["vcomp140.dll (thiếu)"])
    monkeypatch.setattr(vc_runtime, "install", lambda: calls.append("install") or "")
    run._check_vc_runtime()
    assert calls == ["install"] and "vcomp140.dll" in capsys.readouterr().out


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


def test_messages_survive_a_windows_cp1252_pipe(monkeypatch):
    import io

    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", pipe)
    run._safe_console()
    print("Đang tải bản mới")
    pipe.flush()
    assert raw.getvalue() == b"?ang t?i b?n m?i\n"


def test_an_unsigned_runtime_installer_is_never_run(monkeypatch):
    from app import vc_runtime

    ran = []
    monkeypatch.setattr(vc_runtime.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"x"))
    monkeypatch.setattr(vc_runtime, "signed_by_microsoft", lambda path: False)
    monkeypatch.setattr(vc_runtime.subprocess, "run", lambda args: ran.append(args) or _Ran(0))
    with pytest.raises(RuntimeError, match="chữ ký"):
        vc_runtime.install()
    assert ran == []


@pytest.mark.parametrize("output, code, ok", [
    ("Valid|CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US", 0, True),
    ("Valid|CN=Evil Corp, O=Evil Corp", 0, False),
    ("HashMismatch|CN=Microsoft Corporation, O=Microsoft Corporation", 0, False),
    ("NotSigned|", 0, False),
    ("", 1, False),
])
def test_only_a_valid_microsoft_signature_passes(output, code, ok, monkeypatch):
    from app import vc_runtime

    class Result:
        stdout, returncode = output, code

    seen = {}
    monkeypatch.setattr(vc_runtime.subprocess, "run", lambda args, **kw: seen.update(kw) or Result())
    assert vc_runtime.signed_by_microsoft(Path("C:/t/vc.exe")) is ok
    assert seen["env"]["MANGA_VC_FILE"] == str(Path("C:/t/vc.exe")), "the path is passed as data, not as script"


def test_downloads_refuse_anything_but_https(tmp_path):
    with pytest.raises(ValueError):
        install.download("http://example.invalid/m.onnx", tmp_path / "m.onnx", None)


def test_the_desktop_icon_opens_the_app_window_and_quotes_odd_paths():
    from pathlib import PureWindowsPath

    script = install.shortcut_script(PureWindowsPath(r"C:\Users\Tuấn\.venv\Scripts\pythonw.exe"),
                                     PureWindowsPath(r"C:\Users\O'Neil\app\run.py"), PureWindowsPath(r"C:\a\favicon.ico"))
    assert "'C:\\Users\\Tuấn\\.venv\\Scripts\\pythonw.exe'" in script
    assert "$link.Arguments = '\"C:\\Users\\O''Neil\\app\\run.py\" --window'" in script
    assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script


def test_the_app_window_shows_a_loading_page_then_the_app(monkeypatch):
    import types

    calls = []

    class Window:
        def load_url(self, url):
            calls.append(("load", url))

    fake = types.SimpleNamespace(
        settings={},
        create_window=lambda title, url=None, html=None, **kw: calls.append(("create", url, bool(html))) or Window(),
        start=lambda func=None: func and func(),
    )
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert run._show_window("http://127.0.0.1:8000", on_ready=lambda: calls.append(("ready",)))
    assert calls == [("create", None, True), ("ready",), ("load", "http://127.0.0.1:8000")]
    assert fake.settings == {"ALLOW_DOWNLOADS": True, "OPEN_EXTERNAL_LINKS_IN_BROWSER": True}


def test_without_a_window_toolkit_the_app_falls_back_to_the_browser(monkeypatch):
    import types

    def no_gui(func=None):
        raise RuntimeError("no GTK or Qt")

    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(
        settings={}, create_window=lambda *a, **kw: object(), start=no_gui))
    assert run._show_window("http://127.0.0.1:8000") is False
    monkeypatch.setitem(sys.modules, "webview", None)  # not installed at all
    assert run._show_window("http://127.0.0.1:8000") is False
