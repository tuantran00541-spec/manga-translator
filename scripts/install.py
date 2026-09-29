"""Install or update: sync the code, a Python 3.12 environment, the dependencies, the models and a `manga` command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
TORCH = "torch==2.5.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
LAMA_URL = "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"
LAMA_SHA256 = "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9"
CTD_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
CTD_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
# Kept across updates: the environment, models, chapters and logs of the user.
KEEP = frozenset({".venv", "models", "data", "logs", ".cache"})
MARKER = ".manga-install.json"
FREE_GB_NEEDED = 8


def say(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def default_home() -> Path:
    """One folder for uv, Python and the command; a path with accents breaks PaddleOCR, so it falls back."""
    if os.environ.get("MANGA_HOME"):
        return Path(os.environ["MANGA_HOME"])
    if WINDOWS:
        home = Path(os.environ["LOCALAPPDATA"]) / "manga-translator"
        return home if str(home).isascii() else Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "manga-translator"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "manga-translator"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def run(*args, cwd: Path | None = None) -> None:
    subprocess.run([str(a) for a in args], check=True, cwd=cwd)


def find_uv() -> str | None:
    uv = os.environ.get("MANGA_UV")
    return uv if uv and Path(uv).is_file() else shutil.which("uv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str | None, attempts: int = 4) -> None:
    """Fetch ``url`` to ``path`` unless a file is there, resuming broken downloads and refusing a wrong hash."""
    if path.is_file():
        return
    if not url.startswith("https://"):
        raise ValueError(f"only https downloads are allowed: {url}")
    part = path.with_name(path.name + ".part")
    print(f"    {url}", flush=True)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if have and response.status == 206 else "wb"
                with part.open(mode) as out:
                    shutil.copyfileobj(response, out, 1 << 20)
            break
        except OSError as exc:
            if attempt == attempts:
                raise SystemExit(f"Không tải được {path.name}: {exc}. Kiểm tra mạng rồi chạy lại bộ cài.") from exc
            print(f"    mạng lỗi ({exc}), thử lại lần {attempt + 1}…", flush=True)
            time.sleep(3 * attempt)
    if expected and sha256(part) != expected:
        part.unlink()
        raise SystemExit(f"Tải {path.name} bị lỗi (sai mã kiểm tra), hãy chạy lại bộ cài.")
    part.replace(path)


def sync_code(source: Path, target: Path) -> None:
    """Make ``target`` hold the code of ``source`` while keeping the user's environment, models and chapters."""
    if target.exists() and any(target.iterdir()) and not (target / MARKER).is_file():
        raise SystemExit(f"{target} đã có dữ liệu không phải của bộ cài; chọn thư mục khác bằng MANGA_HOME.")
    target.mkdir(parents=True, exist_ok=True)
    for entry in target.iterdir():
        if entry.name in KEEP or entry.name == MARKER:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for entry in source.iterdir():
        if entry.name == MARKER:
            continue
        destination = target / entry.name
        if entry.is_dir():
            # Folders the user keeps get the shipped files merged in; the rest are fresh copies.
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)


def write_marker(target: Path, home: Path, uv: str | None, done: bool) -> None:
    """Mark ``target`` as installed by this tool, so a rerun or update may replace its code."""
    (target / MARKER).write_text(json.dumps({
        "mode": "clone" if (target / ".git").exists() else "download",
        "ref": os.environ.get("MANGA_REF", "main"),
        "uv": uv,
        "home": str(home),
        "complete": done,
        "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


def check_disk(target: Path) -> None:
    probe = target if target.exists() else target.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < FREE_GB_NEEDED:
        raise SystemExit(f"Ổ đĩa còn {free_gb:.1f} GB, cần khoảng {FREE_GB_NEEDED} GB trống để cài.")


def ensure_vc_runtime() -> None:
    """Install or update the Microsoft Visual C++ runtime that torch, onnxruntime and paddle need."""
    if not WINDOWS:
        return
    sys.path.insert(0, str(ROOT))
    from app import vc_runtime

    found = vc_runtime.problems()
    if not found:
        return
    say(f"Cài Microsoft Visual C++ Runtime ({', '.join(found)}); Windows sẽ hỏi quyền admin")
    try:
        note = vc_runtime.install()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if note:
        print(f"    {note}")


def python_version(python: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(python), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, check=True).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def ensure_venv(target: Path, uv: str | None) -> Path:
    venv = target / ".venv"
    python = venv_python(venv)
    version = python_version(python) if python.is_file() else None
    if version and (3, 10) <= version <= (3, 12):
        return python
    if uv:
        run(uv, "venv", "--clear", "--python", "3.12", venv)
    else:
        if not (3, 10) <= sys.version_info[:2] <= (3, 12):
            raise SystemExit("Cần Python 3.10–3.12; hãy dùng lệnh cài một dòng trong README để tự có Python 3.12.")
        if venv.exists():
            shutil.rmtree(venv)
        run(sys.executable, "-m", "venv", venv)
    return python


def install_packages(python: Path, target: Path, uv: str | None) -> None:
    requirements = target / "requirements.txt"
    if uv:
        # uv takes torch (and only torch) from the CPU index; the default Linux build pulls gigabytes of CUDA.
        run(uv, "pip", "install", "--python", python, "-r", requirements, "--torch-backend", "cpu")
    else:
        run(python, "-m", "pip", "install", "-q", "--upgrade", "pip")
        if sys.platform != "darwin":
            run(python, "-m", "pip", "install", "-q", TORCH, "--index-url", TORCH_CPU_INDEX)
        run(python, "-m", "pip", "install", "-q", "-r", requirements)
    # Chromium only serves chapter links from script-heavy sites, so a failure here is not fatal.
    if subprocess.run([str(python), "-m", "playwright", "install", "chromium"]).returncode != 0:
        print("    Chưa cài được Chromium: vẫn dùng được ảnh/ZIP, chỉ tải chương từ vài trang web là cần nó.")


def build_ctd(target: Path, uv: str | None) -> None:
    """Export models/ctd_seg.onnx in a throwaway environment, as CI does."""
    model = target / "models" / "ctd_seg.onnx"
    if model.is_file():
        return
    cache = target / ".cache"
    cache.mkdir(exist_ok=True)
    raw = cache / "comictextdetector.pt.onnx"
    download(CTD_URL, raw, CTD_SHA256)
    build = cache / "ctd-build"
    python = venv_python(build)
    packages = ["numpy==1.26.4", "onnx==1.17.0", "onnx2torch==1.5.15", "onnxruntime==1.20.1"]
    if uv:
        run(uv, "venv", "--clear", "--python", "3.12", build)
        run(uv, "pip", "install", "--python", python, TORCH, *packages, "--torch-backend", "cpu")
    else:
        run(sys.executable, "-m", "venv", "--clear", build)
        torch_index = [] if sys.platform == "darwin" else ["--index-url", TORCH_CPU_INDEX]
        run(python, "-m", "pip", "install", "-q", TORCH, *torch_index)
        run(python, "-m", "pip", "install", "-q", *packages)
    run(python, target / "scripts" / "export_ctd_onnx.py", raw, model)
    shutil.rmtree(cache / "ctd-build", ignore_errors=True)
    raw.unlink(missing_ok=True)


def with_dir(path_value: str, folder: str) -> str:
    """PATH with ``folder`` added once, in front."""
    parts = [p for p in path_value.split(os.pathsep) if p]
    if any(os.path.normcase(p.rstrip("\\/")) == os.path.normcase(folder.rstrip("\\/")) for p in parts):
        return path_value
    return os.pathsep.join([folder] + parts)


def launcher_text(python: Path, entry: Path, windows: bool = WINDOWS, local_appdata: str | None = None) -> str:
    """The `manga` shim: runs the app's Python on run.py; on Windows it copes with paths that have accents."""
    if not windows:
        return f'#!/bin/sh\nexec "{python}" "{entry}" --open "$@"\n'
    python_s, entry_s = str(python), str(entry)
    if local_appdata and python_s.lower().startswith(local_appdata.lower() + "\\"):
        # cmd expands the variable itself, so the user name never has to be written in a code page.
        python_s = "%LOCALAPPDATA%" + python_s[len(local_appdata):]
        entry_s = "%LOCALAPPDATA%" + entry_s[len(local_appdata):] if entry_s.lower().startswith(local_appdata.lower()) else entry_s
    line = f'"{python_s}" "{entry_s}" --open %*'
    if line.isascii():
        return f"@echo off\r\n{line}\r\n"
    # cmd reads a batch file in the console code page; switch to UTF-8 for the path, then switch back.
    return ('@echo off\r\nfor /f "tokens=2 delims=:." %%c in (\'chcp\') do set "_mt_cp=%%c"\r\n'
            f"chcp 65001 >nul\r\n{line}\r\nchcp %_mt_cp% >nul\r\n")


def add_to_user_path_windows(folder: Path) -> None:
    import ctypes
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            value, kind = "", winreg.REG_EXPAND_SZ
        updated = with_dir(value, str(folder))
        if updated != value:
            # Keep the value's type: rewriting REG_EXPAND_SZ as a plain string breaks %USERPROFILE% entries.
            winreg.SetValueEx(key, "Path", 0, kind, updated)
    # Tell Windows the environment changed, so windows opened from now on see the command.
    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)


RC_LINE = 'export PATH="{folder}:$PATH"  # manga-translator'
# The file each shell reads in a new terminal; created when missing (a new Mac account has no ~/.zshrc).
PRIMARY_RC = {"zsh": ".zshrc", "fish": ".config/fish/config.fish",
              "bash": ".bash_profile" if sys.platform == "darwin" else ".bashrc"}


def add_to_shell_path(folder: Path, home: Path | None = None) -> list[Path]:
    """Add ``folder`` to PATH once, in the user's shell start-up file and in ~/.profile when it exists."""
    home = home or Path.home()
    primary = PRIMARY_RC.get(os.path.basename(os.environ.get("SHELL", "")), ".profile")
    files = [home / primary] + [home / name for name in (".bashrc", ".bash_profile", ".zshrc", ".profile")
                                if name != primary and (home / name).is_file()]
    changed = []
    for rc in files:
        line = f"fish_add_path {folder}  # manga-translator" if rc.name == "config.fish" else RC_LINE.format(folder=folder)
        text = rc.read_text(encoding="utf-8") if rc.is_file() else ""
        if str(folder) in text:
            continue
        rc.parent.mkdir(parents=True, exist_ok=True)
        with rc.open("a", encoding="utf-8") as handle:
            handle.write(("\n" if text and not text.endswith("\n") else "") + line + "\n")
        changed.append(rc)
    return changed


def install_command(python: Path, target: Path, home: Path) -> Path:
    entry = target / "run.py"
    if WINDOWS:
        folder = home / "bin"
        folder.mkdir(parents=True, exist_ok=True)
        shim = folder / "manga.cmd"
        data = launcher_text(python, entry, local_appdata=os.environ.get("LOCALAPPDATA")).encode("utf-8")
        # `manga update` runs from this very file, and cmd reads it as it goes: rewrite it only if it changes.
        if not shim.is_file() or shim.read_bytes() != data:
            shim.write_bytes(data)
        add_to_user_path_windows(folder)
        return shim
    folder = Path.home() / ".local" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    shim = folder / "manga"
    shim.write_text(launcher_text(python, entry), encoding="utf-8")
    shim.chmod(0o755)
    if str(folder) not in os.environ.get("PATH", "").split(os.pathsep):
        for rc in add_to_shell_path(folder):
            print(f"    Đã thêm {folder} vào PATH trong {rc}")
    return shim


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")  # a console that cannot show Vietnamese still gets the messages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT, help="where the app is installed (default: this folder)")
    args = parser.parse_args(argv)
    target = args.target.resolve()
    home = default_home()
    uv = find_uv()

    check_disk(target)
    if target != ROOT:
        say(f"Cập nhật mã nguồn vào {target}")
        sync_code(ROOT, target)
    write_marker(target, home, uv, done=False)
    ensure_vc_runtime()
    say("Chuẩn bị Python 3.12 (.venv)")
    python = ensure_venv(target, uv)
    say("Cài thư viện (lần đầu mất vài phút)")
    install_packages(python, target, uv)
    say("Tải và chuẩn bị model")
    (target / "models").mkdir(exist_ok=True)
    download(LAMA_URL, target / "models" / "lama-manga-dynamic.onnx", LAMA_SHA256)
    build_ctd(target, uv)
    say("Tạo lệnh `manga`")
    shim = install_command(python, target, home)
    print(f"    {shim}")
    write_marker(target, home, uv, done=True)
    say("Xong. Mở một cửa sổ cmd/terminal MỚI và gõ:  manga")
    print("    `manga update` để cập nhật bản mới.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # The failing tool has already printed why; a Python traceback on top only hides it.
        name = Path(str(exc.cmd[0])).name if exc.cmd else "?"
        sys.exit(f"\nBước vừa rồi bị lỗi ({name}, mã {exc.returncode}). Chạy lại lệnh cài; "
                 "nếu vẫn lỗi, gửi đoạn chữ phía trên để được hỗ trợ.")
    except KeyboardInterrupt:
        sys.exit("\nĐã dừng. Chạy lại lệnh cài để tiếp tục từ chỗ dở.")
