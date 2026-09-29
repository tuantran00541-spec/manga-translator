"""Install once: a private Python environment, the dependencies, the models and a `manga` command on PATH."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MODELS = ROOT / "models"
WINDOWS = os.name == "nt"
TORCH = "torch==2.5.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
LAMA_URL = "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"
LAMA_SHA256 = "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9"
CTD_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
CTD_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"


def say(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def run(*args) -> None:
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str) -> None:
    """Fetch ``url`` to ``path`` unless a file is already there, and refuse a download whose hash is wrong."""
    if path.is_file():
        return
    part = path.with_suffix(path.suffix + ".part")
    print(f"    {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if sha256(part) != expected:
        part.unlink()
        raise SystemExit(f"Tải {path.name} bị lỗi (sai mã kiểm tra), hãy chạy lại bộ cài.")
    part.replace(path)


def install_packages(python: Path) -> None:
    run(python, "-m", "pip", "install", "-q", "--upgrade", "pip")
    if sys.platform != "darwin":
        # The CPU build of torch; the default one on Linux pulls gigabytes of CUDA.
        run(python, "-m", "pip", "install", "-q", TORCH, "--index-url", TORCH_CPU_INDEX)
    run(python, "-m", "pip", "install", "-q", "-r", ROOT / "requirements.txt")
    run(python, "-m", "playwright", "install", "chromium")


def build_ctd() -> None:
    """Export models/ctd_seg.onnx in a throwaway environment, as CI does."""
    target = MODELS / "ctd_seg.onnx"
    if target.is_file():
        return
    build = ROOT / ".cache" / "ctd-build"
    raw = ROOT / ".cache" / "comictextdetector.pt.onnx"
    raw.parent.mkdir(parents=True, exist_ok=True)
    download(CTD_URL, raw, CTD_SHA256)
    if not venv_python(build).is_file():
        run(sys.executable, "-m", "venv", build)
    python = venv_python(build)
    torch_args = [] if sys.platform == "darwin" else ["--index-url", TORCH_CPU_INDEX]
    run(python, "-m", "pip", "install", "-q", TORCH, *torch_args)
    run(python, "-m", "pip", "install", "-q", "numpy==1.26.4", "onnx==1.17.0", "onnx2torch==1.5.15", "onnxruntime==1.20.1")
    run(python, ROOT / "scripts" / "export_ctd_onnx.py", raw, target)
    shutil.rmtree(ROOT / ".cache" / "ctd-build", ignore_errors=True)


def with_dir(path_value: str, folder: str) -> str:
    """PATH with ``folder`` appended once."""
    parts = [p for p in path_value.split(os.pathsep) if p]
    if any(os.path.normcase(p.rstrip("\\/")) == os.path.normcase(folder.rstrip("\\/")) for p in parts):
        return path_value
    return os.pathsep.join(parts + [folder])


def launcher_text(python: Path, windows: bool = WINDOWS) -> str:
    entry = ROOT / "run.py"
    if windows:
        return f'@echo off\r\n"{python}" "{entry}" --open %*\r\n'
    return f'#!/bin/sh\nexec "{python}" "{entry}" --open "$@"\n'


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
            winreg.SetValueEx(key, "Path", 0, kind, updated)
    # Tell Windows the environment changed, so new cmd windows see the command.
    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)


def install_command(python: Path) -> Path:
    if WINDOWS:
        folder = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "manga-translator"
        folder.mkdir(parents=True, exist_ok=True)
        shim = folder / "manga.cmd"
        shim.write_text(launcher_text(python), encoding="utf-8")
        add_to_user_path_windows(folder)
        return shim
    folder = Path.home() / ".local" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    shim = folder / "manga"
    shim.write_text(launcher_text(python), encoding="utf-8")
    shim.chmod(0o755)
    if str(folder) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"    Thêm {folder} vào PATH (ví dụ trong ~/.bashrc hoặc ~/.zshrc) để gõ được `manga`.")
    return shim


def main() -> None:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("Cần Python 3.10–3.12 (khuyên dùng 3.12 từ python.org), "
                         f"máy đang dùng {sys.version.split()[0]}.")
    say("Tạo môi trường Python riêng (.venv)")
    if not venv_python(VENV).is_file():
        run(sys.executable, "-m", "venv", VENV)
    python = venv_python(VENV)
    say("Cài thư viện (lần đầu mất vài phút)")
    install_packages(python)
    say("Tải và chuẩn bị model")
    MODELS.mkdir(exist_ok=True)
    download(LAMA_URL, MODELS / "lama-manga-dynamic.onnx", LAMA_SHA256)
    build_ctd()
    say("Tạo lệnh `manga`")
    shim = install_command(python)
    print(f"    {shim}")
    say("Xong. Mở một cửa sổ cmd/terminal MỚI và gõ:  manga")
    print("    `manga update` để cập nhật bản mới.")


if __name__ == "__main__":
    main()
