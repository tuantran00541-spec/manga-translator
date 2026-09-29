"""The Microsoft Visual C++ runtime that torch, onnxruntime and paddle load from Windows; standard library only."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
# These are not shipped in the wheels; Python brings its own vcruntime140.dll, so that one is not listed.
DLLS = ("msvcp140.dll", "vcruntime140_1.dll", "vcomp140.dll")
# Code built with MSVC 14.40+ (onnxruntime among it) crashes in std::mutex on an older msvcp140.dll.
MIN_VERSION = (14, 40)
# Installer exit codes that leave a working runtime: done, a newer one already there, done but restart pending.
OK_CODES = {0: "", 1638: "", 3010: "Windows cần khởi động lại để hoàn tất; app vẫn chạy được ngay."}
CANCELLED = 1602


def system32() -> Path:
    return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"


def file_version(path: Path) -> tuple[int, ...] | None:
    """The file version of a Windows DLL, or None when it cannot be read."""
    try:
        api = ctypes.windll.version
    except AttributeError:
        return None
    size = api.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not api.GetFileVersionInfoW(str(path), 0, size, data):
        return None
    info, length = ctypes.c_void_p(), ctypes.c_uint()
    if not api.VerQueryValueW(data, "\\", ctypes.byref(info), ctypes.byref(length)) or length.value < 52:
        return None
    ms, ls = ctypes.cast(info, ctypes.POINTER(ctypes.c_uint32 * 13)).contents[2:4]
    return ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def problems(folder: Path | None = None, version_of=file_version) -> list[str]:
    """What is wrong with the runtime: missing DLLs, or one older than MIN_VERSION."""
    if os.environ.get("MANGA_VC_ASSUME_MISSING") == "1":  # lets CI exercise the install on a machine that has it
        return ["msvcp140.dll (giả lập thiếu)"]
    folder = folder or system32()
    found = []
    for name in DLLS:
        path = folder / name
        if not path.is_file():
            found.append(f"{name} (thiếu)")
            continue
        version = version_of(path)
        if version is not None and version[:2] < MIN_VERSION:
            found.append(f"{name} (bản {'.'.join(map(str, version))} quá cũ)")
    return found


SIGNER = "O=Microsoft Corporation"
# Reads the file's Authenticode signature; the path comes in through the environment, never inside the command.
_SIGNATURE_PS = ("$s = Get-AuthenticodeSignature -LiteralPath $env:MANGA_VC_FILE; "
                 "Write-Output ('{0}|{1}' -f $s.Status, $s.SignerCertificate.Subject)")


def signed_by_microsoft(path: Path) -> bool:
    """True only for a file with a valid Authenticode signature from Microsoft Corporation."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _SIGNATURE_PS],
        capture_output=True, text=True, env={**os.environ, "MANGA_VC_FILE": str(path)},
    )
    status, _, subject = result.stdout.strip().partition("|")
    return result.returncode == 0 and status == "Valid" and SIGNER in subject


def install() -> str:
    """Download, verify and run the official installer; returns a note for the user, raises RuntimeError on failure."""
    folder = Path(tempfile.mkdtemp(prefix="manga-vc-"))  # unpredictable, so nothing can be planted in its place
    target = folder / "vc_redist.x64.exe"
    try:
        try:
            urllib.request.urlretrieve(REDIST_URL, target)  # a fixed https URL
        except OSError as exc:
            raise RuntimeError(f"Không tải được Visual C++ Runtime ({exc}). Tải tay tại {REDIST_URL}, cài rồi chạy lại.") from exc
        if not signed_by_microsoft(target):
            raise RuntimeError("Bộ cài Visual C++ tải về không có chữ ký hợp lệ của Microsoft nên không chạy. "
                               f"Tải tay tại {REDIST_URL}.")
        code = subprocess.run([str(target), "/install", "/passive", "/norestart"]).returncode
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if code == CANCELLED:
        raise RuntimeError("Bạn đã từ chối quyền admin. Chạy lại và bấm Yes, hoặc tự cài từ " + REDIST_URL)
    if code not in OK_CODES:
        raise RuntimeError(f"Cài Visual C++ Runtime lỗi (mã {code}). Tải tay tại {REDIST_URL}, cài rồi chạy lại.")
    os.environ.pop("MANGA_VC_ASSUME_MISSING", None)
    left = problems()
    if left:
        raise RuntimeError(f"Visual C++ Runtime vẫn chưa đủ ({', '.join(left)}). Khởi động lại máy rồi thử lại, "
                           f"hoặc tự cài từ {REDIST_URL}.")
    return OK_CODES[code]
