import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO = "tuantran00541-spec/manga-translator"
MARKER = BASE_DIR / ".manga-install.json"

if sys.stdout is None or sys.stderr is None:
    # Started from the desktop icon (pythonw) there is no console; the log goes to a file instead.
    (BASE_DIR / "logs").mkdir(exist_ok=True)
    _console = open(BASE_DIR / "logs" / "console.log", "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = sys.stdout or _console
    sys.stderr = sys.stderr or _console

if os.name == "nt" and not str(Path.home()).isascii() and "PADDLE_PDX_CACHE_HOME" not in os.environ:
    # PaddleOCR cannot load its models from a user folder with accents.
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "manga-translator" / "paddlex")

# Only pure-Python modules load here: Windows locks loaded .pyd files, and `manga update` may replace them.
from app.config import HOST, PORT, RELOAD, WORKERS, check_models, ensure_directories  # noqa: E402
from app.logging_config import CONSOLE, logger  # noqa: E402

VERSION = "0.3.1"


def _url(port: int) -> str:
    host = "127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST
    return f"http://{host}:{port}"


def _health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
            body = json.loads(response.read() or b"{}")
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _is_ours(url: str) -> bool:
    """True when this app answers at ``url``, even degraded by missing models, not some other program on the port."""
    body = _health(url)
    return bool(body) and body.get("status") in ("ok", "degraded") and "models_missing" in body


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST, port))
        except OSError:
            return False
    return True


def _pick_port() -> int | None:
    """The configured port, or the next free one when another program holds it; None when this app already runs."""
    for port in range(PORT, PORT + 20):
        if _is_ours(_url(port)):
            return None
        if _port_free(port):
            return port
    raise SystemExit(f"Các cổng {PORT}–{PORT + 19} đều bận; đặt biến PORT sang cổng khác rồi chạy lại.")


def _running_port() -> int | None:
    for port in range(PORT, PORT + 20):
        if _is_ours(_url(port)):
            return port
    return None


def _launch_dir() -> Path | None:
    """The folder `manga` was typed in, for the Agent tab; None from an icon, home, a drive root or the app's own folder."""
    if not (sys.stdin and sys.stdin.isatty()):
        return None
    try:
        here = Path.cwd().resolve()
    except OSError:
        return None
    if here in (Path.home().resolve(), Path(here.anchor)) or here == BASE_DIR or BASE_DIR in here.parents:
        return None
    return here


def _page(url: str) -> str:
    """The app URL, carrying the launch folder for the Agent tab when there is one."""
    folder = _launch_dir()
    return f"{url}/?workspace={urllib.parse.quote(str(folder))}" if folder else url


def _open_when_up(url: str, show=webbrowser.open) -> None:
    for _ in range(240):
        if _is_ours(url):
            show(_page(url))
            return
        time.sleep(0.5)


def _print_link(url: str) -> None:
    print(f"Manga Translator: {url}\nCtrl+C để tắt.", flush=True)


def _quiet_console() -> None:
    """Only warnings and errors reach the console; everything still goes to logs/app.log."""
    logger.remove(CONSOLE)
    logger.add(sys.stderr, level="WARNING", format="{message}", diagnose=False)
    os.environ["PYTHONWARNINGS"] = "ignore"
    import warnings
    warnings.simplefilter("ignore")


def _install_info() -> dict:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _run_installer(source: Path, info: dict) -> None:
    """Run the installer of ``source`` on this folder with the same uv and home as the first install."""
    uv = info.get("uv") if info.get("uv") and Path(info["uv"]).is_file() else shutil.which("uv")
    env = {**os.environ, "MANGA_REF": str(info.get("ref") or "main"), "MANGA_CHANNEL": str(info.get("channel") or "ref")}
    if info.get("home"):
        env["MANGA_HOME"] = str(info["home"])
        env["UV_PYTHON_INSTALL_DIR"] = str(Path(info["home"]) / "python")
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
        env["UV_CACHE_DIR"] = str(Path(info["home"]) / "uv-cache")
    script = [str(source / "scripts" / "install.py"), "--target", str(BASE_DIR)]
    if uv:
        env["MANGA_UV"] = uv
        command = [uv, "run", "--no-project", "--python", "3.12", *script]
    else:
        command = [sys.executable, *script]
    subprocess.run(command, check=True, env=env)
    if uv:
        subprocess.run([uv, "cache", "prune"], env=env, capture_output=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_pin(info: dict) -> str | None:
    """Pinned SHA-256 for the update ZIP: MANGA_UPDATE_SHA256 wins, then the install marker's update_sha256."""
    for raw in (os.environ.get("MANGA_UPDATE_SHA256"), info.get("update_sha256")):
        pin = str(raw or "").strip().lower()
        if not pin:
            continue
        if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise SystemExit(f"Mã pin SHA-256 không hợp lệ ({str(raw)[:16]}…); phải là 64 ký tự hex.")
        return pin
    return None


def _verify_archive(archive: Path, ref: str, info: dict) -> None:
    """Refuse the update ZIP when a pinned SHA-256 is set and does not match."""
    pin = _update_pin(info)
    if pin is None:
        return
    digest = _sha256_file(archive)
    if digest != pin:
        raise SystemExit(
            f"Không cập nhật: SHA-256 của bản tải về ({digest[:16]}…) không khớp mã pin ({pin[:16]}…). "
            "File có thể đã bị thay đổi ở nguồn hoặc trên đường truyền; từ chối cài đặt."
        )


def _latest_release() -> str | None:
    """The newest release named like V0.4.0, skipping drafts, pre-releases and data releases."""
    url = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}), timeout=30) as response:
            releases = json.load(response)
    except (OSError, ValueError):
        return None
    return next((r["tag_name"] for r in releases if not r.get("draft") and not r.get("prerelease")
                 and re.match(r"[Vv]?\d+\.\d+", str(r.get("tag_name", "")))), None)


def _ensure_temp_dir() -> None:
    """Fall back to a folder inside the install when the system offers no usable temp dir,
    so a deleted or unwritable Windows Temp folder cannot block an update."""
    try:
        tempfile.gettempdir()
        return
    except FileNotFoundError:
        pass
    fallback = BASE_DIR / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(fallback)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(fallback)


def _update() -> None:
    """Get the latest code and rerun the installer, which keeps models, chapters and the environment."""
    _ensure_temp_dir()
    if _running_port() is not None:
        raise SystemExit("Manga Translator đang chạy; tắt cửa sổ đang chạy nó rồi gõ lại `manga update`.")
    info = _install_info()
    if (BASE_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(BASE_DIR), "pull", "--ff-only"], check=True)
        _run_installer(BASE_DIR, info)
        return
    ref = str(info.get("ref") or "main")
    if info.get("channel") == "release":
        ref = _latest_release() or ref
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "source.zip"
        print(f"Đang tải bản mới ({ref})…", flush=True)
        urllib.request.urlretrieve(f"https://codeload.github.com/{REPO}/zip/{ref}", archive)
        _verify_archive(archive, ref, info)
        shutil.unpack_archive(archive, work)
        source = next(p for p in Path(work).iterdir() if p.is_dir())
        _run_installer(source, {**info, "ref": ref})


def _safe_console() -> None:
    """Vietnamese messages must not crash when Windows sends output to a cp1252 pipe or file."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> None:
    _safe_console()
    parser = argparse.ArgumentParser(prog="manga", description="Manga Translator")
    parser.add_argument("command", nargs="?", choices=["update"], help="update: get the latest version")
    parser.add_argument("--open", action="store_true", help="open the app in the browser once it is up")
    parser.add_argument("--window", action="store_true", help="show the app in its own window instead of the browser")
    parser.add_argument("--link", action="store_true", help="print only the app link, without opening a browser")
    parser.add_argument("--version", action="version", version=f"Manga Translator {VERSION} ({BASE_DIR})")
    args = parser.parse_args(argv)
    if args.command == "update":
        _update()
        return

    port = _pick_port() if args.open or args.window or args.link else PORT
    if port is None:
        url = _url(_running_port() or PORT)
        if args.link:
            _print_link(_page(url))
            return
        logger.info(f"Manga Translator is already running at {url}")
        if not (args.window and _show_window(_page(url))):
            webbrowser.open(_page(url))
        return
    if port != PORT:
        logger.warning(f"Port {PORT} is used by another program; using {port}")
    if args.window:
        _serve_in_window(port)
        return
    _serve(port, args.open, quiet=args.link)


def _check_vc_runtime() -> None:
    """On Windows, fix a missing or old Visual C++ runtime before onnxruntime or paddle fails with a cryptic WinError 126."""
    if os.name != "nt":
        return
    from app import vc_runtime

    found = vc_runtime.problems()
    if not found:
        return
    print(f"Thiếu Microsoft Visual C++ Runtime ({', '.join(found)}); đang cài, Windows sẽ hỏi quyền admin…", flush=True)
    try:
        note = vc_runtime.install()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(note or "Đã cài xong Visual C++ Runtime.", flush=True)


LOADING_PAGE = """<!doctype html><meta charset="utf-8"><body style="margin:0;height:100vh;display:grid;place-items:center;
font:16px system-ui,sans-serif;background:#fff;color:#111"><div style="text-align:center"><b style="font-size:20px">
Manga Translator</b><p>Đang khởi động…</p></div></body>"""


def _show_window(url: str, on_ready=None) -> bool:
    """Show ``url`` in a native window until it is closed; False when this system cannot open one."""
    try:
        import webview
    except ImportError:
        return False
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    window = webview.create_window("Manga Translator", html=LOADING_PAGE if on_ready else None,
                                   url=None if on_ready else url, width=1440, height=920, min_size=(960, 640))

    def load_when_up():
        on_ready()
        window.load_url(url)

    try:
        webview.start(load_when_up if on_ready else None)
    except Exception as exc:  # no GUI toolkit (a bare Linux box): the caller falls back to the browser
        logger.warning(f"Cannot open the app window ({exc}); using the browser")
        return False
    return True


def _serve_in_window(port: int) -> None:
    """The server in the background and the app in its own window; closing the window stops the server."""
    import uvicorn

    _prepare()
    url = _url(port)
    server = uvicorn.Server(uvicorn.Config("app.main:app", host=HOST, port=port, workers=1, log_level="info"))
    thread = threading.Thread(target=server.run, name="server", daemon=True)
    thread.start()

    def wait_until_up():
        while thread.is_alive() and not _is_ours(url):
            time.sleep(0.3)

    if _show_window(_page(url), on_ready=wait_until_up):
        server.should_exit = True
        thread.join(timeout=10)
        return
    threading.Thread(target=_open_when_up, args=(url,), daemon=True).start()
    thread.join()


def _prepare() -> None:
    """Checks every start makes before the server runs."""
    _check_vc_runtime()
    import cv2

    ensure_directories()
    cv2.setNumThreads(1)

    missing = check_models()
    if missing:
        logger.warning("=" * 60)
        logger.warning("MISSING MODELS — server will start but processing will fail")
        logger.warning("Missing files in models/:")
        for name in missing:
            logger.warning(f"  • {name}")
        logger.warning("Run the installer again (install.bat / install.sh) or see docs/INSTALL.md → Models")
        logger.warning("=" * 60)
    else:
        logger.info("All required ONNX models found in models/")


def _serve(port: int, open_browser: bool, quiet: bool = False) -> None:
    import uvicorn

    if quiet:
        _quiet_console()
        print("Manga Translator đang khởi động…", flush=True)
    _prepare()
    logger.info(f"Starting Manga Translator on http://{HOST}:{port}")
    if HOST == "0.0.0.0":
        logger.warning(
            "Bound to 0.0.0.0 — accessible from network. "
            "Use firewall + auth if public."
        )
    if open_browser or quiet:
        threading.Thread(target=_open_when_up, args=(_url(port), _print_link if quiet else webbrowser.open), daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=port,
        reload=RELOAD,
        workers=1 if RELOAD else max(1, WORKERS),
        log_level="warning" if quiet else "info",
        access_log=not quiet,
    )


if __name__ == "__main__":
    main()
