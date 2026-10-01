import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
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
from app.logging_config import logger  # noqa: E402

VERSION = "0.3.0"


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
    """True when this app answers at ``url``, not some other program on the port."""
    body = _health(url)
    return bool(body) and body.get("status") == "ok" and "models_missing" in body


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


def _open_when_up(url: str) -> None:
    for _ in range(240):
        if _is_ours(url):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _install_info() -> dict:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _run_installer(source: Path, info: dict) -> None:
    """Run the installer of ``source`` on this folder with the same uv and home as the first install."""
    uv = info.get("uv") if info.get("uv") and Path(info["uv"]).is_file() else shutil.which("uv")
    env = {**os.environ, "MANGA_REF": str(info.get("ref") or "main")}
    if info.get("home"):
        env["MANGA_HOME"] = str(info["home"])
        env["UV_PYTHON_INSTALL_DIR"] = str(Path(info["home"]) / "python")
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
    script = [str(source / "scripts" / "install.py"), "--target", str(BASE_DIR)]
    if uv:
        env["MANGA_UV"] = uv
        command = [uv, "run", "--no-project", "--python", "3.12", *script]
    else:
        command = [sys.executable, *script]
    subprocess.run(command, check=True, env=env)


def _update() -> None:
    """Get the latest code and rerun the installer, which keeps models, chapters and the environment."""
    if _running_port() is not None:
        raise SystemExit("Manga Translator đang chạy; tắt cửa sổ đang chạy nó rồi gõ lại `manga update`.")
    info = _install_info()
    if (BASE_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(BASE_DIR), "pull", "--ff-only"], check=True)
        _run_installer(BASE_DIR, info)
        return
    ref = str(info.get("ref") or "main")
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "source.zip"
        print(f"Đang tải bản mới ({ref})…", flush=True)
        urllib.request.urlretrieve(f"https://codeload.github.com/{REPO}/zip/refs/heads/{ref}", archive)
        shutil.unpack_archive(archive, work)
        source = next(p for p in Path(work).iterdir() if p.is_dir())
        _run_installer(source, info)


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
    parser.add_argument("--version", action="version", version=f"Manga Translator {VERSION} ({BASE_DIR})")
    args = parser.parse_args(argv)
    if args.command == "update":
        _update()
        return

    port = _pick_port() if args.open or args.window else PORT
    if port is None:
        url = _url(_running_port() or PORT)
        logger.info(f"Manga Translator is already running at {url}")
        if not (args.window and _show_window(url)):
            webbrowser.open(url)
        return
    if port != PORT:
        logger.warning(f"Port {PORT} is used by another program; using {port}")
    if args.window:
        _serve_in_window(port)
        return
    _serve(port, args.open)


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

    if _show_window(url, on_ready=wait_until_up):
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
        logger.warning("Run the installer again (install.bat / install.sh) or see README.md → Models")
        logger.warning("=" * 60)
    else:
        logger.info("All required ONNX models found in models/")


def _serve(port: int, open_browser: bool) -> None:
    import uvicorn

    _prepare()
    logger.info(f"Starting Manga Translator on http://{HOST}:{port}")
    if HOST == "0.0.0.0":
        logger.warning(
            "Bound to 0.0.0.0 — accessible from network. "
            "Use firewall + auth if public."
        )
    if open_browser:
        threading.Thread(target=_open_when_up, args=(_url(port),), daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=port,
        reload=RELOAD,
        workers=1 if RELOAD else max(1, WORKERS),
        log_level="info",
    )


if __name__ == "__main__":
    main()
