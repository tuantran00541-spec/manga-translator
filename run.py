import argparse
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn


from app.config import BASE_DIR, HOST, PORT, RELOAD, WORKERS, ensure_directories, check_models
from app.logging_config import logger
import cv2


def _local_url() -> str:
    host = "127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST
    return f"http://{host}:{PORT}"


def _is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1):
            return True
    except OSError:
        return False


def _open_when_up(url: str) -> None:
    for _ in range(240):
        if _is_up(url):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _update() -> None:
    """Pull the latest code and rerun the installer, which keeps what is already there."""
    if (BASE_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(BASE_DIR), "pull", "--ff-only"], check=True)
    subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "install.py")], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="manga", description="Manga Translator")
    parser.add_argument("command", nargs="?", choices=["update"], help="update: get the latest version")
    parser.add_argument("--open", action="store_true", help="open the app in the browser once it is up")
    args = parser.parse_args(argv)
    if args.command == "update":
        _update()
        return

    url = _local_url()
    if args.open and _is_up(url):
        logger.info(f"Manga Translator is already running at {url}")
        webbrowser.open(url)
        return

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

    logger.info(f"Starting Manga Translator on http://{HOST}:{PORT}")
    if HOST == "0.0.0.0":
        logger.warning(
            "Bound to 0.0.0.0 — accessible from network. "
            "Use firewall + auth if public."
        )
    if args.open:
        threading.Thread(target=_open_when_up, args=(url,), daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        workers=1 if RELOAD else max(1, WORKERS),
        log_level="info",
    )


if __name__ == "__main__":
    main()
