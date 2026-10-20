import json
import os
import subprocess
import sys
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import BASE_DIR

router = APIRouter()

EXTENSION_DIR = BASE_DIR / "extension"
_seen: dict[str, str | None] = {"version": None}


class ExtensionHello(BaseModel):
    version: str


def version() -> str:
    """The extension version on disk, which Chrome only picks up when the user reloads it."""
    return json.loads((EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8"))["version"]


def _status() -> dict:
    seen = _seen["version"]
    return {"path": str(EXTENSION_DIR), "seen": seen is not None, "outdated": seen is not None and seen != version()}


@router.get("/api/extension")
def extension_status() -> dict:
    return _status()


@router.post("/api/extension/hello")
def extension_hello(req: ExtensionHello) -> dict:
    _seen["version"] = req.version
    return _status()


@router.post("/api/extension/open")
def open_extension_folder(request: Request) -> dict:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Only the app itself can open this folder")
    try:
        if sys.platform == "win32":
            os.startfile(EXTENSION_DIR)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(EXTENSION_DIR)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise HTTPException(409, f"Không mở được thư mục: {exc}") from exc
    return {"path": str(EXTENSION_DIR)}
