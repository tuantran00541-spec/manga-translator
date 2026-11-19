import io
import json
import subprocess
import sys
import zipfile
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import BASE_DIR

router = APIRouter()

EXTENSION_DIR = BASE_DIR / "extension"
_seen: dict[str, str | None] = {"version": None}


class ExtensionHello(BaseModel):
    version: str = Field(..., min_length=1, max_length=64)


def version() -> str:
    manifest = json.loads((EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8"))
    ver = manifest.get("version") if isinstance(manifest, dict) else None
    if not isinstance(ver, str) or not ver.strip():
        raise ValueError("Extension manifest missing version")
    return ver


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
    client_host = (request.client.host if request.client else "").strip().lower()
    if client_host and client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "Only local requests can open this folder")
    sec_fetch = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if sec_fetch not in ("", "same-origin", "none"):
        raise HTTPException(403, "Only the app itself can open this folder")
    origin = (request.headers.get("origin") or "").strip()
    host = (request.headers.get("host") or "").strip().lower()
    if not origin or not host:
        raise HTTPException(403, "Only the app itself can open this folder")
    parsed_origin = urlsplit(origin)
    if parsed_origin.scheme not in ("http", "https") or parsed_origin.netloc.lower() != host:
        raise HTTPException(403, "Only the app itself can open this folder")
    if not EXTENSION_DIR.is_dir():
        raise HTTPException(404, f"Không tìm thấy thư mục extension tại: {EXTENSION_DIR}")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer.exe", str(EXTENSION_DIR)])
        else:
            subprocess.Popen(
                ["open" if sys.platform == "darwin" else "xdg-open", str(EXTENSION_DIR)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
    except OSError as exc:
        raise HTTPException(409, f"Không mở được thư mục: {exc}") from exc
    return {"path": str(EXTENSION_DIR)}


@router.get("/api/extension/download")
def download_extension_zip() -> Response:
    if not EXTENSION_DIR.is_dir():
        raise HTTPException(404, f"Không tìm thấy thư mục extension tại: {EXTENSION_DIR}")
    base_resolved = EXTENSION_DIR.resolve()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(EXTENSION_DIR.iterdir()):
            if path.is_symlink() or not path.is_file() or path.name.startswith("."):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(base_resolved):
                continue
            zf.write(resolved, arcname=f"manga-translator-extension/{path.name}")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="manga-translator-extension.zip"'},
    )
