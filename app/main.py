from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, OUTPUT_DIR, PROCESSED_DIR, check_models, ensure_directories
from app.logging_config import logger
from app.manifest_utils import cleanup_stale_temp_artifacts
from app.routers import automation, chapters, editor, export, image, ocr, render, render_commit, translation, visual_qc
from app.security import MAX_REQUEST_BYTES, MAX_UPLOAD_TOTAL_BYTES


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    removed_temp_artifacts = 0
    for root in (PROCESSED_DIR, OUTPUT_DIR):
        if not root.exists():
            continue
        for chapter_dir in root.iterdir():
            if chapter_dir.is_dir():
                removed_temp_artifacts += cleanup_stale_temp_artifacts(chapter_dir)
    if removed_temp_artifacts:
        logger.info("Removed {} stale temporary artifact(s) from previous runs", removed_temp_artifacts)

    missing = check_models()
    if missing:
        logger.warning(
            "Models missing: %s — /api/process_pages will fail until models are placed in models/",
            ", ".join(missing),
        )
    else:
        logger.info("ONNX models OK")
    yield


app = FastAPI(lifespan=lifespan, title="Manga Translator", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


class _RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Enforce request limits on bytes actually received, not only headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        limit = (
            MAX_UPLOAD_TOTAL_BYTES
            if path == "/api/chapter/upload"
            else MAX_REQUEST_BYTES
        )
        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared_size = int(content_length.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                response = JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
                await response(scope, receive, send)
                return
            if declared_size > limit:
                response = JSONResponse(
                    {"detail": "Request too large"}, status_code=413
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            response = JSONResponse(
                {"detail": "Request too large"}, status_code=413
            )
            await response(scope, receive, send)


app.add_middleware(RequestSizeLimitMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


app.include_router(chapters.router)
app.include_router(automation.router)
app.include_router(translation.router)
app.include_router(ocr.router)
app.include_router(editor.router)
app.include_router(render_commit.router)
app.include_router(render.router)
app.include_router(image.router)
app.include_router(export.router)
app.include_router(visual_qc.router)


@app.get("/health")
def health():
    missing = check_models()
    return {
        "status": "ok" if not missing else "degraded",
        "models_missing": missing,
        "version": app.version,
    }


@app.get("/")
def index():
    return FileResponse("app/templates/index.html")


def _remove_duplicate_routes() -> None:
    """Keep the first canonical handler for exact duplicate path/method routes."""
    seen: set[tuple[str, frozenset[str]]] = set()
    unique_routes = []
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = frozenset(getattr(route, "methods", set()) or set())
        if path and methods:
            key = (str(path), methods)
            if key in seen:
                logger.warning(
                    "Ignoring duplicate route %s %s from %s",
                    ",".join(sorted(methods)),
                    path,
                    getattr(route, "name", "unknown"),
                )
                continue
            seen.add(key)
        unique_routes.append(route)
    app.router.routes[:] = unique_routes


_remove_duplicate_routes()
