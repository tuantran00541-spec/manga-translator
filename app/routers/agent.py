from __future__ import annotations

import ipaddress
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent.session import MODES, SessionManager
from app.ai_providers import CLOUD_PROVIDER_ID, PROVIDERS
from app.config import BASE_DIR
from app.routers.visual_qc import _resolve_configured_provider
from app.secret_store import SecretStoreUnavailable, get_provider_api_key, list_provider_configs

AGENT_HEADER = "1"


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def guard(request: Request, x_manga_agent: str | None = Header(default=None)) -> None:
    """Agent calls change files and run commands, so they need the switch on, this machine, and a header a foreign page cannot send without a preflight."""
    if os.getenv("MANGA_AGENT_MODE", "1").strip().lower() in {"0", "false", "no", "off"}:
        raise HTTPException(404, "Agent mode is turned off (MANGA_AGENT_MODE=0)")
    # MANGA_ALLOWED_HOSTS may open the app to the LAN; the agent stays on this machine.
    if not is_loopback(request.client.host if request.client else ""):
        raise HTTPException(403, "Agent mode only answers this machine")
    if x_manga_agent != AGENT_HEADER:
        raise HTTPException(403, "Agent requests need the X-Manga-Agent header")


def resolve(provider_id: str):
    """A configured provider and its key, read at every turn so the key never lands in the session file."""
    if provider_id == CLOUD_PROVIDER_ID:
        raise ValueError("Manga Cloud only serves A.I mode")
    provider = _resolve_configured_provider(provider_id)
    api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    if not api_key:
        raise ValueError(f"{provider.label} API key is not configured")
    return provider, api_key


router = APIRouter(prefix="/api/agent", tags=["agent"], dependencies=[Depends(guard)])
sessions = SessionManager(BASE_DIR / "data" / "agent", resolve)


class SessionRequest(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=200)
    workspace: str = Field(min_length=1, max_length=1000)
    mode: str = "ask"
    effort: str = Field(default="", max_length=20)


class SessionUpdate(BaseModel):
    mode: str


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class ApprovalRequest(BaseModel):
    allow: bool
    note: str = Field(default="", max_length=2000)


def _session(session_id: str):
    try:
        return sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Agent session not found") from exc


def _providers() -> list[dict]:
    """Every built-in and custom provider that has a key."""
    found = [(p.id, p.label, p.default_translation_model or p.default_qc_model) for p in PROVIDERS.values()]
    try:
        found += [(str(c.get("id") or ""), str(c.get("label") or c.get("id") or ""), "") for c in list_provider_configs()]
    except SecretStoreUnavailable:
        pass
    return [{"id": i, "label": label, "model": model} for i, label, model in found
            if i and i != CLOUD_PROVIDER_ID and get_provider_api_key(i, provider_label=label)]


@router.get("/config")
async def config() -> dict:
    return {"providers": await run_in_threadpool(_providers), "modes": list(MODES), "workspace": str(Path.home())}


@router.get("/sessions")
def list_sessions() -> list[dict]:
    return sessions.list()


@router.post("/sessions")
def create_session(req: SessionRequest) -> dict:
    try:
        resolve(req.provider)
        return sessions.create(req.provider, req.model, req.workspace, req.mode, req.effort).summary()
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return _session(session_id).view()


@router.patch("/sessions/{session_id}")
def update_session(session_id: str, req: SessionUpdate) -> dict:
    if req.mode not in MODES:
        raise HTTPException(400, f"Unknown mode: {req.mode}")
    session = _session(session_id)
    with session.lock:
        session.data["mode"] = req.mode
        session.save()
    return session.summary()


@router.post("/sessions/{session_id}/messages")
def send_message(session_id: str, req: MessageRequest) -> dict:
    session = _session(session_id)
    try:
        session.send(req.text)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return session.summary()


@router.post("/sessions/{session_id}/approval")
def approve(session_id: str, req: ApprovalRequest) -> dict:
    session = _session(session_id)
    if not session.data.get("pending"):
        raise HTTPException(409, "Nothing is waiting for approval")
    session.approve(req.allow, req.note)
    return {"ok": True}


@router.post("/sessions/{session_id}/stop")
def stop(session_id: str) -> dict:
    _session(session_id).stop()
    return {"ok": True}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    _session(session_id)
    sessions.delete(session_id)
    return {"ok": True}
