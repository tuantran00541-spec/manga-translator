from __future__ import annotations

import ipaddress
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent import sandbox
from app.agent.session import MODES, AgentSessionManager
from app.agent.tools import Workspace
from app.ai_providers import CLOUD_PROVIDER_ID, validate_model_name
from app.config import BASE_DIR
from app.routers.visual_qc import _resolve_configured_provider
from app.secret_store import SecretStoreUnavailable, get_provider_api_key

AGENT_HEADER = "1"


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def agent_enabled() -> bool:
    return os.getenv("MANGA_AGENT_MODE", "1").strip().lower() not in {"0", "false", "no", "off"}


def guard(request: Request, x_manga_agent: str | None = Header(default=None)) -> None:
    """Agent calls change files and run commands, so they need the switch on, this machine, and a header a foreign page cannot send without a preflight."""
    if not agent_enabled():
        raise HTTPException(404, "Agent mode is turned off (MANGA_AGENT_MODE=0)")
    # MANGA_ALLOWED_HOSTS may open the app to the LAN; the agent stays on this machine.
    if not is_loopback(request.client.host if request.client else ""):
        raise HTTPException(403, "Agent mode only answers this machine")
    if x_manga_agent != AGENT_HEADER:
        raise HTTPException(403, "Agent requests need the X-Manga-Agent header")


router = APIRouter(prefix="/api/agent", tags=["agent"], dependencies=[Depends(guard)])
agent_sessions = AgentSessionManager(BASE_DIR / "data" / "agent")


class SessionRequest(BaseModel):
    provider: str
    model: str
    mode: str = "edits"
    workspace: str = ""
    sandbox: str = "workspace-write"
    network: bool = False


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class ApprovalRequest(BaseModel):
    decision: str
    note: str = Field(default="", max_length=2000)


class SessionUpdate(BaseModel):
    mode: str | None = None
    model: str | None = None
    sandbox: str | None = None
    network: bool | None = None


def _session(session_id: str):
    try:
        return agent_sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Agent session not found") from exc


async def _provider_and_key(provider_id: str):
    if provider_id == CLOUD_PROVIDER_ID:
        raise HTTPException(400, "Manga Cloud only serves A.I mode")
    try:
        provider = await run_in_threadpool(_resolve_configured_provider, provider_id)
        api_key = await run_in_threadpool(get_provider_api_key, provider.id, provider_label=provider.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")
    return provider, api_key


@router.get("/config")
def agent_config() -> dict:
    return {"workspace": str(BASE_DIR), "modes": list(MODES), "sandboxes": list(sandbox.MODES), "sandbox_backend": sandbox.backend()}


@router.post("/sessions")
async def create_session(req: SessionRequest) -> dict:
    provider, api_key = await _provider_and_key(req.provider)
    try:
        model = validate_model_name(req.model, default="")
        if req.sandbox not in sandbox.MODES:
            raise ValueError(f"sandbox must be one of {', '.join(sandbox.MODES)}")
        workspace = Workspace(req.workspace.strip() or BASE_DIR, sandbox.Policy(req.sandbox, req.network))
        session = await run_in_threadpool(agent_sessions.create, provider, api_key, model, workspace, req.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return session.snapshot()


@router.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> dict:
    if session_id in agent_sessions.sessions:
        return agent_sessions.sessions[session_id].snapshot()
    try:
        data = agent_sessions.saved(session_id)
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(404, "Saved agent session not found") from exc
    provider, api_key = await _provider_and_key(str(data.get("provider") or ""))
    try:
        mode, network = (data.get("sandbox") or ["workspace-write", False])[:2]
        workspace = Workspace(data["workspace"], sandbox.Policy(mode, bool(network)))
        session = await run_in_threadpool(agent_sessions.create, provider, api_key, str(data.get("model") or ""), workspace,
                                          data.get("mode") if data.get("mode") in MODES else "edits", session_id=session_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    session.restore(data)
    return session.snapshot()


@router.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": agent_sessions.listing()}


@router.get("/sessions/{session_id}")
def session_events(session_id: str, after: int = 0) -> dict:
    return _session(session_id).snapshot(after)


@router.post("/sessions/{session_id}/messages")
def send_message(session_id: str, req: MessageRequest) -> dict:
    session = _session(session_id)
    try:
        if req.text.startswith("/") and not req.text.startswith("//"):
            return session.command(req.text)
        session.send(req.text)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"sent": True}


@router.post("/sessions/{session_id}/approval")
def approve(session_id: str, req: ApprovalRequest) -> dict:
    if req.decision not in {"allow", "allow_all", "deny"}:
        raise HTTPException(400, "decision must be allow, allow_all or deny")
    try:
        _session(session_id).decide(req.decision, req.note)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.post("/sessions/{session_id}/trust/mcp/{name}")
async def trust_mcp(session_id: str, name: str) -> dict:
    session = _session(session_id)
    try:
        await run_in_threadpool(session.trust_mcp, name)
    except (KeyError, StopIteration) as exc:
        raise HTTPException(404, "MCP server not found") from exc
    return session.snapshot(len(session.events))


@router.post("/sessions/{session_id}/trust/hooks")
def trust_hooks(session_id: str) -> dict:
    session = _session(session_id)
    session.trust_hooks()
    return session.snapshot(len(session.events))


@router.post("/sessions/{session_id}/stop")
def stop(session_id: str) -> dict:
    _session(session_id).stop()
    return {"ok": True}


@router.patch("/sessions/{session_id}")
def update_session(session_id: str, req: SessionUpdate) -> dict:
    session = _session(session_id)
    try:
        if req.mode is not None:
            if req.mode not in MODES:
                raise ValueError(f"mode must be one of {', '.join(MODES)}")
            session.mode = req.mode
        if req.model is not None:
            session.model = validate_model_name(req.model, default="")
        if req.sandbox is not None or req.network is not None:
            policy = session.workspace.policy
            session.set_policy(req.sandbox or policy.mode, policy.network if req.network is None else req.network)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return session.snapshot(len(session.events))


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    try:
        agent_sessions.delete(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Agent session not found") from exc
    return {"ok": True}
