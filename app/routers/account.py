from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import cloud
from app.secret_store import SecretStoreUnavailable, delete_cloud_token, set_cloud_token

router = APIRouter(prefix="/api/account", tags=["account"])


class LoginStartRequest(BaseModel):
    email: str = Field(min_length=3, max_length=256)


class LoginVerifyRequest(BaseModel):
    email: str = Field(min_length=3, max_length=256)
    code: str = Field(pattern="^[0-9]{6}$")


class CheckoutRequest(BaseModel):
    plan: str = Field(pattern="^(plus|pro)$")
    provider: str = Field(pattern="^(payos|lemonsqueezy)$")


def _require_tiers() -> None:
    if not cloud.tiers_enabled():
        raise HTTPException(404, "Plans are disabled")


@router.get("")
async def account_status() -> dict:
    return await run_in_threadpool(cloud.entitlements, fresh=True)


@router.post("/login/start")
async def start_login(req: LoginStartRequest) -> dict:
    _require_tiers()
    return await run_in_threadpool(cloud.login_start, req.email)


@router.post("/login/verify")
async def verify_login(req: LoginVerifyRequest) -> dict:
    _require_tiers()
    session = await run_in_threadpool(cloud.login_verify, req.email, req.code)
    try:
        await run_in_threadpool(set_cloud_token, str(session.get("token") or ""))
    except ValueError as exc:
        raise HTTPException(502, "Manga Cloud returned no session") from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    cloud.invalidate()
    return await run_in_threadpool(cloud.entitlements, fresh=True)


@router.post("/logout")
async def logout() -> dict:
    await run_in_threadpool(cloud.logout)
    try:
        await run_in_threadpool(delete_cloud_token)
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    cloud.invalidate()
    return await run_in_threadpool(cloud.entitlements, fresh=True)


@router.get("/plans")
async def plans() -> dict:
    _require_tiers()
    return await run_in_threadpool(cloud.billing_plans)


@router.post("/checkout")
async def checkout(req: CheckoutRequest) -> dict:
    _require_tiers()
    result = await run_in_threadpool(cloud.checkout, req.plan, req.provider)
    url = str(result.get("checkout_url") or "")
    if urlparse(url).scheme != "https":
        raise HTTPException(502, "Manga Cloud returned an invalid checkout link")
    return result
