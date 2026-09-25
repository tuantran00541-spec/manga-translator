from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import cloud
from app.secret_store import SecretStoreUnavailable, delete_cloud_token, set_cloud_token

router = APIRouter(prefix="/api/account", tags=["account"])


class CloudTokenRequest(BaseModel):
    token: str = Field(min_length=8, max_length=512)


@router.get("")
async def account_status() -> dict:
    return await run_in_threadpool(cloud.entitlements, fresh=True)


@router.post("/token")
async def save_cloud_token(req: CloudTokenRequest) -> dict:
    try:
        await run_in_threadpool(set_cloud_token, req.token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    cloud.invalidate()
    return await run_in_threadpool(cloud.entitlements, fresh=True)


@router.delete("/token")
async def delete_saved_cloud_token() -> dict:
    try:
        await run_in_threadpool(delete_cloud_token)
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    cloud.invalidate()
    return await run_in_threadpool(cloud.entitlements, fresh=True)
