from __future__ import annotations

import os
import threading
import time

import requests
from fastapi import HTTPException

from app.ai_providers import CLOUD_PROVIDER_ID, AIProvider
from app.logging_config import logger
from app.secret_store import SecretStoreUnavailable, get_cloud_token

ALL_FEATURES = ("visual_qc", "byok", "custom_providers")
ENTITLEMENT_TTL_SECONDS = 30
TIMEOUT = (5, 15)

_cache: dict = {}
_cache_lock = threading.Lock()


def tiers_enabled() -> bool:
    return os.getenv("MANGA_TIERS", "0") == "1"


def cloud_base() -> str:
    return os.getenv("MANGA_CLOUD_URL", "http://127.0.0.1:8100/v1").rstrip("/")


def cloud_provider(model: str = "") -> AIProvider:
    return AIProvider(
        id=CLOUD_PROVIDER_ID,
        label="Manga Cloud",
        protocol="openai",
        api_base=cloud_base(),
        default_qc_model=model or "manga-cloud",
        default_translation_model=model or "manga-cloud",
        env_names=(),
        request_profile="standard",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=False,
        builtin=True,
    )


def _token() -> str | None:
    try:
        return get_cloud_token()
    except SecretStoreUnavailable:
        return None


def _message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:300]
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])[:300]
    return f"HTTP {response.status_code}"


def _call(method: str, path: str, token: str, json: dict | None = None) -> requests.Response:
    return requests.request(
        method, f"{cloud_base()}{path}", headers={"Authorization": f"Bearer {token}"},
        json=json, timeout=TIMEOUT, allow_redirects=False,
    )


def invalidate() -> None:
    with _cache_lock:
        _cache.clear()


def entitlements(*, fresh: bool = False) -> dict:
    if not tiers_enabled():
        return {"tiers": False, "signed_in": False, "plan": "unlimited", "features": list(ALL_FEATURES)}
    token = _token()
    if not token:
        return {"tiers": True, "signed_in": False, "plan": "free", "features": [], "quota": None}
    with _cache_lock:
        cached = _cache.get(token)
        if cached and not fresh and time.monotonic() - cached[0] < ENTITLEMENT_TTL_SECONDS:
            return cached[1]
    try:
        response = _call("GET", "/me", token)
    except requests.RequestException:
        return {"tiers": True, "signed_in": True, "offline": True, "plan": "free", "features": [], "quota": None}
    if response.status_code == 401:
        return {"tiers": True, "signed_in": False, "invalid_token": True, "plan": "free", "features": [], "quota": None}
    if not response.ok:
        return {"tiers": True, "signed_in": True, "offline": True, "plan": "free", "features": [], "quota": None,
                "detail": _message(response)}
    data = {"tiers": True, "signed_in": True, **response.json()}
    with _cache_lock:
        _cache[token] = (time.monotonic(), data)
    return data


def require_feature(feature: str) -> None:
    if not tiers_enabled():
        return
    data = entitlements()
    if feature not in (data.get("features") or []):
        raise HTTPException(402, f"Tính năng này cần gói cao hơn (gói hiện tại: {data.get('plan_label') or data.get('plan')})")


def reserve_job() -> dict:
    token = _token()
    if not token:
        raise HTTPException(401, "Chưa đăng nhập Manga Cloud")
    try:
        response = _call("POST", "/jobs", token)
    except requests.RequestException as exc:
        raise HTTPException(503, "Không kết nối được Manga Cloud") from exc
    invalidate()
    if response.status_code in (401, 402):
        raise HTTPException(response.status_code, _message(response))
    if not response.ok:
        raise HTTPException(502, f"Manga Cloud: {_message(response)}")
    return response.json()


def finish_job(job_token: str, outcome: str) -> dict | None:
    try:
        response = _call("POST", "/jobs/finish", job_token, {"outcome": outcome})
    except requests.RequestException:
        logger.warning("Manga Cloud finish call failed; the gateway expires the job on its own")
        return None
    invalidate()
    return response.json() if response.ok else None


def _public(method: str, path: str, json: dict | None = None, token: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.request(method, f"{cloud_base()}{path}", headers=headers, json=json,
                                    timeout=TIMEOUT, allow_redirects=False)
    except requests.RequestException as exc:
        raise HTTPException(503, "Không kết nối được Manga Cloud") from exc
    if not response.ok:
        status = response.status_code if response.status_code in (400, 401, 402, 409, 429, 503) else 502
        raise HTTPException(status, _message(response))
    return response.json()


def login_start(email: str) -> dict:
    return _public("POST", "/auth/start", {"email": email})


def login_verify(email: str, code: str) -> dict:
    return _public("POST", "/auth/verify", {"email": email, "code": code})


def logout() -> None:
    token = _token()
    if token:
        try:
            _public("POST", "/auth/logout", token=token)
        except HTTPException:
            logger.warning("Manga Cloud logout call failed; the local token is removed anyway")
    invalidate()


def billing_plans() -> dict:
    return _public("GET", "/billing/plans")


def checkout(plan: str, provider: str) -> dict:
    token = _token()
    if not token:
        raise HTTPException(401, "Chưa đăng nhập Manga Cloud")
    return _public("POST", "/billing/checkout", {"plan": plan, "provider": provider}, token=token)
