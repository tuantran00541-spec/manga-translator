from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gateway.billing import Billing, BillingConfig, BillingError, billing_from_env
from gateway.mailer import MailUnavailable, Mailer, mailer_from_env
from gateway.plans import PLANS, get_plan
from gateway.store import InvalidToken, LoginRejected, QuotaExceeded, Store

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,63}$")
MAX_OUTPUT_TOKENS = 8192


@dataclass(frozen=True)
class Upstream:
    base: str
    api_key: str
    model: str
    input_usd_per_m: float
    output_usd_per_m: float

    def cost(self, usage: dict) -> float:
        prompt = max(0, int(usage.get("prompt_tokens") or 0))
        completion = max(0, int(usage.get("completion_tokens") or 0))
        return (prompt * self.input_usd_per_m + completion * self.output_usd_per_m) / 1_000_000

    def send(self, payload: dict) -> tuple[int, dict]:
        response = requests.post(
            f"{self.base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 300),
            allow_redirects=False,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": "upstream returned no JSON"}}
        return response.status_code, body if isinstance(body, dict) else {}


def upstream_from_env() -> Upstream:
    return Upstream(
        base=os.getenv("GATEWAY_UPSTREAM_BASE", "https://api.deepseek.com"),
        api_key=os.getenv("GATEWAY_UPSTREAM_KEY", ""),
        model=os.getenv("GATEWAY_UPSTREAM_MODEL", "deepseek-v4-flash-vision-exp"),
        input_usd_per_m=float(os.getenv("GATEWAY_PRICE_INPUT_PER_M", "0.28")),
        output_usd_per_m=float(os.getenv("GATEWAY_PRICE_OUTPUT_PER_M", "0.42")),
    )


class EmailRequest(BaseModel):
    email: str = Field(max_length=256)


class VerifyRequest(BaseModel):
    email: str = Field(max_length=256)
    code: str = Field(min_length=6, max_length=6, pattern="^[0-9]{6}$")


class PlanRequest(BaseModel):
    plan: str
    days: int | None = Field(default=None, ge=1, le=3660)


class CheckoutRequest(BaseModel):
    plan: str
    provider: str = Field(pattern="^(payos|lemonsqueezy)$")


class FinishRequest(BaseModel):
    outcome: str = Field(pattern="^(completed|failed|cancelled)$")


def _bearer(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Missing bearer token")
    return token.strip()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _email(value: str) -> str:
    email = value.strip().lower()
    if not EMAIL_RE.fullmatch(email):
        raise HTTPException(400, "Email không hợp lệ")
    return email


def create_app(store: Store, upstream: Upstream, admin_key: str, *, mailer: Mailer | None = None,
               billing: BillingConfig | None = None) -> FastAPI:
    app = FastAPI(title="Manga Cloud gateway")
    mailer = mailer or Mailer(api_key="", sender="", dev_mode=True)
    payments = Billing(store, billing or BillingConfig())

    def account(authorization: str | None = Header(default=None)):
        try:
            return store.account_for_token(_bearer(authorization))
        except InvalidToken as exc:
            raise HTTPException(401, "Invalid account token") from exc

    def job(authorization: str | None = Header(default=None)):
        try:
            return store.active_job_for_token(_bearer(authorization))
        except InvalidToken as exc:
            raise HTTPException(401, "Invalid or finished job token") from exc

    def admin(x_admin_key: str | None = Header(default=None)) -> None:
        if not admin_key or not hmac.compare_digest(x_admin_key or "", admin_key):
            raise HTTPException(403, "Admin key required")

    def entitlements(row) -> dict:
        plan = get_plan(row["plan"])
        usage = store.usage(row["id"])
        return {
            "account_id": row["id"],
            "email": row["email"],
            "plan": plan.id,
            "plan_label": plan.label,
            "plan_expires_at": row.get("plan_expires_at"),
            "features": sorted(plan.features),
            "quota": {
                "period": usage["period"],
                "limit": plan.chapters_per_month,
                "used": usage["used"],
                "remaining": max(0, plan.chapters_per_month - usage["used"]),
                "cost_usd": usage["cost_usd"],
            },
            "limits": {"max_cost_per_chapter_usd": plan.max_cost_per_chapter_usd, "model": upstream.model},
        }

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "plans": sorted(PLANS)}

    @app.post("/v1/auth/start")
    def login_start(req: EmailRequest) -> dict:
        email = _email(req.email)
        try:
            code = store.start_login(email)
            mailer.send_login_code(email, code)
        except LoginRejected as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except MailUnavailable as exc:
            raise HTTPException(503, "Không gửi được email đăng nhập") from exc
        result = {"sent": True, "email": email}
        if mailer.dev_mode and not mailer.api_key:
            result["dev_code"] = code
        return result

    @app.post("/v1/auth/verify")
    def login_verify(req: VerifyRequest) -> dict:
        try:
            account_id, token = store.verify_login(_email(req.email), req.code)
        except LoginRejected as exc:
            raise HTTPException(exc.status, exc.message) from exc
        return {"account_id": account_id, "token": token}

    @app.post("/v1/auth/logout")
    def logout(authorization: str | None = Header(default=None)) -> dict:
        store.logout(_bearer(authorization))
        return {"ok": True}

    @app.get("/v1/billing/plans")
    def billing_plans() -> dict:
        return payments.config.public()

    @app.post("/v1/billing/checkout")
    def checkout(req: CheckoutRequest, row=Depends(account)) -> dict:
        try:
            return payments.checkout(row, req.plan, req.provider)
        except BillingError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.post("/v1/billing/payos/webhook")
    def payos_webhook(body: dict):
        try:
            return payments.payos_webhook(body)
        except BillingError as exc:
            return _error(exc.status, "webhook_rejected", exc.message)

    @app.post("/v1/billing/lemonsqueezy/webhook")
    async def lemonsqueezy_webhook(request: Request, x_signature: str | None = Header(default=None)):
        raw = await request.body()
        try:
            return payments.lemonsqueezy_webhook(raw, x_signature or "")
        except BillingError as exc:
            return _error(exc.status, "webhook_rejected", exc.message)

    @app.get("/v1/me")
    def me(row=Depends(account)) -> dict:
        return entitlements(row)

    @app.post("/v1/jobs")
    def reserve(row=Depends(account)):
        try:
            job_id, token, cap = store.reserve_job(row["id"])
        except QuotaExceeded:
            plan = get_plan(row["plan"])
            return _error(402, "quota_exceeded",
                          f"Gói {plan.label} đã dùng hết {plan.chapters_per_month} chương A.I mode tháng này")
        return {"job_id": job_id, "job_token": token, "cost_cap_usd": cap, "model": upstream.model,
                "entitlements": entitlements(row)}

    @app.post("/v1/jobs/finish")
    def finish(req: FinishRequest, row=Depends(job)) -> dict:
        return store.finish_job(row["id"], req.outcome)

    @app.post("/v1/chat/completions")
    async def chat(payload: dict, row=Depends(job)):
        if payload.get("stream"):
            return _error(400, "stream_unsupported", "Streaming is not supported")
        try:
            store.begin_request(row["id"])
        except QuotaExceeded:
            return _error(402, "cost_cap", "This chapter reached its A.I cost cap")
        forwarded = dict(payload)
        forwarded["model"] = upstream.model
        try:
            requested = int(payload.get("max_tokens") or MAX_OUTPUT_TOKENS)
        except (TypeError, ValueError):
            requested = MAX_OUTPUT_TOKENS
        forwarded["max_tokens"] = max(1, min(requested, MAX_OUTPUT_TOKENS))
        try:
            status, body = await run_in_threadpool(upstream.send, forwarded)
        except requests.RequestException:
            return _error(502, "upstream_unreachable", "A.I upstream is unreachable")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        total = store.add_cost(
            row["id"], upstream.cost(usage),
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
        )
        if status >= 400:
            detail = body.get("error") if isinstance(body.get("error"), dict) else body
            message = str(detail.get("message") or detail.get("detail") or "")[:200] if isinstance(detail, dict) else ""
            if upstream.api_key:
                message = message.replace(upstream.api_key, "***")
            return _error(status if status in (400, 429) else 502, "upstream_error",
                          f"Upstream HTTP {status}" + (f": {message}" if message else ""))
        body.setdefault("usage", {})["gateway_job_cost_usd"] = round(total, 6)
        return body

    @app.post("/v1/admin/accounts", dependencies=[Depends(admin)])
    def admin_create_account(req: EmailRequest) -> dict:
        account_id, token = store.create_account(_email(req.email))
        return {"account_id": account_id, "token": token}

    @app.post("/v1/admin/accounts/{account_id}/plan", dependencies=[Depends(admin)])
    def set_plan(account_id: str, req: PlanRequest) -> dict:
        expires = store.now() + req.days * 86400 if req.days else None
        try:
            store.set_plan(account_id, req.plan, expires)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Account not found") from exc
        return {"account_id": account_id, "plan": req.plan, "plan_expires_at": expires}

    return app


def app_from_env() -> FastAPI:
    db_path = Path(os.getenv("GATEWAY_DB", "gateway-data/gateway.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_app(Store(db_path), upstream_from_env(), os.getenv("GATEWAY_ADMIN_KEY", ""),
                      mailer=mailer_from_env(), billing=billing_from_env())
