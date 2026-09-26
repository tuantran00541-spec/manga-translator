from __future__ import annotations

import hmac
import json
import os
import re
import time
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
    reasoning_tokens: int = 0
    # Extra attempts after a dropped connection, a timeout, 429 or 5xx.
    retries: int = 2
    retry_wait_s: float = 2.0

    def cost(self, usage: dict) -> float:
        prompt = max(0, int(usage.get("prompt_tokens") or 0))
        completion = max(0, int(usage.get("completion_tokens") or 0))
        return (prompt * self.input_usd_per_m + completion * self.output_usd_per_m) / 1_000_000

    def send(self, payload: dict, trace: dict | None = None) -> tuple[int, dict]:
        """POST upstream, retrying transient failures; ``trace`` records attempts."""
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            if trace is not None:
                trace["attempts"] = attempt + 1
            try:
                response = requests.post(
                    f"{self.base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=(10, 300),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if trace is not None:
                    trace.setdefault("statuses", []).append(type(exc).__name__)
                if last:
                    raise
                time.sleep(self.retry_wait_s * 2 ** attempt)
                continue
            if trace is not None:
                trace.setdefault("statuses", []).append(response.status_code)
            if response.status_code in RETRY_STATUSES and not last:
                time.sleep(min(RETRY_AFTER_MAX_S, _retry_after(response) or self.retry_wait_s * 2 ** attempt))
                continue
            break
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": "upstream returned no JSON"}}
        return response.status_code, body if isinstance(body, dict) else {}


RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_AFTER_MAX_S = 20.0


def _retry_after(response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return None


def _request_shape(payload: dict) -> dict:
    """What a request carries: the opening words of its prompt, its images and size."""
    head, images = "", 0
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                images += 1
            elif part.get("type") == "text" and not head:
                head = str(part.get("text") or "")
    return {"prompt_head": " ".join(head.split())[:80], "images": images}


def _trace_line(path: str, entry: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def upstream_from_env() -> Upstream:
    return Upstream(
        base=os.getenv("GATEWAY_UPSTREAM_BASE", "https://api.deepseek.com"),
        api_key=os.getenv("GATEWAY_UPSTREAM_KEY", ""),
        model=os.getenv("GATEWAY_UPSTREAM_MODEL", "deepseek-v4-flash-vision-exp"),
        input_usd_per_m=float(os.getenv("GATEWAY_PRICE_INPUT_PER_M", "0.28")),
        output_usd_per_m=float(os.getenv("GATEWAY_PRICE_OUTPUT_PER_M", "0.42")),
        reasoning_tokens=max(0, int(os.getenv("GATEWAY_UPSTREAM_REASONING_TOKENS", "0") or 0)),
        retries=max(0, int(os.getenv("GATEWAY_UPSTREAM_RETRIES", "2") or 0)),
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
    # GATEWAY_TRACE_PATH: append a JSON line per upstream request (timing, size, tokens).
    trace_path = os.getenv("GATEWAY_TRACE_PATH", "")

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
            store.cancel_login(email)
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
        # Add the reasoning budget on top of the client's answer budget.
        forwarded["max_tokens"] = max(1, min(requested, MAX_OUTPUT_TOKENS)) + upstream.reasoning_tokens
        trace: dict = {}
        started = time.perf_counter()
        try:
            status, body = await run_in_threadpool(upstream.send, forwarded, trace)
        except requests.RequestException:
            status, body = 0, {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        if trace_path:
            # One line per request: where the time and tokens of an A.I run go.
            prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
            completion_details = (usage.get("completion_tokens_details")
                                  if isinstance(usage.get("completion_tokens_details"), dict) else {})
            choices = body.get("choices") if isinstance(body.get("choices"), list) else []
            _trace_line(trace_path, {
                "t": round(time.time(), 3), "ms": round((time.perf_counter() - started) * 1000),
                "status": status, **trace, **_request_shape(payload),
                "request_kb": round(len(json.dumps(payload)) / 1024, 1), "max_tokens": forwarded["max_tokens"],
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
                "finish_reason": (choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None),
            })
        if status == 0:
            return _error(502, "upstream_unreachable", "A.I upstream is unreachable")
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
