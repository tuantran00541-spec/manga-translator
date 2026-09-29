from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gateway.billing import Billing, BillingConfig, BillingError, billing_from_env
from gateway.mailer import MailUnavailable, Mailer, mailer_from_env
from gateway.store import InvalidToken, LoginRejected, QuotaExceeded, Store

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,63}$")
MAX_OUTPUT_TOKENS = 8192
MAX_BODY_BYTES = 32 * 1024 * 1024  # a request carries at most a few JPEG slices
MAX_IMAGES = 24
PURGE_EVERY_S = 6 * 3600
# Only these fields reach the upstream; tools, logprobs and the like would add cost or reach beyond the job.
FORWARDED_FIELDS = frozenset({"messages", "max_tokens", "response_format", "temperature", "top_p", "seed", "stop"})
ROLES = frozenset({"system", "user", "assistant"})
# A client may ask for more thinking on one request; the gateway adds room for it.
REASONING_BUDGETS = {"none": 0, "minimal": 1024, "low": 4096, "medium": 8192, "high": 16384}


STAGES = frozenset({"scan", "glossary", "review", "translate"})  # A.I mode checkpoints a client may name


@dataclass(frozen=True)
class Route:
    """One upstream model and its prices in USD per million tokens."""
    model: str
    input_usd_per_m: float
    output_usd_per_m: float
    cached_usd_per_m: float | None = None

    def cost(self, usage: dict) -> float:
        prompt = max(0, int(usage.get("prompt_tokens") or 0))
        completion = max(0, int(usage.get("completion_tokens") or 0))
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        cached = min(prompt, max(0, int(details.get("cached_tokens") or 0))) if self.cached_usd_per_m is not None else 0
        return ((prompt - cached) * self.input_usd_per_m + cached * (self.cached_usd_per_m or 0.0)
                + completion * self.output_usd_per_m) / 1_000_000


@dataclass(frozen=True)
class Upstream:
    base: str
    api_key: str
    model: str
    input_usd_per_m: float
    output_usd_per_m: float
    reasoning_tokens: int = 0
    reasoning_effort: str = ""  # "low", "medium" or "high"; empty leaves the model's default
    cached_usd_per_m: float | None = None  # price of cache-hit prompt tokens; None bills them as input
    # Extra attempts after a dropped connection, a timeout, 429 or 5xx.
    retries: int = 2
    retry_wait_s: float = 2.0
    stages: dict = field(default_factory=dict)  # checkpoint name -> Route that answers it
    fallbacks: tuple = ()  # Routes tried, in order, when a model is overloaded or failing

    @property
    def default(self) -> Route:
        return Route(self.model, self.input_usd_per_m, self.output_usd_per_m, self.cached_usd_per_m)

    def route(self, stage: str | None) -> Route:
        return self.stages.get(stage) or self.default

    def chain(self, stage: str | None) -> list[Route]:
        """The stage's model, then each fallback that is a different model."""
        first = self.route(stage)
        return [first] + [route for route in self.fallbacks if route.model != first.model]

    def priced(self, model: str | None) -> Route:
        for route in (self.default, *self.stages.values(), *self.fallbacks):
            if route.model == model:
                return route
        return self.default

    def cost(self, usage: dict, model: str | None = None) -> float:
        return self.priced(model).cost(usage)

    def send(self, payload: dict, trace: dict | None = None, models: list[str] | None = None) -> tuple[int, dict]:
        """POST upstream, retrying transient failures on the next model in ``models``; ``trace`` records attempts."""
        models = models or [payload.get("model")]
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            # An overloaded model is not waited on while another one can answer.
            if models[min(attempt, len(models) - 1)] is not None:
                payload = {**payload, "model": models[min(attempt, len(models) - 1)]}
            if trace is not None:
                trace["attempts"] = attempt + 1
                if payload.get("model") is not None:
                    trace["model"] = payload["model"]
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
                if attempt + 1 >= len(models):
                    time.sleep(self.retry_wait_s * 2 ** attempt)
                continue
            if trace is not None:
                trace.setdefault("statuses", []).append(response.status_code)
            if response.status_code in RETRY_STATUSES and not last:
                if attempt + 1 >= len(models):
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


IMAGE_TOKENS_HELD = 3000  # held per image before the upstream reports the real count


def _held_usd(upstream: Route, payload: dict, max_tokens: int) -> float:
    """Most a request is expected to cost: its text and images in, its whole token budget out."""
    chars, images = 0, 0
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if isinstance(part, str):
                chars += len(part)
            elif isinstance(part, dict) and part.get("type") == "image_url":
                images += 1
            elif isinstance(part, dict):
                chars += len(str(part.get("text") or ""))
    prompt = chars / 4 + images * IMAGE_TOKENS_HELD
    return (prompt * upstream.input_usd_per_m + max_tokens * upstream.output_usd_per_m) / 1_000_000


def _trace_line(path: str, entry: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _routes_from_env(name: str) -> list[tuple[str | None, Route]]:
    """Routes from a JSON list of {"stage"?, "model", "price": [input, output, cached?]} in an env var."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    routes = []
    for item in json.loads(raw):
        price = [float(v) for v in item.get("price") or []] + [0.0, 0.0]
        stage = item.get("stage")
        if stage is not None and stage not in STAGES:
            raise ValueError(f"{name}: unknown stage {stage!r}")
        routes.append((stage, Route(str(item["model"]), price[0], price[1],
                                    price[2] if len(item.get("price") or []) > 2 else None)))
    return routes


def upstream_from_env() -> Upstream:
    return Upstream(
        base=os.getenv("GATEWAY_UPSTREAM_BASE", "https://api.deepseek.com"),
        api_key=os.getenv("GATEWAY_UPSTREAM_KEY", ""),
        model=os.getenv("GATEWAY_UPSTREAM_MODEL", "deepseek-v4-flash-vision-exp"),
        input_usd_per_m=float(os.getenv("GATEWAY_PRICE_INPUT_PER_M", "0.28")),
        output_usd_per_m=float(os.getenv("GATEWAY_PRICE_OUTPUT_PER_M", "0.42")),
        reasoning_tokens=max(0, int(os.getenv("GATEWAY_UPSTREAM_REASONING_TOKENS", "0") or 0)),
        reasoning_effort=os.getenv("GATEWAY_UPSTREAM_REASONING_EFFORT", "").strip().lower(),
        cached_usd_per_m=(float(os.environ["GATEWAY_PRICE_CACHED_PER_M"])
                          if os.getenv("GATEWAY_PRICE_CACHED_PER_M", "").strip() else None),
        retries=max(0, int(os.getenv("GATEWAY_UPSTREAM_RETRIES", "2") or 0)),
        stages={stage: route for stage, route in _routes_from_env("GATEWAY_STAGE_MODELS")},
        fallbacks=tuple(route for _stage, route in _routes_from_env("GATEWAY_FALLBACK_MODELS")),
    )


class EmailRequest(BaseModel):
    email: str = Field(max_length=256)


class VerifyRequest(BaseModel):
    email: str = Field(max_length=256)
    code: str = Field(min_length=6, max_length=6, pattern="^[0-9]{6}$")


class CreditRequest(BaseModel):
    amount_usd: float = Field(ge=-1000, le=1000)
    note: str = Field(min_length=1, max_length=200)


class CheckoutRequest(BaseModel):
    provider: str = Field(pattern="^(payos|lemonsqueezy)$")
    amount: float = Field(gt=0, le=100_000_000)


class FinishRequest(BaseModel):
    outcome: str = Field(pattern="^(completed|failed|cancelled)$")


def _bearer(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Missing bearer token")
    return token.strip()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _messages_problem(messages) -> str | None:
    """Why a chat request's messages cannot be forwarded, or None."""
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"
    images = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            return "each message needs a system, user or assistant role"
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            return "message content must be text or a list of parts"
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text" and isinstance(part.get("text"), str):
                continue
            url = (part.get("image_url") or {}).get("url") if kind == "image_url" and isinstance(part.get("image_url"), dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/"):
                return "only text parts and inline data:image URLs are accepted"
            images += 1
    if images > MAX_IMAGES:
        return f"at most {MAX_IMAGES} images per request"
    return None


def client_ip(request: Request) -> str:
    """The caller's address; run uvicorn with --proxy-headers behind a trusted proxy so this is the real one."""
    return request.client.host if request.client else ""


def _email(value: str) -> str:
    email = value.strip().lower()
    if not EMAIL_RE.fullmatch(email):
        raise HTTPException(400, "Email không hợp lệ")
    return email


def create_app(store: Store, upstream: Upstream, admin_key: str, *, mailer: Mailer | None = None,
               billing: BillingConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def purge_forever():
            while True:
                await run_in_threadpool(store.purge)
                await asyncio.sleep(PURGE_EVERY_S)

        task = asyncio.create_task(purge_forever())
        try:
            yield
        finally:
            task.cancel()

    # No generated API docs: the gateway is not a public API to browse.
    app = FastAPI(title="Manga Cloud gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    mailer = mailer or Mailer(api_key="", sender="", dev_mode=True)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            return _error(400, "bad_length", "Invalid Content-Length")
        if length > MAX_BODY_BYTES or (request.method == "POST" and "chunked" in request.headers.get("transfer-encoding", "")):
            return _error(413, "too_large", "Request body is too large")
        return await call_next(request)

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
        chapter = payments.chapter_price_usd()
        return {
            "account_id": row["id"],
            "email": row["email"],
            "balance_usd": row["balance_usd"],
            "available_usd": row["available_usd"],
            "fee_percent": round(store.fee_rate * 100, 2),
            "chapter_estimate_usd": chapter,
            "chapters_left": int(max(0.0, row["available_usd"]) // chapter) if chapter > 0 else None,
            "usage_30d": store.usage(row["id"], store.now() - 30 * 86400),
            "limits": {"model": upstream.model},
        }

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/v1/auth/start")
    def login_start(req: EmailRequest, request: Request) -> dict:
        email = _email(req.email)
        try:
            code = store.start_login(email, client_ip(request))
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
    def login_verify(req: VerifyRequest, request: Request) -> dict:
        try:
            account_id, token = store.verify_login(_email(req.email), req.code, client_ip(request))
        except LoginRejected as exc:
            raise HTTPException(exc.status, exc.message) from exc
        return {"account_id": account_id, "token": token}

    @app.post("/v1/auth/logout")
    def logout(authorization: str | None = Header(default=None)) -> dict:
        store.logout(_bearer(authorization))
        return {"ok": True}

    @app.post("/v1/auth/logout-all")
    def logout_all(row=Depends(account)) -> dict:
        return {"ok": True, "sessions": store.logout_all(row["id"])}

    @app.get("/v1/me/payments")
    def my_payments(row=Depends(account)) -> dict:
        return {"payments": store.payments(row["id"])}

    @app.get("/v1/me/ledger")
    def my_ledger(row=Depends(account)) -> dict:
        return {"ledger": store.ledger(row["id"])}

    @app.delete("/v1/me")
    def delete_me(row=Depends(account)):
        # The balance is kept under a hash of the email and returns if the same email signs in again.
        store.delete_account(row["id"])
        return {"ok": True}

    @app.get("/v1/billing/topups")
    def billing_topups() -> dict:
        return payments.public()

    @app.post("/v1/billing/checkout")
    def checkout(req: CheckoutRequest, row=Depends(account)) -> dict:
        try:
            return payments.checkout(row, req.provider, req.amount)
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
        chapter = payments.chapter_price_usd()
        try:
            job_id, token, cap = store.reserve_job(row["id"], chapter)
        except QuotaExceeded:
            return _error(402, "insufficient_balance",
                          f"Số dư ${row['available_usd']:.2f} chưa đủ cho một chương (khoảng ${chapter:.2f}); "
                          "hãy nạp thêm hoặc dùng key A.I của bạn")
        return {"job_id": job_id, "job_token": token, "cost_cap_usd": cap, "model": upstream.model,
                "entitlements": entitlements(row)}

    @app.post("/v1/jobs/finish")
    def finish(req: FinishRequest, row=Depends(job)) -> dict:
        return store.finish_job(row["id"], req.outcome)

    @app.post("/v1/chat/completions")
    async def chat(payload: dict, row=Depends(job), x_mt_stage: str | None = Header(default=None)):
        if payload.get("stream"):
            return _error(400, "stream_unsupported", "Streaming is not supported")
        problem = _messages_problem(payload.get("messages"))
        if problem:
            return _error(400, "bad_request", problem)
        # One answer per request from the model the gateway picks for the checkpoint; anything else is dropped.
        stage = x_mt_stage if x_mt_stage in STAGES else None
        chain = upstream.chain(stage)
        forwarded = {key: payload[key] for key in FORWARDED_FIELDS if key in payload}
        forwarded["model"] = chain[0].model
        try:
            requested = int(payload.get("max_tokens") or MAX_OUTPUT_TOKENS)
        except (TypeError, ValueError):
            requested = MAX_OUTPUT_TOKENS
        # Add the reasoning budget on top of the client's answer budget.
        extra = upstream.reasoning_tokens
        asked = str(payload.get("reasoning_effort") or "").strip().lower()
        forwarded.pop("reasoning_effort", None)
        if asked in REASONING_BUDGETS:
            forwarded["reasoning_effort"] = asked
            extra = max(extra, REASONING_BUDGETS[asked])
        elif upstream.reasoning_effort:
            forwarded["reasoning_effort"] = upstream.reasoning_effort
            extra = max(extra, REASONING_BUDGETS.get(upstream.reasoning_effort, 0))
        forwarded["max_tokens"] = max(1, min(requested, MAX_OUTPUT_TOKENS)) + extra
        held = max(_held_usd(route, forwarded, forwarded["max_tokens"]) for route in chain)
        try:
            store.begin_request(row["id"], held)
        except QuotaExceeded as exc:
            if str(exc) == "requests":
                return _error(402, "request_cap", "This chapter reached its A.I request cap")
            if str(exc) == "balance":
                return _error(402, "insufficient_balance", "Số dư Manga Cloud đã hết; hãy nạp thêm")
            return _error(402, "cost_cap", "This chapter reached its A.I cost cap")
        trace: dict = {}
        started = time.perf_counter()
        try:
            status, body = await run_in_threadpool(upstream.send, forwarded, trace, [route.model for route in chain])
        except requests.RequestException:
            status, body = 0, {}
        except BaseException:
            store.add_cost(row["id"], 0.0, released_usd=held)
            raise
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        if trace_path:
            # One line per request: where the time and tokens of an A.I run go.
            prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
            completion_details = (usage.get("completion_tokens_details")
                                  if isinstance(usage.get("completion_tokens_details"), dict) else {})
            choices = body.get("choices") if isinstance(body.get("choices"), list) else []
            _trace_line(trace_path, {
                "t": round(time.time(), 3), "ms": round((time.perf_counter() - started) * 1000),
                "status": status, "stage": stage, **trace, **_request_shape(payload),
                "request_kb": round(len(json.dumps(payload)) / 1024, 1), "max_tokens": forwarded["max_tokens"],
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
                "finish_reason": (choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None),
            })
        total = store.add_cost(
            row["id"], upstream.cost(usage, trace.get("model")),
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), released_usd=held,
        )
        if status == 0:
            return _error(502, "upstream_unreachable", "A.I upstream is unreachable")
        if status >= 400:
            detail = body.get("error") if isinstance(body.get("error"), dict) else body
            message = str(detail.get("message") or detail.get("detail") or "")[:200] if isinstance(detail, dict) else ""
            if upstream.api_key:
                message = message.replace(upstream.api_key, "***")
            return _error(status if status in (400, 429) else 502, "upstream_error",
                          f"Upstream HTTP {status}" + (f": {message}" if message else ""))
        body.setdefault("usage", {})["gateway_job_cost_usd"] = round(total, 6)
        return body

    @app.get("/v1/admin/accounts", dependencies=[Depends(admin)])
    def admin_find_account(email: str) -> dict:
        try:
            row = store.account_by_email(_email(email))
        except KeyError as exc:
            raise HTTPException(404, "Account not found") from exc
        return {**entitlements(row), "created_at": row["created_at"], "payments": store.payments(row["id"]),
                "ledger": store.ledger(row["id"])}

    @app.get("/v1/admin/stats", dependencies=[Depends(admin)])
    def admin_stats() -> dict:
        return store.stats()

    @app.post("/v1/admin/accounts", dependencies=[Depends(admin)])
    def admin_create_account(req: EmailRequest) -> dict:
        account_id, token = store.create_account(_email(req.email))
        return {"account_id": account_id, "token": token}

    @app.post("/v1/admin/accounts/{account_id}/credit", dependencies=[Depends(admin)])
    def admin_credit(account_id: str, req: CreditRequest) -> dict:
        try:
            balance = store.adjust(account_id, req.amount_usd, req.note)
        except KeyError as exc:
            raise HTTPException(404, "Account not found") from exc
        return {"account_id": account_id, "balance_usd": balance}

    return app


def app_from_env() -> FastAPI:
    db_path = Path(os.getenv("GATEWAY_DB", "gateway-data/gateway.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fee_rate = float(os.getenv("GATEWAY_FEE_PERCENT", "5")) / 100
    return create_app(Store(db_path, fee_rate=fee_rate), upstream_from_env(), os.getenv("GATEWAY_ADMIN_KEY", ""),
                      mailer=mailer_from_env(), billing=billing_from_env())
