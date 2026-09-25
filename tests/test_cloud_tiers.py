from __future__ import annotations

import asyncio
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests
import uvicorn
from fastapi import FastAPI, HTTPException

import app.routers.ai_mode as ai_mode_router
from app import cloud
from app.ai_mode.job import AIModeJobManager
from app.ai_mode.vision_json import request_vision_json
from gateway.app import Upstream, create_app
from gateway.store import Store

ADMIN = "admin-secret"
UPSTREAM_KEY = "upstream-secret"
CALL_COST = (100_000 * 0.28 + 10_000 * 0.42) / 1_000_000


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(app: FastAPI) -> tuple[str, uvicorn.Server]:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server


class Stack:
    def __init__(self, tmp_path):
        self.clock = [time.mktime((2026, 9, 10, 12, 0, 0, 0, 0, 0))]
        self.upstream_calls: list[dict] = []
        upstream = FastAPI()

        @upstream.post("/chat/completions")
        def completions(payload: dict):
            self.upstream_calls.append(payload)
            return {
                "model": payload["model"],
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 100_000, "completion_tokens": 10_000},
            }

        self.upstream_url, self._up = _serve(upstream)
        self.store = Store(tmp_path / "gateway.sqlite", clock=lambda: self.clock[0])
        gateway = create_app(
            self.store,
            Upstream(self.upstream_url, UPSTREAM_KEY, "vision-upstream", 0.28, 0.42),
            ADMIN,
        )
        self.url, self._gw = _serve(gateway)
        self.api = f"{self.url}/v1"

    def signup(self, email="reader@example.com") -> tuple[str, str]:
        body = requests.post(f"{self.api}/admin/accounts", json={"email": email},
                             headers={"X-Admin-Key": ADMIN}, timeout=5).json()
        return body["account_id"], body["token"]

    def set_plan(self, account_id: str, plan: str, key: str = ADMIN) -> requests.Response:
        return requests.post(
            f"{self.api}/admin/accounts/{account_id}/plan", json={"plan": plan},
            headers={"X-Admin-Key": key}, timeout=5,
        )

    def chat(self, token: str) -> requests.Response:
        return requests.post(
            f"{self.api}/chat/completions", headers={"Authorization": f"Bearer {token}"},
            json={"model": "anything-expensive", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 999_999},
            timeout=10,
        )

    def close(self) -> None:
        self._gw.should_exit = True
        self._up.should_exit = True


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    s = Stack(tmp_path)
    monkeypatch.setenv("MANGA_TIERS", "1")
    monkeypatch.setenv("MANGA_CLOUD_URL", s.api)
    cloud.invalidate()
    yield s
    cloud.invalidate()
    s.close()


def _sign_in(stack, monkeypatch, email="reader@example.com") -> str:
    account_id, token = stack.signup(email)
    monkeypatch.setenv("MANGA_CLOUD_TOKEN", token)
    cloud.invalidate()
    return account_id


def test_free_plan_allows_three_chapters_then_refuses_with_402(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    assert cloud.entitlements()["quota"]["remaining"] == 3
    for _ in range(3):
        cloud.reserve_job()
    with pytest.raises(HTTPException) as refused:
        cloud.reserve_job()
    assert refused.value.status_code == 402
    assert "Free" in refused.value.detail
    quota = cloud.entitlements(fresh=True)["quota"]
    assert (quota["used"], quota["remaining"]) == (3, 0)


def test_only_a_live_job_token_reaches_the_ai_and_the_gateway_picks_the_model(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    account_token = cloud._token()
    assert stack.chat(account_token).status_code == 401, "an account token alone must not buy A.I calls"
    job = cloud.reserve_job()
    ok = stack.chat(job["job_token"])
    assert ok.status_code == 200
    sent = stack.upstream_calls[-1]
    assert sent["model"] == "vision-upstream", "clients cannot pick a pricier model"
    assert sent["max_tokens"] == 8192
    cloud.finish_job(job["job_token"], "completed")
    assert stack.chat(job["job_token"]).status_code == 401, "a finished job cannot be reused"
    assert stack.chat("mcj_forged").status_code == 401


def test_cost_cap_stops_one_chapter_from_draining_the_plan(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    job = cloud.reserve_job()
    assert job["cost_cap_usd"] == 0.10
    statuses = [stack.chat(job["job_token"]).status_code for _ in range(6)]
    allowed = int(np.ceil(0.10 / CALL_COST))
    assert statuses == [200] * allowed + [402] * (6 - allowed)
    assert cloud.entitlements(fresh=True)["quota"]["cost_usd"] == pytest.approx(allowed * CALL_COST)


def test_a_failed_job_is_refunded_only_when_it_never_called_the_ai(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    unused = cloud.reserve_job()
    assert cloud.finish_job(unused["job_token"], "failed")["refunded"] is True
    spent = cloud.reserve_job()
    stack.chat(spent["job_token"])
    assert cloud.finish_job(spent["job_token"], "failed")["refunded"] is False
    assert cloud.entitlements(fresh=True)["quota"]["used"] == 1


def test_parallel_reservations_never_exceed_the_quota(stack, monkeypatch):
    _sign_in(stack, monkeypatch)

    def attempt(_):
        try:
            cloud.reserve_job()
            return "ok"
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(12) as pool:
        results = list(pool.map(attempt, range(24)))
    assert results.count("ok") == 3
    assert results.count(402) == 21


def test_upgrade_unlocks_quota_and_features(stack, monkeypatch):
    account_id = _sign_in(stack, monkeypatch)
    with pytest.raises(HTTPException) as locked:
        cloud.require_feature("visual_qc")
    assert locked.value.status_code == 402
    assert stack.set_plan(account_id, "pro", key="wrong").status_code == 403
    assert stack.set_plan(account_id, "plus").status_code == 200
    cloud.invalidate()
    cloud.require_feature("visual_qc")
    with pytest.raises(HTTPException):
        cloud.require_feature("byok")
    data = cloud.entitlements()
    assert data["plan"] == "plus" and data["quota"]["remaining"] == 30


def test_quota_resets_next_month_and_stale_unused_reservations_do_not_count(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    for _ in range(3):
        cloud.reserve_job()
    stack.clock[0] += 7 * 3600
    assert cloud.entitlements(fresh=True)["quota"]["used"] == 0, "abandoned jobs with no A.I calls expire"
    for _ in range(3):
        stack.chat(cloud.reserve_job()["job_token"])
    stack.clock[0] += 31 * 86400
    quota = cloud.entitlements(fresh=True)["quota"]
    assert quota["period"] == "2026-10" and quota["remaining"] == 3


def test_tiers_off_keeps_every_feature_open(monkeypatch):
    monkeypatch.setenv("MANGA_TIERS", "0")
    cloud.invalidate()
    cloud.require_feature("visual_qc")
    cloud.require_feature("byok")
    assert cloud.entitlements()["plan"] == "unlimited"


def test_offline_gateway_falls_back_to_free_features(monkeypatch):
    monkeypatch.setenv("MANGA_TIERS", "1")
    monkeypatch.setenv("MANGA_CLOUD_URL", f"http://127.0.0.1:{_free_port()}/v1")
    monkeypatch.setenv("MANGA_CLOUD_TOKEN", "mc_whatever")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    cloud.invalidate()
    assert cloud.entitlements()["offline"] is True
    with pytest.raises(HTTPException):
        cloud.require_feature("visual_qc")
    with pytest.raises(HTTPException) as down:
        cloud.reserve_job()
    assert down.value.status_code == 503


class CloudCallingRunner:
    def __init__(self, job, provider, api_key):
        self.job, self.provider, self.api_key = job, provider, api_key

    def __getattr__(self, name):
        async def stage():
            if name == "scan":
                image = np.full((32, 32, 3), 255, np.uint8)
                result = await asyncio.to_thread(
                    request_vision_json, self.provider, self.job.settings.model, self.api_key,
                    "scan", [("slice 0", image)],
                )
                assert result.data == {"ok": True}
            if name == "translate":
                from app.routers.translation import _resolve_vision_provider
                from app.routers.visual_qc import _resolve_configured_provider
                from app.secret_store import get_provider_api_key

                assert _resolve_vision_provider("manga-cloud") is self.provider
                assert _resolve_configured_provider("manga-cloud") is self.provider
                assert await asyncio.to_thread(get_provider_api_key, "manga-cloud") == self.api_key
        return stage


def _start(req):
    async def scenario():
        snapshot = await ai_mode_router.start_ai_mode(req)
        job = ai_mode_router.ai_mode_jobs._jobs[snapshot["job_id"]]
        await job.task
        return ai_mode_router.ai_mode_jobs.snapshot(snapshot["job_id"])
    return asyncio.run(scenario())


def test_ai_mode_runs_through_the_gateway_and_spends_one_chapter(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    monkeypatch.setattr(ai_mode_router, "validate_url", lambda url: url)
    monkeypatch.setattr(ai_mode_router, "ai_mode_jobs", AIModeJobManager(runner_factory=CloudCallingRunner))
    snapshot = _start(ai_mode_router.AIModeStartRequest(url="https://example.com/c/1", provider="manga-cloud"))
    assert snapshot["status"] == "completed"
    assert stack.upstream_calls[-1]["model"] == "vision-upstream"
    quota = cloud.entitlements(fresh=True)["quota"]
    assert quota["used"] == 1 and quota["cost_usd"] == pytest.approx(CALL_COST)
    with stack.store._connect() as db:
        assert [row["status"] for row in db.execute("SELECT status FROM jobs")] == ["completed"]


def test_manga_cloud_is_unusable_outside_an_ai_mode_job(stack, monkeypatch):
    from app.routers.translation import _resolve_vision_provider
    from app.secret_store import get_provider_api_key

    _sign_in(stack, monkeypatch)
    with pytest.raises(ValueError):
        _resolve_vision_provider("manga-cloud")
    assert get_provider_api_key("manga-cloud") is None


def test_free_plan_cannot_bypass_the_quota_with_its_own_key(stack, monkeypatch):
    _sign_in(stack, monkeypatch)
    monkeypatch.setattr(ai_mode_router, "validate_url", lambda url: url)
    with pytest.raises(HTTPException) as locked:
        _start(ai_mode_router.AIModeStartRequest(url="https://example.com/c/1", provider="openai"))
    assert locked.value.status_code == 402


def test_app_login_with_email_code_stores_the_session_and_checkout_reports_missing_billing(stack, monkeypatch):
    import app.routers.account as account_router

    saved = {}

    def save(token):
        saved["token"] = token
        monkeypatch.setenv("MANGA_CLOUD_TOKEN", token)

    monkeypatch.delenv("MANGA_CLOUD_TOKEN", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setattr(account_router, "set_cloud_token", save)
    monkeypatch.setattr(account_router, "delete_cloud_token", lambda: monkeypatch.delenv("MANGA_CLOUD_TOKEN"))

    sent = asyncio.run(account_router.start_login(account_router.LoginStartRequest(email="Reader@Example.com")))
    with pytest.raises(HTTPException) as wrong:
        asyncio.run(account_router.verify_login(account_router.LoginVerifyRequest(
            email=sent["email"], code=f"{(int(sent['dev_code']) + 1) % 1_000_000:06d}")))
    assert wrong.value.status_code == 400
    me = asyncio.run(account_router.verify_login(account_router.LoginVerifyRequest(email=sent["email"], code=sent["dev_code"])))
    assert me["signed_in"] and me["email"] == "reader@example.com" and me["quota"]["remaining"] == 3
    assert saved["token"].startswith("mc_")

    with pytest.raises(HTTPException) as unavailable:
        asyncio.run(account_router.checkout(account_router.CheckoutRequest(plan="plus", provider="payos")))
    assert unavailable.value.status_code == 503

    out = asyncio.run(account_router.logout())
    assert out["signed_in"] is False
    assert stack.chat(saved["token"]).status_code == 401
