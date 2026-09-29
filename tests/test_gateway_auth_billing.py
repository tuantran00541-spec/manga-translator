from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timezone

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Header

from gateway.app import Upstream, create_app
from gateway.billing import BillingConfig, lemonsqueezy_signature, payos_data_signature, payos_request_signature
from gateway.mailer import Mailer
from gateway.store import QuotaExceeded, Store

ADMIN = "admin-secret"
CHECKSUM = "test-checksum-key"
LS_SECRET = "ls-webhook-secret"
DAY = 86400

SDK_WEBHOOK_DATA = {
    "orderCode": 123, "amount": 3000, "description": "VQRIO123", "accountNumber": "12345678",
    "reference": "TF230204212323", "transactionDateTime": "2023-02-04 18:25:00", "currency": "VND",
    "paymentLinkId": "124c33293c43417ab7879e14c8d9eb18", "code": "00", "desc": "Thành công",
    "counterAccountBankId": "", "counterAccountBankName": "", "counterAccountName": None,
    "counterAccountNumber": None, "virtualAccountName": "", "virtualAccountNumber": "",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(app: FastAPI):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server


class Client:
    def __init__(self, app: FastAPI):
        self.base, self._server = _serve(app)

    def get(self, path, **kwargs):
        return requests.get(self.base + path, timeout=10, **kwargs)

    def post(self, path, **kwargs):
        return requests.post(self.base + path, timeout=10, **kwargs)

    def delete(self, path, **kwargs):
        return requests.delete(self.base + path, timeout=10, **kwargs)

    def close(self):
        self._server.should_exit = True


def TestClient(app: FastAPI) -> Client:
    client = Client(app)
    _CLIENTS.append(client)
    return client


_CLIENTS: list[Client] = []


class FakeProviders:
    def __init__(self):
        self.payos_requests: list[dict] = []
        self.ls_requests: list[dict] = []
        app = FastAPI()

        @app.post("/v2/payment-requests")
        def payos(body: dict, x_client_id: str = Header(), x_api_key: str = Header()):
            self.payos_requests.append({"body": body, "client": x_client_id, "key": x_api_key})
            if body["signature"] != payos_request_signature(body, CHECKSUM):
                return {"code": "20", "desc": "bad signature", "data": None}
            return {"code": "00", "desc": "success",
                    "data": {"checkoutUrl": f"https://pay.payos.vn/web/{body['orderCode']}"}}

        @app.post("/v1/checkouts")
        def ls(body: dict, authorization: str = Header()):
            self.ls_requests.append({"body": body, "auth": authorization})
            return {"data": {"attributes": {"url": "https://shop.lemonsqueezy.com/checkout/buy/abc"}}}

        self.url, self._server = _serve(app)

    def close(self):
        self._server.should_exit = True


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    fake = FakeProviders()
    clock = [datetime(2026, 9, 10, 12, tzinfo=timezone.utc).timestamp()]
    store = Store(tmp_path / "gw.sqlite", clock=lambda: clock[0])
    config = BillingConfig(
        payos_base=fake.url, payos_client_id="client", payos_api_key="payos-key", payos_checksum_key=CHECKSUM,
        ls_base=fake.url, ls_api_key="ls-key", ls_store_id="7", ls_variant="11", ls_webhook_secret=LS_SECRET,
    )
    app = create_app(store, Upstream("http://127.0.0.1:9", "", "m", 0, 0), ADMIN,
                     mailer=Mailer(api_key="", sender="", dev_mode=True), billing=config)
    client = TestClient(app)
    yield client, store, clock, fake
    fake.close()
    while _CLIENTS:
        _CLIENTS.pop().close()


def _login(client, email="reader@example.com") -> tuple[str, dict]:
    code = client.post("/v1/auth/start", json={"email": email}).json()["dev_code"]
    token = client.post("/v1/auth/verify", json={"email": email, "code": code}).json()["token"]
    return token, {"Authorization": f"Bearer {token}"}


def _fund(client, headers, amount_usd=5.0) -> str:
    account_id = client.get("/v1/me", headers=headers).json()["account_id"]
    response = client.post(f"/v1/admin/accounts/{account_id}/credit", json={"amount_usd": amount_usd, "note": "test"},
                           headers={"X-Admin-Key": ADMIN})
    assert response.status_code == 200
    return account_id


def _job(client) -> dict:
    _token, headers = _login(client)
    _fund(client, headers)
    return {"Authorization": f"Bearer {client.post('/v1/jobs', headers=headers).json()['job_token']}"}


def test_payos_signatures_match_the_official_sdk():
    assert payos_data_signature(SDK_WEBHOOK_DATA, CHECKSUM) == "15cd38e52473536ad13caec70a9d5fd6446d63812d84e3ab2af193feb5dded64"
    request = {"orderCode": 1234567, "amount": 49000, "description": "MT PLUS 34567",
               "cancelUrl": "http://127.0.0.1:8000/?billing=cancelled", "returnUrl": "http://127.0.0.1:8000/?billing=paid"}
    assert payos_request_signature(request, CHECKSUM) == "05a8a8868610af1504c27de18f3db3304568e8192fe766be1112aa148f285af8"


def test_email_code_login_creates_an_empty_wallet_and_a_revocable_session(world):
    client, _, _, _ = world
    _, headers = _login(client, " Reader@Example.com ")
    me = client.get("/v1/me", headers=headers).json()
    assert me["email"] == "reader@example.com" and (me["balance_usd"], me["chapters_left"]) == (0, 0)
    assert me["fee_percent"] == 5 and me["chapter_estimate_usd"] == pytest.approx(0.315)
    client.post("/v1/auth/logout", headers=headers)
    assert client.get("/v1/me", headers=headers).status_code == 401


def test_two_logins_share_one_account(world):
    client, _, clock, _ = world
    _, first = _login(client)
    clock[0] += 61
    _, second = _login(client)
    assert client.get("/v1/me", headers=first).json()["account_id"] == client.get("/v1/me", headers=second).json()["account_id"]


def test_wrong_codes_lock_the_code_and_resend_is_rate_limited(world):
    client, _, clock, _ = world
    code = client.post("/v1/auth/start", json={"email": "a@example.com"}).json()["dev_code"]
    wrong = f"{(int(code) + 1) % 1_000_000:06d}"
    assert client.post("/v1/auth/start", json={"email": "a@example.com"}).status_code == 429
    for _ in range(5):
        assert client.post("/v1/auth/verify", json={"email": "a@example.com", "code": wrong}).status_code == 400
    assert client.post("/v1/auth/verify", json={"email": "a@example.com", "code": code}).status_code == 429
    clock[0] += 61
    fresh = client.post("/v1/auth/start", json={"email": "a@example.com"}).json()["dev_code"]
    clock[0] += 11 * 60
    assert client.post("/v1/auth/verify", json={"email": "a@example.com", "code": fresh}).status_code == 400


def test_login_without_a_mail_provider_is_refused_outside_dev_mode(world, tmp_path):
    app = create_app(Store(tmp_path / "gw.sqlite"), Upstream("http://127.0.0.1:9", "", "m", 0, 0), ADMIN,
                     mailer=Mailer(api_key="", sender="", dev_mode=False))
    response = TestClient(app).post("/v1/auth/start", json={"email": "a@example.com"})
    assert response.status_code == 503 and "dev_code" not in response.text



def test_unreachable_mail_provider_is_a_503_and_retry_is_not_blocked(world, tmp_path):
    app = create_app(Store(tmp_path / "gw.sqlite"), Upstream("http://127.0.0.1:9", "", "m", 0, 0), ADMIN,
                     mailer=Mailer(api_key="k", sender="s", dev_mode=False, api_base="http://127.0.0.1:9"))
    client = TestClient(app)
    for _ in range(2):
        assert client.post("/v1/auth/start", json={"email": "a@example.com"}).status_code == 503

def _payos_webhook(order_code: int, amount: int, *, key: str = CHECKSUM, code: str = "00") -> dict:
    data = {**SDK_WEBHOOK_DATA, "orderCode": order_code, "amount": amount, "code": code}
    return {"code": "00", "desc": "success", "success": True, "data": data, "signature": payos_data_signature(data, key)}


def test_payos_top_up_credits_the_balance_once(world):
    client, _store, _clock, fake = world
    _, headers = _login(client)
    checkout = client.post("/v1/billing/checkout", json={"provider": "payos", "amount": 52000}, headers=headers).json()
    sent = fake.payos_requests[-1]
    assert (sent["client"], sent["key"]) == ("client", "payos-key") and sent["body"]["amount"] == 52000
    assert checkout["checkout_url"].startswith("https://pay.payos.vn/") and checkout["credit_usd"] == 2.0
    order = checkout["order_code"]

    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 52000, key="forged")).status_code == 401
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 52000)).json()["applied"] is True
    me = client.get("/v1/me", headers=headers).json()
    assert me["balance_usd"] == 2.0 and me["chapters_left"] == 6
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 52000)).json()["applied"] is False
    assert client.get("/v1/me", headers=headers).json()["balance_usd"] == 2.0, "a replayed webhook credits nothing"
    ledger = client.get("/v1/me/ledger", headers=headers).json()["ledger"]
    assert [(line["kind"], line["amount_usd"]) for line in ledger] == [("topup", 2.0)]
    payment = client.get("/v1/me/payments", headers=headers).json()["payments"][0]
    assert (payment["amount"], payment["currency"], payment["status"], payment["credit_usd"]) == (52000, "VND", "paid", 2.0)


def test_payos_webhook_ignores_wrong_amounts_failed_payments_and_unknown_orders(world):
    client, _, _, _ = world
    _, headers = _login(client)
    order = client.post("/v1/billing/checkout", json={"provider": "payos", "amount": 50000}, headers=headers).json()["order_code"]
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 50000, code="01")).json()["applied"] is False
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 1000)).json()["applied"] is False
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 50000)).json()["applied"] is False, \
        "a payment of the wrong amount stays refused"
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(123, 3000)).json() == {"success": True, "applied": False}
    assert client.get("/v1/me", headers=headers).json()["balance_usd"] == 0


def test_the_payment_fee_is_added_to_what_the_user_pays():
    config = BillingConfig(payos_fee_percent=1.0)
    assert config.quote("payos", 100000) == {"currency": "VND", "credit": 100000, "pay": 101000,
                                             "credit_usd": pytest.approx(100000 / 26000)}
    card = BillingConfig().quote("lemonsqueezy", 10)
    assert (card["credit_usd"], card["pay_cents"]) == (10, 1106), "5% + $0.50 of the total is the card fee"
    assert card["pay_cents"] * 0.95 - 50 >= 1000


def test_top_up_amounts_and_providers_are_checked(world, tmp_path):
    client, _, _, _ = world
    _, headers = _login(client)
    for body in ({"provider": "payos", "amount": 15000}, {"provider": "payos", "amount": 50500},
                 {"provider": "lemonsqueezy", "amount": 2.5}, {"provider": "lemonsqueezy", "amount": 1000}):
        assert client.post("/v1/billing/checkout", json=body, headers=headers).status_code == 400, body
    assert client.post("/v1/billing/checkout", json={"provider": "payos", "amount": 50000}).status_code == 401
    topups = client.get("/v1/billing/topups").json()
    assert [quote["credit"] for quote in topups["topups"]["payos"]] == [20000, 50000, 100000, 200000]
    assert topups["providers"] == {"payos": True, "lemonsqueezy": True} and topups["fee_percent"] == 5
    bare = TestClient(create_app(Store(tmp_path / "bare.sqlite"), Upstream("http://127.0.0.1:9", "", "m", 0, 0), ADMIN))
    _, bare_headers = _login(bare)
    assert bare.post("/v1/billing/checkout", json={"provider": "payos", "amount": 50000},
                     headers=bare_headers).status_code == 503
    assert bare.get("/v1/billing/topups").json()["providers"] == {"payos": False, "lemonsqueezy": False}


def _ls(client, event: str, ref: str, key=LS_SECRET, **attributes):
    raw = json.dumps({
        "meta": {"event_name": event, "custom_data": {"account_id": "ignored", "payment_ref": ref}},
        "data": {"type": "orders", "attributes": {"status": "paid", "currency": "USD", **attributes}},
    }).encode()
    return client.post("/v1/billing/lemonsqueezy/webhook", data=raw,
                       headers={"X-Signature": lemonsqueezy_signature(raw, key), "Content-Type": "application/json"})


def test_lemonsqueezy_top_up_is_credited_by_its_order_and_taken_back_on_refund(world):
    client, _, _, fake = world
    _, headers = _login(client)
    account_id = client.get("/v1/me", headers=headers).json()["account_id"]
    checkout = client.post("/v1/billing/checkout", json={"provider": "lemonsqueezy", "amount": 10}, headers=headers).json()
    body = fake.ls_requests[-1]["body"]["data"]
    assert checkout["checkout_url"].startswith("https://") and checkout["pay_cents"] == 1106
    custom = body["attributes"]["checkout_data"]["custom"]
    assert custom["account_id"] == account_id and body["attributes"]["custom_price"] == 1106
    assert body["relationships"]["variant"]["data"]["id"] == "11"
    ref = custom["payment_ref"]

    assert _ls(client, "order_created", ref, key="forged", subtotal=1106).status_code == 401
    assert _ls(client, "order_created", ref, subtotal=1106, status="pending").json()["applied"] is False
    assert _ls(client, "order_created", "unknown", subtotal=1106).json()["applied"] is False
    assert _ls(client, "order_created", ref, subtotal=1106).json()["applied"] is True
    assert client.get("/v1/me", headers=headers).json()["balance_usd"] == 10
    assert _ls(client, "order_created", ref, subtotal=1106).json()["applied"] is False
    assert _ls(client, "order_refunded", ref).json()["applied"] is True
    assert client.get("/v1/me", headers=headers).json()["balance_usd"] == 0
    assert _ls(client, "order_refunded", ref).json()["applied"] is False

    other = client.post("/v1/billing/checkout", json={"provider": "lemonsqueezy", "amount": 2}, headers=headers).json()
    other_ref = fake.ls_requests[-1]["body"]["data"]["attributes"]["checkout_data"]["custom"]["payment_ref"]
    assert _ls(client, "order_created", other_ref, subtotal=other["pay_cents"], currency="EUR").json()["applied"] is False
    assert client.get("/v1/me", headers=headers).json()["balance_usd"] == 0


def test_an_empty_wallet_cannot_start_a_chapter(world):
    client, _, _, _ = world
    _, headers = _login(client)
    refused = client.post("/v1/jobs", headers=headers)
    assert refused.status_code == 402 and refused.json()["error"]["code"] == "insufficient_balance"
    assert "nạp thêm" in refused.json()["error"]["message"]
    _fund(client, headers, 0.31)
    assert client.post("/v1/jobs", headers=headers).status_code == 402, "a chapter needs about $0.32 to start"
    _fund(client, headers, 0.01)
    started = client.post("/v1/jobs", headers=headers).json()
    assert started["cost_cap_usd"] == pytest.approx(0.32 / 1.05, abs=1e-6)


def test_a_request_may_ask_for_more_thinking_and_gets_room_for_it(world, monkeypatch):
    client, _store, _clock, _fake = world
    job = _job(client)
    sent = []

    def send(self, payload, trace=None):
        sent.append(payload)
        return 200, {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    monkeypatch.setattr(Upstream, "send", send)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 1000}
    client.post("/v1/chat/completions", headers=job, json={**body, "reasoning_effort": "low"})
    client.post("/v1/chat/completions", headers=job, json={**body, "reasoning_effort": "extreme"})
    assert (sent[0]["reasoning_effort"], sent[0]["max_tokens"]) == ("low", 1000 + 4096)
    assert "reasoning_effort" not in sent[1] and sent[1]["max_tokens"] == 1000, "unknown efforts are dropped"


def test_the_default_thinking_level_also_gets_room(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(Upstream, "send", lambda self, payload, trace=None: sent.append(payload) or (200, {"usage": {}}))
    store = Store(tmp_path / "gw.sqlite")
    client = TestClient(create_app(store, Upstream("http://127.0.0.1:9", "", "m", 0, 0, reasoning_effort="low"), ADMIN,
                                   mailer=Mailer(api_key="", sender="", dev_mode=True)))
    job = _job(client)
    client.post("/v1/chat/completions", headers=job, json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 500})
    assert (sent[0]["reasoning_effort"], sent[0]["max_tokens"]) == ("low", 500 + 4096)


def test_requests_in_flight_together_cannot_pass_the_cost_cap_or_the_balance(tmp_path):
    store = Store(tmp_path / "gw.sqlite")
    account_id, _token = store.create_account("a@example.com")
    with pytest.raises(QuotaExceeded):
        store.reserve_job(account_id, 0.3)
    store.adjust(account_id, 10, "test")
    job_id, _job_token, cap = store.reserve_job(account_id, 0.3)
    assert cap == 2.0, "a large balance still stops a runaway chapter"
    store.begin_request(job_id, 0.8)
    store.begin_request(job_id, 0.8)
    with pytest.raises(QuotaExceeded):
        store.begin_request(job_id, 0.8)  # nothing is billed yet, but the held cost would pass the guard
    store.add_cost(job_id, 0.2, released_usd=0.8)
    store.begin_request(job_id, 0.8)  # the real cost came in lower, so there is room again

    other, _ = store.create_account("b@example.com")
    store.adjust(other, 1.05, "test")
    first, _, cap = store.reserve_job(other, 0.3)
    second, _, _ = store.reserve_job(other, 0.3)
    assert cap == pytest.approx(1.0), "the cap is what the balance pays for after the fee"
    store.begin_request(first, 0.6)
    with pytest.raises(QuotaExceeded, match="balance"):
        store.begin_request(second, 0.6)  # two chapters at once share one balance


def test_a_chapter_is_charged_at_cost_plus_the_fee_in_one_ledger_line(tmp_path):
    store = Store(tmp_path / "gw.sqlite", fee_rate=0.05)
    account_id, _ = store.create_account("a@example.com")
    store.adjust(account_id, 1.0, "welcome")
    job_id, _, _ = store.reserve_job(account_id, 0.3)
    for _ in range(2):
        store.begin_request(job_id, 0.05)
        store.add_cost(job_id, 0.1, released_usd=0.05)
    assert store.account(account_id)["balance_usd"] == pytest.approx(0.79)
    assert store.account(account_id)["held_micros"] == 0
    assert store.finish_job(job_id, "completed")["charged_usd"] == pytest.approx(0.21)
    assert [(line["kind"], line["amount_usd"], line["balance_usd"]) for line in store.ledger(account_id)] == [
        ("chapter", pytest.approx(-0.21), pytest.approx(0.79)), ("adjust", 1.0, 1.0)]
    assert store.usage(account_id, 0) == {"chapters": 1, "charged_usd": pytest.approx(0.21)}
    stats = store.stats()
    assert (stats["ai_cost_usd"], stats["charged_usd"], stats["margin_usd"]) == (
        pytest.approx(0.2), pytest.approx(0.21), pytest.approx(0.01))


def test_a_database_from_the_plan_version_is_refused(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, email TEXT, plan TEXT)")
    with pytest.raises(RuntimeError, match="new database"):
        Store(path)


def test_the_gateway_forwards_one_answer_and_frees_what_it_held(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(Upstream, "send", lambda self, payload, trace=None: sent.append(payload) or (
        200, {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}))
    store = Store(tmp_path / "gw.sqlite")
    client = TestClient(create_app(store, Upstream("http://127.0.0.1:9", "", "m", 1.0, 1.0), ADMIN,
                                   mailer=Mailer(api_key="", sender="", dev_mode=True)))
    job = _job(client)
    content = [{"type": "text", "text": "x" * 400}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}]
    assert client.post("/v1/chat/completions", headers=job,
                       json={"messages": [{"role": "user", "content": content}], "n": 8, "max_tokens": 500}).status_code == 200
    assert "n" not in sent[0]
    with store._connect() as db:
        row = db.execute("SELECT cost_usd, reserved_usd FROM jobs").fetchone()
    assert row["reserved_usd"] == 0 and row["cost_usd"] == pytest.approx(0.0011)


def test_login_codes_are_capped_per_day_and_sessions_expire(world):
    client, _store, clock, _fake = world
    for _ in range(10):
        assert client.post("/v1/auth/start", json={"email": "t@example.com"}).status_code == 200
        clock[0] += 61
    assert client.post("/v1/auth/start", json={"email": "t@example.com"}).status_code == 429, "ten codes a day"
    clock[0] += DAY
    token, headers = _login(client, "t@example.com")
    assert client.get("/v1/me", headers=headers).status_code == 200
    clock[0] += 91 * DAY
    assert client.get("/v1/me", headers=headers).status_code == 401, "sessions end after 90 days"


def test_one_network_can_make_only_a_few_new_accounts_a_day(world, monkeypatch):
    client, _store, clock, _fake = world
    for n in range(3):
        _login(client, f"reader{n}@example.com")
    code = client.post("/v1/auth/start", json={"email": "farm@example.com"}).json()["dev_code"]
    assert client.post("/v1/auth/verify", json={"email": "farm@example.com", "code": code}).status_code == 429
    clock[0] += 61
    assert _login(client, "reader0@example.com")[0], "existing accounts still sign in"
    clock[0] += DAY
    assert _login(client, "farm@example.com")[0], "the count starts again the next day"
    monkeypatch.setattr("gateway.store.LOGIN_STARTS_PER_IP", 1)
    clock[0] += DAY
    assert client.post("/v1/auth/start", json={"email": "x@example.com"}).status_code == 200
    assert client.post("/v1/auth/start", json={"email": "y@example.com"}).status_code == 429


def test_only_known_fields_and_inline_images_reach_the_upstream(world, monkeypatch):
    client, _store, _clock, _fake = world
    sent = []
    monkeypatch.setattr(Upstream, "send", lambda self, payload, trace=None: sent.append(payload) or (200, {"usage": {}}))
    job = _job(client)
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}
    ok = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, image]}],
          "response_format": {"type": "json_object"}, "tools": [{"type": "function"}], "logprobs": True, "n": 4}
    assert client.post("/v1/chat/completions", headers=job, json=ok).status_code == 200
    assert set(sent[0]) == {"messages", "response_format", "model", "max_tokens"}
    remote = {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
    for bad in ([{"role": "user", "content": [remote]}], [{"role": "tool", "content": "x"}],
                [{"role": "user", "content": [image] * 25}], []):
        assert client.post("/v1/chat/completions", headers=job, json={"messages": bad}).status_code == 400
    assert len(sent) == 1


def test_oversized_bodies_and_runaway_request_counts_are_refused(world, monkeypatch):
    client, _store, _clock, _fake = world
    monkeypatch.setattr(Upstream, "send", lambda self, payload, trace=None: (200, {"usage": {}}))
    job = _job(client)
    body = {"messages": [{"role": "user", "content": "x"}]}
    monkeypatch.setattr("gateway.app.MAX_BODY_BYTES", 10)
    assert client.post("/v1/chat/completions", headers=job, json=body).status_code == 413
    monkeypatch.setattr("gateway.app.MAX_BODY_BYTES", 1024)
    monkeypatch.setattr("gateway.store.JOB_MAX_REQUESTS", 2)
    codes = [client.post("/v1/chat/completions", headers=job, json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 402]


def test_signing_out_everywhere_ends_every_session(world):
    client, _store, clock, _fake = world
    _, first = _login(client)
    clock[0] += 61
    _, second = _login(client)
    assert client.post("/v1/auth/logout-all", headers=first).json()["sessions"] == 2
    assert client.get("/v1/me", headers=first).status_code == 401
    assert client.get("/v1/me", headers=second).status_code == 401


def test_a_deleted_account_forgets_the_email_but_keeps_its_balance(world):
    client, store, clock, _fake = world
    _, headers = _login(client)
    _fund(client, headers, 3)
    assert client.delete("/v1/me", headers=headers).json() == {"ok": True}
    assert client.get("/v1/me", headers=headers).status_code == 401
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM accounts WHERE email LIKE '%reader%'").fetchone()[0] == 0
    clock[0] += 61
    _, again = _login(client)
    assert client.get("/v1/me", headers=again).json()["balance_usd"] == 3


def test_admin_can_look_up_an_account_credit_it_and_see_totals(world):
    client, _store, _clock, _fake = world
    _, headers = _login(client)
    account_id = client.get("/v1/me", headers=headers).json()["account_id"]
    admin = {"X-Admin-Key": ADMIN}
    credit = {"amount_usd": 2, "note": "support"}
    assert client.post(f"/v1/admin/accounts/{account_id}/credit", json=credit).status_code == 403
    assert client.post("/v1/admin/accounts/nobody/credit", json=credit, headers=admin).status_code == 404
    assert client.post(f"/v1/admin/accounts/{account_id}/credit", json=credit, headers=admin).json()["balance_usd"] == 2
    client.post("/v1/jobs", headers=headers)
    assert client.get("/v1/admin/stats").status_code == 403
    found = client.get("/v1/admin/accounts", params={"email": "reader@example.com"}, headers=admin).json()
    assert found["balance_usd"] == 2 and found["payments"] == [] and found["ledger"][0]["ref"] == "support"
    stats = client.get("/v1/admin/stats", headers=admin).json()
    assert (stats["accounts"], stats["balances_owed_usd"], stats["jobs"]) == (1, 2, 1)
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_housekeeping_closes_stale_jobs_and_drops_old_sessions(world, tmp_path):
    client, store, clock, _fake = world
    _, headers = _login(client)
    account_id = _fund(client, headers, 1)
    job_id, _, _ = store.reserve_job(account_id, 0.3)
    store.begin_request(job_id, 0.01)
    store.add_cost(job_id, 0.1, released_usd=0.01)
    clock[0] += 7 * 3600
    assert store.purge()["jobs"] == 1
    assert [line["kind"] for line in store.ledger(account_id)] == ["chapter", "adjust"], "an expired job still books its cost"
    clock[0] += 91 * DAY
    assert store.purge()["sessions"] == 1
    store.backup(tmp_path / "copy.sqlite")
    assert Store(tmp_path / "copy.sqlite").stats()["accounts"] == 1
