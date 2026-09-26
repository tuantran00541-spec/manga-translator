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
from gateway.store import Store

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
        ls_base=fake.url, ls_api_key="ls-key", ls_store_id="7", ls_variants={"plus": "11", "pro": "22"},
        ls_webhook_secret=LS_SECRET,
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


def test_payos_signatures_match_the_official_sdk():
    assert payos_data_signature(SDK_WEBHOOK_DATA, CHECKSUM) == "15cd38e52473536ad13caec70a9d5fd6446d63812d84e3ab2af193feb5dded64"
    request = {"orderCode": 1234567, "amount": 49000, "description": "MT PLUS 34567",
               "cancelUrl": "http://127.0.0.1:8000/?billing=cancelled", "returnUrl": "http://127.0.0.1:8000/?billing=paid"}
    assert payos_request_signature(request, CHECKSUM) == "05a8a8868610af1504c27de18f3db3304568e8192fe766be1112aa148f285af8"


def test_email_code_login_creates_a_free_account_and_a_revocable_session(world):
    client, _, _, _ = world
    _, headers = _login(client, " Reader@Example.com ")
    me = client.get("/v1/me", headers=headers).json()
    assert me["email"] == "reader@example.com" and me["plan"] == "free"
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


def test_payos_checkout_and_webhook_upgrade_for_thirty_days(world):
    client, store, clock, fake = world
    _, headers = _login(client)
    checkout = client.post("/v1/billing/checkout", json={"plan": "plus", "provider": "payos"}, headers=headers).json()
    sent = fake.payos_requests[-1]
    assert (sent["client"], sent["key"]) == ("client", "payos-key")
    assert checkout["checkout_url"].startswith("https://pay.payos.vn/") and checkout["amount"] == 49000
    order = checkout["order_code"]

    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 49000, key="forged")).status_code == 401
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 49000)).json()["applied"] is True
    me = client.get("/v1/me", headers=headers).json()
    assert me["plan"] == "plus" and me["plan_expires_at"] == pytest.approx(clock[0] + 30 * DAY)
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 49000)).json()["applied"] is False

    clock[0] += 10 * DAY
    renew = client.post("/v1/billing/checkout", json={"plan": "plus", "provider": "payos"}, headers=headers).json()
    client.post("/v1/billing/payos/webhook", json=_payos_webhook(renew["order_code"], 49000))
    assert client.get("/v1/me", headers=headers).json()["plan_expires_at"] == pytest.approx(clock[0] + 50 * DAY)

    clock[0] += 51 * DAY
    assert client.get("/v1/me", headers=headers).json()["plan"] == "free"


def test_payos_webhook_ignores_wrong_amounts_failed_payments_and_unknown_orders(world):
    client, _, _, _ = world
    _, headers = _login(client)
    order = client.post("/v1/billing/checkout", json={"plan": "pro", "provider": "payos"}, headers=headers).json()["order_code"]
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 1000)).json()["applied"] is False
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(order, 129000, code="01")).json()["applied"] is False
    assert client.post("/v1/billing/payos/webhook", json=_payos_webhook(123, 3000)).json() == {"success": True, "applied": False}
    assert client.get("/v1/me", headers=headers).json()["plan"] == "free"


def _ls(client, event: str, account_id: str, status: str, variant="22", key=LS_SECRET, **attributes):
    raw = json.dumps({
        "meta": {"event_name": event, "custom_data": {"account_id": account_id}},
        "data": {"type": "subscriptions", "attributes": {"status": status, "variant_id": variant, **attributes}},
    }).encode()
    return client.post("/v1/billing/lemonsqueezy/webhook", data=raw,
                       headers={"X-Signature": lemonsqueezy_signature(raw, key), "Content-Type": "application/json"})


def test_lemonsqueezy_checkout_carries_the_account_and_webhooks_drive_the_plan(world):
    client, _, clock, fake = world
    _, headers = _login(client)
    account_id = client.get("/v1/me", headers=headers).json()["account_id"]
    checkout = client.post("/v1/billing/checkout", json={"plan": "pro", "provider": "lemonsqueezy"}, headers=headers).json()
    body = fake.ls_requests[-1]["body"]["data"]
    assert checkout["checkout_url"].startswith("https://")
    assert body["attributes"]["checkout_data"]["custom"]["account_id"] == account_id
    assert body["relationships"]["variant"]["data"]["id"] == "22"

    assert _ls(client, "subscription_created", account_id, "active", key="forged").status_code == 401
    renews = datetime.fromtimestamp(clock[0] + 30 * DAY, timezone.utc).isoformat().replace("+00:00", "Z")
    assert _ls(client, "subscription_created", account_id, "active", renews_at=renews).json()["applied"] is True
    assert client.get("/v1/me", headers=headers).json()["plan"] == "pro"

    ends = datetime.fromtimestamp(clock[0] + 5 * DAY, timezone.utc).isoformat()
    _ls(client, "subscription_cancelled", account_id, "cancelled", ends_at=ends)
    assert client.get("/v1/me", headers=headers).json()["plan"] == "pro", "paid time is kept after cancelling"
    clock[0] += 6 * DAY
    assert client.get("/v1/me", headers=headers).json()["plan"] == "free"

    _ls(client, "subscription_resumed", account_id, "active", renews_at=renews)
    _ls(client, "subscription_expired", account_id, "expired")
    assert client.get("/v1/me", headers=headers).json()["plan"] == "free"
    assert _ls(client, "subscription_created", account_id, "active", variant="999").json()["applied"] is False


def test_checkout_refuses_free_plans_and_unconfigured_providers(world, tmp_path):
    client, _, _, _ = world
    _, headers = _login(client)
    assert client.post("/v1/billing/checkout", json={"plan": "free", "provider": "payos"}, headers=headers).status_code == 400
    assert client.post("/v1/billing/checkout", json={"plan": "pro", "provider": "payos"}).status_code == 401
    bare = TestClient(create_app(Store(tmp_path / "bare.sqlite"), Upstream("http://127.0.0.1:9", "", "m", 0, 0), ADMIN))
    _, bare_headers = _login(bare)
    assert bare.post("/v1/billing/checkout", json={"plan": "pro", "provider": "payos"}, headers=bare_headers).status_code == 503
    assert bare.get("/v1/billing/plans").json()["providers"] == {"payos": False, "lemonsqueezy": False}
