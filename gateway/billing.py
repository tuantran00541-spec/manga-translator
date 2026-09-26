from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime

import requests

from gateway.store import PLAN_PERIOD_SECONDS, Store

PAID_PLANS = ("plus", "pro")
LS_ACTIVE = frozenset({"active", "on_trial", "past_due"})
LS_GRACE_SECONDS = 3 * 86400


class BillingError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def payos_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or value in ("undefined", "null"):
        return ""
    if isinstance(value, list):
        items = [dict(sorted(item.items())) if isinstance(item, dict) else item for item in value]
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def payos_data_signature(data: dict, key: str) -> str:
    query = "&".join(f"{name}={payos_value(data[name])}" for name in sorted(data))
    return hmac.new(key.encode(), query.encode(), hashlib.sha256).hexdigest()


def payos_request_signature(body: dict, key: str) -> str:
    query = "&".join(f"{name}={payos_value(body[name])}" for name in ("amount", "cancelUrl", "description", "orderCode", "returnUrl"))
    return hmac.new(key.encode(), query.encode(), hashlib.sha256).hexdigest()


def lemonsqueezy_signature(raw: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _iso_ts(value) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class BillingConfig:
    public_url: str = "http://127.0.0.1:8000"
    prices_vnd: dict = field(default_factory=lambda: {"plus": 49000, "pro": 129000})
    prices_usd: dict = field(default_factory=lambda: {"plus": 2.99, "pro": 6.99})
    payos_base: str = "https://api-merchant.payos.vn"
    payos_client_id: str = ""
    payos_api_key: str = ""
    payos_checksum_key: str = ""
    ls_base: str = "https://api.lemonsqueezy.com"
    ls_api_key: str = ""
    ls_store_id: str = ""
    ls_variants: dict = field(default_factory=dict)
    ls_webhook_secret: str = ""

    @property
    def payos_enabled(self) -> bool:
        return bool(self.payos_client_id and self.payos_api_key and self.payos_checksum_key)

    @property
    def lemonsqueezy_enabled(self) -> bool:
        return bool(self.ls_api_key and self.ls_store_id and self.ls_webhook_secret
                    and all(self.ls_variants.get(plan) for plan in PAID_PLANS))

    def public(self) -> dict:
        return {
            "plans": {plan: {"vnd": self.prices_vnd[plan], "usd": self.prices_usd[plan]} for plan in PAID_PLANS},
            "providers": {"payos": self.payos_enabled, "lemonsqueezy": self.lemonsqueezy_enabled},
            "period_days": PLAN_PERIOD_SECONDS // 86400,
        }


def billing_from_env() -> BillingConfig:
    return BillingConfig(
        public_url=os.getenv("GATEWAY_RETURN_URL", "http://127.0.0.1:8000").rstrip("/"),
        prices_vnd={"plus": int(os.getenv("GATEWAY_PRICE_PLUS_VND", "49000")),
                    "pro": int(os.getenv("GATEWAY_PRICE_PRO_VND", "129000"))},
        prices_usd={"plus": float(os.getenv("GATEWAY_PRICE_PLUS_USD", "2.99")),
                    "pro": float(os.getenv("GATEWAY_PRICE_PRO_USD", "6.99"))},
        payos_base=os.getenv("GATEWAY_PAYOS_BASE", "https://api-merchant.payos.vn").rstrip("/"),
        payos_client_id=os.getenv("GATEWAY_PAYOS_CLIENT_ID", ""),
        payos_api_key=os.getenv("GATEWAY_PAYOS_API_KEY", ""),
        payos_checksum_key=os.getenv("GATEWAY_PAYOS_CHECKSUM_KEY", ""),
        ls_base=os.getenv("GATEWAY_LS_BASE", "https://api.lemonsqueezy.com").rstrip("/"),
        ls_api_key=os.getenv("GATEWAY_LS_API_KEY", ""),
        ls_store_id=os.getenv("GATEWAY_LS_STORE_ID", ""),
        ls_variants={"plus": os.getenv("GATEWAY_LS_VARIANT_PLUS", ""), "pro": os.getenv("GATEWAY_LS_VARIANT_PRO", "")},
        ls_webhook_secret=os.getenv("GATEWAY_LS_WEBHOOK_SECRET", ""),
    )


class Billing:
    def __init__(self, store: Store, config: BillingConfig):
        self.store = store
        self.config = config

    def checkout(self, account: dict, plan: str, provider: str) -> dict:
        if plan not in PAID_PLANS:
            raise BillingError(400, "Chỉ mua được gói Plus hoặc Pro")
        if provider == "payos":
            return self._payos_checkout(account, plan)
        if provider == "lemonsqueezy":
            return self._lemonsqueezy_checkout(account, plan)
        raise BillingError(400, "Cổng thanh toán không hợp lệ")

    def _payos_checkout(self, account: dict, plan: str) -> dict:
        config = self.config
        if not config.payos_enabled:
            raise BillingError(503, "Thanh toán QR chưa được cấu hình")
        order_code = int(self.store.now() * 1000) % 10**12 * 100 + secrets.randbelow(100)
        amount = int(config.prices_vnd[plan])
        body = {
            "orderCode": order_code,
            "amount": amount,
            "description": f"MT {plan.upper()} {order_code % 100000:05d}"[:25],
            "cancelUrl": f"{config.public_url}/?billing=cancelled",
            "returnUrl": f"{config.public_url}/?billing=paid",
        }
        body["signature"] = payos_request_signature(body, config.payos_checksum_key)
        self.store.create_payment("payos", str(order_code), account["id"], plan, amount, "VND")
        try:
            response = requests.post(
                f"{config.payos_base}/v2/payment-requests",
                headers={"x-client-id": config.payos_client_id, "x-api-key": config.payos_api_key},
                json=body, timeout=(5, 20), allow_redirects=False,
            )
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise BillingError(502, "Không tạo được link thanh toán QR") from exc
        if data.get("code") != "00" or not (data.get("data") or {}).get("checkoutUrl"):
            raise BillingError(502, f"payOS: {str(data.get('desc') or 'lỗi')[:120]}")
        return {"provider": "payos", "checkout_url": data["data"]["checkoutUrl"], "order_code": order_code,
                "amount": amount, "currency": "VND"}

    def _lemonsqueezy_checkout(self, account: dict, plan: str) -> dict:
        config = self.config
        if not config.lemonsqueezy_enabled:
            raise BillingError(503, "Thanh toán thẻ quốc tế chưa được cấu hình")
        payload = {"data": {
            "type": "checkouts",
            "attributes": {
                "checkout_data": {"email": account["email"], "custom": {"account_id": account["id"], "plan": plan}},
                "product_options": {"redirect_url": f"{config.public_url}/?billing=paid"},
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": str(config.ls_store_id)}},
                "variant": {"data": {"type": "variants", "id": str(config.ls_variants[plan])}},
            },
        }}
        try:
            response = requests.post(
                f"{config.ls_base}/v1/checkouts",
                headers={"Authorization": f"Bearer {config.ls_api_key}", "Accept": "application/vnd.api+json",
                         "Content-Type": "application/vnd.api+json"},
                json=payload, timeout=(5, 20), allow_redirects=False,
            )
            url = response.json()["data"]["attributes"]["url"]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise BillingError(502, "Không tạo được trang thanh toán thẻ") from exc
        return {"provider": "lemonsqueezy", "checkout_url": url, "amount": config.prices_usd[plan], "currency": "USD"}

    def payos_webhook(self, body: dict) -> dict:
        data = body.get("data")
        signature = str(body.get("signature") or "")
        if not isinstance(data, dict) or not self.config.payos_checksum_key:
            raise BillingError(400, "Invalid webhook")
        expected = payos_data_signature(data, self.config.payos_checksum_key)
        if not hmac.compare_digest(expected, signature):
            raise BillingError(401, "Invalid signature")
        if str(data.get("code")) != "00":
            return {"success": True, "applied": False}
        result = self.store.complete_payment("payos", str(data.get("orderCode")), int(data.get("amount") or 0))
        return {"success": True, "applied": bool(result and result.get("applied"))}

    def lemonsqueezy_webhook(self, raw: bytes, signature: str) -> dict:
        secret = self.config.ls_webhook_secret
        if not secret or not hmac.compare_digest(lemonsqueezy_signature(raw, secret), signature or ""):
            raise BillingError(401, "Invalid signature")
        try:
            body = json.loads(raw)
            meta = body["meta"]
            attributes = body["data"]["attributes"]
        except (ValueError, KeyError, TypeError) as exc:
            raise BillingError(400, "Invalid webhook") from exc
        event = str(meta.get("event_name") or "")
        if not event.startswith("subscription_"):
            return {"applied": False}
        account_id = str((meta.get("custom_data") or {}).get("account_id") or "")
        variants = {str(v): plan for plan, v in self.config.ls_variants.items()}
        plan = variants.get(str(attributes.get("variant_id")))
        if not account_id or plan is None:
            return {"applied": False}
        status = str(attributes.get("status") or "")
        now = self.store.now()
        if status in LS_ACTIVE:
            renews = _iso_ts(attributes.get("renews_at")) or now + PLAN_PERIOD_SECONDS
            expires = renews + LS_GRACE_SECONDS
        elif status == "cancelled":
            expires = _iso_ts(attributes.get("ends_at")) or now
        else:
            expires = now
        try:
            self.store.set_plan(account_id, plan if expires > now else "free", expires if expires > now else None)
        except KeyError:
            return {"applied": False}
        return {"applied": True, "status": status}
