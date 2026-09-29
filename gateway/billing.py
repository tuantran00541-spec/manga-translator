from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import uuid
from dataclasses import dataclass

import requests

from gateway.store import Store, usd

VND_PER_USD = 26000
TOPUP_VND = (20000, 50000, 100000, 200000)
TOPUP_USD = (2, 5, 10, 20)
# What one long webtoon chapter costs in A.I, before the fee; it only sizes the "about N chapters" hint.
CHAPTER_ESTIMATE_USD = 0.30


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


@dataclass(frozen=True)
class BillingConfig:
    public_url: str = "http://127.0.0.1:8000"
    vnd_per_usd: float = VND_PER_USD
    topups_vnd: tuple = TOPUP_VND
    topups_usd: tuple = TOPUP_USD
    chapter_estimate_usd: float = CHAPTER_ESTIMATE_USD
    # Payment-provider fees are added to what the user pays, so the balance gets the full top-up.
    payos_fee_percent: float = 0.0
    ls_fee_percent: float = 5.0
    ls_fee_fixed_usd: float = 0.50
    payos_base: str = "https://api-merchant.payos.vn"
    payos_client_id: str = ""
    payos_api_key: str = ""
    payos_checksum_key: str = ""
    ls_base: str = "https://api.lemonsqueezy.com"
    ls_api_key: str = ""
    ls_store_id: str = ""
    ls_variant: str = ""
    ls_webhook_secret: str = ""

    @property
    def payos_enabled(self) -> bool:
        return bool(self.payos_client_id and self.payos_api_key and self.payos_checksum_key)

    @property
    def lemonsqueezy_enabled(self) -> bool:
        return bool(self.ls_api_key and self.ls_store_id and self.ls_variant and self.ls_webhook_secret)

    def quote(self, provider: str, amount) -> dict:
        """What a top-up of ``amount`` (VND for payOS, USD for cards) credits and what the user pays for it."""
        if provider == "payos":
            credit = int(amount)
            if not (min(self.topups_vnd) <= credit <= 10 * max(self.topups_vnd)) or credit % 1000:
                raise BillingError(400, f"Nạp từ {min(self.topups_vnd):,}đ, bội số của 1.000đ".replace(",", "."))
            pay = math.ceil(credit * (1 + self.payos_fee_percent / 100))
            return {"currency": "VND", "credit": credit, "pay": pay, "credit_usd": round(credit / self.vnd_per_usd, 6)}
        if provider == "lemonsqueezy":
            credit = float(amount)
            if not (min(self.topups_usd) <= credit <= 10 * max(self.topups_usd)) or credit != int(credit):
                raise BillingError(400, f"Nạp từ ${min(self.topups_usd)}, số đô chẵn")
            cents = math.ceil((credit + self.ls_fee_fixed_usd) * 100 / (1 - self.ls_fee_percent / 100))
            return {"currency": "USD", "credit": int(credit), "pay": cents / 100, "pay_cents": cents, "credit_usd": credit}
        raise BillingError(400, "Cổng thanh toán không hợp lệ")


def billing_from_env() -> BillingConfig:
    def numbers(name: str, default: tuple) -> tuple:
        raw = os.getenv(name, "").strip()
        return tuple(int(part) for part in raw.split(",") if part.strip()) if raw else default

    return BillingConfig(
        public_url=os.getenv("GATEWAY_RETURN_URL", "http://127.0.0.1:8000").rstrip("/"),
        vnd_per_usd=float(os.getenv("GATEWAY_VND_PER_USD", str(VND_PER_USD))),
        topups_vnd=numbers("GATEWAY_TOPUPS_VND", TOPUP_VND),
        topups_usd=numbers("GATEWAY_TOPUPS_USD", TOPUP_USD),
        chapter_estimate_usd=float(os.getenv("GATEWAY_CHAPTER_ESTIMATE_USD", str(CHAPTER_ESTIMATE_USD))),
        payos_fee_percent=float(os.getenv("GATEWAY_PAYOS_FEE_PERCENT", "0")),
        ls_fee_percent=float(os.getenv("GATEWAY_LS_FEE_PERCENT", "5")),
        ls_fee_fixed_usd=float(os.getenv("GATEWAY_LS_FEE_FIXED_USD", "0.50")),
        payos_base=os.getenv("GATEWAY_PAYOS_BASE", "https://api-merchant.payos.vn").rstrip("/"),
        payos_client_id=os.getenv("GATEWAY_PAYOS_CLIENT_ID", ""),
        payos_api_key=os.getenv("GATEWAY_PAYOS_API_KEY", ""),
        payos_checksum_key=os.getenv("GATEWAY_PAYOS_CHECKSUM_KEY", ""),
        ls_base=os.getenv("GATEWAY_LS_BASE", "https://api.lemonsqueezy.com").rstrip("/"),
        ls_api_key=os.getenv("GATEWAY_LS_API_KEY", ""),
        ls_store_id=os.getenv("GATEWAY_LS_STORE_ID", ""),
        ls_variant=os.getenv("GATEWAY_LS_VARIANT", ""),
        ls_webhook_secret=os.getenv("GATEWAY_LS_WEBHOOK_SECRET", ""),
    )


class Billing:
    def __init__(self, store: Store, config: BillingConfig):
        self.store = store
        self.config = config

    def chapter_price_usd(self) -> float:
        """What a typical chapter takes from the balance, fee included."""
        return usd(self.store.price_micros(self.config.chapter_estimate_usd))

    def public(self) -> dict:
        config = self.config
        return {
            "fee_percent": round(self.store.fee_rate * 100, 2),
            "chapter_estimate_usd": self.chapter_price_usd(),
            "providers": {"payos": config.payos_enabled, "lemonsqueezy": config.lemonsqueezy_enabled},
            "topups": {
                "payos": [config.quote("payos", amount) for amount in config.topups_vnd],
                "lemonsqueezy": [config.quote("lemonsqueezy", amount) for amount in config.topups_usd],
            },
        }

    def checkout(self, account: dict, provider: str, amount) -> dict:
        quote = self.config.quote(provider, amount)
        if provider == "payos":
            return self._payos_checkout(account, quote)
        return self._lemonsqueezy_checkout(account, quote)

    def _payos_checkout(self, account: dict, quote: dict) -> dict:
        config = self.config
        if not config.payos_enabled:
            raise BillingError(503, "Thanh toán QR chưa được cấu hình")
        order_code = int(self.store.now() * 1000) % 10**12 * 100 + secrets.randbelow(100)
        body = {
            "orderCode": order_code,
            "amount": quote["pay"],
            "description": f"MT NAP {order_code % 100000:05d}"[:25],
            "cancelUrl": f"{config.public_url}/?billing=cancelled",
            "returnUrl": f"{config.public_url}/?billing=paid",
        }
        body["signature"] = payos_request_signature(body, config.payos_checksum_key)
        self.store.create_payment("payos", str(order_code), account["id"], quote["credit_usd"], quote["pay"], "VND")
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
        return {"provider": "payos", "checkout_url": data["data"]["checkoutUrl"], "order_code": order_code, **quote}

    def _lemonsqueezy_checkout(self, account: dict, quote: dict) -> dict:
        config = self.config
        if not config.lemonsqueezy_enabled:
            raise BillingError(503, "Thanh toán thẻ quốc tế chưa được cấu hình")
        # The order id is not known until payment, so the webhook finds the top-up by this reference.
        ref = uuid.uuid4().hex
        payload = {"data": {
            "type": "checkouts",
            "attributes": {
                "custom_price": quote["pay_cents"],
                "checkout_data": {"email": account["email"], "custom": {"account_id": account["id"], "payment_ref": ref}},
                "product_options": {"redirect_url": f"{config.public_url}/?billing=paid",
                                    "enabled_variants": [int(config.ls_variant)] if config.ls_variant.isdigit() else []},
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": str(config.ls_store_id)}},
                "variant": {"data": {"type": "variants", "id": str(config.ls_variant)}},
            },
        }}
        self.store.create_payment("lemonsqueezy", ref, account["id"], quote["credit_usd"], quote["pay_cents"], "USD")
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
        return {"provider": "lemonsqueezy", "checkout_url": url, **quote}

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
        ref = str((meta.get("custom_data") or {}).get("payment_ref") or "")
        if not ref:
            return {"applied": False}
        if event == "order_created":
            if attributes.get("status") != "paid":
                return {"applied": False}
            # A store in another currency, or a changed price, never credits the balance.
            paid = int(attributes.get("subtotal") or 0) if attributes.get("currency") == "USD" else -1
            result = self.store.complete_payment("lemonsqueezy", ref, paid)
            return {"applied": bool(result and result.get("applied"))}
        if event == "order_refunded":
            return {"applied": self.store.refund_payment("lemonsqueezy", ref)}
        return {"applied": False}

