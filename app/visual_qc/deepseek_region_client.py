from __future__ import annotations

import json
import os
import threading

import requests

from app.visual_qc.batch_protocol import RegionBatchDecision, parse_region_batch_decisions
from app.visual_qc.contact_sheet import ContactSheet
from app.visual_qc.regions import QCRegion
from app.visual_qc.request_builder import build_region_batch_payload

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_VISUAL_QC_MODEL",
    "deepseek-v4-flash-vision-exp",
)
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
_INVALID_RESPONSE_MESSAGE = "DeepSeek returned an invalid structured response"


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value >= 0 else default


DEFAULT_TIMEOUT_SECONDS = _int_env(
    "DEEPSEEK_VISUAL_QC_TIMEOUT_SECONDS",
    120,
    15,
    300,
)
DEFAULT_MAX_OUTPUT_TOKENS = _int_env(
    "DEEPSEEK_VISUAL_QC_MAX_OUTPUT_TOKENS",
    800,
    128,
    1200,
)
_BUDGET_INPUT_TOKENS = _int_env(
    "DEEPSEEK_VISUAL_QC_BUDGET_INPUT_TOKENS",
    8192,
    1024,
    32768,
)
_BUDGET_CACHE_HIT_USD_PER_M = _float_env(
    "DEEPSEEK_VISUAL_QC_BUDGET_CACHE_HIT_USD_PER_M",
    0.014,
)
_BUDGET_CACHE_MISS_USD_PER_M = _float_env(
    "DEEPSEEK_VISUAL_QC_BUDGET_CACHE_MISS_USD_PER_M",
    0.44,
)
_BUDGET_OUTPUT_USD_PER_M = _float_env(
    "DEEPSEEK_VISUAL_QC_BUDGET_OUTPUT_USD_PER_M",
    1.32,
)


class DeepSeekRegionQCTimeout(RuntimeError):
    pass


class DeepSeekBudgetExceeded(RuntimeError):
    pass


def _redact_secret(text: object, secret: str) -> str:
    value = str(text)
    return value.replace(secret, "[redacted]") if secret else value


def _safe_error_detail(response: requests.Response, secret: str) -> str:
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                detail = str(err["message"])
            elif payload.get("message"):
                detail = str(payload["message"])
    except ValueError:
        pass
    if not detail:
        detail = (response.text or "").strip() or "unknown error"
    return _redact_secret(detail, secret)[:500]


def _extract_output_text(body: dict) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No DeepSeek completion choice found")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid DeepSeek completion choice")
    if choice.get("finish_reason") == "length":
        raise ValueError("DeepSeek structured response was truncated")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("No DeepSeek completion message found")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("No DeepSeek text output found")


def _ordered_decisions(
    parsed: object,
    expected_ids: list[str],
    regions_by_id: dict[str, QCRegion],
) -> list[RegionBatchDecision]:
    parsed_decisions = parse_region_batch_decisions(parsed, regions_by_id)
    by_id = {decision.region_id: decision for decision in parsed_decisions}
    ordered: list[RegionBatchDecision] = []
    for region_id in expected_ids:
        decision = by_id.get(region_id)
        if decision is None:
            region = regions_by_id.get(region_id)
            if region is None:
                continue
            decision = RegionBatchDecision(
                page_index=region.page_index,
                region_id=region.region_id,
                status="ambiguous",
                issues=(),
            )
        ordered.append(decision)
    return ordered


class DeepSeekRegionQC:
    def __init__(
        self,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        *,
        budget_usd: float = 0.08,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        budget = float(budget_usd)
        if budget <= 0 or budget > 0.15:
            raise ValueError("DeepSeek visual QC budget must be > 0 and <= $0.15")
        self.model = model
        self.timeout_seconds = int(timeout_seconds)
        self.budget_usd = budget
        self.max_output_tokens = max(128, min(1200, int(max_output_tokens)))
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._estimated_cost_usd = 0.0
        self._requests = 0
        self._prompt_tokens = 0
        self._cache_hit_tokens = 0
        self._cache_miss_tokens = 0
        self._completion_tokens = 0

    def _reservation_cost(self) -> float:
        return (
            _BUDGET_INPUT_TOKENS * _BUDGET_CACHE_MISS_USD_PER_M
            + self.max_output_tokens * _BUDGET_OUTPUT_USD_PER_M
        ) / 1_000_000

    def _reserve(self) -> float:
        reservation = self._reservation_cost()
        with self._lock:
            projected = self._estimated_cost_usd + self._reserved_usd + reservation
            if projected > self.budget_usd + 1e-12:
                raise DeepSeekBudgetExceeded(
                    f"DeepSeek visual QC budget cap ${self.budget_usd:.3f} reached"
                )
            self._reserved_usd += reservation
            self._requests += 1
        return reservation

    def _release(self, reservation: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reservation)

    def _charge_unknown(self, reservation: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reservation)
            self._estimated_cost_usd += reservation

    def _commit_usage(self, body: dict, reservation: float) -> None:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            self._charge_unknown(reservation)
            return
        try:
            prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
            hit_tokens = max(0, int(usage.get("prompt_cache_hit_tokens") or 0))
            raw_miss = usage.get("prompt_cache_miss_tokens")
            miss_tokens = (
                max(0, int(raw_miss))
                if raw_miss is not None
                else max(0, prompt_tokens - hit_tokens)
            )
            completion_tokens = max(0, int(usage.get("completion_tokens") or 0))
        except (TypeError, ValueError, OverflowError):
            self._charge_unknown(reservation)
            return
        cost = (
            hit_tokens * _BUDGET_CACHE_HIT_USD_PER_M
            + miss_tokens * _BUDGET_CACHE_MISS_USD_PER_M
            + completion_tokens * _BUDGET_OUTPUT_USD_PER_M
        ) / 1_000_000
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reservation)
            self._estimated_cost_usd += cost
            self._prompt_tokens += prompt_tokens
            self._cache_hit_tokens += hit_tokens
            self._cache_miss_tokens += miss_tokens
            self._completion_tokens += completion_tokens

    def usage_snapshot(self) -> dict:
        with self._lock:
            spent = self._estimated_cost_usd
            return {
                "requests": self._requests,
                "prompt_tokens": self._prompt_tokens,
                "cache_hit_tokens": self._cache_hit_tokens,
                "cache_miss_tokens": self._cache_miss_tokens,
                "completion_tokens": self._completion_tokens,
                "estimated_cost_usd": round(spent, 6),
                "budget_usd": round(self.budget_usd, 6),
                "remaining_budget_usd": round(max(0.0, self.budget_usd - spent), 6),
            }

    def _payload(self, sheet: ContactSheet, mode: str) -> dict:
        shared = build_region_batch_payload(sheet, model=self.model, mode=mode)
        prompt = str(shared["input"][0]["text"])
        image_data = str(shared["input"][1]["data"])
        prompt += (
            "\n\nReturn JSON only. The top-level JSON object must be "
            '{"regions":[{"region_id":"...","status":"pass|flagged|ambiguous","issues":[]}]}. '
            "Every issue must include issue_type, confidence, box_2d, reason, and recommended_action."
        )
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }

    def inspect(
        self,
        sheet: ContactSheet,
        regions_by_id: dict[str, QCRegion],
        api_key: str,
        *,
        mode: str,
    ) -> list[RegionBatchDecision]:
        secret = (api_key or "").strip()
        if not secret:
            raise ValueError("DeepSeek API key is not configured")
        expected_ids = [item.region_id for item in sheet.items]
        if not expected_ids:
            return []
        reservation = self._reserve()
        try:
            response = requests.post(
                DEEPSEEK_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                },
                json=self._payload(sheet, mode),
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
            )
        except requests.Timeout as exc:
            self._charge_unknown(reservation)
            raise DeepSeekRegionQCTimeout(
                f"DeepSeek did not respond within {self.timeout_seconds}s; retry the QC request"
            ) from exc
        except requests.RequestException as exc:
            self._release(reservation)
            raise RuntimeError(
                f"DeepSeek request failed: {_redact_secret(exc, secret)}"
            ) from exc
        if not response.ok:
            self._release(reservation)
            raise RuntimeError(
                f"DeepSeek API returned HTTP {response.status_code}: "
                f"{_safe_error_detail(response, secret)}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            self._charge_unknown(reservation)
            raise RuntimeError(_INVALID_RESPONSE_MESSAGE) from exc
        if not isinstance(body, dict):
            self._charge_unknown(reservation)
            raise RuntimeError(_INVALID_RESPONSE_MESSAGE)
        self._commit_usage(body, reservation)
        try:
            parsed = json.loads(_extract_output_text(body))
        except (ValueError, TypeError) as exc:
            raise RuntimeError(_INVALID_RESPONSE_MESSAGE) from exc
        return _ordered_decisions(parsed, expected_ids, regions_by_id)
