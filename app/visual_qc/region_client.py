from __future__ import annotations

import json
import os

import requests

from app.visual_qc.batch_protocol import RegionBatchDecision, parse_region_batch_decisions
from app.visual_qc.contact_sheet import ContactSheet
from app.visual_qc.regions import QCRegion
from app.visual_qc.request_builder import build_region_batch_payload

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_VISUAL_QC_MODEL", "gemini-3.7-flash")
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
try:
    DEFAULT_TIMEOUT_SECONDS = max(15, min(300, int(os.getenv("GEMINI_VISUAL_QC_TIMEOUT_SECONDS", "120"))))
except ValueError:
    DEFAULT_TIMEOUT_SECONDS = 120


class GeminiRegionQCTimeout(RuntimeError):
    pass


def _extract_output_text(body: dict) -> str:
    for step in reversed(body.get("steps") or []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content") or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("No text model output found")


def _safe_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
            if payload.get("message"):
                return str(payload["message"])[:500]
    except ValueError:
        pass
    text = (response.text or "").strip()
    return text[:500] if text else "unknown error"


class GeminiRegionQC:
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        self.model = model
        self.timeout_seconds = timeout_seconds

    def inspect(self, sheet: ContactSheet, regions_by_id: dict[str, QCRegion], api_key: str, *, mode: str) -> list[RegionBatchDecision]:
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is not configured")
        expected_ids = [item.region_id for item in sheet.items]
        if not expected_ids:
            return []
        payload = build_region_batch_payload(sheet, model=self.model, mode=mode)
        try:
            response = requests.post(
                GEMINI_INTERACTIONS_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key.strip(),
                    "Api-Revision": "2026-05-20",
                },
                json=payload,
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
            )
        except requests.Timeout as exc:
            raise GeminiRegionQCTimeout(
                f"Gemini did not respond within {self.timeout_seconds}s; retry the QC request"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        if not response.ok:
            raise RuntimeError(
                f"Gemini API returned HTTP {response.status_code}: {_safe_error_detail(response)}"
            )
        try:
            body = response.json()
            parsed = json.loads(_extract_output_text(body))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini returned an invalid structured response") from exc

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
