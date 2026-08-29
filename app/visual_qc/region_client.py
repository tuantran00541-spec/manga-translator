from __future__ import annotations

import json

import requests

from app.visual_qc.batch_protocol import RegionBatchDecision, parse_region_batch_decisions
from app.visual_qc.contact_sheet import ContactSheet
from app.visual_qc.gemini_interactions import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GEMINI_INTERACTIONS_URL,
    extract_output_text,
    safe_error_detail,
)
from app.visual_qc.regions import QCRegion
from app.visual_qc.request_builder import build_region_batch_payload


class GeminiRegionQCTimeout(RuntimeError):
    pass


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
                f"Gemini API returned HTTP {response.status_code}: {safe_error_detail(response)}"
            )
        try:
            body = response.json()
            parsed = json.loads(extract_output_text(body))
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
