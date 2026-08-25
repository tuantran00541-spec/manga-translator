from __future__ import annotations

import pytest
import requests

from app.visual_qc.contact_sheet import ContactSheet, ContactSheetItem
from app.visual_qc.deepseek_region_client import DeepSeekRegionQC
from app.visual_qc.regions import QCRegion


class _ErrorResponse:
    ok = False
    status_code = 400

    def __init__(self, secret: str):
        self.secret = secret
        self.text = f"upstream echoed {secret}"

    def json(self):
        return {"error": {"message": f"bad credential {self.secret}"}}


def _inputs():
    import numpy as np

    region = QCRegion(0, "P0001-R01", (0, 0, 20, 20), (), (), 0.1, False)
    sheet = ContactSheet(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        items=(ContactSheetItem(region.region_id, 0, region.bbox, (0, 0, 20, 20)),),
        scale=1.0,
    )
    return region, sheet


def test_http_error_never_echoes_deepseek_api_key(monkeypatch):
    secret = "deepseek-secret-do-not-leak"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _ErrorResponse(secret))
    region, sheet = _inputs()
    client = DeepSeekRegionQC(budget_usd=0.02)
    with pytest.raises(RuntimeError) as captured:
        client.inspect(sheet, {region.region_id: region}, secret, mode="region-clean")
    assert secret not in str(captured.value)
    assert "[redacted]" in str(captured.value)


def test_requests_exception_never_echoes_deepseek_api_key(monkeypatch):
    secret = "deepseek-secret-do-not-leak"

    def fail(*args, **kwargs):
        raise requests.RequestException(f"transport failed for {secret}")

    monkeypatch.setattr(requests, "post", fail)
    region, sheet = _inputs()
    client = DeepSeekRegionQC(budget_usd=0.02)
    with pytest.raises(RuntimeError) as captured:
        client.inspect(sheet, {region.region_id: region}, secret, mode="region-clean")
    assert secret not in str(captured.value)
    assert "[redacted]" in str(captured.value)
