from __future__ import annotations

import pytest
import requests

import gateway.app as gateway_app
from gateway.app import Upstream


class _Response:
    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body


def _upstream(**kwargs) -> Upstream:
    return Upstream(base="https://upstream.test/v1", api_key="k", model="m",
                    input_usd_per_m=0.1, output_usd_per_m=0.2, retry_wait_s=0.0, **kwargs)


def _replay(monkeypatch, outcomes):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gateway_app.requests, "post", post)
    monkeypatch.setattr(gateway_app.time, "sleep", lambda seconds: None)
    return calls


def test_dropped_connection_and_5xx_are_retried(monkeypatch):
    calls = _replay(monkeypatch, [requests.ConnectionError("reset"), _Response(503), _Response(200, {"ok": True})])
    assert _upstream().send({}) == (200, {"ok": True})
    assert len(calls) == 3


def test_a_request_error_is_not_retried(monkeypatch):
    calls = _replay(monkeypatch, [_Response(400, {"error": {"message": "bad image"}})])
    assert _upstream().send({})[0] == 400
    assert len(calls) == 1


def test_the_last_failure_is_returned_or_raised(monkeypatch):
    calls = _replay(monkeypatch, [_Response(429, headers={"Retry-After": "1"})] * 3)
    assert _upstream(retries=2).send({})[0] == 429 and len(calls) == 3

    _replay(monkeypatch, [requests.Timeout("slow")] * 3)
    with pytest.raises(requests.Timeout):
        _upstream(retries=2).send({})
