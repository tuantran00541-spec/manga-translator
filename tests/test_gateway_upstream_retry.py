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


def test_send_traces_every_attempt(monkeypatch):
    _replay(monkeypatch, [requests.ConnectionError("reset"), _Response(503), _Response(200, {"ok": True})])
    trace: dict = {}
    assert _upstream(retries=2).send({}, trace)[0] == 200
    assert trace == {"attempts": 3, "statuses": ["ConnectionError", 503, 200]}


def test_request_shape_counts_images_and_names_the_prompt():
    payload = {"messages": [
        {"role": "system", "content": "You letter manga.\n  Keep it short."},
        {"role": "user", "content": [
            {"type": "text", "text": "slice 3"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
        ]},
    ]}
    assert gateway_app._request_shape(payload) == {"prompt_head": "You letter manga. Keep it short.", "images": 2}


def test_request_shape_reads_a_prompt_sent_as_user_text():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Scan these slices"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}
    assert gateway_app._request_shape(payload) == {"prompt_head": "Scan these slices", "images": 1}
