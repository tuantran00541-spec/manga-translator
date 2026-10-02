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


def test_cache_hits_are_billed_at_the_cached_price():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 800}}
    billed = Upstream("b", "k", "m", 1.0, 2.0, cached_usd_per_m=0.02).cost(usage)
    assert billed == (200 * 1.0 + 800 * 0.02 + 100 * 2.0) / 1_000_000
    assert Upstream("b", "k", "m", 1.0, 2.0).cost(usage) == (1000 * 1.0 + 100 * 2.0) / 1_000_000


def test_each_checkpoint_has_its_model_and_an_overloaded_one_hands_over_at_once(monkeypatch):
    from gateway.app import Route, upstream_from_env

    monkeypatch.setenv("GATEWAY_STAGE_MODELS", '[{"stage": "review", "model": "muse", "price": [0.1, 0.2]},'
                                               ' {"stage": "review", "model": "mimo", "price": [0.15, 0.3]}]')
    upstream = upstream_from_env()
    assert [route.model for route in upstream.chain("review")] == ["muse", "mimo"]
    assert upstream.chain("scan") == [upstream.default], "a checkpoint without its own models uses the default"
    sent, slept = [], []
    monkeypatch.setattr(gateway_app.requests, "post", lambda url, **kwargs: sent.append(kwargs["json"]["model"]) or (
        _Response(503) if len(sent) == 1 else _Response(200, {"ok": True})))
    monkeypatch.setattr(gateway_app.time, "sleep", slept.append)
    trace: dict = {}
    assert upstream.send({"model": "muse"}, trace, ["muse", "mimo"])[0] == 200
    assert sent == ["muse", "mimo"] and not slept and trace["model"] == "mimo"
    assert upstream.cost({"prompt_tokens": 1_000_000}, "mimo") == Route("mimo", 0.15, 0.3).cost({"prompt_tokens": 1_000_000})

    efforts = []
    monkeypatch.setattr(gateway_app.requests, "post", lambda url, **kwargs: efforts.append(
        kwargs["json"].get("reasoning_effort")) or _Response(200, {"ok": True}))
    upstream.send({"model": "x", "reasoning_effort": "low"}, None, [Route("muse", 0.1, 0.2, None, "none")])
    upstream.send({"model": "x", "reasoning_effort": "low"}, None, [Route("luna", 0.1, 0.5)])
    upstream.send({"model": "x", "reasoning_effort": "low"}, None, [Route("qwen", 0.1, 0.1, None, "omit")])
    assert efforts == ["none", "low", None], "a model's own thinking level wins; otherwise the request's is kept"

    monkeypatch.setenv("GATEWAY_STAGE_MODELS", '[{"stage": "scan", "model": "qwen", "price": [0, 0], "effort": "omit"}]')
    assert upstream_from_env().stages["scan"][0].reasoning_effort == "omit"

    monkeypatch.setenv("GATEWAY_STAGE_MODELS", '[{"stage": "read", "model": "x", "price": [0, 0]}]')
    with pytest.raises(ValueError):
        upstream_from_env()


def test_only_manga_cloud_requests_name_their_checkpoint():
    from app.ai_mode.vision_json import stage_headers
    from app.ai_providers import PROVIDERS
    from app.cloud import cloud_provider

    assert stage_headers(cloud_provider(), "review") == {"X-MT-Stage": "review"}
    assert stage_headers(PROVIDERS["openai"], "review") == {}
