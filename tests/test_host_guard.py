import asyncio

from app.main import LocalHostOnlyMiddleware, _host_name


def _call(host: str) -> int:
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/", "headers": [(b"host", host.encode())]}
    asyncio.run(LocalHostOnlyMiddleware(app, {"127.0.0.1", "localhost", "::1"})(scope, receive, send))
    return sent[0]["status"]


def test_only_loopback_hosts_reach_the_app():
    assert _call("127.0.0.1:8000") == 200 and _call("localhost:8000") == 200 and _call("[::1]:8000") == 200
    assert _call("evil.example:8000") == 403, "a page rebinding its domain to this machine is refused"
    assert _call("") == 403


def test_host_names_drop_the_port():
    assert _host_name("LocalHost:8000") == "localhost" and _host_name("[::1]:80") == "::1" and _host_name("::1") == "::1"


def test_a_crashing_keyring_backend_reads_as_unavailable_storage(monkeypatch):
    import keyring

    from app import secret_store

    class Panic(BaseException):
        pass

    def crash(*_args):
        raise Panic("rust panic")

    monkeypatch.setattr(keyring, "get_password", crash)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    status = secret_store.provider_key_status("gemini")
    assert status["source"] == "unavailable" and "Panic" in status["detail"]
