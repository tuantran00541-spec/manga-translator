"""A relay on the runner that adds the API key itself, so the containers under test never hold it."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

KEY = open(sys.argv[1], encoding="utf-8").read().strip()
STATS, PORT, UPSTREAM = sys.argv[2], int(sys.argv[3]), sys.argv[4].rstrip("/")
# Limits keep a run polite to the provider: tokens a minute, requests at once, a dollar cap, and a stop on the first key rejection.
LIMITS = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
PRICE = LIMITS.get("price") or {}
SLOTS = threading.BoundedSemaphore(int(LIMITS.get("concurrent") or 64))
# The agent calls http://relay/v1/<path>; only these paths go on, to UPSTREAM/<path>.
ALLOWED = {"/v1/chat/completions": "/chat/completions", "/v1/models": "/models"}
MAX_BODY = 16_000_000
# Status counts and the first 429 show whether the provider starts throttling the key.
stats = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "status": {}, "first_429": None, "usd": 0.0, "stopped": None}
recent: list[tuple[float, int]] = []
lock = threading.Lock()


def flush() -> None:
    with lock:
        open(STATS, "w", encoding="utf-8").write(json.dumps(stats))
    os.chmod(STATS, 0o644)


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status: int, data: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_any(self):
        if self.path.split("?")[0] not in ALLOWED:
            return self.reply(404, b'{"error": "path not allowed"}')
        body = b""
        if self.command == "POST":
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                return self.reply(413, b'{"error": "body too large"}')
            body = self.rfile.read(size)
        if stats["stopped"]:
            return self.reply(503, json.dumps({"error": f"relay stopped: {stats['stopped']}"}).encode())
        self.wait_for_budget()
        with SLOTS:
            self.forward(body)

    def wait_for_budget(self):
        """Holds the request while the last minute already used the token allowance."""
        cap = int(LIMITS.get("tokens_per_minute") or 0)
        while cap:
            with lock:
                now = time.time()
                recent[:] = [(t, n) for t, n in recent if now - t < 60]
                if sum(n for _, n in recent) < cap:
                    return
            time.sleep(1)

    def forward(self, body: bytes):
        if stats["stopped"]:
            return self.reply(503, json.dumps({"error": f"relay stopped: {stats['stopped']}"}).encode())
        with lock:
            stats["requests"] += 1
        try:
            up = requests.request(self.command, UPSTREAM + ALLOWED[self.path.split("?")[0]], data=body or None, stream=True, timeout=(15, 600),
                                  headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")})
        except requests.RequestException as exc:
            with lock:
                stats["errors"] += 1
            return self.reply(502, json.dumps({"error": type(exc).__name__}).encode())
        self.send_response(up.status_code)
        self.send_header("Content-Type", up.headers.get("Content-Type", "application/json"))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        tail = bytearray()
        try:
            for chunk in up.iter_content(chunk_size=None):
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    tail += chunk
                    if len(tail) > 4_000_000:
                        del tail[:-2_000_000]
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass
        with lock:
            stats["status"][str(up.status_code)] = stats["status"].get(str(up.status_code), 0) + 1
            if up.status_code == 429 and not stats["first_429"]:
                stats["first_429"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if up.status_code >= 400:
                stats["errors"] += 1
            if up.status_code in (401, 403) and not stats["stopped"]:
                stats["stopped"] = f"the provider rejected the key (HTTP {up.status_code})"
        for piece in tail.decode("utf-8", "ignore").split("\n"):
            piece = piece[5:].strip() if piece.startswith("data:") else piece
            try:
                usage = json.loads(piece).get("usage")
            except (ValueError, AttributeError):
                continue
            if isinstance(usage, dict):
                prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                with lock:
                    stats["prompt_tokens"] += prompt
                    stats["completion_tokens"] += completion
                    recent.append((time.time(), prompt + completion))
                    # Every prompt token is priced as uncached, so the dollar count is an upper bound.
                    stats["usd"] += prompt * float(PRICE.get("input") or 0) + completion * float(PRICE.get("output") or 0)
        with lock:
            stats["usd"] += float(PRICE.get("request") or 0)
            if LIMITS.get("max_usd") and stats["usd"] >= float(LIMITS["max_usd"]) and not stats["stopped"]:
                stats["stopped"] = f"the run spent its ${LIMITS['max_usd']} cap"
        flush()

    do_GET = do_POST = handle_any


if __name__ == "__main__":
    flush()
    ThreadingHTTPServer(("0.0.0.0", PORT), Relay).serve_forever()
