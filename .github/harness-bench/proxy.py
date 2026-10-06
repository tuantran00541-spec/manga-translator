"""A relay to PolarGrid on the runner that adds the key itself, so the containers under test never hold it."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

UPSTREAM = "https://api.mia-01.edge.polargrid.ai"
KEY = open(sys.argv[1], encoding="utf-8").read().strip()
STATS, PORT = sys.argv[2], int(sys.argv[3])
ALLOWED = {"/v1/chat/completions", "/v1/models"}
MAX_BODY = 16_000_000
stats = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
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
        with lock:
            stats["requests"] += 1
        try:
            up = requests.request(self.command, UPSTREAM + self.path, data=body or None, stream=True, timeout=(15, 600),
                                  headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")})
        except requests.RequestException as exc:
            with lock:
                stats["errors"] += 1
            return self.reply(502, json.dumps({"error": type(exc).__name__}).encode())
        self.send_response(up.status_code)
        self.send_header("Content-Type", up.headers.get("Content-Type", "application/json"))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        tail = b""
        try:
            for chunk in up.iter_content(chunk_size=None):
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    tail = (tail + chunk)[-8000:]
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass
        if up.status_code >= 400:
            with lock:
                stats["errors"] += 1
        for piece in tail.decode("utf-8", "ignore").split("\n"):
            piece = piece[5:].strip() if piece.startswith("data:") else piece
            try:
                usage = json.loads(piece).get("usage")
            except (ValueError, AttributeError):
                continue
            if isinstance(usage, dict):
                with lock:
                    stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        flush()

    do_GET = do_POST = handle_any


if __name__ == "__main__":
    flush()
    ThreadingHTTPServer(("0.0.0.0", PORT), Relay).serve_forever()
