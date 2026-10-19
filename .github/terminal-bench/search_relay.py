"""Holds the TinyFish key for the bench and forwards only its search and fetch calls, so task containers never see the key."""
import http.server
import json
import sys
import threading

import requests

KEY = open(sys.argv[1], encoding="utf-8").read().strip()
STATS, PORT = sys.argv[2], int(sys.argv[3])
ROUTES = {"/search": ("GET", "https://api.search.tinyfish.ai"), "/fetch": ("POST", "https://api.fetch.tinyfish.ai")}
stats = {"search": 0, "fetch": 0, "errors": 0}
lock = threading.Lock()


class Relay(http.server.BaseHTTPRequestHandler):
    def _forward(self) -> None:
        path, _, query = self.path.partition("?")
        method, upstream = ROUTES.get(path, (None, None))
        if method != self.command:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        headers = {"X-API-Key": KEY, "Accept": "application/json", **({"Content-Type": "application/json"} if body else {})}
        try:
            up = requests.request(method, upstream + (f"?{query}" if query else ""), data=body, headers=headers, timeout=(10, 90))
            status, data = up.status_code, up.content
        except requests.RequestException as exc:
            status, data = 502, json.dumps({"error": type(exc).__name__}).encode()
        with lock:
            stats[path.strip("/")] += 1
            stats["errors"] += status >= 400
            with open(STATS, "w", encoding="utf-8") as f:
                json.dump(stats, f)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _forward

    def log_message(self, *args) -> None:
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Relay).serve_forever()
