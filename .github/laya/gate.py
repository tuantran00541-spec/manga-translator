"""Serve the tuned Laya stop judge over HTTP on the bench runner, on CPU (int8: prefix optional): POST /score {"state"} -> {"verified"}."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import laya
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.agent.stopgate import QUESTION  # noqa: E402


def load(path: str):
    """Load a checkpoint; an int8: prefix quantizes its linear layers to int8, the export the bench runs."""
    int8 = path.startswith("int8:")
    agent = laya.load(path.removeprefix("int8:"), device="cpu")
    if int8:
        # Only the encoder, which holds nearly all the weights; the heads read .weight as a tensor, which quantized linears lack.
        agent.model.encoder = torch.ao.quantization.quantize_dynamic(agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8)
    return agent


def serve(path: str, port: int) -> None:
    torch.set_num_threads(2)
    agent, lock = load(path), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            try:
                state = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["state"])[:20_000]
                with lock:
                    answer = agent.predict_batch([state], QUESTION, batch_size=1)[0]["answers"]["verified"]["noul"]
                self._reply(200, {"verified": float(answer)})
            except Exception as exc:  # one bad request must not stop the judge for the other trials
                self._reply(400, {"error": f"{type(exc).__name__}: {exc}"[:300]})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(sys.argv[1], int(sys.argv[2]))
