"""Run our agent on AutomationBench: each task's world is an MCP server, and the bench's own rubric scores the end state."""

import asyncio
import copy
import json
import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from automationbench.domains import get_domain_dataset
from automationbench.rubric import create_rubric, partial_credit
from automationbench.runner import AutomationBenchEnv, compute_allowed_services, strip_none_values
from automationbench.schema.world import WorldState
from automationbench.tools.api import API_TOOLS

CFG = json.loads(os.environ["AB_CONFIG"])
OUT = Path(os.environ["OUT"])
AGENT_USER = "agent"
FUNCS = {f.__name__: f for f in API_TOOLS}
WORLDS: dict[str, dict] = {}
LOCK = threading.Lock()


def sample() -> list[tuple[str, list, dict]]:
    """An even sample of each domain's tasks: (task name, prompt messages, info)."""
    rows = []
    for domain in CFG["domains"]:
        ds = get_domain_dataset(domain)
        n, k = len(ds), min(CFG["per_domain"], len(ds))
        for i in range(k):
            row = ds[i * n // k]
            info = row["info"] if isinstance(row["info"], dict) else json.loads(row["info"])
            rows.append((info["task_name"], row["prompt"], info))
    return rows


def new_world(info: dict) -> dict:
    """The task's starting world, built the way the bench's own environment builds it."""
    info = copy.deepcopy(info)
    initial = strip_none_values(info.get("initial_state", {}))
    info["assertions"] = [strip_none_values(a) for a in info.get("assertions", [])]
    world = WorldState(**initial)
    world.meta.allowed_services = compute_allowed_services(initial, info["assertions"], info.get("zapier_tools", []))
    return {"world": world, "info": info, "initial_state": copy.deepcopy(initial), "calls": 0}


class Handler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP with plain JSON answers; the path's random token picks the task's world."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        token = self.path.strip("/").split("/")[-1]
        msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        task = WORLDS.get(token)
        if task is None:
            return self._send(404, {})
        if "id" not in msg:
            return self._send(202, None)
        method, params = msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                      "serverInfo": {"name": "automationbench", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name, args = params.get("name"), dict(params.get("arguments") or {})
            args = {k: v for k, v in args.items() if not (isinstance(v, dict) and not v)}
            with LOCK:
                task["calls"] += 1
                try:
                    func = FUNCS[name]
                    text = func(world=task["world"], **args) if name == "api_fetch" else func(**args)
                    result = {"content": [{"type": "text", "text": str(text)}]}
                except Exception as exc:
                    result = {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
        else:
            return self._send(200, {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "unknown method"}})
        self._send(200, {"jsonrpc": "2.0", "id": msg["id"], "result": result})

    def do_DELETE(self):
        self._send(200, None)

    def _send(self, code: int, body):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def score(task: dict) -> dict:
    """The bench's partial credit; a task passes only when every scored assertion holds."""
    state = {"info": task["info"], "world": task["world"], "initial_state": task["initial_state"]}
    credit = partial_credit(state)
    return {"partial": credit, "passed": credit == 1.0,
            "failed": [r["type"] for r in state.get("_assertion_results", []) if not r["passed"] and not r.get("excluded")]}


async def run_one(name: str, prompt: list, info: dict, port: int, gate: asyncio.Semaphore) -> dict:
    token = secrets.token_hex(16)
    WORLDS[token] = task = new_world(info)
    async with gate:
        home = Path(f"/home/{AGENT_USER}/runs/{name}")
        subprocess.run(["sudo", "-u", AGENT_USER, "mkdir", "-p", f"{home}/home/.manga-agent", f"{home}/ws"], check=True)
        profile = {"advisor": True, "token_budget": 10 ** 12, "max_steps": CFG["max_steps"],
                   "max_output_tokens": CFG["max_output_tokens"], "models": {".": {"request": CFG["request"]}}}
        mcp = {"mcpServers": {"world": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp/{token}"}}}
        for file, data in (("profile.json", profile), ("mcp.json", mcp)):
            subprocess.run(["sudo", "-u", AGENT_USER, "tee", f"{home}/home/.manga-agent/{file}"], input=json.dumps(data).encode(),
                           stdout=subprocess.DEVNULL, check=True)
        # An arm's note (such as a push toward chains) follows the task's own prompt.
        instruction = "\n\n".join([m["content"] for m in prompt] + ([CFG["note"]] if CFG.get("note") else []))
        command = (f"cd {home}/ws && HOME={home}/home AGENT_API_KEY=relay PYTHONPATH=/opt/chaos "
                   f"{shlex.quote(CFG['agent_python'])} -m app.agent.cli exec "
                   f"{shlex.quote(instruction)} --base {shlex.quote(CFG['base'])} --model {shlex.quote(CFG['model'])} --provider bench "
                   f"--allow-private-base --workspace {home}/ws --mode bypass --sandbox full-access --timeout-min {CFG['minutes']} --json")
        started = time.time()
        proc = await asyncio.create_subprocess_exec("sudo", "-u", AGENT_USER, "bash", "-c", command,
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), CFG["minutes"] * 60 + 120)
        except asyncio.TimeoutError:
            proc.kill()
            out, err = b"", b"killed at the time limit"
        seconds = time.time() - started
    events = [json.loads(line) for line in out.decode(errors="replace").splitlines() if line.startswith("{")]
    (OUT / "events").mkdir(parents=True, exist_ok=True)
    (OUT / "events" / f"{name}.jsonl").write_text(out.decode(errors="replace") + "\n# stderr\n" + err.decode(errors="replace")[-4000:])
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    with LOCK:
        row = {"task": name, **score(task), "api_calls": task["calls"], "seconds": round(seconds),
               "status": result.get("status"), "usage": result.get("usage")}
    del WORLDS[token]
    print(json.dumps({k: row[k] for k in ("task", "passed", "partial", "api_calls", "seconds", "status")}), flush=True)
    return row


def write_summary(rows: list[dict]) -> None:
    """Rows and pass counts per domain into OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    by_domain: dict[str, list] = {}
    for r in rows:
        by_domain.setdefault(r["task"].split(".")[0], []).append(r["passed"])
    summary = {d: f"{sum(v)}/{len(v)}" for d, v in by_domain.items()}
    summary["total"] = f"{sum(r['passed'] for r in rows)}/{len(rows)}"
    summary["partial_mean"] = round(sum(r["partial"] for r in rows) / max(1, len(rows)), 3)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


async def main() -> None:
    global TOOLS
    env = AutomationBenchEnv(dataset=get_domain_dataset("simple"), rubric=create_rubric(), toolset="api")
    TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"], "inputSchema": t["function"]["parameters"]}
             for t in env._all_oai_tools]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gate = asyncio.Semaphore(CFG["concurrent"])
    tasks = sample()
    print(f"{len(tasks)} tasks", flush=True)
    rows = await asyncio.gather(*(run_one(n, p, i, server.server_address[1], gate) for n, p, i in tasks))
    write_summary(rows)


def native_summary(path: str) -> None:
    """Pass counts from the bench's own runner export, in the same shape as ours."""
    outputs = json.load(open(path))["outputs"]
    write_summary([{"task": o["task"], "passed": o["reward"] == 1.0, "partial": o["reward"]} for o in outputs])


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        print(",".join(name for name, _, _ in sample()))
    elif sys.argv[1:2] == ["native-summary"]:
        native_summary(sys.argv[2])
    else:
        asyncio.run(main())
