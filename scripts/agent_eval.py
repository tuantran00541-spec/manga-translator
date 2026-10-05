"""Repeatable scenarios for the Agent on a real model: outcome checks plus steps, tokens and time, written as JSON."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--app-dir", default=".", help="the code under test")
parser.add_argument("--corpus", default=".", help="the checkout whose app/agent files the explore scenario reads")
parser.add_argument("--base", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--key-env", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--scenarios", default="bugfix,explore,bigout,helpers,writers")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.app_dir).resolve()))

from app.agent import sandbox  # noqa: E402
from app.agent.session import AgentSessionManager  # noqa: E402
from app.agent.tools import Workspace  # noqa: E402
from app.ai_providers import resolve_provider  # noqa: E402

SHOP = ('def total(prices, discount=0.0):\n    """Sum of prices after a fractional discount, rounded to cents."""\n'
        "    subtotal = sum(prices)\n    return round(subtotal * discount, 2)\n")
TESTS = ("from shop import total\n\ndef test_no_discount():\n    assert total([1.5, 2.5]) == 4.0\n\n"
         "def test_discount():\n    assert total([10, 10], 0.25) == 15.0\n")
PROMPTS = {
    "bugfix": "The tests fail. Find and fix the bug with the smallest change, then run the tests.",
    "explore": "Read every .py file in this folder (they are in the src directory). For each file give its name, its line count and the name of its longest function. Finish with one table.",
    "bigout": "Run `cat server.log` and tell me which user caused the ERROR line and its error code.",
    "helpers": "Run two helpers at the same time with spawn_agent: an explore helper that says what test_shop.py checks and one that says what shop.py does (never the same job twice), then wait_agent for both and answer in two sentences.",
    "writers": "Start two coder helpers at the same time with spawn_agent: one creates a.txt containing the text A, the other creates b.txt containing the text B. Wait for both and confirm.",
}
provider = resolve_provider(args.label, label=args.label, protocol="openai", api_base=args.base)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def project(name):
    root = Path(tempfile.mkdtemp())
    if name in ("bugfix", "helpers"):
        (root / "shop.py").write_text(SHOP)
        (root / "test_shop.py").write_text(TESTS)
    elif name == "explore":
        src = root / "src"
        src.mkdir()
        for path in sorted((Path(args.corpus) / "app" / "agent").glob("*.py")):
            shutil.copy(path, src / path.name)
    elif name == "bigout":
        lines = [f"INFO request {i} ok user=u{i % 97}" for i in range(30000)]
        lines[17431] = "ERROR code=E4711 user=zeta failed to commit"
        (root / "server.log").write_text("\n".join(lines) + "\n")
    return root


def tests_pass(root):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root, capture_output=True).returncode == 0


def checks_for(name, root, session, events):
    final = next((e["text"] for e in reversed(events) if e["type"] == "assistant" and e["text"]), "")
    tools = [(e["name"], e["ok"], e["output"]) for e in events if e["type"] == "tool"]
    if name == "bugfix":
        edits = [i for i, t in enumerate(tools) if t[0] in ("edit_file", "edit_lines", "apply_patch", "write_file")]
        runs = [i for i, t in enumerate(tools) if t[0] == "run_command"]
        return {"tests_pass": tests_pass(root), "checked_after_last_edit": bool(edits) and bool(runs) and max(runs) > max(edits)}
    if name == "explore":
        stems = [p.stem for p in (root / "src").glob("*.py") if p.stem != "__init__"]
        return {"names_most_files": sum(s in final for s in stems) >= 0.8 * len(stems), "files": len(stems)}
    if name == "bigout":
        return {"found_code": "E4711" in final, "found_user": "zeta" in final}
    if name == "helpers":
        return {"spawned_two": sum(t[0] == "spawn_agent" and t[1] for t in tools) >= 2, "answered": len(final) > 40}
    if name == "writers":
        return {"a_made": (root / "a.txt").exists() and "A" in (root / "a.txt").read_text(),
                "b_made": (root / "b.txt").exists() and "B" in (root / "b.txt").read_text(),
                "queued_once": any(t[2].startswith("Queued ") for t in tools)}
    return {}


def run_one(name, repeat):
    root = project(name)
    agents = AgentSessionManager(Path(tempfile.mkdtemp()), home=Path(tempfile.mkdtemp()))
    session = agents.create(provider, os.environ[args.key_env], args.model, Workspace(root, sandbox.Policy("workspace-write", False)), "edits")
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            if session.status == "waiting" and session.pending:
                session.decide("deny", "no approvals in this eval")
            time.sleep(0.4)

    threading.Thread(target=watcher, daemon=True).start()
    start = time.time()
    session.send(PROMPTS[name])
    time.sleep(1)
    while session.status != "idle" and time.time() - start < 900:
        time.sleep(1)
    if session.status != "idle":
        session.stop()
        time.sleep(3)
    stop.set()
    events = list(session.events)
    checks = checks_for(name, root, session, events)
    ok_keys = [k for k, v in checks.items() if isinstance(v, bool) and k not in ("queued_once",)]
    row = {"label": args.label, "model": args.model, "scenario": name, "repeat": repeat, "ok": all(checks[k] for k in ok_keys), "checks": checks,
           "seconds": round(time.time() - start, 1), "steps": sum(e["type"] == "assistant" for e in events), "usage": session.usage,
           "tools": {n: [e["name"] for e in events if e["type"] == "tool"].count(n) for n in sorted({e["name"] for e in events if e["type"] == "tool"})},
           "errors": [e["text"] for e in events if e["type"] == "error"], "notices": [e["text"] for e in events if e["type"] == "notice"][:5]}
    agents.close_all()
    return row


rows = []
for repeat in range(args.repeats):
    for name in args.scenarios.split(","):
        try:
            row = run_one(name, repeat)
        except Exception as exc:  # one broken scenario must not hide the others
            row = {"label": args.label, "model": args.model, "scenario": name, "repeat": repeat, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
(out / f"{args.label}-{args.model}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
