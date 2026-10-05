"""Repeatable scenarios for the Agent on a real model: outcome checks plus steps, tokens and time, written as JSON."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--app-dir", default=".", help="the code under test")
parser.add_argument("--base", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--key-env", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--timeout-min", type=int, default=90)
parser.add_argument("--scenarios", default="research,build,debug,server,longrun")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.app_dir).resolve()))

from app.agent import sandbox  # noqa: E402
from app.agent.session import AgentSessionManager  # noqa: E402
from app.agent.tools import Workspace  # noqa: E402
from app.ai_providers import resolve_provider  # noqa: E402

INVENTORY = '''class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, name, qty, price):
        if name in self.items:
            self.items[name]["qty"] = qty
        else:
            self.items[name] = {"qty": qty, "price": price}

    def remove(self, name, qty):
        item = self.items[name]
        if qty > item["qty"]:
            raise ValueError("not enough stock")
        item["qty"] -= qty
        if item["qty"] == 0:
            del self.items[name]

    def total_value(self):
        return sum(i["qty"] * i["price"] for i in self.items.values())

    def low_stock(self, threshold):
        return sorted(n for n, i in self.items.items() if i["qty"] < threshold)
'''
INVENTORY_TESTS = '''import pytest
from inventory import Inventory


def test_add_accumulates():
    inv = Inventory()
    inv.add("nut", 5, 0.1)
    inv.add("nut", 3, 0.1)
    assert inv.items["nut"]["qty"] == 8


def test_remove_deletes_empty_items():
    inv = Inventory()
    inv.add("nut", 2, 0.1)
    inv.remove("nut", 2)
    assert "nut" not in inv.items
    with pytest.raises(KeyError):
        inv.remove("nut", 1)


def test_remove_too_many():
    inv = Inventory()
    inv.add("nut", 2, 0.1)
    with pytest.raises(ValueError):
        inv.remove("nut", 3)


def test_total_value_rounds_to_cents():
    inv = Inventory()
    inv.add("nut", 3, 0.1)
    assert inv.total_value() == 0.3


def test_low_stock_is_inclusive():
    inv = Inventory()
    inv.add("a", 5, 1)
    inv.add("b", 4, 1)
    assert inv.low_stock(5) == ["a", "b"]
'''
PROMPTS = {
    "research": "Find out on the web: the latest stable release of Python (its version number and release date) and the year Python 3.0 was released. "
                "Write answer.md with both answers and the URLs you used as sources.",
    "build": "In this empty folder build a small Python command-line tool wordfreq.py that reads text files (or stdin) and prints the N most common words, "
             "with flags --top N, --ignore-case and --min-length N. Write a pytest file test_wordfreq.py that covers every flag, and run the tests until they pass.",
    "debug": "The tests in this project fail and I do not know why. Find the causes, fix the code (not the tests), and prove all tests pass.",
    "server": "Start a web server in the background that serves a page whose title is 'Hello Agent' on port 8765. Fetch it from the command line, "
              "confirm the title, then stop the server and say what you did.",
    "longrun": "Implement a Markdown to HTML converter in Python (md2html.py with a function convert(text) -> str) supporting headings, paragraphs, bold, "
               "italics, inline code, fenced code blocks, ordered and unordered lists, blockquotes and links, with a thorough pytest suite (test_md2html.py). "
               "Keep going until the whole suite passes and every feature is tested.",
}
provider = resolve_provider(args.label, label=args.label, protocol="openai", api_base=args.base)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def project(name):
    root = Path(tempfile.mkdtemp())
    if name == "debug":
        (root / "inventory.py").write_text(INVENTORY)
        (root / "test_inventory.py").write_text(INVENTORY_TESTS)
    return root


def tests_pass(root):
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root, capture_output=True, text=True)
    return done.returncode == 0, done.stdout


def checks_for(name, root, events):
    final = next((e["text"] for e in reversed(events) if e["type"] == "assistant" and e["text"]), "")
    calls = [c for e in events if e["type"] == "assistant" for c in e.get("calls", [])]
    names = [c["name"] for c in calls]
    if name == "research":
        text = (root / "answer.md").read_text() if (root / "answer.md").exists() else ""
        return {"answer_file": bool(text), "has_2008": "2008" in text, "has_version": any(t.startswith("3.") for t in text.replace(",", " ").split()),
                "has_source_url": "http" in text, "used_web": any(n in ("web_search", "web_fetch") for n in names)}
    if name == "build":
        passed, report = tests_pass(root)
        return {"tool_exists": (root / "wordfreq.py").exists(), "tests_pass": passed, "enough_tests": report.count("passed") and int(report.split(" passed")[0].split()[-1]) >= 4 if passed else False}
    if name == "debug":
        passed, _ = tests_pass(root)
        tests = (root / "test_inventory.py").read_text()
        return {"tests_pass": passed, "tests_untouched": tests == INVENTORY_TESTS or tests.count("def test_") >= 5}
    if name == "server":
        used_bg = any(c["name"] == "run_command" and c["args"].get("background") for c in calls)
        left = subprocess.run(["pgrep", "-f", "8765"], capture_output=True, text=True).stdout.split()
        return {"used_background_job": used_bg, "said_hello_agent": "Hello Agent" in final, "server_stopped": not left}
    if name == "longrun":
        passed, report = tests_pass(root)
        count = int(report.split(" passed")[0].split()[-1]) if passed and " passed" in report else 0
        return {"converter_exists": (root / "md2html.py").exists(), "tests_pass": passed, "enough_tests": count >= 15}
    return {}


def run_one(name, repeat):
    root = project(name)
    agents = AgentSessionManager(Path(tempfile.mkdtemp()), home=Path(tempfile.mkdtemp()))
    # Everything on: all tools, a sandbox confining writes to the folder, the network on only where the task is a server.
    session = agents.create(provider, os.environ[args.key_env], args.model, Workspace(root, sandbox.Policy("workspace-write", name == "server")), "auto")
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            if session.status == "waiting" and session.pending:
                if session.pending["name"] == "ask_user":
                    session.decide("allow", "Use your best judgment and do not ask again.")
                else:
                    session.decide("deny", "no approvals in this eval")
            time.sleep(0.4)

    threading.Thread(target=watcher, daemon=True).start()
    start = time.time()
    session.send(PROMPTS[name])
    time.sleep(1)
    while session.status != "idle" and time.time() - start < args.timeout_min * 60:
        time.sleep(2)
    timed_out = session.status != "idle"
    if timed_out:
        session.stop()
        time.sleep(3)
    stop.set()
    events = list(session.events)
    checks = checks_for(name, root, events)
    tool_names = [e["name"] for e in events if e["type"] == "tool"]
    row = {"label": args.label, "model": args.model, "scenario": name, "repeat": repeat, "ok": all(v for v in checks.values() if isinstance(v, bool)), "timed_out": timed_out,
           "checks": checks, "seconds": round(time.time() - start), "steps": sum(e["type"] == "assistant" for e in events), "usage": session.usage,
           "tools": {n: tool_names.count(n) for n in sorted(set(tool_names))}, "failed_tools": sum(1 for e in events if e["type"] == "tool" and not e["ok"]),
           "helpers": sum(1 for e in events if e["type"] == "subagent" and e["state"] == "done"),
           "masked": sum(1 for m in session.history if m.get("masked")), "offloaded": sum(1 for m in session.history if "full output saved to" in m.get("content", "")),
           "skills": [c["args"].get("name") for e in events if e["type"] == "assistant" for c in e.get("calls", []) if c["name"] == "skill"],
           "errors": [e["text"] for e in events if e["type"] == "error"][:3], "notices": [e["text"] for e in events if e["type"] == "notice"][:5]}
    clipped = [{**e, "output": e["output"][:1500]} if e["type"] == "tool" else e for e in events]
    (out / f"events-{args.label}-{name}-{repeat}.json").write_text(json.dumps(clipped, ensure_ascii=False, indent=1), encoding="utf-8")
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
