"""Windows agent probe: can a model call tools at all when the app runs on Windows?

Prints platform/sandbox diagnostics, checks sandbox.run directly, then drives a
real agent session (MiniRouter, deepseek-v4.1-flash-1017) through three questions
that each need a tool. Every session event is saved to agent-probe-events.json and
a VERDICT line ends the summary. Exit 0 only when tools demonstrably worked.
"""
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

# A stock Windows console is cp1252: printing Vietnamese would crash the probe
# before the session even starts, so force UTF-8 with replacement.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

summary = []


def note(line):
    summary.append(line)
    (APP / "agent-probe-summary.txt").write_text("\n".join(summary), encoding="utf-8")
    try:
        print(line, flush=True)
    except Exception:
        pass


note(f"platform: {platform.platform()} | python {sys.version.split()[0]} | os.name={os.name}")

from app.agent import sandbox

note(f"sandbox backend: {sandbox.backend()} | engine settings: {sandbox.settings()} | docker_ready: {sandbox.docker_ready()}")
note(f"bash: {shutil.which('bash')} | cmd: {shutil.which('cmd')} | powershell: {shutil.which('powershell')}")

ws = APP / "agent-probe-ws"
ws.mkdir(exist_ok=True)
(ws / "hello.txt").write_text("NOI DUNG BI MAT 12345\n", encoding="utf-8")

try:
    code, out = sandbox.run("echo SANDBOX_OK", sandbox.Policy("workspace-write", False), ws, 60)
    note(f"sandbox.run direct: exit={code} out={out.strip()[:200]!r}")
except Exception as exc:
    note(f"sandbox.run direct FAILED: {type(exc).__name__}: {exc}")

key = os.environ.get("MINIROUTER_API_KEY", "").strip()
if not key:
    note("VERDICT: NO KEY - MINIROUTER_API_KEY secret is not set for this run")
    (APP / "agent-probe-summary.txt").write_text("\n".join(summary), encoding="utf-8")
    sys.exit(2)

from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace
from app.ai_providers import resolve_provider

home = Path.home()
manager = AgentSessionManager(home / ".manga-agent", home=home)
provider = resolve_provider("minirouter", label="minirouter", protocol="openai",
                            api_base="https://api.minirouter.sh/v1")
session = manager.create(provider, [key], "deepseek/deepseek-v4.1-flash-1017",
                         Workspace(str(ws), sandbox.Policy("workspace-write", False)), "bypass")
session.set_deadline(10 * 60)

QUESTIONS = [
    "bạn đang đứng ở thư mục nào",
    "đọc file hello.txt trong thư mục hiện tại và cho tôi biết nội dung của nó",
    "tạo file ketqua.txt với nội dung là số 42, sau đó đọc lại để kiểm chứng",
]

all_events = []
tool_calls = 0


def drain(seen):
    global tool_calls
    deadline = time.time() + 240
    snap = session.snapshot(seen)
    while True:
        for ev in snap["events"]:
            all_events.append(ev)
            seen = max(seen, ev["seq"])
            calls = ev.get("calls") or []
            for c in calls:
                tool_calls += 1
                note(f"TOOL CALL: {c.get('name')} {json.dumps(c.get('args', {}), ensure_ascii=False)[:200]}")
            if ev.get("type") == "assistant" and ev.get("text"):
                note(f"ASSISTANT: {ev['text'][:300]}")
            elif ev.get("type") != "assistant":
                note(f"EVENT {ev.get('type')}: {json.dumps(ev, ensure_ascii=False)[:300]}")
        if snap["status"] == "waiting":
            session.decide("allow", "probe auto-approve")
        elif snap["status"] == "idle":
            return seen
        if time.time() > deadline:
            session.stop()
            note("TIMEOUT waiting for the session")
            return seen
        time.sleep(0.5)
        snap = session.snapshot(seen)


seen = 0
for q in QUESTIONS:
    note(f"=== USER: {q}")
    session.send(q)
    seen = drain(seen)
    (APP / "agent-probe-events.json").write_text(json.dumps(all_events, ensure_ascii=False, indent=1), encoding="utf-8")

usage = session.snapshot(seen).get("usage")
manager.close_all()

(APP / "agent-probe-events.json").write_text(json.dumps(all_events, ensure_ascii=False, indent=1), encoding="utf-8")
created = (ws / "ketqua.txt").exists()
note(f"ketqua.txt created on disk: {created}")
note(f"total tool calls: {tool_calls} | usage: {json.dumps(usage, ensure_ascii=False)}")
if tool_calls and created:
    verdict = "TOOLS WORK on Windows"
elif tool_calls:
    verdict = "MODEL CALLED TOOLS but the work did not land (execution broken)"
else:
    verdict = "MODEL NEVER CALLED ANY TOOL"
note(f"VERDICT: {verdict}")
(APP / "agent-probe-summary.txt").write_text("\n".join(summary), encoding="utf-8")
sys.exit(0 if verdict.startswith("TOOLS WORK") else 1)
