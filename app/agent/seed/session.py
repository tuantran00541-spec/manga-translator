"""SeedSession: the tool-less harness loop.

The model starts with exactly ONE tool — ``define`` — and no other way to
act. Every turn it may define a capability (Python code against the fixed
``harness`` API) or call one it already defined. The tool surface therefore
grows as the session goes: the model builds its own harness.

Persisted capabilities (``persist: true``, approved by the human) are saved
under ``~/.seed/caps/`` and auto-loaded next session. Everything defined
and called is written to the audit log.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

DEFINE_SPEC = {
    "name": "define",
    "description": (
        "Define a new capability (tool) for yourself. You start with NO other tools: "
        "no read_file, no shell, no web search. To do anything, first define what you need. "
        "Capability code is Python and MUST define `def main(args: dict) -> str`. "
        "A global object named `harness` is available inside your code — it is your ONLY "
        "contact with the outside world (no imports except: json, re, math, datetime, "
        "collections, itertools, functools, hashlib, base64, random, string, urllib.parse). "
        "harness.read(path)->str, harness.write(path, text)->str, harness.ls(pattern)->[str], "
        "harness.run(cmd, timeout=30)->str ('exit=N' + output, OS-sandboxed, no network), "
        "harness.fetch(url)->str, harness.spawn(argv)->handle for line-based subprocesses "
        "(e.g. an MCP server you wrote), harness.log(msg). "
        "Redefining an existing name replaces it (old version stays in the audit log)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Capability name, e.g. read_file"},
            "description": {"type": "string", "description": "What it does, shown in your tool list"},
            "parameters": {"type": "object", "description": "JSON schema of its arguments"},
            "code": {"type": "string", "description": "Python source defining def main(args)"},
            "persist": {"type": "boolean",
                        "description": "Keep for future sessions (asks the human first). Default false: dies with this session."},
        },
        "required": ["name", "description", "code"],
    },
}

SYSTEM_PROMPT = """You are Seed, an agent inside a harness that has NO built-in tools.

You begin with exactly one tool: `define`. There is no read_file, no shell, no web search, nothing else. If you want to read a file, you must FIRST define a `read_file` capability yourself, using the `harness` API described in the define tool.

How to work:
1. Look at the task. Decide the 2-4 capabilities you need (e.g. read_file, write_file, maybe a search).
2. Define them one by one with the `define` tool. Keep each capability small and composable.
3. Call them to do the task. If a capability is clumsy, redefine it better.
4. When the task is done, reply with a short summary and NO tool calls.

Rules:
- Capability code must define `def main(args: dict) -> str`. A global `harness` object is pre-injected; do not import anything except the allowed stdlib modules.
- Do not define dozens of tools up front. Define what the task needs, use it, then extend if you must.
- If a capability call returns an error, read it: it tells you what went wrong. Fix the capability or your arguments.
- `persist: true` asks the human to keep a capability for future sessions. Only persist tools that proved genuinely useful.
- You may also write MCP servers as files (harness.write) and talk to them via harness.spawn — an MCP client is just another capability.
"""


class SeedSession:
    """A conversation in which the model builds its own tools."""

    def __init__(self, provider, api_key: str, model: str, workspace: str | Path,
                 approve=None, home: str | Path | None = None, complete=None,
                 max_steps: int = 40):
        from .harness import Harness
        from .registry import Registry

        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.approve = approve or (lambda prompt: True)
        self._complete = complete  # injectable for tests
        self.home = Path(home) if home else Path.home() / ".seed"
        self.caps_dir = self.home / "caps"
        self.caps_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = uuid.uuid4().hex[:12]
        self.audit: list[dict] = []
        self.harness = Harness(self.workspace, self._audit)
        self.registry = Registry(self.harness, self._audit)
        self.messages: list[dict] = []
        self._load_persisted()

    # -- audit ----------------------------------------------------------

    def _audit(self, kind: str, detail: str) -> None:
        entry = {"t": time.time(), "session": self.session_id, "kind": kind, "detail": detail}
        self.audit.append(entry)
        try:
            with open(self.home / f"audit-{self.session_id}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # -- persistence -----------------------------------------------------

    def _load_persisted(self) -> None:
        for path in sorted(self.caps_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.registry.define(
                    data["name"], data.get("description", ""), data.get("parameters"),
                    data["code"], persisted=True,
                )
            except Exception as exc:  # noqa: BLE001 — a bad saved cap must not kill boot
                self._audit("capability.load_failed", f"{path.name}: {exc}")

    def _persist(self, cap) -> None:
        path = self.caps_dir / f"{cap.name}.json"
        path.write_text(json.dumps({
            "name": cap.name, "description": cap.description,
            "parameters": cap.parameters, "code": cap.code,
            "version": cap.version,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- the loop ----------------------------------------------------------

    def run(self, task: str) -> str:
        self.messages = [{"role": "user", "content": task}]
        self._audit("session.start", task[:200])
        step = 0
        while self.max_steps <= 0 or step < self.max_steps:
            step += 1
            specs = [DEFINE_SPEC] + self.registry.specs()
            turn = self._turn(specs)
            text = (turn.get("text") or "").strip()
            calls = turn.get("calls") or []
            if text:
                self._audit("assistant.text", text[:300])
            if not calls:
                self._audit("session.done", text[:200])
                return text or "(empty reply)"
            self.messages.append({
                "role": "assistant", "content": text,
                "calls": [{"id": c["id"], "name": c["name"],
                           "args": c.get("args") or c.get("arguments") or {}} for c in calls],
            })
            for call in calls:
                name = call["name"]
                args = call.get("args") or call.get("arguments") or {}
                output = self._dispatch(name, args)
                self.messages.append({"role": "tool", "id": call["id"], "name": name, "content": output})
        self._audit("session.step_limit", "")
        return "(stopped: step limit reached)"

    def _turn(self, specs: list[dict]) -> dict:
        if self._complete is not None:
            return self._complete(self.messages, specs)
        from app.agent import client
        import json as _json

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in self.messages:
            role = m.get("role")
            if role == "assistant" and m.get("calls"):
                messages.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": _json.dumps(c.get("args") or c.get("arguments") or {})}}
                        for c in m["calls"]
                    ],
                })
            elif role == "tool":
                messages.append({"role": "tool", "tool_call_id": m.get("id"),
                                 "content": m.get("content") or ""})
            else:
                messages.append({"role": role, "content": m.get("content") or ""})
        return client.complete(self.provider, self.api_key, self.model,
                               messages, tools=specs)

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "define":
            return self._do_define(args)
        try:
            return self.registry.call(name, args)
        except Exception as exc:  # noqa: BLE001 — model sees the error
            return f"error: {exc}"

    def _do_define(self, args: dict) -> str:
        from .registry import CapabilityError

        name = (args.get("name") or "").strip()
        try:
            result = self.registry.define(
                name, args.get("description") or "",
                args.get("parameters"), args.get("code") or "",
            )
        except CapabilityError as exc:
            return f"error: {exc}"
        if args.get("persist"):
            cap = self.registry._caps[name]
            if self.approve(f"Keep capability '{name}' v{cap.version} for future sessions?"):
                self._persist(cap)
                cap.persisted = True
                return result + " Persisted for future sessions."
            return result + " Not persisted (human declined); it dies with this session."
        return result
