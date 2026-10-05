"""Agent sessions: the tool-calling loop and the approval gate."""
from __future__ import annotations

import json
import platform
import threading
import time
import uuid

from app.agent import client
from app.agent.tools import COMMAND_TOOLS, EDIT_TOOLS, ToolError, Workspace
from app.logging_config import logger

MODES = ("ask", "edits", "auto")
MAX_STEPS = 60
MAX_HISTORY_CHARS = 400_000
MAX_SESSIONS = 50
SYSTEM_PROMPT = """You are a coding agent working inside the Manga Translator app, like Claude Code or Codex.
The workspace root is {root} on {system}. All paths are relative to it.
Look before you change: list, search and read the code first. Make small, exact edits with edit_file.
Run the project's tests or the command that proves your change works. Report briefly what you did and what is left.
Reply in the language the user writes in."""


class AgentSession:
    """One conversation with one model, its history, events for the UI and the approval gate."""

    def __init__(self, session_id: str, provider, api_key: str, model: str, workspace: Workspace, mode: str,
                 complete=client.complete):
        self.id = session_id
        self.provider, self.api_key, self.model = provider, api_key, model
        self.workspace, self.mode = workspace, mode
        self.complete = complete
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.status = "idle"
        self.text_tools = False
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.created_at = time.time()
        self.pending: dict | None = None
        self._decision: dict | None = None
        self._stop = False
        self._lock = threading.Condition()
        self._thread: threading.Thread | None = None

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events) + 1, "type": kind, "time": round(time.time(), 3), **data})
            self._lock.notify_all()

    def snapshot(self, after: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "status": self.status, "provider": self.provider.id, "provider_label": self.provider.label,
                    "model": self.model, "mode": self.mode, "workspace": str(self.workspace.root), "text_tools": self.text_tools,
                    "usage": dict(self.usage), "pending": self.pending,
                    "events": [e for e in self.events if e["seq"] > after]}

    def send(self, text: str) -> None:
        with self._lock:
            if self.status in ("running", "waiting"):
                raise RuntimeError("The agent is still working")
            self.status, self._stop = "running", False
        self.history.append({"role": "user", "content": text})
        self.emit("user", text=text)
        self._thread = threading.Thread(target=self._loop, name=f"agent-{self.id}", daemon=True)
        self._thread.start()

    def decide(self, decision: str, note: str = "") -> None:
        with self._lock:
            if self.status != "waiting" or self.pending is None:
                raise RuntimeError("Nothing is waiting for approval")
            self._decision = {"decision": decision, "note": note}
            self._lock.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()

    def _needs_approval(self, name: str) -> bool:
        if self.mode == "auto":
            return False
        return name in COMMAND_TOOLS or (name in EDIT_TOOLS and self.mode == "ask")

    def _wait_for_decision(self, call: dict) -> dict:
        with self._lock:
            self.pending, self._decision, self.status = call, None, "waiting"
        self.emit("approval", call=call)
        with self._lock:
            while self._decision is None and not self._stop:
                self._lock.wait(timeout=1.0)
            decision = self._decision or {"decision": "deny", "note": "stopped"}
            self.pending, self.status = None, "running"
        return decision

    def _trim(self) -> None:
        """Oldest tool results give way first when the conversation grows past what a model can take."""
        size = sum(len(json.dumps(item, ensure_ascii=False)) for item in self.history)
        for item in self.history:
            if size <= MAX_HISTORY_CHARS:
                return
            if item["role"] == "tool" and len(item["content"]) > 200:
                size -= len(item["content"]) - 40
                item["content"] = "[older tool output removed to save room]"

    def _turn(self) -> dict:
        system = SYSTEM_PROMPT.format(root=self.workspace.root, system=platform.system())
        try:
            return self.complete(self.provider, self.api_key, self.model,
                                 client.render(self.history, system, self.text_tools), use_tools=not self.text_tools)
        except client.ToolsUnsupported as exc:
            self.text_tools = True
            self.emit("notice", text=f"Model không nhận gọi công cụ kiểu gốc, chuyển sang gọi công cụ bằng văn bản ({exc}).")
            return self.complete(self.provider, self.api_key, self.model,
                                 client.render(self.history, system, True), use_tools=False)

    def _run_call(self, call: dict) -> tuple[str, bool]:
        if call.get("error"):
            return call["error"], False
        if self._needs_approval(call["name"]):
            decision = self._wait_for_decision(call)
            if decision["decision"] == "allow_all":
                self.mode = "auto"
            elif decision["decision"] != "allow":
                note = f" Note from the user: {decision['note']}" if decision.get("note") else ""
                return f"The user refused this {call['name']} call.{note}", False
        try:
            return self.workspace.run(call["name"], call["args"]), True
        except ToolError as exc:
            return f"Error: {exc}", False
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}", False

    def _loop(self) -> None:
        try:
            for _ in range(MAX_STEPS):
                if self._stop:
                    self.emit("notice", text="Đã dừng.")
                    break
                self._trim()
                turn = self._turn()
                for key in self.usage:
                    self.usage[key] += int(turn["usage"].get(key) or 0)
                calls = turn["calls"]
                self.history.append({"role": "assistant", "content": turn["text"], "calls": calls})
                self.emit("assistant", text=turn["text"], reasoning=turn["reasoning"][-4000:], calls=calls)
                if not calls:
                    break
                for call in calls:
                    if self._stop:
                        output, ok = "Stopped by the user before this call ran.", False
                    else:
                        output, ok = self._run_call(call)
                    self.history.append({"role": "tool", "id": call["id"], "name": call["name"], "content": output})
                    self.emit("tool", id=call["id"], name=call["name"], ok=ok, output=output)
            else:
                self.emit("notice", text=f"Dừng sau {MAX_STEPS} bước; nhắn tiếp để agent làm tiếp.")
        except Exception as exc:
            logger.opt(exception=True).warning("Agent session {} failed", self.id)
            self.emit("error", text=str(exc)[:1000])
        finally:
            with self._lock:
                self.status = "idle"
                self._lock.notify_all()
            self.emit("done")


class AgentSessionManager:
    """Live sessions by id; they last until the app stops."""

    def __init__(self):
        self.sessions: dict[str, AgentSession] = {}

    def create(self, provider, api_key: str, model: str, workspace: Workspace, mode: str, complete=client.complete) -> AgentSession:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        idle = [s for s in self.sessions.values() if s.status == "idle"]
        while len(self.sessions) >= MAX_SESSIONS and idle:
            self.sessions.pop(min(idle, key=lambda s: s.created_at).id, None)
            idle = [s for s in self.sessions.values() if s.status == "idle"]
        session = AgentSession(uuid.uuid4().hex[:16], provider, api_key, model, workspace, mode, complete)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            raise KeyError(session_id)
        session.stop()
