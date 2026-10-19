"""Agent sessions: the loop that calls the model and runs its tools, with approval, stop and saved history."""
from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from app.agent import client
from app.agent.tools import CHANGES, SPECS, Workspace, clip

# ask: the user approves every file change and command; edits: file changes in the folder run freely, commands are still asked.
MODES = ("ask", "edits")
FILE_CHANGES = {"write_file", "edit_file"}
MAX_STEPS = 200  # model calls in one turn
COMPACT_CHARS = 400_000  # history size that starts shortening old tool results
KEEP_RECENT = 12  # the newest messages are never shortened
OLD_RESULT_CHARS = 2000
AGENTS_FILE_CHARS = 20_000

SYSTEM = """You are Chaos, a coding agent working in the folder {root} on {os}.
Use the tools to read files, change them and run commands. Look before you change anything, keep each edit small, and check your work by running it.
When the task is done, or you need the user, stop calling tools and reply with a short summary."""


def system_prompt(root: Path) -> str:
    text = SYSTEM.format(root=root, os=platform.system() or "an unknown OS")
    rules = root / "AGENTS.md"
    if rules.is_file():
        text += f"\n\nThe folder's AGENTS.md:\n{rules.read_text(encoding='utf-8', errors='replace')[:AGENTS_FILE_CHARS]}"
    return text


def compact(messages: list[dict]) -> None:
    """Shorten old tool results, oldest first, until the history fits again."""
    size = sum(len(m.get("content") or "") for m in messages)
    for message in messages[:-KEEP_RECENT]:
        if size <= COMPACT_CHARS:
            return
        content = message.get("content") or ""
        if message["role"] == "tool" and len(content) > OLD_RESULT_CHARS:
            message["content"] = content[:OLD_RESULT_CHARS] + f"\n… [older result cut from {len(content)} characters]"
            size -= len(content) - len(message["content"])


class Session:
    def __init__(self, data: dict, folder: Path, resolve: Callable):
        self.data = data
        self.folder = folder
        self.resolve = resolve  # provider id -> (provider, api key)
        self.lock = threading.Lock()
        self.answer = threading.Event()
        self.decision: dict = {}
        self.stopping = False
        self.thread: threading.Thread | None = None
        data["status"] = "idle" if data.get("status") in {"running", "waiting"} else data.get("status", "idle")
        data["pending"] = None

    @property
    def id(self) -> str:
        return self.data["id"]

    def save(self) -> None:
        self.data["updated"] = time.time()
        path = self.folder / f"{self.id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def summary(self) -> dict:
        keys = ("id", "title", "provider", "model", "mode", "effort", "workspace", "status", "updated")
        return {k: self.data.get(k) for k in keys}

    def view(self) -> dict:
        with self.lock:
            return {**self.summary(), "messages": list(self.data["messages"]), "pending": self.data.get("pending"),
                    "error": self.data.get("error")}

    def send(self, text: str) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("The agent is still working")
            self.data["messages"].append({"role": "user", "content": text})
            self.data["title"] = self.data.get("title") or text.strip().splitlines()[0][:60]
            self.data.update(status="running", error=None)
            self.stopping = False
            self.save()
        self.thread = threading.Thread(target=self._turn, name=f"agent-{self.id}", daemon=True)
        self.thread.start()

    def approve(self, allow: bool, note: str = "") -> None:
        self.decision = {"allow": allow, "note": note}
        self.answer.set()

    def stop(self) -> None:
        self.stopping = True
        self.answer.set()

    def _turn(self) -> None:
        try:
            provider, api_key = self.resolve(self.data["provider"])
            workspace = Workspace(self.data["workspace"])
            for _ in range(MAX_STEPS):
                if self.stopping:
                    break
                messages = [{"role": "system", "content": system_prompt(workspace.root)}] + self.data["messages"]
                reply = client.chat(provider, self.data["model"], api_key, messages, SPECS, self.data.get("effort") or "")
                calls = reply["calls"]
                with self.lock:
                    message = {"role": "assistant", "content": reply["content"]}
                    if calls:
                        message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                            "name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}} for c in calls]
                    self.data["messages"].append(message)
                    self.save()
                if not calls:
                    break
                for call in calls:
                    result = self._run(workspace, call)
                    with self.lock:
                        self.data["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": result})
                        compact(self.data["messages"])
                        self.save()
            else:
                self.data["error"] = f"Stopped after {MAX_STEPS} model calls in one turn"
        except Exception as exc:  # The turn ends with the reason shown, whatever broke.
            self.data["error"] = str(exc) or exc.__class__.__name__
        with self.lock:
            self.data.update(status="error" if self.data.get("error") else "idle", pending=None)
            self.save()

    def _run(self, workspace: Workspace, call: dict) -> str:
        if call.get("error"):
            return f"Error: {call['error']}"
        if self.stopping:
            return "Not run: the user stopped the agent."
        if call["name"] in CHANGES and not (call["name"] in FILE_CHANGES and self.data.get("mode") == "edits"):
            self.answer.clear()
            with self.lock:
                self.data.update(status="waiting", pending={"id": call["id"], "name": call["name"], "args": call["args"]})
            self.answer.wait()
            with self.lock:
                self.data.update(status="running", pending=None)
            if self.stopping:
                return "Not run: the user stopped the agent."
            if not self.decision.get("allow"):
                note = self.decision.get("note") or ""
                return "The user declined this call." + (f" They said: {note}" if note else "")
        return clip(workspace.run(call["name"], call["args"]))


class SessionManager:
    def __init__(self, folder: Path, resolve: Callable):
        self.folder = folder
        self.resolve = resolve
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def create(self, provider: str, model: str, workspace: str, mode: str = "ask", effort: str = "") -> Session:
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        root = Workspace(workspace).root
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {"id": uuid.uuid4().hex[:12], "title": "", "provider": provider, "model": model, "mode": mode,
                "effort": effort, "workspace": str(root), "status": "idle", "messages": [], "created": time.time()}
        session = Session(data, self.folder, self.resolve)
        session.save()
        with self.lock:
            self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
            path = self.folder / f"{session_id}.json"
            if not session_id.isalnum() or not path.is_file():
                raise KeyError(session_id)
            session = Session(json.loads(path.read_text(encoding="utf-8")), self.folder, self.resolve)
            self.sessions[session_id] = session
            return session

    def list(self) -> list[dict]:
        found = []
        for path in self.folder.glob("*.json") if self.folder.is_dir() else []:
            try:
                found.append(self.get(path.stem).summary())
            except (KeyError, ValueError, OSError):
                continue
        return sorted(found, key=lambda s: -(s.get("updated") or 0))

    def delete(self, session_id: str) -> None:
        session = self.get(session_id)
        session.stop()
        with self.lock:
            self.sessions.pop(session_id, None)
        (self.folder / f"{session_id}.json").unlink(missing_ok=True)
