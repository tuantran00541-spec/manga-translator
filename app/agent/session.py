"""Agent sessions: the tool loop, approvals, hooks, skills, MCP tools, sub-agents, compaction and saved state."""
from __future__ import annotations

import atexit
import json
import os
import platform
import threading
import time
import uuid
from pathlib import Path

from app.agent import client, context, mcp, sandbox, skills
from app.agent.tools import KIND, SPECS, ToolError, Workspace, clip
from app.logging_config import logger

MODES = ("ask", "edits", "auto")
MAX_STEPS = 80
SUBAGENT_STEPS = 30
COMPACT_AT = 300_000
MAX_HISTORY_CHARS = 450_000
MAX_SESSIONS = 50
HOOK_TIMEOUT = 60
SUBAGENT_TOOLS = {"list_dir", "read_file", "search", "glob", "web_fetch", "skill"}
SYSTEM_PROMPT = """You are a coding agent inside the Manga Translator app, working like Claude Code or Codex.
Workspace root: {root} on {system}. Paths are relative to it.
{sandbox}
Work in small verified steps: look first (list_dir, glob, search, read_file), then change files with apply_patch or edit_file,
then run the project's tests or the command that proves the change works. Use todo_write to plan work with several steps
and keep it current. Use task to send a read-only helper to research a question when that saves you reading.
Do not re-read a file you just changed; the tool reports failure. Fix root causes; keep changes minimal and in the code's style.
End with a short report of what changed, how you checked it, and anything left. Reply in the language the user writes in."""
SESSION_SPECS = {
    "skill": {"name": "skill", "description": "Load a skill's full instructions by name before doing a task it covers.",
              "parameters": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
    "todo_write": {"name": "todo_write", "description": "Replace the task's plan; each item has content and a status of "
                                                       "pending, in_progress or completed. Keep one item in_progress.",
                   "parameters": {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
                       "type": "object", "required": ["content", "status"], "properties": {
                           "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}}}}}},
    "task": {"name": "task", "description": "Send a read-only helper agent to research something in the workspace or on the web; "
                                            "it returns one report. Give it a complete, standalone instruction.",
             "parameters": {"type": "object", "required": ["description", "prompt"], "properties": {
                 "description": {"type": "string", "description": "A few words naming the job."}, "prompt": {"type": "string"}}}},
}
BUILTIN_COMMANDS = {
    "help": "Xem các lệnh", "compact": "Tóm gọn hội thoại để giải phóng chỗ (có thể ghi điều cần giữ)",
    "init": "Viết AGENTS.md mô tả dự án này", "skills": "Xem các skill tìm thấy",
    "mcp": "Xem MCP server và công cụ của chúng", "model": "Đổi model: /model TÊN",
    "mode": "Đổi cách duyệt: /mode ask|edits|auto",
    "sandbox": "Đổi sandbox: /sandbox read-only|workspace-write|full-access [net]",
    "clear": "Mở phiên mới",
}
INIT_PROMPT = ("Study this repository: its layout, how to install, build, run, test and lint it, its code style and conventions. "
               "Then write AGENTS.md at the root with short sections a coding agent needs (overview, commands, structure, "
               "conventions, gotchas). If AGENTS.md exists, improve it instead of replacing what is still right.")


class AgentSession:
    """One conversation with one model in one workspace."""

    def __init__(self, session_id: str, provider, api_key: str, model: str, workspace: Workspace, mode: str, *,
                 complete=client.complete, store: Path | None = None, trust: context.TrustStore | None = None,
                 depth: int = 0, home: Path | None = None):
        self.id, self.provider, self.api_key, self.model = session_id, provider, api_key, model
        self.workspace, self.mode, self.complete, self.store, self.depth = workspace, mode, complete, store, depth
        self.trust = trust or context.TrustStore(None)
        self.home = home
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.todos: list[dict] = []
        self.status = "idle"
        self.text_tools = False
        self.title = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.created_at = self.updated_at = time.time()
        self.pending: dict | None = None
        self._decision: dict | None = None
        self._stop = False
        self._lock = threading.Condition()
        self.allowed_tools: set[str] | None = SUBAGENT_TOOLS if depth else None
        self.skills = skills.discover(workspace.root, home)
        workspace.read_roots = [s.folder for s in self.skills.values()]
        self.commands = context.commands(workspace.root, home) if not depth else {}
        self.hooks = context.hooks(workspace.root, home) if not depth else {"user": {}, "workspace": {}}
        self.mcp_servers: dict[str, mcp.Server] = {}
        self.mcp_status: dict[str, dict] = {}
        self.mcp_tools: dict[str, tuple[str, dict]] = {}
        self._mcp_ready = depth > 0

    # Events, state and persistence.

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events) + 1, "type": kind, "time": round(time.time(), 3), **data})
            self.updated_at = time.time()
            self._lock.notify_all()

    def snapshot(self, after: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "status": self.status, "provider": self.provider.id, "provider_label": self.provider.label,
                    "model": self.model, "mode": self.mode, "workspace": str(self.workspace.root), "title": self.title,
                    "sandbox": {"mode": self.workspace.policy.mode, "network": self.workspace.policy.network,
                                "backend": sandbox.backend()},
                    "text_tools": self.text_tools, "usage": dict(self.usage), "pending": self.pending, "todos": self.todos,
                    "skills": [{"name": s.name, "description": s.description} for s in self.skills.values()],
                    "mcp": list(self.mcp_status.values()), "hooks": self._hook_summary(),
                    "commands": [{"name": k, "description": v} for k, v in BUILTIN_COMMANDS.items()]
                    + [{"name": c["name"], "description": c["description"]} for c in self.commands.values()],
                    "events": [e for e in self.events if e["seq"] > after]}

    def save(self) -> None:
        if self.store is None:
            return
        data = {"id": self.id, "provider": self.provider.id, "model": self.model, "mode": self.mode, "title": self.title,
                "workspace": str(self.workspace.root), "sandbox": [self.workspace.policy.mode, self.workspace.policy.network],
                "created_at": self.created_at, "updated_at": self.updated_at, "usage": self.usage, "todos": self.todos,
                "text_tools": self.text_tools, "history": self.history, "events": self.events}
        tmp = self.store.with_suffix(".tmp")
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.store)
        except OSError:
            logger.opt(exception=True).warning("Could not save agent session {}", self.id)

    def restore(self, data: dict) -> None:
        self.history, self.events = list(data.get("history") or []), list(data.get("events") or [])
        self.todos, self.title = list(data.get("todos") or []), str(data.get("title") or "")
        self.usage.update(data.get("usage") or {})
        self.text_tools = bool(data.get("text_tools"))
        self.created_at = float(data.get("created_at") or self.created_at)

    def set_policy(self, mode: str, network: bool) -> None:
        if mode not in sandbox.MODES:
            raise ValueError(f"sandbox must be one of {', '.join(sandbox.MODES)}")
        self.workspace.policy = sandbox.Policy(mode, bool(network))

    # Starting a turn, approvals and stopping.

    def send(self, text: str) -> None:
        with self._lock:
            if self.status in ("running", "waiting"):
                raise RuntimeError("The agent is still working")
            self.status, self._stop = "running", False
        self.title = self.title or text.strip().splitlines()[0][:80]
        self.history.append({"role": "user", "content": text})
        self.emit("user", text=text)
        threading.Thread(target=self._loop, name=f"agent-{self.id}", daemon=True).start()

    def command(self, text: str) -> dict:
        """A slash command: built-ins act at once, custom ones become a message to the agent."""
        name, _, args = text.strip()[1:].partition(" ")
        args = args.strip()
        if name == "help":
            rows = [f"/{k} — {v}" for k, v in BUILTIN_COMMANDS.items()] + [f"/{c['name']} — {c['description']}" for c in self.commands.values()]
            return {"message": "\n".join(rows)}
        if name == "skills":
            return {"message": "\n".join(f"{s.name}: {s.description}" for s in self.skills.values()) or "Không có skill nào."}
        if name == "mcp":
            self._ensure_mcp()
            rows = [f"{m['name']} ({m['scope']}): {m['state']}" + (f", {m['tools']} công cụ" if m.get("tools") else "")
                    + (f" — {m['error']}" if m.get("error") else "") for m in self.mcp_status.values()]
            return {"message": "\n".join(rows) or "Chưa cấu hình MCP server nào (.mcp.json, ~/.claude.json, ~/.codex/config.toml)."}
        if name == "model" and args:
            self.model = args
            return {"message": f"Model: {args}"}
        if name == "mode" and args in MODES:
            self.mode = args
            return {"message": f"Quyền: {args}"}
        if name == "sandbox" and args:
            parts = args.split()
            self.set_policy(parts[0], len(parts) > 1 and parts[1] in ("net", "network", "on"))
            return {"message": sandbox.Policy(parts[0], self.workspace.policy.network).describe(self.workspace.root)}
        if name == "compact":
            with self._lock:
                if self.status != "idle":
                    raise RuntimeError("The agent is still working")
                self.status = "running"
            threading.Thread(target=self._compact_job, args=(args,), daemon=True).start()
            return {"message": "Đang tóm gọn hội thoại…"}
        if name == "init":
            self.send(INIT_PROMPT)
            return {"sent": True}
        if name in self.commands:
            self.send(context.expand_command(self.commands[name]["body"], args))
            return {"sent": True}
        raise ValueError(f"Unknown command /{name}; type /help")

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

    def close(self) -> None:
        self.stop()
        for server in self.mcp_servers.values():
            server.close()
        self.mcp_servers.clear()

    # MCP servers.

    def _ensure_mcp(self) -> None:
        if self._mcp_ready:
            return
        self._mcp_ready = True
        for row in mcp.configured(self.workspace.root, self.home):
            name, digest = row["name"], mcp.config_hash(row["config"])
            status = {"name": name, "scope": row["scope"], "source": row["source"], "digest": digest, "tools": 0, "error": ""}
            self.mcp_status[name] = status
            if row["config"].get("disabled") or row["config"].get("enabled") is False:
                status["state"] = "disabled"
            elif row["scope"] == "workspace" and not self.trust.trusted(self.workspace.root, f"mcp:{name}", digest):
                status["state"] = "untrusted"
            else:
                self._start_mcp(name, row["config"])

    def _start_mcp(self, name: str, config: dict) -> None:
        status = self.mcp_status[name]
        try:
            server = mcp.Server(name, config, self.workspace.root)
        except Exception as exc:
            status.update(state="failed", error=str(exc)[:300])
            return
        self.mcp_servers[name] = server
        for tool in server.tools:
            self.mcp_tools[mcp.tool_name(name, tool["name"])] = (name, tool)
        status.update(state="running", tools=len(server.tools), error="")

    def trust_mcp(self, name: str) -> None:
        self._ensure_mcp()
        status = self.mcp_status.get(name)
        if status is None:
            raise KeyError(name)
        row = next(r for r in mcp.configured(self.workspace.root, self.home) if r["name"] == name)
        self.trust.allow(self.workspace.root, f"mcp:{name}", mcp.config_hash(row["config"]))
        if status["state"] != "running":
            self._start_mcp(name, row["config"])

    # Hooks.

    def _hook_summary(self) -> dict:
        workspace = self.hooks.get("workspace") or {}
        digest = mcp.config_hash(workspace) if workspace else ""
        return {"user": sum(len(v) for v in (self.hooks.get("user") or {}).values()),
                "workspace": sum(len(v) for v in workspace.values()),
                "workspace_trusted": bool(digest) and self.trust.trusted(self.workspace.root, "hooks", digest)}

    def trust_hooks(self) -> None:
        workspace = self.hooks.get("workspace") or {}
        if workspace:
            self.trust.allow(self.workspace.root, "hooks", mcp.config_hash(workspace))

    def _run_hooks(self, event: str, call: dict, output: str | None = None) -> tuple[bool, str]:
        """Matching hooks for one call; a PreToolUse hook exiting 2 blocks the call with its message."""
        rows = list((self.hooks.get("user") or {}).get(event, []))
        if self._hook_summary()["workspace_trusted"]:
            rows += (self.hooks.get("workspace") or {}).get(event, [])
        notes = []
        policy = self.workspace.policy if self.workspace.policy.mode == "full-access" else sandbox.Policy("workspace-write", False)
        payload = {"hook_event_name": event, "tool_name": context.CLAUDE_NAMES.get(call["name"], call["name"]),
                   "tool_input": call["args"], "cwd": str(self.workspace.root), "session_id": self.id}
        if output is not None:
            payload["tool_response"] = output
        for row in rows:
            if not context.hook_matches(row["matcher"], call["name"]):
                continue
            code, err = sandbox.run(row["command"], policy, self.workspace.root, HOOK_TIMEOUT, json.dumps(payload), split=True)
            if code == 2 and event == "PreToolUse":
                return False, f"Blocked by a hook ({row['command']}): {err.strip()}"
            if code != 0 or err.strip():
                notes.append(f"[hook {row['command']} exit {code}] {err.strip()}"[:2000])
        return True, "\n".join(notes)

    # Tools.

    def specs(self) -> list[dict]:
        rows = [s for s in SPECS if self.allowed_tools is None or s["name"] in self.allowed_tools]
        if self.skills:
            rows.append(SESSION_SPECS["skill"])
        if not self.depth:
            rows += [SESSION_SPECS["todo_write"], SESSION_SPECS["task"]]
            for name, (server, tool) in self.mcp_tools.items():
                rows.append({"name": name, "description": f"[MCP {server}] {tool.get('description') or tool['name']}"[:1024],
                             "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}}})
        return rows

    def _needs_approval(self, call: dict) -> bool:
        name = call["name"]
        if self.mode == "auto" or name in SESSION_SPECS:
            return False
        if name in self.mcp_tools:
            read_only = (self.mcp_tools[name][1].get("annotations") or {}).get("readOnlyHint")
            return not (self.mode == "edits" and read_only)
        kind = KIND.get(name, "exec")
        if kind == "read":
            return False
        if self.mode == "ask":
            return True
        if kind == "exec":
            # In edits mode commands run on their own only inside a working OS sandbox.
            confined = sandbox.backend() != "none" and self.workspace.policy.mode != "full-access"
            return bool(call["args"].get("outside_sandbox")) or not confined
        return False

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

    def _session_tool(self, call: dict) -> str:
        args = call["args"]
        if call["name"] == "skill":
            found = self.skills.get(str(args.get("name") or ""))
            if found is None:
                raise ToolError(f"No skill named {args.get('name')!r}; known: {', '.join(self.skills) or 'none'}")
            return skills.load(found)
        if call["name"] == "todo_write":
            items = [{"content": str(i.get("content", ""))[:300], "status": i.get("status") if i.get("status") in
                      ("pending", "in_progress", "completed") else "pending"} for i in args.get("items") or [] if isinstance(i, dict)]
            self.todos = items[:50]
            self.emit("todos", items=self.todos)
            done = sum(i["status"] == "completed" for i in self.todos)
            return f"Plan saved: {done}/{len(self.todos)} done."
        return self._subagent(str(args.get("description") or "task"), str(args.get("prompt") or ""))

    def _subagent(self, description: str, prompt: str) -> str:
        if not prompt.strip():
            raise ToolError("task needs a prompt")
        helper = Workspace(self.workspace.root, sandbox.Policy("read-only", False))
        child = AgentSession(f"{self.id}-sub", self.provider, self.api_key, self.model, helper, "auto",
                             complete=self.complete, depth=self.depth + 1, home=self.home)
        child.text_tools = self.text_tools
        self.emit("subagent", description=description, state="started")
        child.history.append({"role": "user", "content": prompt})
        child.status = "running"
        child._loop(max_steps=SUBAGENT_STEPS)
        for key in self.usage:
            self.usage[key] += child.usage[key]
        answer = next((e["text"] for e in reversed(child.events) if e["type"] == "assistant" and e["text"]), "")
        tools_used = sum(1 for e in child.events if e["type"] == "tool")
        self.emit("subagent", description=description, state="done", tools=tools_used)
        return answer or "The helper finished without a report."

    def _run_call(self, call: dict) -> tuple[str, bool]:
        if call.get("error"):
            return call["error"], False
        known = {s["name"] for s in self.specs()}
        if call["name"] not in known:
            return f"Unknown tool {call['name']!r}; available: {', '.join(sorted(known))}", False
        if self._needs_approval(call):
            decision = self._wait_for_decision(call)
            if decision["decision"] == "allow_all":
                self.mode = "auto"
            elif decision["decision"] != "allow":
                note = f" Note from the user: {decision['note']}" if decision.get("note") else ""
                return f"The user refused this {call['name']} call.{note}", False
        if call["name"] not in SESSION_SPECS:
            allowed, message = self._run_hooks("PreToolUse", call)
            if not allowed:
                return message, False
        try:
            if call["name"] in SESSION_SPECS:
                output, ok = self._session_tool(call), True
            elif call["name"] in self.mcp_tools:
                server, tool = self.mcp_tools[call["name"]]
                output, ok = self.mcp_servers[server].call_tool(tool["name"], call["args"])
            else:
                output, ok = self.workspace.run(call["name"], call["args"]), True
        except ToolError as exc:
            output, ok = f"Error: {exc}", False
        except (OSError, mcp.MCPError) as exc:
            output, ok = f"Error: {type(exc).__name__}: {exc}", False
        if call["name"] not in SESSION_SPECS:
            _, notes = self._run_hooks("PostToolUse", call, output)
            if notes:
                output = f"{output}\n{notes}"
        return output, ok

    # The model turn and the loop.

    def system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT.format(root=self.workspace.root, system=platform.system(),
                                      sandbox=self.workspace.policy.describe(self.workspace.root))]
        if not self.depth:
            parts.append(context.instructions(self.workspace.root, self.home))
        else:
            parts.append("You are a read-only helper: research and answer with one complete report; you cannot change files.")
        parts.append(skills.catalog(self.skills))
        return "\n\n".join(p for p in parts if p)

    def _turn(self) -> dict:
        system, specs = self.system_prompt(), self.specs()
        try:
            return self.complete(self.provider, self.api_key, self.model, client.render(self.history, system, self.text_tools, specs),
                                 tools=None if self.text_tools else specs)
        except client.ToolsUnsupported as exc:
            self.text_tools = True
            self.emit("notice", text=f"Model không nhận gọi công cụ kiểu gốc, chuyển sang gọi công cụ bằng văn bản ({exc}).")
            return self.complete(self.provider, self.api_key, self.model, client.render(self.history, system, True, specs), tools=None)

    def _size(self) -> int:
        return sum(len(json.dumps(item, ensure_ascii=False)) for item in self.history)

    def _transcript(self, items: list[dict]) -> str:
        rows = []
        for item in items:
            if item["role"] == "user":
                rows.append(f"USER: {item['content']}")
            elif item["role"] == "assistant":
                calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:300]})" for c in item.get("calls") or [])
                rows.append(f"ASSISTANT: {item.get('content') or ''}" + (f"\nCALLS: {calls}" if calls else ""))
            else:
                rows.append(f"TOOL {item['name']}: {item['content'][:1500]}")
        return clip("\n\n".join(rows), 200_000)

    def compact(self, focus: str = "") -> bool:
        """Replace everything before the latest user message with a summary the model writes."""
        last_user = max((i for i, item in enumerate(self.history) if item["role"] == "user"), default=0)
        head, tail = self.history[:last_user], self.history[last_user:]
        if not head:
            return False
        ask = ("Summarise this conversation between a user and a coding agent so the agent can continue without it: "
               "the goals, decisions, files read and changed with key details, commands run and results, open problems "
               "and next steps. Be specific and complete." + (f" Focus on: {focus}" if focus else ""))
        messages = [{"role": "system", "content": "You write precise handover summaries."},
                    {"role": "user", "content": f"{ask}\n\n<conversation>\n{self._transcript(head)}\n</conversation>"}]
        summary = self.complete(self.provider, self.api_key, self.model, messages, tools=None)["text"]
        if not summary:
            return False
        self.history = [{"role": "user", "content": f"[Summary of the earlier conversation]\n{summary}"},
                        {"role": "assistant", "content": "Understood; continuing from that summary.", "calls": []}] + tail
        self.emit("notice", text=f"Đã tóm gọn {len(head)} mục hội thoại cũ.")
        return True

    def _compact_job(self, focus: str) -> None:
        try:
            if not self.compact(focus):
                self.emit("notice", text="Chưa có gì để tóm gọn.")
        except Exception as exc:
            self.emit("error", text=f"Không tóm gọn được: {exc}"[:1000])
        finally:
            with self._lock:
                self.status = "idle"
            self.emit("done")
            self.save()

    def _make_room(self) -> None:
        if self._size() > COMPACT_AT and not self.depth:
            try:
                self.compact()
            except Exception as exc:
                logger.warning("Agent session {} could not compact: {}", self.id, exc)
        size = self._size()
        for item in self.history:
            if size <= MAX_HISTORY_CHARS:
                return
            if item["role"] == "tool" and len(item["content"]) > 200:
                size -= len(item["content"]) - 40
                item["content"] = "[older tool output removed to save room]"

    def _loop(self, max_steps: int = MAX_STEPS) -> None:
        try:
            self._ensure_mcp()
            for _ in range(max_steps):
                if self._stop:
                    self.emit("notice", text="Đã dừng.")
                    break
                self._make_room()
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
                self.save()
            else:
                self.emit("notice", text=f"Dừng sau {max_steps} bước; nhắn tiếp để agent làm tiếp.")
        except Exception as exc:
            logger.opt(exception=True).warning("Agent session {} failed", self.id)
            self.emit("error", text=str(exc)[:1000])
        finally:
            with self._lock:
                self.status = "idle"
                self._lock.notify_all()
            self.emit("done")
            self.save()


class AgentSessionManager:
    """Live sessions by id, saved under a folder so they can be reopened after a restart."""

    def __init__(self, store_dir: Path | None = None, trust: context.TrustStore | None = None, home: Path | None = None):
        self.store_dir, self.home = store_dir, home
        self.trust = trust or context.TrustStore(store_dir / "trust.json" if store_dir else None)
        self.sessions: dict[str, AgentSession] = {}
        atexit.register(self.close_all)

    def _path(self, session_id: str) -> Path | None:
        return self.store_dir / "sessions" / f"{session_id}.json" if self.store_dir else None

    def create(self, provider, api_key: str, model: str, workspace: Workspace, mode: str, complete=client.complete,
               session_id: str | None = None) -> AgentSession:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        idle = sorted((s for s in self.sessions.values() if s.status == "idle"), key=lambda s: s.updated_at)
        while len(self.sessions) >= MAX_SESSIONS and idle:
            self.sessions.pop(idle.pop(0).id).close()
        session_id = session_id or uuid.uuid4().hex[:16]
        session = AgentSession(session_id, provider, api_key, model, workspace, mode, complete=complete,
                               store=self._path(session_id), trust=self.trust, home=self.home)
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def saved(self, session_id: str) -> dict:
        path = self._path(session_id)
        if path is None or not path.is_file() or not session_id.isalnum():
            raise KeyError(session_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def listing(self) -> list[dict]:
        rows = {s.id: {"id": s.id, "title": s.title, "model": s.model, "status": s.status, "updated_at": s.updated_at,
                       "workspace": str(s.workspace.root), "live": True} for s in self.sessions.values()}
        folder = self.store_dir / "sessions" if self.store_dir else None
        for path in sorted(folder.glob("*.json")) if folder and folder.is_dir() else []:
            if path.stem in rows:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows[path.stem] = {"id": path.stem, "title": data.get("title", ""), "model": data.get("model", ""), "status": "saved",
                               "updated_at": data.get("updated_at", 0), "workspace": data.get("workspace", ""), "live": False}
        return sorted(rows.values(), key=lambda r: r["updated_at"], reverse=True)[:100]

    def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            session.close()
        path = self._path(session_id)
        if path is not None and session_id.isalnum() and path.is_file():
            path.unlink()
        elif session is None:
            raise KeyError(session_id)

    def close_all(self) -> None:
        for session in list(self.sessions.values()):
            session.close()
