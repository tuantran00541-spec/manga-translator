from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


from app.agent import (
    agents,
    client,
    context,
    mcp,
    memory,
    models,
    registry,
    rules,
    sandbox,
    skill_install,
    skills,
)
from app.agent.session_config import *  # noqa: F401, F403
from app.agent.tools import SPECS, ToolError
from app.logging_config import logger


def _ses(name: str):
    return getattr(sys.modules["app.agent.session"], name)


class AgentCommandsMixin:
    """Command, plugin, MCP, hook, and skill handlers for AgentSession."""

    def _build_registry(self, home_path: Path) -> registry.Registry:
        reg = registry.Registry({s["name"] for s in SPECS} | set(SESSION_SPECS))
        registry.load_plugins(reg, self.workspace.root, home_path, self.trust, self.profile.get("plugins"))
        return reg

    def _enabled(self, name: str) -> bool:
        plugin = self.registry.tools.get(name)
        return (plugin.group if plugin else registry.group_of(name)) not in self.disabled

    def trust_plugins(self) -> None:
        home_path = self.home if self.home is not None else Path.home()
        sources = registry.plugin_sources(self.workspace.root, home_path)
        self.trust.allow(self.workspace.root, "plugins", registry.sources_digest(sources))
        for scope, path, data in sources:
            if scope == "user" and data is not None:
                registry.trust_user_plugin(self.trust, home_path, path, data)
        self.registry = self._build_registry(home_path)

    def _tools_command(self, args: str) -> str:
        verb, _, name = args.strip().partition(" ")
        created, name = self._root().created, name.strip()
        if verb == "trust":
            self._root().trust_tools()
            return f"Đã tin cậy tool của dự án: {', '.join(n for n in created if created[n].get('scope') == 'workspace') or 'không có'}."
        if verb in ("show", "rm", "export") and name not in created:
            return f"Không có tool {name!r}; đã tạo: {', '.join(created) or 'chưa có'}"
        if verb == "show":
            return f"{created[name]['description']}\n\n{created[name]['code']}"
        if verb == "rm":
            return self._create({"name": name, "code": ""})
        if verb == "export":
            return self._export_tool(created[name])
        if verb:
            return "Dùng: /tools, /tools show TÊN, /tools rm TÊN, /tools trust, /tools export TÊN"
        rows = [f"  {t['name']} ({t.get('kind', 'exec')}, {t.get('scope', 'user')}): {t.get('about') or t['description']}"[:200] for t in created.values()]
        rows += [f"  {n} (dự án, chưa tin cậy: /tools trust)" for n in getattr(self._root(), "untrusted_tools", [])]
        return "\n".join(["Tool đã tạo:"] + rows) if rows else "Chưa có tool nào; model tạo tool bằng create."

    def _export_tool(self, tool: dict) -> str:
        if re.search(r"\btools\.", tool["code"]):
            return "Tool này gọi tool khác qua tools.…, nên chỉ chạy được trong Chaos; chưa xuất được ra MCP."
        folder = self._outputs_dir() / "mcp"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{tool['name']}_server.py"
        spec = {"name": tool["name"], "description": tool.get("about") or tool["description"], "inputSchema": tool["parameters"]}
        path.write_text(MCP_EXPORT.format(code=tool["code"], spec=json.dumps(spec, ensure_ascii=False), name=tool["name"]), encoding="utf-8")
        return (f"Đã xuất {path}. Thêm vào MCP của agent khác, ví dụ:\n"
                + json.dumps({"mcpServers": {tool["name"]: {"command": "python3", "args": [str(path)]}}}, indent=1))

    def _plugins_command(self, args: str) -> str:
        parts = args.split()
        if parts[:1] == ["reload"]:
            home_path = self.home if self.home is not None else Path.home()
            self.registry = self._build_registry(home_path)
            self._refresh_mcp()
            for child in self.children.values():
                child.registry = self.registry
        elif parts:
            return "Dùng: /plugins hoặc /plugins reload"
        return self._plugins_report()

    def _plugin_write(self, args: dict) -> str:
        name = str(args.get("name") or "").strip()
        if not registry.NAME_RE.match(name):
            raise ToolError("name must be lower_case letters, digits and _, starting with a letter")
        home_path = self.home if self.home is not None else Path.home()
        scope = "workspace" if args.get("scope") == "workspace" else "user"
        folder = self.workspace.root / ".agents" / "plugins" if scope == "workspace" else home_path / ".manga-agent" / "plugins"
        path = folder / f"{name}.py"
        before = registry.plugin_sources(self.workspace.root, home_path)
        was_trusted = (not any(s == "workspace" for s, _, _ in before)
                       or self.trust.trusted(self.workspace.root, "plugins", registry.sources_digest(before)))
        if args.get("remove"):
            if not path.exists():
                raise ToolError(f"no {scope} plugin {name}")
            path.unlink()
            done = f"Removed plugin {name}."
        else:
            code = str(args.get("code") or "")
            try:
                compile(code, str(path), "exec")
            except SyntaxError as exc:
                raise ToolError(f"Syntax error: {exc}") from exc
            folder.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            if scope == "user":
                registry.trust_user_plugin(self.trust, home_path, path, code.encode("utf-8"))
            done = f"Wrote plugin {name} ({path})."
        if scope == "workspace" and was_trusted:
            self.trust.allow(self.workspace.root, "plugins", registry.sources_digest(registry.plugin_sources(self.workspace.root, home_path)))
        root = self._root()
        root.registry = root._build_registry(home_path)
        for child in root.children.values():
            child.registry = root.registry
        row = next((p for p in root.registry.plugins if p["name"] == name and p["scope"] == scope), None)
        if row is None:
            return done
        if row["state"] == "untrusted":
            return f"{done} It is not loaded: this project has other plugins the user has not trusted; they can trust them with /plugins."
        return f"{done} State: {row['state']}" + (f"; error: {row['error']}" if row["error"] else "; it is active from the next step.")

    def _plugins_report(self) -> str:
        rows = ["Plugin:"] + [f"  {p['name']} ({p['scope']}): {p['state']}" + (f" — {p['error']}" if p["error"] else "") for p in self.registry.plugins]
        rows += [f"Tính năng đã tắt: {', '.join(sorted(self.disabled)) or 'không'}", f"Vòng lặp: {self.profile['loop']}",
                 f"Nhóm có thể tắt trong profile.json: {', '.join(registry.ALL_GROUPS)}"]
        return "\n".join(rows)

    def _hook_call(self, fn, *args) -> str:
        try:
            return str(fn(self, *args) or "")
        except Exception as exc:
            logger.warning("Agent plugin hook failed: {}", exc)
            return ""

    def _fire(self, event: str, *args) -> None:
        for fn in self.registry.hooks[event]:
            self._hook_call(fn, *copy.deepcopy(args))


    def emit(self, kind: str, **data) -> None:
        with self._lock:
            event = {"seq": len(self.events) + 1, "type": kind, "time": round(time.time(), 3), **data}
            self.events.append(event)
            self.updated_at = time.time()
            self._lock.notify_all()

    def _info(self) -> dict:
        return {"disabled": sorted(self.disabled), "plugins": {"rows": self.registry.plugins,
                                                                "needs_trust": any(p["state"] == "untrusted" for p in self.registry.plugins)},
                "agents": [{"name": a.name, "description": a.description} for a in self.agents.values()],
                "skills": [{"name": s.name, "description": s.description} for s in self.skills.values()],
                "mcp": list(self.mcp_status.values()), "replaced": self.replaced, "hooks": self._hook_summary(),
                "created": [{"name": n, "kind": c.get("kind", "exec"), "scope": c.get("scope", "user")} for n, c in self._root().created.items()],
                "untrusted_tools": list(getattr(self._root(), "untrusted_tools", [])),
                "commands": [{"name": k, "description": v} for k, v in self._builtin_commands().items()]
                + [{"name": k, "description": d} for k, (d, _) in self.registry.commands.items()]
                + [{"name": s.name, "description": ("" if s.manual else "Đang bật: " if s.name in self.active_skills else "Bật skill: ") + s.description[:120]}
                   for s in self.skills.values()]
                + [{"name": c["name"], "description": c["description"]} for c in self.commands.values()]
                + [{"name": k, "description": str(p.get("description") or p["name"])[:120]} for k, (_, p) in self._mcp_prompts().items()]}

    def snapshot(self, after: int = 0, rev: str = "") -> dict:
        with self._lock:
            sandbox_info = {"mode": self.workspace.policy.mode, "network": self.workspace.policy.network, "backend": sandbox.backend()}
            info = self._info()
            digest = hashlib.sha256(json.dumps([info, sandbox_info], sort_keys=True, default=str).encode()).hexdigest()[:12]
            snap = {"id": self.id, "status": self.status, "provider": self.provider.id, "provider_label": self.provider.label,
                    "model": self.model, "mode": self.mode, "workspace": str(self.workspace.root), "title": self.title,
                    "sandbox": sandbox_info, "text_tools": self.text_tools, "usage": dict(self.usage), "stats": self._stats(), "live": self._live_view(),
                    "pending": self.pending, "todos": self.todos, "key_prompt": self.key_prompt, "keys": len(self.api_keys),
                    "plan_mode": self.plan_mode, "effort": self.effort, "goal": self.goal and self.goal["text"], "queued": len(self.queue),
                    "rate_limited": dict(self.rate_limited), "info_rev": digest, "events": [e for e in self.events if e["seq"] > after]}
            if rev != digest:
                snap.update(info)
            return snap

    def _live_view(self) -> dict | None:
        live = self.live
        if not live:
            return None
        return {"text": live["text"][-1500:], "reasoning": live["reasoning"][-600:], "tools": live["tools"]}

    def _spent(self) -> int:
        return self.usage["prompt_tokens"] - self.usage["cached_tokens"] + self.usage["completion_tokens"]

    def _stats(self) -> dict:
        prompt, cached = self.usage["prompt_tokens"], self.usage["cached_tokens"]
        price = models.prices(self.model, self.profile["prices"])
        cost = None
        if price.get("in") is not None and price.get("out") is not None:
            cost = round(((prompt - cached) * price["in"] + cached * price.get("cached", price["in"]) + self.usage["completion_tokens"] * price["out"]) / 1e6, 4)
        return {"model_seconds": round(self.stats["model_s"], 1), "tool_seconds": round(self.stats["tool_s"], 1),
                "cache_pct": round(100 * cached / prompt) if prompt else 0, "cost": cost}

    def save(self) -> None:
        if self.store is None or self._closing:
            return
        data = {"id": self.id, "provider": self.provider.id, "model": self.model, "mode": self.mode, "title": self.title,
                "workspace": str(self.workspace.root), "sandbox": [self.workspace.policy.mode, self.workspace.policy.network],
                "created_at": self.created_at, "updated_at": self.updated_at, "usage": self.usage, "todos": self.todos, "notes": self.notes,
                "text_tools": self.text_tools, "plan_mode": self.plan_mode, "effort": self.effort, "goal": self.goal, "active_skills": self.active_skills, "schedules": self.schedules, "compact_at": self.compact_at,
                "history": [{k: v for k, v in h.items() if k != "images"} for h in self.history], "events": self.events,
                "tainted": self.tainted, "tainted_files": sorted(self._tainted_files), "taint_latch": self._taint_latch}
        tmp = self.store.with_name(f"{self.store.stem}.tmp-{os.getpid()}-{threading.get_ident()}")
        card = {k: data[k] for k in ("title", "model", "updated_at", "workspace")}
        with self._save_lock:
            try:
                self.store.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.store)
                meta = self.store.with_name(self.store.stem + ".meta.json")
                meta.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
                os.chmod(meta, 0o600)
            except (OSError, TypeError, ValueError):
                logger.opt(exception=True).warning("Could not save agent session {}", self.id)
            finally:
                tmp.unlink(missing_ok=True)

    def restore(self, data: dict) -> None:
        self.history, self.events = list(data.get("history") or []), list(data.get("events") or [])
        self.todos, self.title = list(data.get("todos") or []), str(data.get("title") or "")
        self.notes = str(data.get("notes") or "")
        self.effort = data.get("effort") if data.get("effort") in EFFORTS else ""
        if isinstance(data.get("compact_at"), int) and data["compact_at"] >= 0:
            self.compact_at = data["compact_at"]
        self.schedules = [t for t in (data.get("schedules") or []) if isinstance(t, dict) and {"id", "prompt", "next_at", "every"} <= set(t)][:MAX_SCHEDULES]
        if self.schedules:
            self._start_scheduler()
        self.usage.update(data.get("usage") or {})
        self.text_tools = bool(data.get("text_tools"))
        self.plan_mode = bool(data.get("plan_mode"))
        if isinstance(data.get("active_skills"), list):
            self.active_skills = [n for n in data["active_skills"] if isinstance(n, str) and n in self.skills]
        goal = data.get("goal")
        self.goal = {"text": str(goal.get("text", "")), "turns": int(goal.get("turns", 0))} if isinstance(goal, dict) else None
        self.created_at = float(data.get("created_at") or self.created_at)
        self.tainted = bool(data.get("tainted"))
        self._taint_latch = bool(data.get("taint_latch"))
        self._tainted_files = {str(p) for p in (data.get("tainted_files") or []) if isinstance(p, str)}

    def set_model(self, model: str) -> None:
        self.model = model
        self.quirks = models.quirks(model, self.profile["models"])
        self._apply_caps()

    def _apply_caps(self) -> None:
        window = self.quirks.get("context_window")
        if isinstance(window, int) and 4_000 <= window <= 10_000_000:
            self.compact_at = int(window * WINDOW_SHARE)
        if isinstance(self.quirks.get("max_output"), int) and self.quirks["max_output"] > 0:
            self._max_out = self.quirks["max_output"]
        if self.quirks.get("tools") in ("text", "native"):
            self.text_tools = self.quirks["tools"] == "text"

    def set_effort(self, effort: str) -> None:
        effort = EFFORT_WORDS.get(effort.strip().lower(), effort.strip().lower())
        if effort not in EFFORTS:
            raise ValueError("effort must be one of: default, low, medium, high, xhigh")
        self.effort = effort

    def set_policy(self, mode: str, network: bool) -> None:
        if mode not in sandbox.MODES:
            raise ValueError(f"sandbox must be one of {', '.join(sandbox.MODES)}")
        self.workspace.policy = sandbox.Policy(mode, bool(network))


    def _builtin_commands(self) -> dict[str, str]:
        return {k: v for k, v in BUILTIN_COMMANDS.items() if registry.COMMAND_GROUPS.get(k) not in self.disabled}

    def _rel(self, path: str) -> str:
        try:
            return self.workspace.rel(self.workspace.resolve(path))
        except ToolError:
            return str(path)

    def _references(self, text: str) -> tuple[str, list[str]]:
        blocks, names = [], []
        for raw in REF.findall(text):
            name = raw.rstrip(".,;:!?)]}\"'")
            if not name or name in names or len(names) >= MAX_REFS:
                continue
            try:
                path = self.workspace.resolve(name)
            except ToolError:
                continue
            tool = "read_file" if path.is_file() else "list_dir"
            if not path.exists() or rules.check(self.rules, tool, {"path": name}, self._rel) == "deny":
                continue
            try:
                body = self.workspace.run(tool, {"path": name} if tool == "read_file" else {"path": name, "depth": 2})
            except ToolError:
                continue
            names.append(name)
            blocks.append(f'<attached path="{name}">\n{body}\n</attached>')
        return ("\n\n" + "\n".join(blocks)) if blocks else "", names

    def send(self, text: str, scheduled: bool = False) -> bool:
        extra, names = self._references(text)
        with self._lock:
            queued = self.status in ("running", "waiting")
            if queued:
                self.queue.append(text + extra)
            else:
                self.status, self._stop = "running", False
        self.title = self.title or (text.strip().splitlines() or [""])[0][:80]
        self.emit("user", text=text, refs=names, queued=queued)
        if queued:
            return True
        self._new_request(text + extra, scheduled=scheduled)
        update = self._context_update()
        if update:
            self.history.append({"role": "user", "content": update})
            self.cache_log["context updates"] += 1
        self.history.append({"role": "user", "content": text + extra, "from_user": True})
        threading.Thread(target=self._loop, name=f"agent-{self.id}", daemon=True).start()
        return False

    def _new_request(self, text: str, scheduled: bool = False) -> None:
        if not self.depth:
            self.checkpoints.begin()
        if not any(item.get("tainted") for item in self.history) and not self._inherit_taint and not self._taint_latch:
            self.tainted = False
        else:
            self.tainted = True
        self._inherit_taint = False
        self._edited = self._plan_nudged = self._continued = self._check_nudged = self._handoff_nudged = False
        self._last_command: str | None = None
        self._failing: dict[str, int] = {}
        self._commands = 0
        self._fallback_at = 0
        self._request_no += 1
        self.chain_budget = {}
        self._prewalk = bool(self.profile["prewalk_model"]) and self.profile["prewalk_model"] != self.model and not self.depth
        self._skills_loaded = []
        self._turn_usage = self._spent()
        self._fired = {}
        hosts = {h.lower() for h in URL_RE.findall(text)}
        if scheduled:
            self._schedule_hosts |= hosts
        else:
            self.web_ok |= hosts
        self._last_request = text

    def command(self, text: str) -> dict:
        name, _, args = text.strip()[1:].partition(" ")
        args = args.strip()
        if registry.COMMAND_GROUPS.get(name) in self.disabled:
            raise ValueError(f"/{name} đang tắt trong profile.json")
        if name == "help":
            rows = [f"/{k} — {v}" for k, v in self._builtin_commands().items()] + [f"/{k} — {d}" for k, (d, _) in self.registry.commands.items()]
            rows += [f"/{k} — {str(p.get('description') or p['name'])[:120]}" for k, (_, p) in self._mcp_prompts().items()]
            return {"message": "\n".join(rows + [f"/{c['name']} — {c['description']}" for c in self.commands.values()])}
        if name == "plugins":
            return {"message": self._plugins_command(args)}
        if name == "tools":
            return {"message": self._tools_command(args)}
        if name in self.registry.commands:
            result = self.registry.commands[name][1](self, args)
            return result if isinstance(result, dict) else {"message": str(result)}
        if name == "skills":
            if args.split()[:1] == ["add"]:
                return {"message": self._install_skills(args[3:].strip())}
            if args.split()[:1] == ["off"]:
                return {"message": self._deactivate_skill(args[3:].strip())}
            rows = [f"/{s.name}{' (đang bật)' if s.name in self.active_skills else ''}{' (có sẵn)' if s.builtin else ''}: {s.description[:150]}"
                    for s in self.skills.values()]
            return {"message": "\n".join(rows) or "Không có skill nào."}
        if name == "models":
            return {"message": self._models_command(args)}
        if name == "intent":
            return {"message": self._intent(write=False) or "Chưa có bản hình dung nào: nó được viết khi chuỗi đầu tiên của một yêu cầu chạy."}
        if name == "mcp":
            self._ensure_mcp()
            rows = [f"{m['name']} ({m['scope']}): {m['state']}" + (f", {m['tools']} công cụ" if m.get("tools") else "")
                    + (f" — {m['error']}" if m.get("error") else "") for m in self.mcp_status.values()]
            configured = self.profile["replace"]
            rows += [f"{name} → {target}" + ("" if target in self.mcp_tools else " (chưa kết nối, đang dùng tool có sẵn)") for name, target in configured.items()]
            rows += [f"{name}: vai {role}" for name, role in self.profile["roles"].items()]
            return {"message": "\n".join(rows) or "Chưa cấu hình MCP server nào (.mcp.json, ~/.claude.json, ~/.codex/config.toml)."}
        if name == "model":
            if args:
                self.set_model(args)
            return {"message": f"Model: {self.model}"}
        if name == "effort":
            if args:
                self.set_effort(args)
            return {"message": f"Mức suy luận: {self.effort or 'mặc định'}"}
        if name == "mode" and args in MODES:
            self.mode = args
            return {"message": f"Quyền: {args}"}
        if name == "sandbox" and args:
            parts = args.split()
            self.set_policy(parts[0], len(parts) > 1 and parts[1] in ("net", "network", "on"))
            return {"message": sandbox.Policy(parts[0], self.workspace.policy.network).describe(self.workspace.root)}
        if name == "plan":
            self.plan_mode = args != "off"
            if self.plan_mode and args:
                self.send(args)
            return {"message": "Đang lập kế hoạch: agent chỉ đọc cho tới khi bạn duyệt." if self.plan_mode else "Đã tắt chế độ lập kế hoạch."}
        if name == "goal":
            if args == "off" or (not args and self.goal):
                text, self.goal = (self.goal or {}).get("text"), None if args == "off" else self.goal
                return {"message": "Đã bỏ mục tiêu." if args == "off" else f"Mục tiêu: {text}"}
            if not args:
                return {"message": "Chưa có mục tiêu. Dùng /goal MỤC TIÊU."}
            self.goal = {"text": args[:2000], "turns": 0}
            self.send(GOAL_PROMPT.format(text=args))
            return {"sent": True}
        if name == "cache":
            rows = [f"{k}: {v}" for k, v in sorted(self.cache_log.items())]
            return {"message": f"Token lấy từ cache: {self._stats()['cache_pct']}%\n" + ("\n".join(rows) or "Chưa có yêu cầu nào.")}
        if name == "receipts":
            return {"message": self._receipts()}
        if name == "undo":
            with self._lock:
                if self.status != "idle":
                    raise RuntimeError("The agent is still working")
            done = [self.workspace.rel(Path(p)) for p in self.checkpoints.undo()]
            self.workspace.seen.clear()
            if not done:
                return {"message": "Không có gì để hoàn tác (chỉ hoàn tác file do công cụ sửa file đã đổi, không gồm lệnh shell)."}
            self.history.append({"role": "user", "content": "[The user undid your last turn's file changes: " + ", ".join(done) + "]"})
            self.history.append({"role": "assistant", "content": "Understood; those files are back as they were.", "calls": []})
            self.emit("notice", text="Đã hoàn tác: " + ", ".join(done))
            return {"message": "Đã hoàn tác " + ", ".join(done)}
        if name == "memory":
            return {"message": self._memory_command(args)}
        if name == "agents":
            return {"message": agents.catalog(self.agents)}
        if name == "rules":
            return {"message": rules.describe(self.rules)}
        if name == "compact" and args.split()[:1] == ["auto"]:
            return {"message": self._compact_auto(args.split()[1:])}
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
        if name in self.skills and self.skills[name].manual:
            self.send(skills.load(self.skills[name]) + (f"\n\nUser input: {args}" if args else ""))
            return {"sent": True}
        if name in self.skills:
            return self._activate_skill(name, args)
        if name in self.commands:
            self.send(context.expand_command(self.commands[name]["body"], args))
            return {"sent": True}
        if name.startswith("mcp:") and name.count(":") == 2:
            self.send(self._mcp_prompt(name, args))
            return {"sent": True}
        raise ValueError(f"Unknown command /{name}; type /help")

    def _mcp_resource(self, args: dict) -> str:
        server = self._root().mcp_servers.get(str(args.get("server") or ""))
        if server is None:
            raise ToolError(f"no running MCP server {args.get('server')!r}; running: {', '.join(self._root().mcp_servers) or 'none'}")
        self.tainted = True
        try:
            if not args.get("uri"):
                rows = server.resources()
                return "\n".join(f"{r.get('uri')} — {r.get('name') or ''} {r.get('description') or ''}".strip()[:300] for r in rows[:200]) or "(no resources)"
            return server.read_resource(str(args["uri"]))
        except mcp.MCPError as exc:
            raise ToolError(f"MCP {server.name}: {exc}") from exc

    def _mcp_prompts(self) -> dict[str, tuple[str, dict]]:
        return {f"mcp:{name}:{p['name']}": (name, p) for name, server in self.mcp_servers.items() for p in getattr(server, "prompts", [])}

    def _mcp_prompt(self, command: str, args: str) -> str:
        self._ensure_mcp()
        found = self._mcp_prompts().get(command)
        if found is None:
            raise ValueError(f"Không có prompt /{command}")
        server, prompt = found
        names = [str(a.get("name")) for a in prompt.get("arguments") or [] if isinstance(a, dict) and a.get("name")]
        pairs = dict(re.findall(r"(\w+)=(\S+)", args))
        if not pairs and args and names:
            pairs = {names[0]: args}
        try:
            return self.mcp_servers[server].get_prompt(prompt["name"], pairs)
        except mcp.MCPError as exc:
            raise ValueError(f"MCP {server}: {exc}") from exc

    def reload_skills(self) -> None:
        self.skills = skills.discover(self.workspace.root, self.home)
        self._set_read_roots()

    def _activate_skill(self, name: str, args: str) -> dict:
        if name in self.active_skills:
            if args:
                self.send(args)
                return {"sent": True}
            return {"message": f"Skill {name} đang bật."}
        self.active_skills.append(name)
        self.save()
        body = f"\n\n{skills.load(self.skills[name])}" if self._pinned is not None else ""
        self.send(f"The user turned on skill {name}; follow it for the rest of this session.{body}" + (f"\n\nUser input: {args}" if args else ""))
        return {"sent": True}

    def _deactivate_skill(self, name: str) -> str:
        if name not in self.active_skills:
            return f"Skill {name or '?'} không bật."
        self.active_skills.remove(name)
        self.save()
        return f"Đã tắt skill {name}; nó rời khỏi ngữ cảnh khi phiên tóm gọn hoặc mở lại."

    def _install_skills(self, spec: str) -> str:
        home = self.home if self.home is not None else Path.home()
        try:
            names = skill_install.install(spec, home)
        except (ValueError, OSError) as exc:
            return f"Không cài được: {exc}"
        self.reload_skills()
        return f"Đã cài {len(names)} skill vào ~/.manga-agent/skills: {', '.join(names)}. Đọc kỹ SKILL.md của chúng trước khi tin."

    def _memory_command(self, args: str) -> str:
        home = self.home if self.home is not None else Path.home()
        parts = args.split(maxsplit=2)
        try:
            if parts[:1] == ["add"] and len(parts) > 1:
                memory.add(home, self.workspace.root, "project", args[4:])
            elif parts[:1] == ["rm"] and len(parts) == 3 and parts[2].isdigit():
                memory.remove(home, self.workspace.root, parts[1], int(parts[2]))
        except ValueError as exc:
            return str(exc)
        return memory.prompt(home, self.workspace.root) or "Chưa có ghi nhớ nào."

    @staticmethod
    def _decision_key(call: dict) -> str:
        return str(call.get("id") or id(call))

    def decide(self, decision: str, note: str = "") -> None:
        with self._lock:
            if self.status != "waiting" or self.pending is None:
                raise RuntimeError("Nothing is waiting for approval")
            self._decisions[self._decision_key(self.pending)] = {"decision": decision, "note": note}
            self._lock.notify_all()

    def _root(self) -> "AgentSession":
        return self.parent._root() if self.parent is not None else self

    @property
    def _base_complete(self):
        return self.complete.raw if isinstance(self.complete, _ses("_Keyed")) else self.complete

    def _complete_keyed(self, provider, api_key, *args, **kwargs):
        root = self._root()
        while True:
            key = root.api_key if api_key in root.api_keys else api_key
            token = client.ASK_ON_LIMIT.set(key == root.api_key and len(root.api_keys) > 1)
            extra = client.REQUEST_EXTRA.set(self._request_fields())
            try:
                adapter = self._adapter(provider)
                if adapter is not None:
                    return self._via_adapter(adapter, provider, key, *args, **kwargs)
                return self._raw_complete(provider, key, *args, **kwargs)
            except client.KeyLimited as exc:
                if not root._ask_next_key(key, str(exc)):
                    raise
            finally:
                client.REQUEST_EXTRA.reset(extra)
                client.ASK_ON_LIMIT.reset(token)

    def _adapter(self, provider):
        name = self.profile["adapters"].get(provider.id, provider.id)
        return self.registry.providers.get(name)

    def _via_adapter(self, adapter, provider, key, model, messages, *, tools=None, on_delta=None, max_tokens=None):
        out = adapter(self, {"model": model, "messages": messages, "tools": tools, "api_key": key,
                             "api_base": provider.api_base, "max_tokens": max_tokens, "fields": self._request_fields()})
        if not isinstance(out, dict) or not isinstance(out.get("text", ""), str):
            raise RuntimeError(f"model adapter for {provider.id} returned {type(out).__name__}, not a turn dict")
        calls = [c for c in out.get("calls") or [] if isinstance(c, dict) and c.get("name")]
        return {"text": out.get("text") or "", "reasoning": str(out.get("reasoning") or ""), "usage": dict(out.get("usage") or {}),
                "calls": [{"id": str(c.get("id") or uuid.uuid4().hex[:12]), "name": str(c["name"]),
                           "args": c.get("args") if isinstance(c.get("args"), dict) else {}} for c in calls]}

    def _request_fields(self) -> dict:
        fields = dict(self.quirks.get("request")) if isinstance(self.quirks.get("request"), dict) else {}
        if self._root().effort:
            fields["reasoning_effort"] = self._root().effort
        clock, late = self._root()._clock, self.quirks.get("late_request")
        if clock and isinstance(late, dict) and time.time() - clock[0] > clock[1] * (1 - LATE_SHARE):
            fields.update(late)
        return fields

    def _ask_next_key(self, key: str, detail: str) -> bool:
        with self._key_lock:
            if key != self.api_key:
                return True
            index = self.api_keys.index(key)
            following = (index + 1) % len(self.api_keys)
            with self._lock:
                self.key_prompt = {"index": index + 1, "next": following + 1, "count": len(self.api_keys),
                                   "provider": self.provider.label, "detail": detail[:300]}
                self._key_choice = None
            self.emit("key_limit", **self.key_prompt)
            with self._lock:
                while self._key_choice is None and not self._stop:
                    self._lock.wait(timeout=1.0)
                choice, self.key_prompt = self._key_choice, None
            if choice != "next":
                self.emit("notice", text=f"Đã dừng: key {index + 1}/{len(self.api_keys)} ({self.provider.label}) bị giới hạn.")
                return False
            self.api_key = self.api_keys[following]
            self.emit("notice", text=f"Đã đổi sang key {following + 1}/{len(self.api_keys)} ({self.provider.label}).")
            return True

    def choose_key(self, choice: str) -> None:
        with self._lock:
            if self.key_prompt is None:
                raise RuntimeError("No key is waiting for a decision")
            self._key_choice = choice
            self._lock.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self.queue.clear()
            self.goal = None
            self._lock.notify_all()
        for child in list(self.children.values()):
            child.held = None
            child.stop()
        with self._usage_lock:
            for job in list(self.jobs.values()):
                job.stop()
            self.jobs.clear()

    def close(self) -> None:
        self._closing = True
        if self.profile.get("keep_jobs"):
            with self._usage_lock:
                for job in list(self.jobs.values()):
                    if job.alive:
                        job.detach()
                self.jobs.clear()
        self.stop()
        self._join_turn()
        for child in list(self.children.values()):
            child.close()
            if child.copy is not None:
                shutil.rmtree(child.copy, ignore_errors=True)
        if self._repl is not None:
            self._repl.close()
        for server in self.mcp_servers.values():
            server.close()
        self.mcp_servers.clear()

    def _join_turn(self, timeout: float = 30.0) -> None:
        me = threading.current_thread()
        for thread in threading.enumerate():
            if thread is not me and thread.name == f"agent-{self.id}":
                thread.join(timeout)


    @property
    def replaced(self) -> dict[str, str]:
        tools = self._root().mcp_tools
        return {name: target for name, target in self.profile["replace"].items() if target in tools}

    def _role(self, name: str) -> str | None:
        return self.profile["roles"].get(name) or self._root().mcp_kinds.get(name)

    def _ensure_mcp(self) -> None:
        with self._mcp_lock:
            if self._mcp_ready:
                return
            self._mcp_ready = True
        starts = []
        for row in mcp.configured(self.workspace.root, self.home):
            name, digest = row["name"], mcp.config_hash(row["config"])
            status = {"name": name, "scope": row["scope"], "source": row["source"], "digest": digest, "tools": 0, "error": "", "state": "waiting"}
            self.mcp_status[name] = status
            if row["config"].get("disabled") or row["config"].get("enabled") is False:
                status["state"] = "disabled"
            elif row["scope"] == "workspace" and not self.trust.trusted(self.workspace.root, f"mcp:{name}", digest):
                status["state"] = "untrusted"
            else:
                starts.append(row)
        if starts:
            with ThreadPoolExecutor(max_workers=min(8, len(starts))) as pool:
                list(pool.map(self._start_mcp, starts))

    def _start_mcp(self, row: dict) -> None:
        name, status = row["name"], self.mcp_status[row["name"]]
        try:
            server = mcp.Server(name, row["config"], self.workspace.root)
        except Exception as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            return
        server.media_dir = self._outputs_dir() / "mcp-media"
        with self._mcp_lock:
            self.mcp_servers[name] = server
            self._offer_mcp_tools(name, row["scope"])
        status.update(state="running", tools=len(server.tools), prompts=len(server.prompts), error="")

    def _offer_mcp_tools(self, name: str, scope: str) -> None:
        server = self.mcp_servers[name]
        for tool_name in [n for n, (s, _) in self.mcp_tools.items() if s == name]:
            self.mcp_tools.pop(tool_name, None)
            self.mcp_kinds.pop(tool_name, None)
        kinds = server.config.get("kinds") if scope == "user" and isinstance(server.config.get("kinds"), dict) else {}
        for tool in server.tools:
            full = mcp.tool_name(name, tool["name"])
            self.mcp_tools[full] = (name, tool)
            if kinds.get(tool["name"]) in mcp.KINDS:
                self.mcp_kinds[full] = kinds[tool["name"]]

    def _sync_mcp_tools(self) -> None:
        for name, server in list(self.mcp_servers.items()):
            if not getattr(server, "tools_stale", False):
                continue
            try:
                server.refresh_tools()
            except mcp.MCPError as exc:
                logger.warning("MCP server {} could not list its tools again: {}", name, exc)
                continue
            with self._mcp_lock:
                self._offer_mcp_tools(name, self.mcp_status.get(name, {}).get("scope", "workspace"))
            self.mcp_status.get(name, {})["tools"] = len(server.tools)
            self.emit("notice", text=f"MCP {name} cập nhật: {len(server.tools)} công cụ.")

    def _stop_mcp(self, name: str) -> None:
        with self._mcp_lock:
            for tool_name in [n for n, (server, _) in self.mcp_tools.items() if server == name]:
                self.mcp_tools.pop(tool_name, None)
                self.mcp_kinds.pop(tool_name, None)
            server = self.mcp_servers.pop(name, None)
        if server is not None:
            server.close()

    def reload_mcp(self) -> None:
        with self._lock:
            busy = self.status != "idle"
            self._mcp_stale = busy
        if not busy:
            self._refresh_mcp()

    def _refresh_mcp(self) -> None:
        self._mcp_stale = False
        for name in list(self.mcp_servers):
            self._stop_mcp(name)
        self.mcp_status.clear()
        self._mcp_ready = False
        self._ensure_mcp()

    def trust_mcp(self, name: str) -> None:
        self._ensure_mcp()
        status = self.mcp_status.get(name)
        if status is None:
            raise KeyError(name)
        self.trust.allow(self.workspace.root, f"mcp:{name}", status["digest"])
        if status["state"] == "untrusted":
            row = next((r for r in mcp.configured(self.workspace.root, self.home) if r["name"] == name), None)
            if row is not None:
                self._start_mcp(row)


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
