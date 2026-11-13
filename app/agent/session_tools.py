from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


from app.agent import (
    agents,
    chaincheck,
    gitguard,
    mcp,
    models,
    registry,
    repl,
    rules,
    sandbox,
    skills,
)
from app.agent.session_config import *  # noqa: F401, F403
from app.agent.session_config import _as_function, _signature
from app.agent.tools import KIND, SPECS, ToolError, Workspace
from app.logging_config import logger


def _ses(name: str):
    return getattr(sys.modules["app.agent.session"], name)


class AgentToolsMixin:
    """Tool specification, approval, created tools, and chain setup for AgentSession."""

    def specs(self, listed: list[dict] | None = None) -> list[dict]:
        sees = models.sees_images(self.provider.id, self.model, self.quirks["vision"] if isinstance(self.quirks.get("vision"), bool) else self.profile["vision"])
        swap = self.replaced
        tools = self._root().mcp_tools
        rows = []
        for row in self._all_specs() if listed is None else listed:
            if row["name"] in swap.values():
                continue
            if row["name"] in swap:
                server, tool = tools[swap[row["name"]]]
                row = {"name": row["name"], "description": f"[MCP {server}] {tool.get('description') or tool['name']}"[:1024],
                       "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}}}
            if self._enabled(row["name"]) and (sees or row["name"] != "view_image"):
                rows.append(row)
        return rows

    def _all_specs(self) -> list[dict]:
        every = {s["name"] for s in SPECS} | {"skill"}
        if self.depth:
            allowed = self.agent.allowed(every) if self.agent else agents.READ_TOOLS
            rows = [s for s in SPECS if s["name"] in allowed]
            return rows + ([SESSION_SPECS["skill"]] if self.skills and "skill" in allowed else [])
        return self._lean_specs()

    def _lean_specs(self) -> list[dict]:
        rows = [s for s in SPECS if s["name"] in (("read_file",) if self.plan_mode else LEAN_TOOLS)]
        if self.workspace.policy.mode != "read-only":
            rows.append(SESSION_SPECS["create"])
            rows += [{k: tool[k] for k in ("name", "description", "parameters")} for tool in self.created.values()]
        if self.skills:
            rows.append(SESSION_SPECS["skill"])
        if self.quirks.get("direct_tools"):
            rows += [t.spec for t in self.registry.tools.values() if not self.plan_mode or t.kind == "read"]
            rows += [{"name": name, "description": f"[MCP {server}] {tool.get('description') or tool['name']}"[:1024],
                      "parameters": tool.get("inputSchema") or OBJ} for name, (server, tool) in self._root().mcp_tools.items()
                     if not self.plan_mode or self._role(name) == "read"]
        if self.jobs:
            rows += [SESSION_SPECS["job_output"], SESSION_SPECS["job_input"], SESSION_SPECS["job_stop"]]
        if self.plan_mode:
            rows.append(SESSION_SPECS["exit_plan_mode"])
        elif self.goal:
            rows.append(SESSION_SPECS["goal_done"])
        return rows

    def library_specs(self) -> list[dict]:
        listing = agents.catalog(self.agents)
        task, spawn, fan = (({**SESSION_SPECS[n], "description": SESSION_SPECS[n]["description"].format(
            agents=listing if n == "task" else "the same as for task.")}) for n in ("task", "spawn_agent", "fan_out"))
        rows = list(SPECS)
        if self.skills:
            rows.append(SESSION_SPECS["skill"])
        rows += [SESSION_SPECS["job_output"], SESSION_SPECS["job_input"], SESSION_SPECS["job_stop"], SESSION_SPECS["checkpoint"]]
        rows += [SESSION_SPECS["schedule_create"], SESSION_SPECS["schedule_list"], SESSION_SPECS["schedule_delete"],
                 SESSION_SPECS["todo_write"], SESSION_SPECS["ask_user"],
                 SESSION_SPECS["memory"], SESSION_SPECS["plugin_write"], task, spawn, fan, SESSION_SPECS["wait_agent"], SESSION_SPECS["send_input"], SESSION_SPECS["close_agent"]]
        rows += [t.spec for t in self.registry.tools.values()]
        if self.goal or self.profile["notes_context"]:
            rows += [SESSION_SPECS["context_notes"], SESSION_SPECS["new_context"]]
        rows += [{"name": name, "description": f"[MCP {server}] {tool.get('description') or tool['name']}"[:1024],
                  "parameters": tool.get("inputSchema") or OBJ} for name, (server, tool) in self.mcp_tools.items()]
        if any("resources" in getattr(s, "capabilities", {}) for s in self._root().mcp_servers.values()):
            rows.append(SESSION_SPECS["mcp_resource"])
        if self.plan_mode:
            reads = agents.READ_TOOLS | {"checkpoint", "todo_write", "task", "fan_out", "ask_user", "memory", "spawn_agent", "wait_agent", "send_input", "close_agent"}
            reads |= {n for n, t in self.registry.tools.items() if t.kind == "read"}
            reads |= {n for n, (_, tool) in self.mcp_tools.items()
                      if self._role(n) == "read" or (not self._role(n) and (tool.get("annotations") or {}).get("readOnlyHint"))}
            rows = [r for r in rows if r["name"] in reads]
        return rows

    def _needs_approval(self, call: dict, verdict: str | None = None) -> bool:
        name = call["name"]
        plugin = self.registry.tools.get(name)
        if plugin and plugin.always_ask:
            return True
        if name == "run_command" and call["args"].get("outside_sandbox") and self.workspace.policy.mode != "full-access":
            return True
        if name == "job_input" and self._typing_outside(call):
            return True
        if name == "schedule_create":
            return self.mode not in HANDS_OFF
        if name == "memory":
            return self.mode not in HANDS_OFF and call["args"].get("action") != "list"
        if name == "create":
            return False
        created = self._root().created.get(name)
        if created is not None and created.get("kind") == "read" and self._confined() and verdict != "ask":
            return False
        if name == "checkpoint":
            return call["args"].get("action") == "restore" and self.mode == "ask" and verdict != "allow"
        if self.mode == "bypass" or name in SESSION_SPECS or verdict == "allow":
            return False
        if verdict == "ask":
            return True
        if name in URL_TOOLS and self.mode == "edits" and (urlparse(str(call["args"].get("url") or "")).hostname or "") not in self.web_ok:
            return True
        if name in self.mcp_tools and self._role(name) != "read" and self.tainted and self.profile["untrusted_guard"]:
            return True
        if name in self.mcp_tools and not self._role(name):
            tool = self.mcp_tools[name][1] or {}
            return not (self.mode == "edits" and mcp.harness_read_only(tool.get("name")))
        kind = plugin.kind if plugin else self._role(name) or KIND.get(name, "exec")
        if kind == "read":
            return False
        if self.tainted and self.profile["untrusted_guard"] and not (
                name in URL_TOOLS and (urlparse(str(call["args"].get("url") or "")).hostname or "") in self.web_ok):
            return True
        via_mcp = name in self.replaced or name in self.mcp_tools
        if self.mode == "ask":
            return via_mcp or not (name == "run_command" and not call["args"].get("outside_sandbox") and rules.safe_readonly(str(call["args"].get("command") or "")))
        if kind == "exec":
            confined = (sandbox.full() and self.workspace.policy.mode != "full-access" and not via_mcp
)
            return bool(call["args"].get("outside_sandbox")) or not confined
        if kind == "edit" and via_mcp:
            try:
                self.workspace.resolve(str(call["args"]["path"]), write=True)
            except (KeyError, ToolError):
                return True
        return False

    def _typing_outside(self, call: dict) -> bool:
        job = self.jobs.get(str(call["args"].get("id") or "").strip().lower()) if call["name"] == "job_input" else None
        return bool(job and job.outside and self.workspace.policy.mode != "full-access")

    def _always_ask(self, call: dict) -> bool:
        plugin = self.registry.tools.get(call["name"])
        return bool(plugin and plugin.always_ask) or call["name"] in ("memory", "plugin_write") or self._typing_outside(call) or bool(
            call["name"] == "run_command" and call["args"].get("outside_sandbox") and self.workspace.policy.mode != "full-access")

    def _user_messages(self) -> list[str]:
        root = self._root()
        return [h["content"] for h in root.history if h["role"] == "user" and not h["content"].startswith(("[", "<"))]

    def _receipts(self) -> str:
        calls = {c["id"]: c for e in self.events if e["type"] == "assistant" for c in e.get("calls") or []}
        rows = []
        for e in self.events:
            stamp = time.strftime("%H:%M:%S", time.localtime(e["time"]))
            if e["type"] == "tool":
                call = calls.get(e["id"], {"args": {}})
                args, kind = call.get("args") or {}, KIND.get(e["name"])
                if e["name"] == "run_command":
                    first = str(e["output"]).rsplit("[", 1)[-1].rstrip("]\n")
                    rows.append(f"{stamp} chạy   {str(args.get('command', ''))[:100]}  [{first}]")
                elif kind == "edit":
                    paths = ", ".join(self.workspace.rel(p) for p in self.workspace.targets(e["name"], args)) or "?"
                    rows.append(f"{stamp} sửa    {paths}" + ("" if e["ok"] else "  [lỗi]"))
                elif kind == "net" or e["name"].startswith("mcp__"):
                    rows.append(f"{stamp} mạng   {e['name']} {str(args.get('url') or args.get('query') or '')[:80]}")
            elif e["type"] == "approval":
                rows.append(f"{stamp} duyệt  {e['call']['name']}")
        return "\n".join(rows[-200:]) or "Chưa có file nào được sửa, lệnh nào được chạy."

    def _remember(self, category: str, pattern: str) -> None:
        home_path = self.home if self.home is not None else Path.home()
        try:
            rules.save_allow(home_path, category, pattern)
        except OSError as exc:
            logger.warning("Agent session {} could not save a rule: {}", self.id, exc)
            return
        self.rules = rules.load(self.workspace.root, home_path)
        self.emit("notice", text=f"Đã lưu luật: luôn cho phép {category} {pattern}")

    def _wait_for_decision(self, call: dict) -> dict:
        if self.parent is not None:
            return self.parent._wait_for_decision({**call, "agent": self.nick or (self.agent.name if self.agent else "")})
        key = self._decision_key(call)
        with self._approval_lock:
            with self._lock:
                previous_status, previous_pending = self.status, self.pending
                self.pending, self.status = call, "waiting"
                self._decisions[key] = None
            self.emit("approval", call=call)
            asked = time.time()
            with self._lock:
                while self._decisions.get(key) is None and not self._stop:
                    self._lock.wait(timeout=1.0)
                self.asked_s += time.time() - asked
                decision = self._decisions.pop(key, None) or {"decision": "deny", "note": "stopped"}
                self.pending, self.status = previous_pending, previous_status
            return decision

    def _tools_folder(self, scope: str = "user") -> Path:
        if scope == "workspace":
            return self.workspace.root / ".agents" / "tools"
        return (self.home if self.home is not None else Path.home()) / ".manga-agent" / "tools"

    def _workspace_tools_digest(self) -> str:
        h = hashlib.sha256()
        for path in sorted(self._tools_folder("workspace").glob("*.json")):
            try:
                h.update(path.name.encode() + b"\0" + path.read_bytes())
            except OSError:
                continue
        return h.hexdigest()

    def _load_created(self) -> dict[str, dict]:
        found = {}
        self.untrusted_tools: list[str] = []
        trusted = self.trust.trusted(self.workspace.root, "tools", self._workspace_tools_digest())
        for scope in ("user", "workspace"):
            for path in sorted(self._tools_folder(scope).glob("*.json")):
                try:
                    tool = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not (isinstance(tool, dict) and CREATED_NAME.fullmatch(str(tool.get("name"))) and tool.get("name") not in BUILTIN_NAMES and isinstance(tool.get("code"), str)):
                    continue
                if scope == "workspace" and not trusted:
                    self.untrusted_tools.append(tool["name"])
                    continue
                tool["scope"], tool["kind"] = scope, "read" if tool.get("kind") == "read" else "exec"
                found[tool["name"]] = tool
        return found

    def trust_tools(self) -> None:
        self.trust.allow(self.workspace.root, "tools", self._workspace_tools_digest())
        self._root().created.update(self._load_created())

    def _create(self, args: dict) -> str:
        asked = str(args.get("name") or "").strip()
        name = re.sub(r"[^a-z0-9_]+", "_", asked.lower()).strip("_")[:41]
        if name and not name[0].isalpha():
            name = f"t_{name}"[:41]
        if not CREATED_NAME.fullmatch(name):
            raise ToolError("name must be lower_case letters, digits and _, starting with a letter")
        created = self._root().created
        if name in BUILTIN_NAMES or name in self.registry.tools or name in self._root().mcp_tools:
            raise ToolError(f"{name} is reserved; pick another name, e.g. shell, read, edit")
        old = created.get(name)
        scope = args.get("scope") if args.get("scope") in ("user", "workspace") else (old or {}).get("scope", "user")
        path = self._tools_folder(scope) / f"{name}.json"
        code = re.sub(r"\A\s*```[\w+-]*[ \t]*\n(.*?)\n?```\s*\Z", r"\1", str(args.get("code") or ""), flags=re.S)
        if not code.strip():
            if name not in created:
                raise ToolError(f"No created tool {name}; created: {', '.join(created) or 'none'}")
            gone = created.pop(name)
            (self._tools_folder(gone.get("scope", "user")) / f"{name}.json").unlink(missing_ok=True)
            if gone.get("scope") == "workspace":
                self.trust.allow(self.workspace.root, "tools", self._workspace_tools_digest())
            return f"Deleted tool {name}."
        code = _as_function(name, code)
        try:
            compile(code, f"{name}.py", "exec")
        except SyntaxError as exc:
            raise ToolError(f"Syntax error: {exc}") from exc
        signature, doc, inferred = _signature(name, code)
        params = args.get("parameters") if isinstance(args.get("parameters"), dict) and args["parameters"].get("properties") else inferred
        params = {"type": "object", "properties": {}, **params}
        about = str(args.get("description") or doc or (old or {}).get("about", (old or {}).get("description")) or "").strip()
        kind = args.get("kind") if args.get("kind") in ("read", "exec") else (old or {}).get("kind", "exec")
        tool = {"name": name, "description": (f"{signature}: {about}" if about and signature not in about else about or signature)[:1024],
                "about": about, "parameters": params, "code": code, "kind": kind}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tool, ensure_ascii=False, indent=1), encoding="utf-8")
        if old and old.get("scope", "user") != scope:
            (self._tools_folder(old.get("scope", "user")) / f"{name}.json").unlink(missing_ok=True)
        if "workspace" in (scope, (old or {}).get("scope")):
            self.trust.allow(self.workspace.root, "tools", self._workspace_tools_digest())
        created[name] = {**tool, "scope": scope}
        takes = ", ".join(params["properties"]) or "no arguments"
        loose = "" if "def " in code else " It has no function, so it reads its arguments from the dict args."
        where = "with this project" if scope == "workspace" else "for later sessions"
        how = "read-only and offline, without asking" if kind == "read" and self._confined() else "in the sandbox"
        return (f"{'Updated' if old else 'Created'} tool {tool['description'].rstrip('.')}. Call {name} directly as a tool ({takes}); "
                f"it runs {how} and is kept {where}.{loose}" + (f" The name {asked!r} was not valid, so it is {name}." if asked != name else ""))

    def _confined(self) -> bool:
        return sandbox.backend() != "none" and self.workspace.policy.mode != "full-access"

    def _looks_only(self, name: str) -> bool:
        plugin = self.registry.tools.get(name)
        if plugin is not None:
            return plugin.kind == "read"
        if name in self._root().mcp_tools:
            return self._role(name) == "read"
        return name in LOOK_ONLY or KIND.get(name) == "read" or name in ("skill", "_search", "_describe", "_source", "_intent", "_chain", "_long")

    def _run_created(self, call: dict) -> str:
        tool = self._root().created[call["name"]]
        args = call["args"] if isinstance(call["args"], dict) else {}
        problems = chaincheck.check(self._with_callees(tool["code"]), self._chain_arguments())
        if problems:
            raise ToolError("The chain was not run, because its code calls tools that cannot work:\n" + "\n".join(problems[:10]))
        looks = tool.get("kind") == "read"
        policy = sandbox.Policy("read-only", False) if looks else sandbox.Policy("read-only", self.workspace.policy.network) if self.plan_mode else self.workspace.policy
        state = self._outputs_dir() / "state"
        state.mkdir(parents=True, exist_ok=True)
        docker = sandbox.backend() == "docker" and policy.mode != "full-access"
        self._write_library(state)
        run_id = uuid.uuid4().hex[:12]
        rpc = state / "rpc" / run_id
        rpc.mkdir(parents=True)
        where = "/tmp" if docker else str(state)
        prelude = (f"import builtins as _b, os as _os, sys as _sys\n_os.environ['AGENT_STATE'] = {where!r}\n_sys.path.insert(0, _os.path.join({where!r}, 'lib'))\n"
                   f"_os.environ['AGENT_RPC'] = _os.path.join({where!r}, 'rpc', {run_id!r})\nfrom agent_tools import tools\n"
                   "memo = _b.memo = getattr(_b, 'memo', {})\n")
        script = prelude + f"args = __import__('json').loads({json.dumps(json.dumps(args, ensure_ascii=False, default=str))})\n" + tool["code"] + CREATED_CALL.format(args=json.dumps(args, ensure_ascii=False, default=str), name=tool["name"])
        self._write_private(rpc / "main.py", script)
        target = f"/tmp/rpc/{run_id}/main.py" if docker else str(rpc / "main.py")
        command = (f"python3 {target}" if docker else
                   subprocess.list2cmdline([sys.executable, target]) if os.name == "nt" else f"{shlex.quote(sys.executable)} {shlex.quote(target)}")
        self._intent()
        if not self._watch.ready or self._watch_request != self._request_no:
            self._watch.refresh()
            self._watch_request = self._request_no
        guard = gitguard.snapshot(self.workspace.root)
        top = not self.plan_mode and not str(call.get("id", "")).startswith(("chain-", "script-", "mcp-", "plugin-"))
        if top:
            self._auto_mark(f"before-{tool['name']}")
        if self._repl is None:
            self._repl = repl.Kernel(state)
        job = self._repl.start(run_id, target, policy, self.workspace.root, where, docker)
        note = job.note if job is not None else "[ran in a fresh process because another tool call was still running, so memo started empty]\n"
        if job is None:
            job = sandbox.Job(command, policy, self.workspace.root, scratch=str(state))
            job.close_input()
        server = threading.Thread(target=self._serve_chain, args=(rpc, job, None, looks), name=f"agent-chain-{self.id}", daemon=True)
        server.start()
        output = self._await_job(job, _ses("CREATED_TIMEOUT"), guard)
        if job.code is not None:
            server.join(2)
        changed = self._record_changes()
        if top and re.search(r"^PASS\b", output, re.M):
            self._auto_mark(f"pass-{tool['name']}")
        check = f"\n{INTENT_CHECK}" if not self.depth and self.profile["intent"] and self._intent(write=False) else ""
        return note + output + (f"\n[changed: {', '.join(changed[:30])}{' …' if len(changed) > 30 else ''}]" if changed else "") + check

    def _write_library(self, state: Path) -> None:
        (state / "lib").mkdir(parents=True, exist_ok=True)
        sources = {n: t["code"] for n, t in self._root().created.items()}
        part = state / "lib" / f"agent_tools.{uuid.uuid4().hex[:8]}.part"
        part.write_text(TOOLS_LIBRARY.format(sources=json.dumps(sources, ensure_ascii=False),
                                             remote=json.dumps(sorted(self._chain_targets()))), encoding="utf-8")
        part.replace(state / "lib" / "agent_tools.py")
        (state / "bin").mkdir(exist_ok=True)
        lib = state / "lib"
        if os.name == "nt":
            (state / "bin" / "agent-tool.cmd").write_text(f'@set "PYTHONPATH={lib};%PYTHONPATH%"\r\n@"{sys.executable}" -m agent_tools %*\r\n', encoding="utf-8")
        else:
            tool = state / "bin" / "agent-tool"
            tool.write_text(f'#!/bin/sh\nPYTHONPATH={shlex.quote(str(lib))}${{PYTHONPATH:+:$PYTHONPATH}} exec {shlex.quote(sys.executable)} -m agent_tools "$@"\n', encoding="utf-8")
            tool.chmod(0o755)
        self._write_shims(state / "bin")
        self._write_fallbacks(state)

    def _write_fallbacks(self, state: Path) -> None:
        if os.name == "nt" or getattr(self, "_fallbacks", None) == state:
            return
        from app.agent import fallbacks

        bin_dir, lib = state / "bin", state / "lib"
        (lib / "agent_fallbacks.py").write_text(Path(fallbacks.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        python = shlex.quote(sys.executable)
        note = shlex.quote(FALLBACK_NOTE.format(version=".".join(map(str, sys.version_info[:2]))))
        for name in ("python3", "python", *fallbacks.COMMANDS):
            if shutil.which(name):
                continue
            if name in fallbacks.COMMANDS:
                body = f'PYTHONPATH={shlex.quote(str(lib))} exec {python} -m agent_fallbacks {name} "$@"'
            else:
                body = (f'[ -e "${{AGENT_STATE:-/tmp}}/.fallback-python" ] || {{ : > "${{AGENT_STATE:-/tmp}}/.fallback-python"; echo {note} >&2; }}\n'
                        f'exec {python} "$@"')
            shim = bin_dir / name
            shim.write_text(f"#!/bin/sh\n# {FALLBACK_MARK}\n{body}\n", encoding="utf-8")
            shim.chmod(0o755)
        self._fallbacks = state

    def _write_shims(self, bin_dir: Path) -> None:
        own = {s["name"] for s in self.specs() + self.specs(self.library_specs())} | {"agent", "think", "vote", "models", "skill", "branch", "adopt"}
        names = {n for n in self._chain_targets() | set(self._root().created) | {"branch", "adopt"} if SHIM_NAME.fullmatch(n)} - {"run_command"}
        names = {n for n in names if n in own or not shutil.which(n)}
        if getattr(self, "_shims", None) == (bin_dir, names):
            return
        for old in bin_dir.iterdir():
            if old.stem not in names and old.stem != "agent-tool" and SHIM_MARK in old.read_text(encoding="utf-8", errors="replace")[:200]:
                old.unlink()
        for name in names:
            if os.name == "nt":
                (bin_dir / f"{name}.cmd").write_text(f'@rem {SHIM_MARK}\r\n@"{bin_dir / "agent-tool.cmd"}" {name} %*\r\n', encoding="utf-8")
            else:
                shim = bin_dir / name
                shim.write_text(f'#!/bin/sh\n# {SHIM_MARK}\nexec {shlex.quote(str(bin_dir / "agent-tool"))} {name} "$@"\n', encoding="utf-8")
                shim.chmod(0o755)
        self._shims = (bin_dir, names)

    def _command_job(self, command: str, policy: "sandbox.Policy", tty: bool = False) -> "sandbox.Job":
        if not self._bridged(policy):
            return sandbox.Job(command, policy, self.workspace.root, tty=tty)
        state = self._outputs_dir() / "state"
        self._write_library(state)
        rpc = state / "rpc" / uuid.uuid4().hex[:12]
        rpc.mkdir(parents=True)
        job = sandbox.Job(command, policy, self.workspace.root, tty=tty, scratch=str(state),
                          env={"PYTHONPATH": str(state / "lib"), "PATH": str(state / "bin"), "AGENT_RPC": str(rpc), "AGENT_STATE": str(state)})
        threading.Thread(target=self._serve_chain, args=(rpc, job, None, self.plan_mode, "chain-shell-"), name=f"agent-bridge-{self.id}", daemon=True).start()
        return job

    def _bridged(self, policy: "sandbox.Policy") -> bool:
        return sandbox.backend() != "docker" or policy.mode == "full-access"

    def _chain_arguments(self) -> dict[str, set[str] | None]:
        known: dict[str, set[str] | None] = {}
        created = self._root().created
        for name in self._chain_targets() | set(created):
            if name in created:
                schema = created[name].get("parameters") or {}
            elif name in ("think", "agent", "skill", "vote", "models", "create"):
                schema = {}
            else:
                schema = (self._target_spec(name) or {}).get("parameters") or {}
            props = schema.get("properties") if isinstance(schema, dict) else None
            known[name] = set(props) if isinstance(props, dict) and props and not schema.get("additionalProperties") else None
        return known

    def _intent(self, write: bool = True) -> str:
        root = self._root()
        path = root._outputs_dir() / "state" / "intent.md"
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if not write or not root.profile["intent"] or root._intent_request == root._request_no or not root._last_request.strip():
            return current
        root._intent_request = root._request_no
        asked = "\n\n".join([m[:3000] for m in root._user_messages()[-4:]] + root._answers[-6:])
        messages = [{"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": (f"Earlier picture:\n{current[:4000]}\n\n" if current else "") + f"What the user said, latest last:\n{asked}"}]
        try:
            reply = root.complete(root.provider, root.api_key, root.model, messages, tools=None)
        except Exception:
            return current
        with root._usage_lock:
            for key in root.usage:
                root.usage[key] += int(reply.get("usage", {}).get(key) or 0)
        text = re.sub(r"^```(?:markdown|md)?\n|\n```\s*$", "", reply["text"].strip())[:6000] + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        root._answers = []
        root.emit("notice", text="Đã ghi cách hiểu yêu cầu (/intent để xem).")
        return text

    def _chain_targets(self) -> set[str]:
        names = {n for n in self.registry.tools if n not in registry.GROUPS} | set(self._root().mcp_tools) | {"agent", "create", "think"}
        names |= {s["name"] for s in self.specs() + self.specs(self.library_specs())} - {"exit_plan_mode", "goal_done"} - set(self._root().created)
        names |= {"vote", "models"} if self.profile["model_roles"] or self._sources() else set()
        return names | ({"skill"} if self.skills else set())

    def _serve_chain(self, folder: Path, job: "sandbox.Job", ws: Workspace | None = None, looks: bool = False, origin: str = "chain-") -> None:
        targets = self._chain_targets()

        def answer(request: Path, data: dict) -> None:
            name, args = str(data.get("name") or ""), data.get("args") if isinstance(data.get("args"), dict) else {}
            try:
                if looks and not self._looks_only(name):
                    output, ok = f"{name!r} changes things, and this tool was created with kind read; create it with kind exec to call it", False
                elif name in ("_chain", "_long"):
                    output, ok = self._spend_chain(name)
                elif name in ("_search", "_describe"):
                    output, ok = self._find_tools(name, args, targets)
                elif name == "_repair":
                    output, ok = self._repair(args)
                elif name == "_intent":
                    output, ok = self._intent() or "no picture of the request was written", True
                elif name == "_source":
                    tool = self._root().created.get(str(args.get("name") or ""))
                    output, ok = (tool["code"], True) if tool else ("no such created tool", False)
                elif name in ("_branch", "_adopt"):
                    output, ok = (("a branch cannot branch or adopt; return what it found and choose at the top", False) if ws is not None
                                  else self._branch(args) if name == "_branch" else self._adopt(args))
                elif ws is not None and name in BRANCH_LOCAL:
                    output, ok = self._run_branch_tool(ws, name, args)
                elif name not in targets:
                    output, ok = f"{name!r} cannot be reached from a tool; reachable: {', '.join(sorted(targets)) or 'none'}", False
                elif name == "skill":
                    found = self.skills.get(str(args.get("name") or ""))
                    output, ok = (skills.load(found), True) if found else (f"No skill named {args.get('name')!r}; known: {', '.join(self.skills)}", False)
                elif name == "think":
                    output, ok = self._think(str(args.get("question") or ""), str(args.get("answer") or ""), str(args.get("model") or "")), True
                elif name == "vote":
                    output, ok = self._vote(args), True
                elif name == "models":
                    output, ok = self._models(str(args.get("source") or ""), str(args.get("query") or "")), True
                elif name == "agent":
                    output, ok = self._spend_chain("agent")
                    if ok:
                        role = {"role": str(args["model"])} if args.get("model") else {}
                        output = self._subagent("chain", str(args.get("prompt") or ""), str(args.get("agent") or "explore"), **role)
                elif name == "create":
                    output, ok = self._create(args), True
                else:
                    output, ok = self._run_call({"id": f"{origin}{uuid.uuid4().hex[:8]}", "name": name, "args": args})
            except Exception as exc:
                output, ok = f"Error: {type(exc).__name__}: {exc}", False
            reply = folder / request.name.replace(".req", ".res")
            part = reply.with_suffix(".part")
            part.write_text(json.dumps({"ok": ok, "output": output}), encoding="utf-8")
            os.chmod(part, 0o600)
            part.replace(reply)

        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                done = job.code is not None
                for request in sorted(folder.glob("*.req")):
                    try:
                        data = json.loads(request.read_text(encoding="utf-8"))
                        request.unlink()
                    except (OSError, ValueError):
                        continue
                    pool.submit(answer, request, data if isinstance(data, dict) else {})
                if done or self._stop:
                    break
                time.sleep(0.02)
        if self._stop:
            self._refuse_pending(folder)
        shutil.rmtree(folder, ignore_errors=True)

    @staticmethod
    def _refuse_pending(folder: Path) -> None:
        for request in sorted(folder.glob("*.req")):
            try:
                reply = folder / request.name.replace(".req", ".res")
                part = reply.with_suffix(".part")
                part.write_text(json.dumps({"ok": False, "output": "the agent stopped before answering this tool call"}), encoding="utf-8")
                os.chmod(part, 0o600)
                part.replace(reply)
                request.unlink()
            except OSError:
                pass

    def _run_branch_tool(self, ws: Workspace, name: str, args: dict) -> tuple[str, bool]:
        call = {"id": f"chain-{uuid.uuid4().hex[:8]}", "name": name, "args": args}
        lock = self.__dict__.setdefault("_branch_ws_lock", threading.Lock())
        with lock:
            real = self.workspace
            call["args"] = self._on_copy(args, ws.root)
            self.workspace = ws
            try:
                return self._run_call(call)
            finally:
                self.workspace = real
