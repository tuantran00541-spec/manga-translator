from __future__ import annotations

import atexit
import collections
import copy
import ctypes
import difflib
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.agent import (
    agents, chaincheck, client, codemode, context, gitguard, isolate,
    mcp, memory, models, registry, repl, roles, rules, sandbox,
    schema as schemas, skill_install, skills, watch,
)
from app.agent.checkpoint import Checkpoints
from app.agent.session_config import *  # noqa: F401, F403
from app.agent.session_config import _as_function, _signature
from app.agent.tools import COMMAND_TIMEOUT, KIND, MAX_COMMAND_TIMEOUT, SPECS, ToolError, Workspace, clip
from app.ai_providers import auth_headers, resolve_provider
from app.logging_config import logger


def _ses(name: str):
    return getattr(sys.modules["app.agent.session"], name)


class AgentChainsMixin:
    """Branch execution, role routing, repair, and change tracking for AgentSession."""

    @staticmethod
    def _write_private(path: Path, text: str) -> None:
        part = path.with_name(f"{path.name}.part-{os.getpid()}-{threading.get_ident()}")
        part.write_text(text, encoding="utf-8")
        os.chmod(part, 0o600)
        part.replace(path)

    @staticmethod
    def _signature(call: dict) -> str:
        return call["name"] + json.dumps(call["args"], sort_keys=True, default=str)

    def _redone(self, call: dict, output: str, ok: bool) -> str:
        self._after_summary += 1
        same = ok and self._before_summary.get(self._signature(call)) == hashlib.sha1(output.encode()).hexdigest()
        if self._after_summary >= POST_SUMMARY_WINDOW:
            self._before_summary = {}
        if not same:
            return ""
        self._repeats += 1
        return ("\n[You made this exact call before the summary and got the same result. The summary may not mention it: check its Progress "
                "before redoing more work.]")

    def _on_copy(self, value, copy: Path):
        if isinstance(value, str):
            return value.replace(str(self.workspace.root), str(copy))
        if isinstance(value, dict):
            return {k: self._on_copy(v, copy) for k, v in value.items()}
        if isinstance(value, list):
            return [self._on_copy(v, copy) for v in value]
        return value

    def _branch(self, args: dict) -> tuple[str, bool]:
        if sandbox.backend() == "docker" and self.workspace.policy.mode != "full-access":
            return "tools.branch needs the local sandbox", False
        branches, check = args.get("branches"), str(args.get("check") or "")
        if not isinstance(branches, dict) or not 2 <= len(branches) <= MAX_BRANCHES or not all(isinstance(c, str) and c.strip() for c in branches.values()):
            return f"tools.branch takes 2 to {MAX_BRANCHES} named pieces of code: tools.branch({{'a': code_a, 'b': code_b}}, check=check_code)", False
        self._drop_branches()
        top = self._outputs_dir() / "branches" / uuid.uuid4().hex[:8]

        def one(name: str, code: str) -> tuple:
            copy = top / (re.sub(r"[^A-Za-z0-9_-]", "_", name)[:40] or "b")
            base = isolate.snapshot(self.workspace.root, copy)
            if base is None:
                return name, copy, None, "the workspace is too big to copy", "", ""
            ws = Workspace(copy, self.workspace.policy)
            out = self._branch_run(ws, code)
            verdict = self._branch_run(ws, check) if check.strip() else "(no check given)"
            return name, copy, base, out, verdict, self._branch_diff(copy, base)

        with ThreadPoolExecutor(max_workers=len(branches)) as pool:
            rows = list(pool.map(lambda item: one(str(item[0]), item[1]), branches.items()))
        self._branches = {name: (copy, base) for name, copy, base, *_ in rows if base is not None}
        table = "\n\n".join(f"== branch {name} ==\n[changes]\n{diff or '(none)'}\n[output]\n{clip(out, 3000)}\n[check]\n{clip(verdict, 3000)}"
                            for name, _, _, out, verdict, diff in rows)
        return (table + "\n\nNothing was applied. Compare the branches against every requirement, then tools.adopt('<name>') (adopt <name> in the shell) "
                "the one to keep or write new attempts, and run the check again on the real workspace before you finish."), True

    def _branch_run(self, ws: Workspace, code: str) -> str:
        state = self._outputs_dir() / "state"
        (state / "lib").mkdir(parents=True, exist_ok=True)
        sources = {n: t["code"] for n, t in self._root().created.items()}
        (state / "lib" / "agent_tools.py").write_text(TOOLS_LIBRARY.format(sources=json.dumps(sources, ensure_ascii=False),
                                                                          remote=json.dumps(sorted(self._chain_targets() | BRANCH_LOCAL))), encoding="utf-8")
        run_id = uuid.uuid4().hex[:12]
        rpc = state / "rpc" / run_id
        rpc.mkdir(parents=True)
        prelude = (f"import builtins as _b, os as _os, sys as _sys\n_os.environ['AGENT_STATE'] = {str(state)!r}\n_sys.path.insert(0, _os.path.join({str(state)!r}, 'lib'))\n"
                   f"_os.environ['AGENT_RPC'] = {str(rpc)!r}\nfrom agent_tools import tools\nargs = {{}}\n")
        self._write_private(rpc / "main.py",
                           prelude + _as_function("branch", code) + CREATED_CALL.format(args="{}", name="branch"))
        target = str(rpc / "main.py")
        command = subprocess.list2cmdline([sys.executable, target]) if os.name == "nt" else f"{shlex.quote(sys.executable)} {shlex.quote(target)}"
        job = sandbox.Job(command, ws.policy, ws.root, scratch=str(state))
        job.close_input()
        server = threading.Thread(target=self._serve_chain, args=(rpc, job, ws), name=f"agent-branch-{self.id}", daemon=True)
        server.start()
        end = time.time() + BRANCH_TIMEOUT
        while job.code is None and time.time() < end and not self._stop:
            time.sleep(0.05)
        finished = job.code is not None
        if finished:
            job.out.join(5)
        output, code_ = job.out.text().strip(), job.code
        job.stop()
        server.join(2)
        return f"{output}\n[{f'exit code {code_}' if finished else f'stopped after {BRANCH_TIMEOUT} s'}]"

    def _branch_diff(self, copy: Path, base: dict[str, str]) -> str:
        now = {p.relative_to(copy).as_posix() for p in isolate._files(copy)}
        rows = []
        for rel in sorted(now | set(base)):
            if rel not in now:
                rows.append(f"deleted {rel}")
                continue
            if base.get(rel) == isolate._hash(copy / rel):
                continue
            try:
                old = (self.workspace.root / rel).read_text(encoding="utf-8").splitlines() if rel in base else []
                new = (copy / rel).read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                rows.append(f"changed {rel} (not text)")
                continue
            rows.append("\n".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm="", n=1)) or f"new empty file {rel}")
        return clip("\n".join(rows), 6000)

    def _adopt(self, args: dict) -> tuple[str, bool]:
        name = str(args.get("name") or "")
        if name not in self._branches:
            return f"no branch {name!r}; open branches: {', '.join(self._branches) or 'none, run tools.branch first'}", False
        copy, base = self._branches[name]
        self._auto_mark(f"before-adopt-{name}")
        applied, conflicts = isolate.merge(self.workspace.root, copy, base, f"branch-{name}", before=self.checkpoints.save)
        self._drop_branches()
        kept = f"; changed on both sides, kept under {isolate.CONFLICTS}/branch-{name}/: {', '.join(conflicts)}" if conflicts else ""
        return f"Adopted branch {name}: {', '.join(applied) or 'no file changes'}{kept}. Run the check on the real workspace before you finish.", True

    def _log_outside(self, call: dict) -> None:
        name, args = call["name"], call["args"] if isinstance(call["args"], dict) else {}
        if name == "run_command" and not rules.safe_readonly(str(args.get("command") or "")):
            self.checkpoints.note_ran(f"run_command: {clip(str(args.get('command') or ''), 200)}")
        elif name in self._root().created or name in self.replaced or (name in self.mcp_tools and self._role(name) != "read"):
            self.checkpoints.note_ran(f"{name}({clip(json.dumps(args, ensure_ascii=False, default=str), 160)})")

    def _auto_mark(self, label: str) -> None:
        self.checkpoints.auto_no += 1
        self.checkpoints.mark(f"{self.checkpoints.auto_no}-{label}", auto=True)

    def _drop_branches(self) -> None:
        for branch, _ in self._branches.values():
            shutil.rmtree(branch, ignore_errors=True)
        self._branches = {}

    def _target_spec(self, name: str) -> dict | None:
        root = self._root()
        if name in root.created:
            return {k: root.created[name][k] for k in ("name", "description", "parameters")}
        if name in root.mcp_tools:
            server, tool = root.mcp_tools[name]
            return {"name": name, "description": f"[MCP {server}] {tool.get('description') or tool['name']}", "parameters": tool.get("inputSchema") or OBJ}
        if name in self.registry.tools:
            return self.registry.tools[name].spec
        extra = {"think": "Ask the model a question from inside the running code: think(question, answer='the form of the answer').",
                 "agent": "Run a helper agent and return its report: agent(prompt, agent='explore').",
                 "skill": f"Return a skill's instructions: skill(name). Skills: {', '.join(self.skills)}.",
                 "models": "List the model providers and roles, or one provider's models with prices: models(source='', query='').",
                 "vote": "Ask several models the same question: vote(question, models=['<source>:<model>' or roles], answer='')."}
        if name in extra:
            return {"name": name, "description": extra[name], "parameters": OBJ}
        return next((dict(s) for s in [*SPECS, *SESSION_SPECS.values()] if s["name"] == name), None)

    def _find_tools(self, kind: str, args: dict, targets: set[str]) -> tuple[str, bool]:
        names = sorted(targets | set(self._root().created))
        if kind == "_describe":
            name = str(args.get("name") or "")
            spec = self._target_spec(name) if name in names else None
            return (json.dumps(spec, ensure_ascii=False), True) if spec else (f"no reachable tool {name!r}; use tools.search(query)", False)
        words = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", str(args.get("query") or ""))}
        rows = []
        for name in names:
            spec = self._target_spec(name) or {"description": ""}
            text = f"{name} {spec.get('description') or ''}".lower()
            score = sum(w in text for w in words) + sum(w in name for w in words)
            if score or not words:
                rows.append((-score, name, " ".join(str(spec.get("description") or "").split())[:160]))
        rows.sort()
        try:
            limit = max(1, min(50, int(args.get("limit") or 15)))
        except (TypeError, ValueError):
            limit = 15
        return json.dumps([{"name": n, "description": d} for _, n, d in rows[:limit]], ensure_ascii=False), True

    def _note_one_off(self, name: str, tool: dict, output: str) -> str:
        if not tool or "[exit code 0]" not in output.removesuffix("\n" + INTENT_CHECK)[-200:]:
            return ""
        key = hashlib.sha1(tool["code"].encode()).hexdigest()
        row = self._one_offs.setdefault(key, {"name": name, "tool": tool, "runs": 0})
        row["runs"] += 1
        if row["runs"] < 2 or name in self._root().created:
            return ""
        self._create({k: tool[k] for k in ("name", "description", "parameters", "code")})
        self.emit("notice", text=f"Đã giữ {name} làm công cụ (chạy tốt 2 lần).")
        return f"\n[this one-off ran cleanly twice, so it is now kept as the tool {name}]"

    def _spend_chain(self, kind: str) -> tuple[str, bool]:
        root = self._root()
        limit, label = {"_chain": (_ses("MAX_CHAINS"), "chains"), "_long": (_ses("MAX_LONG_CHAINS"), "chains of more than 10 steps"),
                        "agent": (_ses("MAX_CHAIN_AGENTS"), "helper agents started from chains")}[kind]
        with self._usage_lock:
            root.chain_budget[kind] = used = root.chain_budget.get(kind, 0) + 1
        if used > limit:
            return f"chain budget used up: at most {limit} {label} per request; finish with what you have or ask the user", False
        return "ok", True

    def _role_target(self, name: str, seen: tuple = ()) -> tuple[str, object, str, str]:
        table = self.profile["model_roles"]
        cfg = table.get(name)
        if cfg is None and ":" in name:
            return self._pick(name)
        if cfg is None:
            raise ToolError(f"no model role {name!r}; roles: {', '.join(table) or 'none'}; or name a model as '<source>:<model id>' from tools.models()")
        if not roles.within(self.home if self.home is not None else Path.home(), name, cfg):
            if cfg["fallback"] and cfg["fallback"] not in (*seen, name):
                return self._role_target(cfg["fallback"], (*seen, name))
            raise ToolError(f"model role {name} has used today's budget and no fallback is left")
        if "provider" in cfg:
            find = self._root().find_provider
            if find is None:
                raise ToolError(f"role {name} names the app provider {cfg['provider']!r}, which this session cannot look up")
            provider, keys = find(cfg["provider"])
            key = keys[0] if keys else ""
        else:
            provider = resolve_provider(f"role-{name}", label=f"role {name}", protocol="openai", api_base=cfg["base_url"])
            key = roles.env_key(cfg)
        if not key:
            raise ToolError(f"role {name} has no API key")
        return name, provider, key, cfg["model"]

    def _sources(self) -> dict[str, str]:
        root = self._root()
        cached = getattr(root, "_source_cache", None)
        if cached is not None and time.time() - cached[0] < 60:
            return cached[1]
        found = {n: c["note"] or c["base_url"] for n, c in self.profile["model_sources"].items()}
        if root.list_providers is not None:
            try:
                found.update({p_id: label for p_id, label in root.list_providers() if p_id not in found})
            except Exception:
                pass
        root._source_cache = (time.time(), found)
        return found

    def _source(self, source: str) -> tuple[object, str, dict]:
        cfg = self.profile["model_sources"].get(source)
        if cfg is not None:
            provider = resolve_provider(f"source-{source}", label=source, protocol="openai", api_base=cfg["base_url"])
            key = roles.env_key(cfg)
        else:
            find = self._root().find_provider
            if find is None or source not in self._sources():
                raise ToolError(f"no model source {source!r}; sources: {', '.join(self._sources()) or 'none'}")
            provider, keys = find(source)
            key = keys[0] if keys else ""
        if not key:
            raise ToolError(f"model source {source} has no API key")
        return provider, key, cfg or {}

    def _pick(self, spec: str) -> tuple[str, object, str, str]:
        source, _, model = spec.partition(":")
        if not model.strip():
            raise ToolError(f"name a model as '<source>:<model id>', not {spec!r}")
        provider, key, cfg = self._source(source)
        if cfg and not roles.within(self.home if self.home is not None else Path.home(), source, cfg):
            raise ToolError(f"model source {source} has used today's budget")
        self._price_gate(source, model.strip())
        return source, provider, key, model.strip()

    def _price_gate(self, source: str, model: str) -> None:
        root = self._root()
        spec = f"{source}:{model}"
        if spec in root._paid_models:
            return
        try:
            price = next((r["price"] for r in json.loads(self._models(source, model)) if r["id"] == model), {})
        except Exception:
            price = {}
        if price.get("in", 0) <= PRICE_CAP[0] and price.get("out", 0) <= PRICE_CAP[1]:
            return
        decision = self._wait_for_decision({"id": f"price-{uuid.uuid4().hex[:8]}", "name": "use_model",
                                            "args": {"model": spec, "per_million_in": price.get("in"), "per_million_out": price.get("out")}})
        if decision["decision"] == "deny":
            raise ToolError(f"{spec} costs ${price.get('in', 0):g} in / ${price.get('out', 0):g} out per million tokens, above the "
                            f"${PRICE_CAP[0]:g} / ${PRICE_CAP[1]:g} limit, and the user did not allow it; pick a cheaper model")
        root._paid_models.add(spec)

    def _models(self, source: str = "", query: str = "") -> str:
        if not source:
            used = roles.today(self.home if self.home is not None else Path.home())
            return json.dumps({"sources": self._sources(), "roles": {n: {"model": c["model"], "note": c["note"], "today": used.get(n, {})}
                                                                      for n, c in self.profile["model_roles"].items()},
                               "use": "pass model='<source>:<model id>' or a role name"}, ensure_ascii=False)
        if source in self.profile["model_roles"]:
            _, provider, key, _ = self._role_target(source)
        else:
            provider, key, _ = self._source(source)
        cache = self._root().__dict__.setdefault("_catalogs", {})
        stamp, rows = cache.get((source, query), (0, None))
        if rows is None or time.time() - stamp > 600:
            rows = roles.catalog(provider.api_base, auth_headers(provider, key), query.strip())
            cache[(source, query)] = (time.time(), rows)
        return json.dumps(rows, ensure_ascii=False)

    def _count_role(self, name: str, requests_made: int, tokens: int) -> None:
        home = self.home if self.home is not None else Path.home()
        cfg = (self.profile.get("model_roles") or {}).get(name) or None
        if cfg is None:
            cfg = (self.profile.get("model_sources") or {}).get(name) or None
        if not roles.add(home, name, requests_made, tokens, cfg):
            logger.warning("model role {} exceeded today's budget; its usage was not counted", name)
        with self._usage_lock:
            row = self._root().role_usage.setdefault(name, {"requests": 0, "tokens": 0})
            row["requests"] += requests_made
            row["tokens"] += tokens

    def _vote(self, args: dict) -> str:
        question, answer = str(args.get("question") or ""), str(args.get("answer") or "")
        names = [str(m) for m in args.get("models") or []][:5] if isinstance(args.get("models"), list) else []
        if len(names) < 2:
            raise ToolError("vote needs models=[two or more model roles]")
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {n: pool.submit(self._think, question, answer, n) for n in names}
        answers = {}
        for n, future in futures.items():
            try:
                answers[n] = future.result()
            except Exception as exc:
                answers[n] = f"Error: {exc}"
        plain = {" ".join(a.lower().strip(" .").split()) for a in answers.values() if not a.startswith("Error:")}
        return json.dumps({"answers": answers, "agree": len(plain) == 1 and len(plain) > 0}, ensure_ascii=False)

    def _think(self, question: str, answer: str = "", model: str = "") -> str:
        if not question.strip():
            raise ToolError("think needs a question")
        shape = f" in this form: {answer}" if answer else ""
        messages = [{"role": "system", "content": f"A program you wrote is running and asks you this question. Reply with the answer only{shape}."},
                    {"role": "user", "content": question[:200_000]}]
        if model:
            used, provider, key, name = self._role_target(model)
            reply = self._raw_complete(provider, key, name, messages, tools=None)
            usage = reply.get("usage") or {}
            self._count_role(used, 1, int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
            return reply["text"].strip()
        reply = self.complete(self.provider, self.api_key, self.model, messages, tools=None)
        with self._usage_lock:
            for key in self.usage:
                self.usage[key] += int(reply.get("usage", {}).get(key) or 0)
        return reply["text"].strip()

    def _repair(self, args: dict) -> tuple[str, bool]:
        code, ok = self._repair_once(args)
        return code, ok

    def _repair_once(self, args: dict) -> tuple[str, bool]:
        name = str(args.get("name") or "")
        tool = self._root().created.get(name)
        if tool is None:
            return "not a created tool", False
        if self.mode not in HANDS_OFF:
            return "live repair runs only in bypass mode, where nobody is asked before code runs", False
        messages = [{"role": "system", "content": "A tool in a program you wrote failed while the program is still running. Reply with the whole corrected "
                                                  "Python code of that tool, same function name and parameters, in one ```python block."},
                    {"role": "user", "content": f"Tool {name}:\n```python\n{tool['code']}\n```\n\nCalled with: {json.dumps(args.get('args'), ensure_ascii=False)[:4000]}\n\n"
                                                f"Error:\n{str(args.get('error') or '')[-4000:]}\n\nWhat the user asked:\n{self._last_request[:3000]}"}]
        reply = self.complete(self.provider, self.api_key, self.model, messages, tools=None)
        with self._usage_lock:
            for key in self.usage:
                self.usage[key] += int(reply.get("usage", {}).get(key) or 0)
        found = re.search(r"```(?:python)?\n(.*?)```", reply["text"], re.S)
        code = (found.group(1) if found else reply["text"]).strip() + "\n"
        try:
            compile(code, f"{name}.py", "exec")
        except SyntaxError as exc:
            return f"the repair did not compile: {exc}", False
        self._create({"name": name, "code": code, "description": tool["description"], "parameters": tool["parameters"]})
        self.emit("notice", text=f"Đã sửa công cụ {name}.")
        return code, True

    def _with_callees(self, code: str) -> str:
        created, seen, todo, parts = self._root().created, set(), [code], [code]
        while todo:
            for name in re.findall(r"\btools\.([a-z][a-z0-9_]*)", todo.pop()):
                if name in created and name not in seen:
                    seen.add(name)
                    todo.append(created[name]["code"])
                    parts.append(f"# tools.{name}:\n{created[name]['code']}")
        return "\n\n".join(parts)

    def _record_changes(self) -> list[str]:
        if not self._watch.ready:
            return []
        changed = []
        for path, old in self._watch.changes():
            if old is not watch.UNKNOWN:
                self.checkpoints.remember(path, old)
            changed.append(self.workspace.rel(path))
        if changed:
            self._edited = True
        return changed
