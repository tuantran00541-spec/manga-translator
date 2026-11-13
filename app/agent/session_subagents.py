from __future__ import annotations

import copy
import difflib
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse


from app.agent import (
    agents,
    codemode,
    gitguard,
    isolate,
    mcp,
    memory,
    rules,
    sandbox,
    schema as schemas,
    skills,
)
from app.agent.session_config import *  # noqa: F401, F403
from app.agent.tools import COMMAND_TIMEOUT, KIND, MAX_COMMAND_TIMEOUT, SPECS, ToolError, Workspace, clip


def _ses(name: str):
    return getattr(sys.modules["app.agent.session"], name)


class AgentSubagentsMixin:
    """Session tool dispatch, jobs, checkpoints, and subagent orchestration for AgentSession."""

    def _session_tool(self, call: dict) -> str:
        args = call["args"]
        if call["name"] == "create":
            if args.get("keep") is False and str(args.get("name") or "").strip() in self._root().created:
                raise ToolError("a one-off script needs a name no kept tool has, or it would replace and then delete that tool")
            made = self._create(args)
            if (not isinstance(args.get("args"), dict) and args.get("keep") is not False) or not str(args.get("code") or "").strip():
                return made
            name = str(args.get("name") or "").strip()
            kept = self._root().created.get(name, {})
            try:
                output, _ = self._run_call({"id": f"{call.get('id', 'create')}-run", "name": name,
                                            "args": args["args"] if isinstance(args.get("args"), dict) else {}})
            finally:
                if args.get("keep") is False:
                    self._create({"name": name, "code": ""})
            if args.get("keep") is not False:
                return f"{made}\n{output}"
            return f"Ran a one-off tool (not kept).\n{output}{self._note_one_off(name, kept, output)}"
        if call["name"] == "mcp_resource":
            return self._mcp_resource(args)
        if call["name"] == "plugin_write":
            return self._plugin_write(args)
        if call["name"] == "skill":
            found = self.skills.get(str(args.get("name") or ""))
            if found is None:
                raise ToolError(f"No skill named {args.get('name')!r}; known: {', '.join(self.skills) or 'none'}")
            if found.name not in self._skills_loaded:
                if len(self._skills_loaded) >= MAX_SKILLS_PER_TURN:
                    raise ToolError(f"Load at most {MAX_SKILLS_PER_TURN} skills per task; you already loaded {', '.join(self._skills_loaded)}. Use those.")
                self._skills_loaded.append(found.name)
            return skills.load(found)
        if call["name"] == "todo_write":
            items = [{"content": str(i.get("content", ""))[:300], "status": i.get("status") if i.get("status") in
                      ("pending", "in_progress", "completed") else "pending"} for i in args.get("items") or [] if isinstance(i, dict)]
            self.todos = items[:50]
            self.emit("todos", items=self.todos)
            done = sum(i["status"] == "completed" for i in self.todos)
            return f"Plan saved: {done}/{len(self.todos)} done."
        if call["name"] == "ask_user":
            decision = self._wait_for_decision(call)
            answer = decision.get("note", "").strip()
            if decision["decision"] == "deny":
                return "The user did not answer." + (f" Note: {answer}" if answer else "")
            if answer:
                root = self._root()
                root._answers.append(f"Asked: {str(args.get('question') or '')[:1000]}\nAnswered: {answer[:2000]}")
                root._intent_request = 0
            return f"The user answered: {answer}" if answer else "The user answered with no text."
        if call["name"] == "exit_plan_mode":
            decision = self._wait_for_decision(call)
            if decision["decision"] == "deny":
                note = decision.get("note", "").strip()
                return "The user did not approve the plan; revise it." + (f" Their note: {note}" if note else "")
            self.plan_mode = False
            if decision["decision"] == "allow_all":
                self.mode = "edits"
            self.emit("notice", text="Đã duyệt kế hoạch.")
            return "The user approved the plan. Implement it now."
        if call["name"] == "goal_done":
            self.goal = None
            self.emit("notice", text="Đã đạt mục tiêu.")
            return "Goal marked done."
        if call["name"] == "memory":
            return self._memory_tool(args)
        if call["name"] == "context_notes":
            if "text" not in args:
                return self.notes or "(the notebook is empty)"
            text = str(args.get("text") or "")
            if len(text) > MAX_NOTES:
                raise ToolError(f"the notebook is limited to {MAX_NOTES} characters; keep only what the next steps need")
            self.notes = text
            return f"Notebook saved ({len(text)} characters)."
        if call["name"] == "new_context":
            self._rollover = True
            return "A new context window starts after this step." + ("" if self.notes else " Your notebook is empty: save it with context_notes now.")
        if call["name"] == "checkpoint":
            return self._checkpoint(args)
        if call["name"] == "job_output":
            return self._job_output(str(args.get("id") or ""), args.get("wait_s"))
        if call["name"] in ("schedule_create", "schedule_list", "schedule_delete"):
            return self._schedule(call["name"], args)
        if call["name"] == "job_input":
            job = self._job(str(args.get("id") or ""))
            try:
                job.write(str(args.get("chars") or ""))
            except OSError as exc:
                raise ToolError(str(exc)) from exc
            return self._job_output(str(args.get("id") or ""), 2 if args.get("wait_s") is None else args.get("wait_s"), settle=True)
        if call["name"] == "job_stop":
            job = self._job(str(args.get("id") or ""))
            job.stop()
            with self._usage_lock:
                for name, other in self.jobs.items():
                    if other is job:
                        del self.jobs[name]
                        break
            return f"Stopped {args.get('id')}. Last output:\n{clip(job.read(), 4000)}"
        if call["name"] == "spawn_agent":
            nick = self._spawn(str(args.get("message") or ""), str(args.get("agent") or "explore"))
            queued = self.children[nick].held is not None
            lead = f"Queued {nick}: helpers that edit files run one at a time, so it starts when the one before it is done." if queued else f"Started {nick}."
            return f"{lead} Do other work, then call wait_agent with ids [\"{nick}\"]; it also reports to you by itself when done."
        if call["name"] == "fan_out":
            return self._fan_out(args.get("jobs"), str(args.get("agent") or "explore"), args.get("schema") if isinstance(args.get("schema"), dict) else None, bool(args.get("as_data")))
        if call["name"] == "wait_agent":
            return self._wait_agents(args.get("ids"), args.get("timeout_s"))
        if call["name"] == "send_input":
            child = self._child(str(args.get("id") or ""))
            child.reported = False
            child.send(str(args.get("message") or ""))
            return f"Sent to {child.nick}; call wait_agent to get its answer."
        if call["name"] == "close_agent":
            child = self._child(str(args.get("id") or ""))
            state = self._state(child)
            child.held = None
            self._collect(child)
            child.close()
            child.closed = True
            return f"Closed {child.nick} (it was {state})."
        return self._subagent(str(args.get("description") or "task"), str(args.get("prompt") or ""), str(args.get("agent") or "explore"))

    def _run_script(self, args: dict, policy: "sandbox.Policy | None" = None) -> str:
        code = str(args.get("code") or "")
        if not code.strip():
            raise ToolError("code is empty")
        limit = max(1, min(3600, int(args.get("timeout") or 120)))
        started_agents = [0]

        def call_tool(name: str, tool_args: dict) -> tuple[str, bool]:
            if name not in SCRIPT_TOOLS:
                return f"{name} cannot be called from a script; allowed: {', '.join(sorted(SCRIPT_TOOLS))}", False
            if name == "fan_out":
                jobs = tool_args.get("jobs") if isinstance(tool_args.get("jobs"), list) else []
                if started_agents[0] + min(len(jobs), MAX_FAN_OUT) > SCRIPT_AGENTS:
                    return f"a script may start at most {SCRIPT_AGENTS} helper agents in all", False
                started_agents[0] += min(len(jobs), MAX_FAN_OUT)
                tool_args = {**tool_args, "as_data": True}
            output, ok = self._run_call({"id": f"script-{uuid.uuid4().hex[:8]}", "name": name, "args": tool_args})
            return output, ok

        code_out, output, calls = codemode.run(code, policy or self.workspace.policy, self.workspace.root, limit, call_tool)
        status = f"[stopped after {limit} s]" if code_out is None else f"[exit code {code_out}]"
        return clip(f"{output.strip()}\n{status} [{calls} tool call{'s' if calls != 1 else ''}]", 400_000)

    def _run_foreground(self, args: dict) -> str:
        command = str(args.get("command") or "").strip()
        if not command:
            raise ToolError("command is empty")
        try:
            limit = max(1, min(MAX_COMMAND_TIMEOUT, int(args.get("timeout") or COMMAND_TIMEOUT)))
        except (TypeError, ValueError) as exc:
            raise ToolError("timeout must be a whole number of seconds") from exc
        policy = sandbox.Policy("full-access", True) if args.get("outside_sandbox") else self.workspace.policy
        root = self.workspace.root
        guard = gitguard.snapshot(root)
        job = self._command_job(command, policy)
        if hasattr(job, "close_input"):
            job.close_input()
        return self._await_job(job, limit, guard)

    def _await_job(self, job: "sandbox.Job", limit: int, guard) -> str:
        root = self.workspace.root
        end, paused = time.time() + limit, self._root().asked_s
        while job.code is None and (time.time() < end + self._root().asked_s - paused or self._root().status == "waiting") and not self._stop:
            time.sleep(0.05)
        undone = gitguard.restore(root, guard)
        warning = f"\n[blocked: the command changed {', '.join(undone)}; git hooks and config run outside the sandbox, so they were put back]" if undone else ""
        with self._usage_lock:
            name = f"job{next(i for i in range(1, 100) if f'job{i}' not in self.jobs)}" if job.code is None and not self._stop and len(self.jobs) < MAX_JOBS else ""
            if name:
                self.jobs[name] = job
        if name:
            return clip(f"{job.read().strip()}\n[still running after {limit} s; it was not stopped but moved to the background as {name}: read it with job_output, "
                        f"stop it with job_stop; you are told when it ends]{warning}", 400_000)
        finished = job.code is not None
        with self._usage_lock:
            kept = f"job{next(i for i in range(1, 100) if f'job{i}' not in self.jobs)}" if (
                finished and not self._stop and _ses("_alive")(job) and len(self.jobs) < MAX_JOBS) else ""
            if kept:
                job.announced = True
                self.jobs[kept] = job
        if kept:
            time.sleep(0.2)
            return clip(f"{job.read().strip()}\n[exit code {job.code}; it left processes running, which keep running as {kept}: "
                        f"read their output with job_output, stop them with job_stop]{warning}", 400_000)
        if finished:
            job.out.join(5)
        output = job.out.text().strip()
        code = job.code
        job.stop()
        status = f"[exit code {code}]" if finished else f"[stopped after {limit} s]"
        return clip(f"{output}\n{status}{warning}", 400_000)

    def _job(self, name: str) -> "sandbox.Job":
        key = name.strip().lower()
        job = self.jobs.get(key) or self.jobs.get(f"job{key}")
        if job is None:
            raise ToolError(f"No job named {name!r}; running: {', '.join(self.jobs) or 'none'}")
        return job

    def _start_job(self, args: dict) -> str:
        with self._usage_lock:
            for name in [n for n, j in self.jobs.items() if not _ses("_alive")(j) and not j.read() and time.time() - j.started > 600]:
                self.jobs.pop(name).stop()
            if len(self.jobs) >= MAX_JOBS:
                raise ToolError(f"{MAX_JOBS} background jobs already; stop some with job_stop")
            command = str(args.get("command") or "").strip()
            if not command:
                raise ToolError("command is empty")
            policy = sandbox.Policy("full-access", True) if args.get("outside_sandbox") else self.workspace.policy
            name = f"job{next(i for i in range(1, 100) if f'job{i}' not in self.jobs)}"
            self.jobs[name] = self._command_job(command, policy, tty=bool(args.get("tty")))
        time.sleep(1.0)
        with self._usage_lock:
            job = self.jobs.get(name)
        if job is None:
            return f"Started {name}, which was already stopped."
        state = "running" if job.code is None else f"exited with code {job.code}" + (", and what it started still runs" if _ses("_alive")(job) else "")
        return f"Started {name} ({state}). Read it with job_output.\n{clip(job.read(), 4000)}".rstrip()

    def _job_output(self, name: str, wait, settle: bool = False) -> str:
        job = self._job(name)
        name = next((n for n, j in self.jobs.items() if j is job), name)
        try:
            end = time.time() + max(0, min(120, int(wait or 0)))
        except (TypeError, ValueError) as exc:
            raise ToolError("wait_s must be a whole number of seconds") from exc
        while job.code is None and time.time() < end and len(job.out.data) == job.cursor and not self._stop:
            time.sleep(0.25)
        while settle and job.code is None and time.time() < end and not self._stop:
            seen = len(job.out.data)
            time.sleep(0.25)
            if len(job.out.data) == seen:
                break
        text = job.read()
        state = "running" if job.code is None else f"exited with code {job.code}" + (", and what it started still runs" if _ses("_alive")(job) else "")
        return f"[{name} {state}]\n{clip(text, 20000) if text else '(no new output)'}"

    def _diff(self, before: dict[str, bytes | None], only: str = "") -> str:
        if only:
            target = str(self.workspace.resolve(only))
            before = {k: v for k, v in before.items() if k == target}
        blocks = []
        for name, old in sorted(before.items()):
            path, rel = Path(name), self.workspace.rel(Path(name))
            try:
                new = path.read_bytes() if path.is_file() else None
            except OSError:
                new = None
            if new == old:
                continue
            if b"\0" in (old or b"")[:8192] or b"\0" in (new or b"")[:8192]:
                blocks.append(f"Binary file {rel} changed")
                continue
            lines = lambda data: (data or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            diff = difflib.unified_diff(lines(old), lines(new), "/dev/null" if old is None else f"a/{rel}", "/dev/null" if new is None else f"b/{rel}")
            blocks.append("".join(diff).rstrip("\n"))
        return clip("\n".join(blocks), 60_000)

    def _checkpoint(self, args: dict) -> str:
        cp, action = self.checkpoints, str(args.get("action") or "show")
        name, path = str(args.get("name") or "").strip(), str(args.get("path") or "")
        if name and action in ("show", "restore") and name not in cp.marks:
            raise ToolError(f"no mark {name!r}; marks: {', '.join(cp.marks) or 'none'}")
        if action == "save":
            if not name:
                raise ToolError("save needs a name")
            held = cp.mark(name, str(args.get("note") or ""))
            return f"Saved mark {name} ({held} changed files so far)."
        if action == "pin":
            paths = args.get("paths") or ([path] if path else [])
            if not paths:
                raise ToolError("pin needs paths")
            for item in map(str, paths):
                rel = self.workspace.rel(self.workspace.resolve(item))
                if rules.check(self.rules, "read_file", {"path": rel}, self._rel) == "deny":
                    raise ToolError(f"{rel} may not be read")
                cp.pins[rel] = str(args.get("note") or "")
            return "Pinned: " + ", ".join(cp.pins)
        if action == "restore":
            if self.workspace.policy.mode == "read-only" or self.plan_mode:
                raise ToolError("Nothing can be restored while the session only reads")
            target = self.workspace.resolve(path, write=True) if path else None
            if target is not None and rules.check(self.rules, "edit_file", {"path": str(target)}, self._rel) == "deny":
                raise ToolError(f"{self.workspace.rel(target)} may not be edited")
            if not name and target is None:
                raise ToolError("restore needs the name of a mark, or a path to put back as it was before this request")
            why = str(args.get("why") or "").strip()
            if name and not why:
                raise ToolError("say why you go back to this mark (why): what failed or what you learned, so the next try does not repeat it")
            state = cp.at(name) if name else cp.current()
            if target is not None and str(target) not in state:
                changed = ", ".join(self.workspace.rel(Path(n)) for n in state) or "none"
                raise ToolError(f"{path!r} was not changed {'since mark ' + name if name else 'by the edit tools in this request'}; changed: {changed}")
            done = cp.restore(state, str(target) if target is not None else "")
            for item in map(Path, done):
                if item.exists():
                    self.workspace.seen[item] = item.stat().st_mtime_ns
                else:
                    self.workspace.seen.pop(item, None)
            ran = cp.ran_since(name) if name else []
            if name:
                cp.drop(name, why)
            where = f"as they were at mark {name}" if name else "as it was before this request"
            text = f"Put back {where}: {', '.join(self.workspace.rel(Path(n)) for n in done)}." if done else f"Nothing to put back; the files are already {where}."
            if ran:
                text += ("\nNot undone, because only files go back: these ran after the mark and their effects outside the files "
                         "(installed packages, started processes, network calls) are still there:\n" + "\n".join(f"- {r}" for r in ran[-15:]))
            return text
        if action != "show":
            raise ToolError("action is show, save, pin or restore")
        before = cp.at(name) if name else cp.current()
        diff = self._diff(before, path) if path or args.get("diff") else self._diff_stat(before)
        parts = [diff or (f"No file changed since mark {name}" if name else "No file changed by the edit tools in this request") + (f" matches {path}" if path else "") + "."]
        if cp.marks:
            parts.append("Marks: " + "; ".join(f"{n}{' (auto)' if m['auto'] else ''}{': ' + m['note'] if m['note'] else ''}" for n, m in cp.marks.items()))
        if cp.pins:
            parts.append("Pinned: " + "; ".join(f"{p}{': ' + n if n else ''}" for p, n in cp.pins.items()))
        if cp.dropped:
            parts.append("Dropped attempts:\n" + "\n".join(f"- back to {d['mark']}: {d['why']}" for d in cp.dropped))
        rel = self._rel(path) if path else ""
        if rel in cp.pins and rules.check(self.rules, "read_file", {"path": rel}, self._rel) != "deny":
            try:
                parts.append(f"== {rel} ==\n" + clip(self.workspace.resolve(rel).read_text(encoding="utf-8", errors="replace"), 20_000))
            except OSError as exc:
                parts.append(f"== {rel} ==\n(cannot read: {exc})")
        return clip("\n\n".join(parts), 60_000)

    def _diff_stat(self, before: dict[str, bytes | None]) -> str:
        rows = []
        for name, old in sorted(before.items()):
            path = Path(name)
            try:
                new = path.read_bytes() if path.is_file() else None
            except OSError:
                new = None
            if new == old:
                continue
            lines = lambda data: (data or b"").decode("utf-8", errors="replace").splitlines()
            added = removed = 0
            for row in list(difflib.unified_diff(lines(old), lines(new), n=0, lineterm=""))[2:]:
                added += row.startswith("+")
                removed += row.startswith("-")
            state = "new" if old is None else "deleted" if new is None else f"+{added} -{removed}"
            rows.append(f"{self.workspace.rel(path)}: {state}")
        return ("Changed files (diff=true or path shows the lines):\n" + "\n".join(rows)) if rows else ""

    def _memory_tool(self, args: dict) -> str:
        home, root, scope = self.home if self.home is not None else Path.home(), self.workspace.root, str(args.get("scope") or "project")
        try:
            if args.get("action") == "add":
                if self.tainted:
                    return ("Refused: this session has read untrusted content (web/MCP), so memory notes are "
                            "disabled to keep attacker instructions out of later sessions.")
                memory.add(home, root, scope, args.get("text") or "")
            elif args.get("action") == "remove":
                memory.remove(home, root, scope, int(args.get("index") or 0))
            elif args.get("action") != "list":
                raise ToolError("action must be add, remove or list")
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return memory.prompt(home, root) or "No notes."

    def _child(self, nick: str) -> "AgentSession":
        child = self.children.get(nick.strip().lower())
        if child is None or child.closed:
            raise ToolError(f"No open agent named {nick!r}; open: {', '.join(n for n, c in self.children.items() if not c.closed) or 'none'}")
        return child

    @staticmethod
    def _state(child: "AgentSession") -> str:
        if child.closed:
            return "closed"
        if child.held is not None:
            return "queued"
        if child.status != "idle":
            return "running"
        return "errored" if any(e["type"] == "error" for e in child.events[-3:]) else "completed"

    @staticmethod
    def _result(child: "AgentSession") -> str:
        errors = [e["text"] for e in child.events if e["type"] == "error"]
        text = next((e["text"] for e in reversed(child.events) if e["type"] == "assistant" and e["text"]), "")
        return (text or (errors[-1] if errors else "The agent finished without a report.")) + child.merge_note

    def _collect(self, child: "AgentSession") -> None:
        if child.reported or child.status != "idle" or child.held is not None:
            return
        child.reported = True
        self._merge_copy(child)
        self.tainted = self.tainted or child.tainted
        with self._usage_lock:
            for key in self.usage:
                self.usage[key] += child.usage[key] - child.counted[key]
                child.counted[key] = child.usage[key]
        self.emit("subagent", description=child.job[1][:80] if child.job else "", agent=child.agent.name, id=child.nick,
                  state="done", tools=sum(1 for e in child.events if e["type"] == "tool"))

    def _merge_copy(self, child: "AgentSession") -> None:
        if child.copy is None:
            return
        applied, conflicts = isolate.merge(self.workspace.root, child.copy, child.base or {}, child.nick, before=self.checkpoints.save)
        shutil.rmtree(child.copy, ignore_errors=True)
        child.copy = None
        note = f"\n\n[Its work ran in a private copy and was merged: {len(applied)} files brought in" + (f" ({', '.join(applied[:12])})" if applied else "") + "."
        if conflicts:
            note += (f" {len(conflicts)} changed on both sides and were NOT merged: {', '.join(conflicts[:12])}. Its versions are in "
                     f"{isolate.CONFLICTS}/{child.nick}/; merge them by hand, then delete that folder.")
        child.merge_note = note + "]"
        self.emit("notice", text=f"Đã gộp từ {child.nick}: {len(applied)} file" + (f", {len(conflicts)} xung đột" if conflicts else ""))

    def _spawn(self, message: str, agent_name: str, dedupe: bool = True, role: str = "") -> str:
        with self._spawn_lock:
            return self._spawn_locked(message, agent_name, dedupe, role)

    def _spawn_locked(self, message: str, agent_name: str, dedupe: bool, role: str = "") -> str:
        if not message.strip():
            raise ToolError("The agent needs a message")
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ToolError(f"No agent named {agent_name!r}; known: {', '.join(self.agents)}")
        job = (agent.name, " ".join(message.split()))
        for nick, other in self.children.items():
            if dedupe and not other.closed and other.job == job:
                raise ToolError(f"{nick} already has this exact job ({self._state(other)}); call wait_agent instead of starting it again")
        running = sum(1 for c in self.children.values() if self._state(c) in ("running", "queued"))
        if running >= MAX_AGENT_THREADS:
            raise ToolError(f"{running} agents are already running (limit {MAX_AGENT_THREADS}); call wait_agent for some of them first")
        if len(self.children) >= MAX_CHILDREN:
            raise ToolError(f"This session already started {MAX_CHILDREN} agents; close finished ones and continue yourself")
        allowed = agent.allowed({s["name"] for s in SPECS} | {"skill"})
        mutating = bool(allowed & (agents.EDIT_TOOLS | {"run_command"}))
        if self.plan_mode and mutating:
            raise ToolError("In plan mode only read-only agents may run")
        busy = next((n for n, c in self.children.items() if c.mutating and self._state(c) in ("running", "queued")), "") if mutating else ""
        nick = next((n for n in NICKNAMES if n not in self.children), f"agent{len(self.children) + 1}")
        copy, base = None, None
        if busy and self.profile["isolate_writers"] and not self.depth:
            copy = Path(tempfile.mkdtemp(prefix="agent-copy-"))
            base = isolate.snapshot(self.workspace.root, copy)
            if base is None:
                shutil.rmtree(copy, ignore_errors=True)
                copy = None
            else:
                busy = ""
        helper = Workspace(copy or self.workspace.root, self.workspace.policy if mutating else sandbox.Policy("read-only", False),
                           read_roots=self.workspace.read_roots)
        used, provider, keys, model = "", self.provider, self.api_keys, agent.model or self.profile.get("subagent_model") or self.model
        if role:
            used, provider, key, model = self._role_target(role)
            keys = [key]
        child = _ses("AgentSession")(f"{self.id}-{nick}", provider, keys, model, helper,
                             self.mode if mutating else "bypass", complete=self._raw_complete, depth=self.depth + 1, home=self.home,
                             trust=self.trust, parent=self, agent=agent)
        child.role = used
        child.text_tools, child.nick, child.job, child.mutating = self.text_tools, nick, job, mutating and copy is None
        child.copy, child.base = copy, base
        child._inherit_taint = self.tainted
        child._tainted_files = self._tainted_files
        self.children[nick] = child
        self.emit("subagent", description=message[:80], agent=agent.name, id=nick, state="started")
        if busy:
            child.held = message
        else:
            child.send(message)
        return nick

    def _start_queued(self) -> None:
        if any(c.mutating and c.held is None and c.status != "idle" and not c.closed for c in self.children.values()):
            return
        for child in self.children.values():
            if child.held is not None and not child.closed and not self._stop:
                message, child.held = child.held, None
                child.send(message)
                return

    def _fan_out(self, jobs, agent_name: str, schema: dict | None = None, as_data: bool = False) -> str:
        if not isinstance(jobs, list) or not jobs:
            raise ToolError("jobs must be a non-empty list of {description, prompt}")
        todo = [(i, str(j.get("description") or f"job {i + 1}")[:60], str(j.get("prompt") or j.get("message") or ""))
                for i, j in enumerate(jobs[:MAX_FAN_OUT]) if isinstance(j, dict)]
        running: dict[str, tuple[int, str]] = {}
        done: dict[int, str] = {}
        repaired: set[str] = set()
        while (todo or running) and not self._stop:
            while todo:
                index, title, prompt = todo[0]
                if schema:
                    prompt += f"\n\nEnd your final reply with one JSON object of this shape and nothing after it:\n{json.dumps(schema)}"
                try:
                    nick = self._spawn(prompt, agent_name)
                except ToolError as exc:
                    if "already running" in str(exc):
                        break
                    todo.pop(0)
                    done[index] = f"### {title}\nNot started: {exc}"
                    continue
                todo.pop(0)
                running[nick] = (index, title)
            self._start_queued()
            for nick in list(running):
                child = self.children[nick]
                if self._state(child) in ("running", "queued"):
                    continue
                index, title = running[nick]
                self._collect(child)
                result = self._result(child)
                if schema:
                    value, why = schemas.report(result, schema)
                    if why and nick not in repaired:
                        repaired.add(nick)
                        child.reported = False
                        child.send(f"Your report does not fit the required shape: {why}. Reply again ending with one JSON object of this shape and nothing after it:\n{json.dumps(schema)}")
                        continue
                    result = json.dumps(value, ensure_ascii=False) if value is not None else f"null (invalid report: {why})"
                    data = value
                else:
                    data = result
                running.pop(nick)
                done[index] = {"job": title, "report": data} if as_data else f"### {title} ({nick})\n{result}"
                child.close()
                child.closed = True
            time.sleep(0.25)
        for nick in running:
            self.children[nick].close()
        if as_data:
            return json.dumps([done.get(i) or {"job": f"job {i + 1}", "report": None} for i in range(len(jobs[:MAX_FAN_OUT]))], ensure_ascii=False)
        return "\n\n".join(done[i] for i in sorted(done)) or "Stopped before any job finished."

    def _wait_agents(self, ids, timeout) -> str:
        names = [str(i).strip().lower() for i in ids] if isinstance(ids, list) and ids else []
        targets = [self._child(n) for n in names] or [c for c in self.children.values() if not c.closed]
        if not targets:
            return "No agents to wait for."
        limit = max(1, min(WAIT_MAX, int(timeout or WAIT_DEFAULT)))
        end = time.time() + limit
        while time.time() < end and not self._stop and any(self._state(c) in ("running", "queued") for c in targets):
            self._start_queued()
            time.sleep(0.25)
        rows, waiting = [], False
        for child in targets:
            state = self._state(child)
            if state in ("running", "queued"):
                waiting = True
                rows.append(f"{child.nick} ({child.agent.name}): still running")
                continue
            self._collect(child)
            rows.append(f"{child.nick} ({child.agent.name}): {state}\n{self._result(child)}")
        return "\n\n".join(rows) + ("\n\n[timed out; call wait_agent again to keep waiting]" if waiting else "")

    def _notifications(self) -> None:
        self._start_queued()
        for name, job in list(self.jobs.items()):
            if not _ses("_alive")(job) and not getattr(job, "announced", False):
                job.announced = True
                changed = self._record_changes()
                note = f"[{name} finished with exit code {job.code}; read the rest of its output with job_output]"
                self.history.append({"role": "user", "content": note + (f"\n[changed: {', '.join(changed[:30])}]" if changed else "")})
        for child in list(self.children.values()):
            if not child.closed and not child.reported and child.status == "idle" and child.held is None:
                self._collect(child)
                note = {"agent_id": child.nick, "agent": child.agent.name, "status": self._state(child), "result": self._result(child)}
                self.history.append({"role": "user", "content": f"<subagent_notification>{json.dumps(note, ensure_ascii=False)}</subagent_notification>"})

    def _subagent(self, description: str, prompt: str, agent_name: str = "explore", role: str = "") -> str:
        nick = self._spawn(prompt, agent_name, dedupe=False, role=role)
        child = self.children[nick]
        while self._state(child) in ("running", "queued") and not self._stop:
            self._start_queued()
            time.sleep(0.2)
        self._collect(child)
        answer = self._result(child)
        if child.role:
            self._count_role(child.role, child.cache_log["requests"], child.usage["prompt_tokens"] + child.usage["completion_tokens"])
        child.close()
        child.closed = True
        return answer

    def _run_call(self, call: dict) -> tuple[str, bool]:
        if call.get("error"):
            return call["error"], False
        offered = {s["name"]: s for s in self.specs()}
        if str(call.get("id", "")).startswith(("chain-", "script-", "mcp-", "plugin-")) and call["name"] not in offered:
            offered |= {s["name"]: s for s in self.specs(self.library_specs()) if s["name"] not in offered}
            if call["name"] not in offered and (spec := self._target_spec(call["name"])):
                offered[call["name"]] = spec
        known = set(offered)
        self.workspace.offered = known
        if call["name"] not in known and TOOL_ALIASES.get(call["name"]) in known:
            call["name"] = TOOL_ALIASES[call["name"]]
        if call["name"] not in known:
            return f"Unknown tool {call['name']!r}; available: {', '.join(sorted(known))}", False
        required = offered[call["name"]].get("parameters", {}).get("required") or []
        missing = [k for k in required if isinstance(call["args"], dict) and k not in call["args"]]
        if missing and not str(call.get("id", "")).startswith("script-"):
            got = ", ".join(call["args"]) if call["args"] else "no arguments at all"
            return (f"{call['name']} needs {', '.join(missing)}; this call arrived with {got}. A very long argument can be cut or garbled "
                    "on the way: write long content to a file first and pass its path where the tool takes one."), False
        signature = self._signature(call)
        coded = str(call.get("id", "")).startswith("chain-shell-")
        if not coded and signature in self._before_summary and self._repeats >= DOOM_LOOP:
            return (f"Blocked: after the summary you have redone {self._repeats} calls exactly as before and got the same results. "
                    "Read the summary's Progress and continue from there, or ask the user."), False
        self._recent = self._recent if coded else (self._recent + [signature])[-DOOM_LOOP:]
        polling = call["name"] == "job_output" and isinstance(call["args"], dict) and getattr(self.jobs.get(str(call["args"].get("id"))), "alive", False)
        if not coded and len(self._recent) == DOOM_LOOP and len(set(self._recent)) == 1 and call["name"] != "wait_agent" and not polling:
            self._recent = []
            self.emit("notice", text=f"Đã chặn {call['name']}: lặp {DOOM_LOOP} lần cùng tham số.")
            return (f"Blocked: you made this exact {call['name']} call {DOOM_LOOP} times in a row. "
                    "Change your approach or ask the user."), False
        verdict = rules.check(self.rules, call["name"], call["args"], self._rel) if isinstance(call["args"], dict) else None
        if verdict == "deny":
            return f"Denied by a permission rule for {call['name']}. Do not retry it; use another way or ask the user.", False
        call = copy.deepcopy(call)
        with self._approval_lock:
            ask = self._needs_approval(call, verdict)
            ask = ask or self._always_ask(call)
            if ask:
                why = "untrusted" if self.tainted and self.profile["untrusted_guard"] else ""
                saved = None if why else rules.remembered(call["name"], call["args"], verdict)
                extra = {**({"why": why} if why else {}), **({"remember": saved[1]} if saved else {})}
                decision = self._wait_for_decision(copy.deepcopy({**call, **extra} if extra else call))
                if decision["decision"] == "allow_always" and saved:
                    self._remember(*saved)
                if decision["decision"] == "allow_all":
                    self.mode = "edits"
                    above = self.parent
                    while above is not None:
                        above.mode = "edits"
                        above = above.parent
                elif decision["decision"] not in ("allow", "allow_always"):
                    note = f" Note from the user: {decision['note']}" if decision.get("note") else ""
                    return f"The user refused this {call['name']} call.{note}", False
                if call["name"] in URL_TOOLS:
                    self.web_ok.add((urlparse(str(call["args"].get("url") or "")).hostname or "").lower())
        if call["name"] not in SESSION_SPECS:
            allowed, message = self._run_hooks("PreToolUse", call)
            if not allowed:
                return message, False
            for fn in self.registry.hooks["pre_tool"]:
                blocked = self._hook_call(fn, copy.deepcopy(call))
                if blocked:
                    return blocked, False
        def execute(_call: dict) -> tuple[str, bool]:
            try:
                if call["name"] in SESSION_SPECS:
                    output, ok = self._session_tool(call), True
                elif call["name"] in self.replaced:
                    root = self._root()
                    server, tool = root.mcp_tools[self.replaced[call["name"]]]
                    if KIND.get(call["name"]) == "edit" and isinstance(call["args"], dict) and call["args"].get("path"):
                        self.checkpoints.save(self.workspace.resolve(str(call["args"]["path"]), write=True))
                    output, ok = root.mcp_servers[server].call_tool(tool["name"], call["args"])
                elif call["name"] in self.mcp_tools:
                    if self._role(call["name"]) == "edit" and isinstance(call["args"], dict) and call["args"].get("path"):
                        self.checkpoints.save(self.workspace.resolve(str(call["args"]["path"]), write=True))
                    server, tool = self.mcp_tools[call["name"]]
                    output, ok = self.mcp_servers[server].call_tool(tool["name"], call["args"])
                elif call["name"] == "run_command" and call["args"].get("background"):
                    output, ok = self._start_job(call["args"]), True
                elif call["name"] == "run_command" and self.profile["timeout_to_background"]:
                    output, ok = self._run_foreground(call["args"]), True
                elif call["name"] == "run_script":
                    output, ok = self._run_script(call["args"]), True
                elif call["name"] in self._root().created:
                    output, ok = self._run_created(call), True
                elif call["name"] in self.registry.tools:
                    try:
                        output, ok = str(self.registry.tools[call["name"]].handler(self, call["args"])), True
                    except ToolError:
                        raise
                    except Exception as exc:
                        output, ok = f"Error: {type(exc).__name__}: {exc}", False
                else:
                    if KIND.get(call["name"]) == "edit" and isinstance(call["args"], dict):
                        for path in self.workspace.targets(call["name"], call["args"]):
                            self.checkpoints.save(path)
                    output, ok = self.workspace.run(call["name"], call["args"]), True
            except ToolError as exc:
                output, ok = f"Error: {exc}", False
            except (OSError, mcp.MCPError) as exc:
                output, ok = f"Error: {type(exc).__name__}: {exc}", False
            if ok:
                self._log_outside(call)
            return output, ok

        output, ok = execute(call)
        name = call["name"]
        role = self._role(name)
        if name in ("web_fetch", "web_search", "web_download") or (name in self.mcp_tools and role in (None, "net")):
            self.tainted = True
        if ok and (KIND.get(name) == "edit" or role == "edit") and name != "web_download":
            self._edited = True
        if name == "run_command" or name in self._root().created:
            self._last_command = output
            self._commands += 1
            args = call["args"] if isinstance(call["args"], dict) else {}
            command = str(args.get("command") or "") if name == "run_command" else f"{name} {json.dumps(args, ensure_ascii=False, default=str)}"
            if _ses("_failed")(output):
                self._failing[command] = self._commands
            else:
                self._failing.pop(command, None)
        if name == "run_command" or name in self.mcp_tools or name in self.replaced or name in self.registry.tools or name in self._root().created:
            output = self._offload(call, output)
        if call["name"] not in SESSION_SPECS:
            _, notes = self._run_hooks("PostToolUse", call, output)
            for fn in self.registry.hooks["post_tool"]:
                notes = "\n".join(n for n in (notes, self._hook_call(fn, call, output)) if n)
            if notes:
                output = f"{output}\n{notes}"
        return output, ok

    def _offload(self, call: dict, output: str) -> str:
        if len(output) <= OFFLOAD_AT:
            return output
        folder = self._outputs_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{re.sub(r'[^A-Za-z0-9_-]', '_', str(call['id']))[:40]}-{call['name'][:20]}.txt"
            self._write_private(path, output)
        except OSError:
            return clip(output)
        return (f"{output[:6000]}\n… [full output saved to {path}: {len(output) - 10000} more characters; "
                f"read it with read_file offset and limit, or search it] …\n{output[-4000:]}")
