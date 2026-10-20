# Ranked changes for Terminal-Bench 2.1

The highest-leverage fixes are the ones that remove friction from the path the models already take: they run `run_command` 90% of the time, chain in the shell, and rarely use `create` for multi-tool steps. The proposals below are ordered by expected task gain per line of code.

## 1. Make the benchmark runner actually use the two fixes already added for reasoning models and task servers

**Problem.** `.github/terminal-bench/run_agent.py` creates the session with the default profile, so `keep_jobs` stays false and `echo_reasoning` stays âDeepSeek onlyâ:

```python
session = manager.create(provider, "relay", args.model, Workspace(args.workspace, sandbox.Policy("full-access", True)), "bypass")
session.set_deadline(args.minutes * 60)
```

That contradicts the measured fixes in the prompt: Beam:free failed until reasoning was echoed, and jobs were stopped on close. In the runner, task-started servers are still killed and non-DeepSeek reasoning models still lose `reasoning_content`.

**Change.**

```python
session = manager.create(provider, "relay", args.model, Workspace(args.workspace, sandbox.Policy("full-access", True)), "bypass")
session.profile["keep_jobs"] = True
session.profile["echo_reasoning"] = "all"
session.echo_reasoning = "all"
session.set_deadline(args.minutes * 60)
```

**Risk:** none for task success; detached jobs remain after timeout, which is intended for grader checks.

**Measure:** rerun Beam:free and PolarGrid; compare âall tasks failed until reasoning fixâ class of failures and tasks needing a started server.

---

## 2. Expose `skill` as a direct lean tool

**Problem.** Built-in skills were read only 5 times in 89 tasks. The prompt advertises them and says âload with the skill tool,â but the lean tool list never offers `skill`:

```python
LEAN_TOOLS = frozenset({"read_file", "edit_file", "run_command"})
...
rows = [s for s in SPECS if s["name"] in (("read_file",) if self.plan_mode else LEAN_TOOLS)]
...
if self.workspace.policy.mode != "read-only":
    rows.append(SESSION_SPECS["create"])
    rows += [{k: tool[k] for k in ("name", "description", "parameters")} for tool in self.created.values()]
```

`_chain_targets` includes `skill`, and `_target_spec` has a skill entry, so it is reachable only through `agent-tool skill` or `create`. That indirection is why models do not use it.

**Change** in `app/agent/session.py`, `_lean_specs`:

```python
if self.skills:
    rows.append(SESSION_SPECS["skill"])
```

Add it next to `create`, not gated by `direct_tools`.

**Risk:** one more tool spec in every prompt (~1 KB). Skills are read-only instructions, so no approval change is needed.

**Measure:** skill loads per task; Terminal-Bench tasks that match a built-in skill (webapp testing, MCP builder, document handling).

---

## 3. Stop the doom loop from blocking legitimate repeated checks

**Problem.** `_run_calls` blocks an identical call after three repetitions:

```python
self._recent = (self._recent + [signature])[-DOOM_LOOP:]
...
if len(self._recent) == DOOM_LOOP and len(set(self._recent)) == 1 and call["name"] != "wait_agent" and not polling:
    ...
    return (f"Blocked: you made this exact {call['name']} call {DOOM_LOOP} times in a row. ...")
```

Terminal tasks legitimately rerun the same `run_command` test after edits. The signature contains only arguments, not workspace state, so a repeated `pytest` command can be blocked even though it is progress.

**Change.** Store the result digest with the signature and only block when both are unchanged:

```python
digest = hashlib.sha1(output.encode()).hexdigest()
self._recent = (self._recent + [f"{signature}\x00{digest}"])[-DOOM_LOOP:]
...
if len(self._recent) == DOOM_LOOP and len(set(self._recent)) == 1 and call["name"] != "wait_agent" and not polling:
```

**Risk:** a model can repeat a genuinely useless command more times, but the token budget and time limit still bound it.

**Measure:** count âBlocked: you made this exact â¦ call 3 timesâ events; rerun rate on terminal tasks.

---

## 4. Give background jobs the same tool bridge as foreground commands

**Problem.** `_run_foreground` writes `agent_tools.py`, sets `PYTHONPATH`/`PATH`/`AGENT_RPC`, and starts `_serve_chain`:

```python
job = sandbox.Job(command, policy, root, scratch=str(state),
                  env={"PYTHONPATH": str(state / "lib"), "PATH": str(state / "bin"), "AGENT_RPC": str(rpc), "AGENT_STATE": str(state)})
threading.Thread(target=self._serve_chain, args=(rpc, job, None, self.plan_mode), name=f"agent-bridge-{self.id}", daemon=True).start()
```

But `_start_job` does not:

```python
self.jobs[name] = sandbox.Job(command, policy, self.workspace.root, tty=bool(args.get("tty")))
```

So `run_command(background=true)` cannot use `agent-tool` or `from agent_tools import tools`, even though the prompt says run_command reaches the whole library. Tasks that start a server and then query it through harness tools cannot do so from the background job.

**Change.** Make `_start_job` use the same library setup as `_run_foreground`, with `looks=self.plan_mode`.

**Risk:** background jobs now have a live RPC thread; if the session stops, `_serve_chain` must drain/refuse pending calls. That is already handled by `_refuse_pending`.

**Measure:** background jobs that call `agent-tool`; tasks requiring a started service plus harness tool calls.

---

## 5. Drain pending RPC requests before a finished command is considered done

**Problem.** `_serve_chain` stops as soon as the shell exits:

```python
done = job.code is not None
...
if done or self._stop:
    break
time.sleep(0.02)
```

If the command launched parallel `agent-tool` calls (`agent-tool a & agent-tool b`), their `.res` files may not have been written yet. The thread exits and those calls fail.

**Change.** Track the last time a `.req` was seen; only exit when the job is done and no request has arrived for a short grace period:

```python
done = job.code is not None
...
if done:
    if not list(folder.glob("*.req")):
        if time.time() - last_req >= 0.2:
            break
    else:
        last_req = time.time()
    time.sleep(0.02)
    continue
```

**Risk:** adds at most ~0.2 s to short commands.

**Measure:** parallel `agent-tool` calls from one shell command.

---

## 6. Preserve failure output during context masking

**Problem.** Older tool outputs are masked in batches:

```python
big = [i for i, item in enumerate(self.history) if item["role"] == "tool" and not item.get("masked") and len(item["content"]) > 300]
old = big[:-MASK_KEEP] if len(big) > MASK_KEEP else []
```

A failed test output can be masked away, forcing the model to rerun the same command to recover the failure text.

**Change.** Keep twice as many tool outputs and never mask outputs containing failure markers:

```python
MARKERS = ("error", "fail", "traceback", "exit code 1", "exit code 2")
big = [i for i, item in enumerate(self.history)
       if item["role"] == "tool" and not item.get("masked") and len(item["content"]) > 300
       and not any(m in item["content"].lower() for m in MARKERS)]
old = big[:-MASK_KEEP * 2] if len(big) > MASK_KEEP * 2 else []
```

**Risk:** slightly larger context; compaction still handles overflow.

**Measure:** repeated identical failing test commands after masking.

---

## 7. Make the shell bridge path explicit when PATH is reset

**Problem.** The bridge depends on `PATH` and `PYTHONPATH`, but a command such as `env -i ...` or a tool that resets `PATH` can make `agent-tool` disappear. The foreground path is:

```python
env={"PYTHONPATH": str(state / "lib"), "PATH": str(state / "bin"), "AGENT_RPC": str(rpc), "AGENT_STATE": str(state)}
```

**Change.** Add `AGENT_TOOL` to that env and to `_start_job`:

```python
"AGENT_TOOL": str(state / "bin" / ("agent-tool.cmd" if os.name == "nt" else "agent-tool"))
```

and update `LEAN_CREATE` to say:

> If `PATH` was reset, call `$AGENT_TOOL` directly; from Python use `PYTHONPATH=$AGENT_STATE/lib python...`.

**Risk:** low; only changes prompt/env.

**Measure:** âcommand not found: agent-toolâ occurrences.

---

## 8. Implement the already-proposed create cleanup

I agree with the four already-proposed items and would implement them as described:

- parse `<invoke>` XML tool calls in `app/agent/client.py` (`parse_text_calls` / `_read_call`);
- make `skill` a direct tool (this is item 2 above);
- strip Markdown fences in `create` (`app/agent/session.py`, `_create`);
- one nudge before a final answer with no passing check (`_steps`).

I would add one refinement to the last item: only nudge when the session has made edits and the last `run_command` did not print a passing marker such as `[exit code 0]` or `PASS`. The harness already tracks `_edited` and tool results, so this is a small condition, not a new subsystem.

## Dead, duplicated, or overbuilt code that could be removed

- `app/agent/codemode.py` is now largely subsumed by `run_command` plus the `agent_tools` bridge. `_lean_specs` does not even offer `run_script` by default. Keep it until usage data is collected, but mark it deprecated and measure `run_script` calls.
- `codemode._run_plain` duplicates execution without the tool bridge. If the file-based bridge is enabled for Docker, this branch can be removed.
- `app/agent/repl.py` (`Kernel`) duplicates process management already present in `sandbox.Job`; it exists only to make created tools share `memo`. It is complex and should be measured before removal, but it is a candidate for simplification.
- `app/agent/chaincheck.py` duplicates target/argument validation that `_run_call` already performs. It is useful as a preflight, but its error messages could be folded into `_run_created`.
- `app/agent/isolate.py` and branch tools (`_branch`, `_adopt`, `_branch_run`) are powerful but likely unused in Terminal-Bench. Keep, but measure branch usage before investing further.
- `schedule_*`, `memory`, `context_notes`/`new_context`, and `checkpoint` are useful for interactive use but are not central to terminal tasks. Do not remove without usage data, but they are candidates for a âcore vs. interactiveâ split.
- The built-in skills are rarely read because of item 2; after exposing `skill` directly, measure whether each built-in skill is ever loaded before deciding whether to keep all of them.

## How to measure

Use the existing `.github/terminal-bench/run_agent.py` event stream and add counters for:

- `skill` calls per task;
- `agent-tool`/`agent_tools` invocations by tool name;
- background jobs that use the bridge;
- blocked doom-loop events;
- create failures by reason (`SyntaxError`, bad name, truncated);
- task success by model and timeout count.

The most important A/B comparisons are: runner with `keep_jobs` and `echo_reasoning="all"`; skill as a direct tool; and the background-job bridge. Those three are small, low-risk, and directly target the measured failure modes.