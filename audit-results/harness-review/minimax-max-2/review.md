# Executive summary

The best return is at three boundaries:

1. **Model-protocol ingress**: malformed text/XML tool calls and lost streamed calls.
2. **Run lifecycle**: the deadline currently protects neither in-flight work nor the final model call.
3. **The `create` boundary**: it is heavily used, but rejects common model output shapes.

I would **not** add more subagents, branches, tools, or rate-limit machinery. The mimo run had no rate limiting, and the measured failures are protocol/lifecycle failures rather than insufficient agent capability.

## Ranked proposals

| Rank | Change | Approx. LOC | Measured evidence | Expected value |
|---|---|---:|---|---|
| 1 | Echo `reasoning_content` back adaptively | 2â6 | reflection/beam:free failed every task until this was done | Highest; potentially converts a 0-pass model to a usable one |
| 2 | Make the deadline hard; do not start work or a report after it | 5â15 minimal, ~20 robust | 21 `AgentTimeoutError`s across the two PolarGrid runs; 35 deadline reports | High; directly addresses task-ending failures |
| 3 | Parse native/ordinary/XML tool-call forms at the client boundary | 15â25 | 30 unreadable calls | High; malformed calls can cascade into the four-turn stop |
| 4 | Make `create` accept common model output shapes | 5â12 | 34/395 create calls failed | High; 8.6% of create calls currently fail |
| 5 | Retry an empty streamed turn without streaming | 2â4 | 8 dropped streamed calls | High gain/LOC; the existing fallback only handles one finish reason |
| 6 | Nudge once when the model finishes without doing work | 6â10 | 4 tasks ended under 5K tokens with no error | Medium; prevents âpremature finalâ from becoming a zero-score task |
| 7 | Offer `skill` directly in the lean tool set | 2â4 | built-in skills used only 5 times in 89 tasks | Medium/uncertain; strongest on skill-matching tasks |
| 8 | Shorten or move the `create` tutorial out of every system prompt | deletion/1â5 | 395 create calls | Medium; reduces the number of opportunities to fail, but may hurt complex tasks |

Ranks 3â5 are close. Implement and measure them separately.

---

# 1. Echo `reasoning_content` for every turn when the model uses it

## Problem and evidence

### `app/agent/session.py`, `AgentSession.__init__`

The default is hard-coded to one provider:

```python
self.echo_reasoning = provider.id == "deepseek" if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
```

Thus `reflection/beam:free`, unless configured separately, does not receive the behavior that made every task fail.

The lower-level support already exists.

### `app/agent/client.py`, `render`

```python
if reasoning == "all":
    message["reasoning_content"] = item.get("reasoning") or ""
```

### `app/agent/session.py`, `_steps`

Reasoning is retained in history only when `echo_reasoning` is enabled:

```python
**({"reasoning": turn["reasoning"] if self.echo_reasoning == "all" else turn["reasoning"][-20000:]}
    if self.echo_reasoning and turn["reasoning"] else {})
```

The measured result is decisive: reflection/beam:free failed every task until the harness sent reasoning back on every turn.

## Why it costs tasks

A reasoning model whose provider requires the previous `reasoning_content` receives an incomplete continuation after its first tool call. It can lose the chain of thought, emit another invalid call, or fail the provider request entirely. This is not a prompt-quality issue; it is a wire-format requirement.

## Concrete change

### Smallest benchmark-only fix

`.github/terminal-bench/run_agent.py`, after creating the session:

```diff
 session = manager.create(provider, "relay", args.model,
                          Workspace(args.workspace, sandbox.Policy("full-access", True)), "bypass")
+session.echo_reasoning = "all"
```

The existing `render(..., reasoning="all")` and history logic then do the rest.

### Better provider-agnostic fix

Detect reasoning after the first response rather than identifying the provider:

```diff
 # app/agent/session.py, AgentSession.__init__
 self.echo_reasoning = provider.id == "deepseek" if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
+self._echo_reasoning_explicit = self.profile["echo_reasoning"] is not None
```

```diff
 # app/agent/session.py, _steps, immediately after turn = self._turn()
 turn = self._turn()
+if turn.get("reasoning") and not self._echo_reasoning_explicit:
+    self.echo_reasoning = "all"
```

The first response has no prior assistant history to echo. After receiving reasoning, the session enables `"all"` before that turn is appended to history, so the next request includes it.

Also avoid sending empty reasoning fields:

```diff
 # app/agent/client.py, render
-if reasoning == "all":
+if reasoning == "all" and item.get("reasoning"):
     message["reasoning_content"] = item.get("reasoning") or ""
```

## Risk

- More input tokens and possibly larger prompts.
- A strict provider that returns `reasoning_content` but rejects it on input could fail. Keep the explicit profile override as an escape hatch.
- `"all"` retains full reasoning in session history; the existing `"all"` path already does this intentionally.

## Measurement

Run reflection/beam:free on the full task set:

- Baseline: 0 passes because every task failed.
- Gate: non-zero pass count and no recurring provider error mentioning `reasoning_content`.
- Also inspect assistant turns after the first tool call and confirm each carries non-empty `reasoning_content`.
- Compare prompt-token usage and wall-clock time; if success rises but cost explodes, use `"all"` only for the turns that contain tool calls.

---

# 2. Make the deadline hard, and skip the model-only final report in Terminal-Bench

## Problem and evidence

### `app/agent/session.py`, `set_deadline`

The internal deadline is only a trigger for a report:

```python
self._deadline = time.time() + seconds - grace
```

### `app/agent/session.py`, `_steps`

The deadline is checked only **before** asking the model:

```python
if time.time() >= self._deadline:
    self.emit("notice", text="Out of time; the agent is writing its final report.")
    self._wrap_up("time limit")
    break
```

After the model returns, its calls are executed unconditionally:

```python
self._run_calls(calls)
```

A command that begins immediately before the deadline can therefore run for another 120 seconds or more.

### `app/agent/client.py`, `complete`

A transient failure can consume several minutes:

```python
for attempt in range(NETWORK_RETRIES + 1):
```

with exponential waits up to 90 seconds per retry. The final report therefore has no hard bound.

There were 21 timeout errors across the two PolarGrid runs and 35 deadline reports.

## Why it costs tasks

At the deadline, a model-only final report cannot change any files and therefore cannot improve a Terminal-Bench score. If that report call stalls or retries, the process can miss the outer task timeout even though the workspace changes are already present.

Likewise, work started just before the deadline can consume the remaining wall clock and prevent cleanup or result emission.

## Concrete change

### Minimal benchmark-safe change

Allow the runner to disable the final model call:

```diff
 # app/agent/session.py, AgentSession.__init__
 self._deadline = self._notice_at = 0.0
+self.wrap_up_on_deadline = True
```

```diff
 # app/agent/session.py, _steps
 if time.time() >= self._deadline:
-    self.emit("notice", text="Out of time; the agent is writing its final report.")
-    self._wrap_up("time limit")
+    if self.wrap_up_on_deadline:
+        self.emit("notice", text="Out of time; the agent is writing its final report.")
+        self._wrap_up("time limit")
+    else:
+        self.emit("notice", text="Out of time; stopping with the current workspace.")
     break
```

```diff
 # .github/terminal-bench/run_agent.py
 session.set_deadline(args.minutes * 60)
+session.wrap_up_on_deadline = False
```

This changes no scored behavior: the final report has no tools and cannot edit the workspace.

### Also prevent work after the deadline

Store a hard deadline separately:

```diff
 # app/agent/session.py
 self._deadline = self._notice_at = 0.0
+self._hard_deadline = 0.0
```

```diff
 # app/agent/session.py, set_deadline
 def set_deadline(self, seconds: float) -> None:
     grace = min(TIME_GRACE_S, seconds * 0.1)
-    self._deadline = time.time() + seconds - grace
+    self._hard_deadline = time.time() + seconds
+    self._deadline = self._hard_deadline - grace
     self._notice_at = self._deadline - grace * 2
```

After a model turn, reject late calls:

```diff
 # app/agent/session.py, _steps
 calls = turn["calls"]
+if self._hard_deadline and time.time() >= self._hard_deadline:
+    calls = []
```

In `_await_job`, stop rather than background a job at the hard deadline:

```diff
 # app/agent/session.py, _await_job
-while job.code is None and (time.time() < end + self._root().asked_s - paused
-                             or self._root().status == "waiting") and not self._stop:
+hard = self._root()._hard_deadline
+while job.code is None and (time.time() < end + self._root().asked_s - paused
+                             or self._root().status == "waiting") and not self._stop:
+    if hard and time.time() >= hard:
+        job.stop()
+        break
     time.sleep(0.05)
```

For the general harness, also pass the hard deadline into `client.complete` and stop retries when it is reached. That requires threading an optional absolute deadline through `complete`, `_complete_once`, `_post`, and `_read_stream`.

## Risk

- Stopping an in-flight command can leave a partially written file. This is preferable to an external process kill only if the hard deadline is genuinely the outer deadline.
- Disabling the final report is unsuitable for the interactive product, but appropriate for Terminal-Bench. Keep the product default enabled.
- Do not remove the grace period; it gives a nearly finished model turn time to return.

## Measurement

Use the existing JSONL events:

- Count `result.status == "timeout"`.
- Count tasks that reached the deadline.
- Record wall-clock time from session start to `done`.
- Compare the two PolarGrid runs and the completed mimo run.

Acceptance target:

- At least halve the 21 timeout events, or recover at least five task outcomes.
- Zero model calls beginning after the hard deadline.
- Terminal-Bench result status should become `done` rather than `timeout`.

---

# 3. Normalize all common tool-call text formats at the provider boundary

## Problem and evidence

### `app/agent/client.py`

The parser recognizes only a tag containing a zero-width character:

```python
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
```

and `_build` only enters that parser when it sees the same exact spelling:

```python
if "<tool_call>" in text:
```

A model that writes ordinary `<tool_call>...</tool_call>` is not parsed.

### `app/agent/client.py`, `_read_call`

The alternatives are also narrow:

```python
named = re.search(r"function=(\w+)", raw)
```

and:

```python
PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
```

Thus `<invoke name="...">`, `<parameter name="...">`, and ordinary `<tool_call>` are not understood. The measured run had **30 unreadable calls** in precisely these forms.

### `app/agent/session.py`, `_steps`

Four consecutive malformed turns stop the run:

```python
if broken >= UNREADABLE_TURNS:
    self.emit("error", text=f"The model wrote unreadable tool calls {broken} turns in a row; stopping to save tokens.")
    break
```

## Why it costs tasks

The malformed call becomes a failed tool result. The model then spends another turn explaining or repairing it, often with the same output format. Several such calls can consume most of the task time or hit the four-turn stop.

This should be fixed where provider output enters the harness, not by asking each model to behave differently.

## Concrete change

### First: accept both spellings of `tool_call`

```diff
-TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
+TOOL_CALL_RE = re.compile(r"<(?:\u200b)?tool_call>\s*(.*?)\s*(?:</(?:\u200b)?tool_call>|(?=<(?:\u200b)?tool_call>)|\Z)", re.S)
```

```diff
-if "<tool_call>" in text:
+if "<tool_call>" in text or "<\u200btool_call>" in text:
```

This is only a few lines and should be measured separately.

### Then: parse `<invoke>` blocks

Add a small provider-boundary normalizer:

```python
INVOKE_RE = re.compile(
    r"<invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</invoke\s*>",
    re.I | re.S,
)
PARAM_RE = re.compile(
    r"<parameter\b[^>]*\bname\s*=\s*[\"']?(?P<name>[\w.-]+)[\"']?[^>]*>"
    r"(?P<value>.*?)</parameter\s*>",
    re.I | re.S,
)

def _xml_calls(text: str, schemas: dict) -> list[dict]:
    calls = []
    for match in INVOKE_RE.finditer(text):
        named = re.search(r"\bname\s*=\s*[\"']?([\w.-]+)", match.group("attrs"))
        if not named:
            continue
        name = named.group(1)
        if schemas and name not in schemas:
            continue
        body = match.group("body").strip()
        try:
            data = json.loads(body)
            args = data.get("arguments", data.get("args", {})) if isinstance(data, dict) else {}
        except ValueError:
            args = {}
            for param in PARAM_RE.finditer(body):
                value = param.group("value").strip()
                try:
                    value = json.loads(value)
                except ValueError:
                    value = html.unescape(value)
                args[param.group("name")] = value
        calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return calls
```

Then, in `_build`:

```python
written = _xml_calls(text, schemas_of(tools))
if not calls:
    calls = written
else:
    # Use XML arguments only to fill an empty native call.
    spare = list(written)
    for call in calls:
        match = next((w for w in spare if w["name"] == call["name"]), None)
        if match and not call["args"]:
            call["args"] = match["args"]
            call.pop("error", None)
            spare.remove(match)
```

For native calls whose `arguments` string is not JSON, run the same parser against `function["arguments"]` before marking the call as an error.

For `<function_calls>...</function_calls>` wrappers containing direct JSON objects, reuse the existing raw JSON-object scan in `_read_call` on each object.

## Risk

- An answer that discusses XML could be mistaken for a tool call. Restrict parsed names to the current tool schemas and only use XML calls when there are no valid native calls, or to fill empty native arguments.
- HTML entity decoding must be used for parameter values.
- Do not silently accept an XML name that is not an offered tool.

## Measurement

Instrument calls by source:

- Native JSON.
- Ordinary `<tool_call>`.
- `<invoke>`.
- Unreadable.

Acceptance target:

- Unreadable calls fall from 30 to at most 2â3 per 89 tasks.
- The number of successful tool calls increases.
- The count of four-consecutive-malformed stops becomes zero.

Test at least:

```text
<tool_call>{"name":"read_file","arguments":{"path":"app.py"}}</tool_call>
<invoke name="read_file"><parameter name="path">app.py</parameter></invoke>
<function_calls><invoke name="read_file">{"path":"app.py"}</invoke></function_calls>
```

---

# 4. Make `create` accept the code and names models actually produce

## Problem and evidence

### `app/agent/session.py`, `_create`

Names are rejected before any useful feedback:

```python
if not CREATED_NAME.fullmatch(name):
    raise ToolError("name must be lower_case letters, digits and _, starting with a letter")
```

### `app/agent/session.py`, `_create`

Code is compiled immediately:

```python
code = _as_function(name, code)
try:
    compile(code, f"{name}.py", "exec")
except SyntaxError as exc:
    raise ToolError(f"Syntax error: {exc}") from exc
```

Models commonly send a Markdown-fenced body, and the task evidence explicitly records bad names, syntax errors, and truncated code. There were 34 failures among 395 create calls.

### `app/agent/models.py`, Qwen quirk

The prompt tells Qwen:

```python
"to hand over long text, save it with write_file and give the path."
```

### `app/agent/session.py`

But `write_file` is not in the lean tool set:

```python
LEAN_TOOLS = frozenset({"read_file", "edit_file", "run_command"})
```

So the model is instructed to use an unavailable tool, while long code remains a single oversized function argument.

## Why it costs tasks

A failed create consumes a full model/tool round after the model has already spent tokens generating the tool. If it happens near the output cutoff, it also intersects the timeout failures.

## Concrete change

### A. Strip Markdown fences

Before `_as_function`:

```python
code = str(args.get("code") or "")
fenced = re.search(r"```(?:python)?\s*(.*?)```", code, re.S | re.I)
if fenced:
    code = fenced.group(1).strip()
```

### B. Normalize harmless name mistakes

```python
raw_name = str(args.get("name") or "").strip()
name = re.sub(r"[^a-z0-9_]+", "_", raw_name.lower()).strip("_")
if not name or name[0].isdigit():
    name = f"tool_{name}"
name = name[:41]
```

Do not silently overwrite an existing tool when normalization changes the name; report the collision instead.

### C. Allow code to come from a file

Extend the create schema:

```python
"required": ["name"],
"properties": {
    "name": {...},
    "code": {"type": "string", ...},
    "code_file": {"type": "string", ...},
    ...
}
```

Resolve and read `code_file` in `_create`:

```python
source = str(args.get("code_file") or "").strip()
if source:
    path = self.workspace.resolve(source)
    code = path.read_text(encoding="utf-8", errors="replace")
else:
    code = str(args.get("code") or "")
```

Update the immediate-run condition in `_session_tool` so `code_file` also counts as code to run.

Finally, expose `write_file` to lean models:

```diff
-LEAN_TOOLS = frozenset({"read_file", "edit_file", "run_command"})
+LEAN_TOOLS = frozenset({"read_file", "write_file", "edit_file", "run_command"})
```

This gives the Qwen advice a real target and avoids putting a large generated program into one JSON argument.

## Risk

- Name normalization can collide with a built-in or existing created tool. Check collisions explicitly.
- `code_file` must be confined to the workspace and read-only roots.
- A stale file can create a different tool than the model intended. Include the source path in the create result.
- More directly offered tools can distract a small model; measure this separately.

## Measurement

Count create failures by category:

- Invalid name.
- Syntax error.
- Missing/truncated argument.
- Other.

Also track:

- Calls per task.
- Failed create calls per 100 tasks.
- Percentage of creates using `code_file`.
- Whether failed creates precede successful task completion.

Acceptance target: fewer than 10 failed creates per 395, without increasing total turns.

---

# 5. Retry an empty streamed tool turn without streaming

## Problem and evidence

### `app/agent/session.py`, `_steps`

There is already a useful fallback, but only for one finish reason:

```python
if debug.get("finish") == "tool_calls" and not self._plain and self._base_complete is client.complete:
    self._plain = True
    self.emit("notice", text=f"The server lost the tool calls while streaming ({used} tokens); turning streaming off for this session.")
```

A relay can drop a call while reporting `finish_reason="stop"` or an empty/custom finish reason. In that case the session retries the same streamed request and can lose the same call repeatedly. There were eight such events.

## Why it costs tasks

Every retry consumes another model response and may cross the deadline. The harness has enough information to know that the streamed representation was unreliable; it should change transport after the first empty turn.

## Concrete change

Broaden the existing fallback:

```diff
-if debug.get("finish") == "tool_calls" and not self._plain and self._base_complete is client.complete:
+if not self._plain and self._base_complete is client.complete:
     self._plain = True
-    self.emit("notice", text=f"The server lost the tool calls while streaming ({used} tokens); turning streaming off for this session.")
+    self.emit("notice", text=f"The streamed reply was empty ({used} tokens); retrying without streaming.")
```

Keep the existing `finish == "length"` branch before it so an output-limited reasoning response still gets a larger budget rather than an immediate transport change.

For providers that omit `index` on a single tool call, also normalize `delta.tool_calls` fragments in `client._read_stream`; repeated full function names should not be concatenated.

## Risk

- A genuinely empty response causes one non-streamed retry. The current loop already retries empty responses, so this mainly changes transport.
- Turning streaming off for the rest of the session can increase latency on later turns.

## Measurement

Count:

- Empty streamed replies.
- Empty non-streamed retries.
- Recovered calls after the transport switch.
- Wall-clock overhead.

The important metric is not merely fewer empty replies; it is the number of previously lost calls that subsequently execute successfully.

---

# 6. Do not accept a first âdoneâ reply with no work

## Problem and evidence

### `app/agent/session.py`, `_steps`

Any assistant text without calls ends the request:

```python
if not calls:
    if self._more_work():
        continue
    break
```

There is no check that the model inspected the task, changed anything, or ran a check. Four measured tasks ended under 5K tokens with no error, exactly matching this path.

## Why it costs tasks

Terminal-Bench tasks normally require workspace changes. A model that says âdoneâ before doing anything gets a zero immediately; there is no later opportunity to correct it.

## Concrete change

Add a request-scoped counter and a one-shot continuation:

```diff
 # AgentSession.__init__
 self._request_tools = 0
+self._final_nudge = False
```

```diff
 # _new_request
 self._request_tools = 0
+self._final_nudge = False
```

Increment in `_run_calls`, then before accepting a no-call reply:

```python
if (not calls and self.require_evidence and not self._request_tools
        and not self._final_nudge):
    self._final_nudge = True
    self.history.append({
        "role": "user",
        "content": "[Continue] No tool has run for this task. Inspect the task files, make the required change, "
                   "and run its check. If the task truly requires no change, explain why and repeat the final answer.",
    })
    continue
```

Enable it only for Terminal-Bench:

```diff
 # .github/terminal-bench/run_agent.py
 session.wrap_up_on_deadline = False
+session.require_evidence = True
```

A second optional condition can catch âedited but never checked,â but that requires tracking the last edit and the last `run_command`; start with the no-tool case to minimize false positives.

## Risk

- A question-only task can be delayed by one turn. Keep this behind a profile/runner flag.
- A model can repeat the same final text. Accept the second reply rather than creating a loop.

## Measurement

For the four known early-stop tasks, verify:

- The first final had no tool call.
- The continuation led to a tool call or an explicit no-change explanation.
- At least two of the four become passing tasks.

Across all tasks, ensure the extra turn does not increase timeout rate.

---

# 7. Offer `skill` directly instead of requiring `create` to reach it

## Problem and evidence

### `app/agent/session.py`, `_lean_specs`

The lean root model receives only:

```python
LEAN_TOOLS = frozenset({"read_file", "write_file", "edit_file", "run_command"})
```

`skill` is included in `library_specs`, so it is reachable only from inside a created script.

### `app/agent/session.py`, `_volatile`

The root catalog is path-based:

```python
rows = {"Skills": skills.catalog(self.skills, paths=not self.depth)}
```

At root depth this tells the model to read a path rather than use a `skill` tool. Built-in skills were read only five times in 89 tasks.

## Why it costs tasks

A skill containing the correct procedure may never activate. Requiring a Python wrapper to load it adds a failure point to an otherwise simple read-only operation.

## Concrete change

Add the skill tool directly when at least one model-invocable skill exists:

```diff
 # _lean_specs
 rows = [s for s in SPECS if s["name"] in (("read_file",) if self.plan_mode else LEAN_TOOLS)]
+if any(not skill.manual for skill in self.skills.values()):
+    rows.append(SESSION_SPECS["skill"])
```

Use the concise catalog at root:

```diff
 # _volatile
-rows = {"Skills": skills.catalog(self.skills, paths=not self.depth)}
+rows = {"Skills": skills.catalog(self.skills, paths=False)}
```

`skills.load` already returns the body, skill path, and list of supporting files, so the model can read referenced files afterward.

## Risk

- One more tool can distract a small model.
- Manual skills should not become model-invocable merely because the tool exists; reject `found.manual` in `_session_tool` if that is the intended policy.
- Skill use is not inherently beneficial; do not force a skill on tasks whose descriptions do not match.

## Measurement

Report:

- Skill calls per task.
- Skill calls on matching versus non-matching tasks.
- Pass rate on the matching subset.
- Tokens spent loading skills.

Do not judge this by total skill count alone; the goal is better outcomes on tasks where the skill contains the needed procedure.

---

# 8. Demote `create` from the default prompt to an advanced mechanism

## Problem and evidence

### `app/agent/session.py`, `LEAN_CREATE`

The prompt contains four worked examples and repeatedly teaches the model to create tools. The measured run called `create` 395 times, with 34 failures. It also tells the model:

```text
reach for create when steps follow each other without needing your judgment in between
```

This encourages `create` even for one edit or one test.

## Why it costs tasks

Every additional abstraction creates another place for malformed arguments, code syntax errors, and timeouts. The simple pathâread/edit/runâis better for most Terminal-Bench tasks.

## Concrete change

Do not remove `create`; keep it open as the owner wants. First shorten the always-on guidance to something like:

```text
Use direct read_file/write_file/edit_file/run_command for ordinary work. Use create only for a repeated
multi-step operation that needs a named reusable tool. Keep one call simple and checkable.
```

Move the long examples into an on-demand `creating-tools` skill.

As an A/B test, also expose `write_file` directly as described in proposal 4.

## Risk

Complex tasks might benefit from the richer chain examples. Measure task categories separately rather than removing the tutorial immediately.

## Measurement

Compare:

- Create calls per task.
- Failed create rate.
- Direct-tool calls per task.
- Total model turns and wall time.
- Pass rate on tasks requiring repeated operations.

The desired result is fewer creates without reducing complex-task success.

---

# Dead, duplicated, or overbuilt code

These are candidates, not all should be removed immediately. Run `rg` over the complete repository and tests before deleting public functions.

| Location | Finding | Suggested action |
|---|---|---|
| `app/agent/registry.py::workspace_digest` | Superseded by `sources_digest`, whose doc says: âUses the same construction as workspace_digestâ. No caller appears in the supplied tree. | Remove `workspace_digest`; keep `sources_digest`. |
| `app/agent/chaincheck.py::check` | The `fixes` callback has a default and one branch, but no supplied caller passes it: `fixes=lambda name, args: False`. | Remove the callback and branch after checking tests. |
| `app/agent/session.py` module constants | `MAX_STEPS = 300` and `TOKEN_BUDGET = 10_000_000` are duplicated by `profile["max_steps"]` and `profile["token_budget"]`; no supplied-tree references use the module constants. | Remove the two constants, or make the profile reference them once rather than duplicating literals. |
| `app/agent/rules.py::SAFE_COMMANDS` | Contains `"env"`, but `safe_readonly` explicitly rejects it: `if words[0] == "env" ...`. | Remove `"env"` from `SAFE_COMMANDS`. |
| `app/agent/codemode.py` and `SPECS["run_script"]` | A second tools-as-Python implementation duplicates `session.TOOLS_LIBRARY`/`CREATED_CALL` and `repl.py`. Its fallback deliberately provides non-working tools: `_Tools.__getattr__` always raises âagent tools cannot be calledâ¦â. | After usage telemetry, consolidate `run_script` onto the `create` RPC path or remove it. Prefer keeping `create` as the one model-written-tool mechanism. |
| `app/agent/builtin_skills/mcp-builder/scripts/connections.py` | A separate MCP client implementation, disconnected from `app/agent/mcp.py`. | Move to a separate examples repository, or remove from the shipped harness. |
| `app/agent/builtin_skills/mcp-builder/scripts/evaluation.py` | Anthropic-specific evaluator using the separate `connections.py`; it does not participate in harness execution. | Keep only if server-authoring evaluation is a deliberate supported workflow; otherwise move/remove with the preceding script. This can also remove the `anthropic` and official `mcp` client dependencies from the product package. |
| `app/agent/builtin_skills/webapp-testing/examples/*` | Static sample scripts, including hard-coded `/mnt/user-data/outputs` paths. They are not runtime code and are merely listed as skill extras. | Move to documentation or remove from the installed skill. Keep `scripts/with_server.py`, which is operational. |
| `app/agent/session.py::LEAN_CREATE` | Not dead, but a large always-on tutorial duplicates the `create` tool schema and examples. | Move most of it to an on-demand skill after the `create` A/B test. |

Do **not** remove the core MCP client, plugin API, checkpoint system, subagents, or branch merge solely from these benchmark numbers. There is no evidence they caused the measured failures, and they are part of the requested open surface.

---

# Measurement protocol

Use the exact task set and keep task timeout, sandbox, model, prompt, and tool ordering fixed.

## Primary metrics

For each arm, report:

- Mean pass count across repeated runs.
- Tasks passing in every run, not just the union.
- `AgentTimeoutError`/timeout rate.
- Unreadable tool calls per 100 tasks.
- Failed `create` calls per 100 tasks.
- Model turns per task.
- Wall-clock seconds per task.
- Completion and prompt tokens.
- Four-consecutive-unreadable stops.
- Tasks that finish after the deadline.
- Skill calls on matching versus non-matching tasks.

The baselineâs 21 versus 16 passes and only nine tasks passing in both runs show substantial variance. A single run is not enough to accept a one-task change.

## Suggested rollout

1. **Reasoning echo**: smoke-test reflection/beam:free, then run the full set.
2. **Deadline behavior**: run the two PolarGrid arms and the completed mimo arm separately.
3. **Tool-call parser**: add ordinary `<tool_call>` support first; measure before adding full XML parsing.
4. **Create hardening**: measure fence extraction, name normalization, and `code_file` separately if possible.
5. **Premature-final nudge** and **direct skill**: run only after protocol and lifecycle failures are low, so their effect is not masked.

Use paired task-level comparisons and report the number of tasks changed from fail to pass and pass to fail. For small deltas, use at least three runs per arm; a single aggregate score difference of one or two tasks is within the observed baseline variance.

Finally, there is no measured justification for spending effort on new rate limiting: the mimo run explicitly had none. The demonstrated wins are in tool-call normalization, reasoning round-tripping, deadline enforcement, and making the already-open `create` mechanism less brittle.