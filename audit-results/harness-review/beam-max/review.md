# Ranked changes most likely to raise task success

| Rank | Change | Why it matters | Approx. lines |
|---|---|---:|---:|
| 1 | Parse XML-ish tool calls (`<invoke>`, `<function_calls>`) | 30 unreadable calls in the measured Qwen run; one unreadable turn can end a whole task | ~40 |
| 2 | Expose `skill` as a top-level tool in lean mode | Built-in skills were read only 5 times in 89 tasks; model cannot call `skill` from the lean tool list | ~5 |
| 3 | Send `reasoning_content` back by default | `reflection/beam:free` failed every task until this was fixed | ~1 |
| 4 | Keep task-started jobs alive in the benchmark runner | Servers the task asked for are killed when the session closes | ~1 |
| 5 | Add a âno passing check seenâ nudge before final | 28 of 30 reflection failures ended with the model saying it was done | ~15 |
| 6 | Strip Markdown fences from `create` code | 34 of 395 `create` calls failed on syntax/truncation issues | ~5 |
| 7 | Retry dropped streamed tool calls immediately | 8 streaming drops cost a full turn each | ~12 |
| 8 | Make `skill` read-only in `KIND` | Avoids approval friction for loading skills | ~1 |

---

## 1. Parse XML-ish tool calls in text mode

**Problem.** The fallback text parser only understands `<tool_call>...</tool_call>` JSON and Qwenâs `<function=...><parameter=...>` form. Measured run: 30 calls were unreadable because the model wrote XML-like `<invoke name=...>` or `<function_calls>` text.

Evidence:
- `app/agent/client.py`
  ```python
  TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
  ```
- `_read_call` only tries:
  ```python
  named = re.search(r"function=(\w+)", raw)
  called = re.search(r'"name"\s*:\s*"(\w+)"', raw)
  ```
- Session stops after 4 unreadable turns:
  ```python
  if broken >= UNREADABLE_TURNS:
      self.emit("error", text=f"The model wrote unreadable tool calls {broken} turns in a row; stopping to save tokens.")
      break
  ```

**Why it costs tasks.** An unreadable call becomes a failed tool result. The model often retries the same malformed style, reaches `UNREADABLE_TURNS = 4`, and the session stops with no useful result.

**Change.** In `parse_text_calls`, also extract `<invoke name="...">...<parameter name="...">value</parameter>...</invoke>` blocks, coerce parameter values with `json.loads` when possible, and remove `<function_calls>...</function_calls>` blocks from the returned text.

Pseudo-diff:
```python
INVOKE_RE = re.compile(r"<invoke\b[^>]*?\bname\s*=\s*[\"']?([A-Za-z0-9_]+)[\"']?[^>]*>(.*?)</invoke>", re.S | re.I)
PARAM_RE = re.compile(r"<parameter\b[^>]*?\bname\s*=\s*[\"']?([A-Za-z0-9_]+)[\"']?[^>]*>(.*?)</parameter>", re.S | re.I)

def parse_text_calls(text, tools=None):
    calls = []
    schemas = schemas_of(tools)

    for raw in TOOL_CALL_RE.findall(text or ""):
        ...

    for match in INVOKE_RE.finditer(text or ""):
        name = match.group(1)
        args = {}
        for param in PARAM_RE.finditer(match.group(2)):
            value = param.group(2).strip()
            try:
                value = json.loads(value)
            except ValueError:
                pass
            args[param.group(1)] = value
        calls.append({"id": uuid.uuid4().hex[:12], "name": name, "args": args})

    text = re.sub(r"<function_calls>.*?</function_calls>", "", text or "", flags=re.S | re.I)
    text = TOOL_CALL_RE.sub("", text).strip()
    return text, calls
```

**Risk.** Nested XML inside a parameter value may be parsed incorrectly. That is acceptable for the measured formats; malformed calls still fall back to the existing unreadable-call error.

**Measure.** Re-run the Qwen benchmark and compare:
- `broken >= UNREADABLE_TURNS` stops,
- count of `Unreadable tool call` errors,
- task success.

---

## 2. Expose `skill` as a top-level tool in lean mode

**Problem.** The lean tool list does not include `skill`, even though the system prompt tells the model to load skills with the `skill` tool. Built-in skills were read only 5 times in 89 tasks.

Evidence:
- `app/agent/session.py`
  ```python
  LEAN_TOOLS = frozenset({"read_file", "edit_file", "run_command"})
  ```
- `_lean_specs()`:
  ```python
  rows = [s for s in SPECS if s["name"] in (("read_file",) if self.plan_mode else LEAN_TOOLS)]
  ```
- `SESSION_SPECS["skill"]` exists, but is only added in `library_specs()`, i.e. reachable through `create` code, not as a direct tool.
- `skills.catalog()` is still placed in the system prompt, so the model is told to use a tool it cannot see.

**Why it costs tasks.** Skills such as PDF, XLSX, DOCX, and webapp testing contain exact procedures. The model cannot invoke them directly, so it improvises or skips them.

**Change.** Add `skill` to the lean tool list when skills exist.

```python
if self.skills:
    rows.append(SESSION_SPECS["skill"])
```
inside `_lean_specs()`, after the base read/edit/run tools.

**Risk.** One more tool in the prompt. It is read-only and only loads instructions, so approval impact is small.

**Measure.** Count `skill` calls and success on document/webapp tasks before and after.

---

## 3. Default `reasoning_content` echoing to true

**Problem.** Non-DeepSeek providers that require echoed reasoning content get no reasoning echoed unless the profile explicitly sets it. `reflection/beam:free` failed every task until this was fixed.

Evidence:
- `app/agent/session.py`
  ```python
  self.echo_reasoning = provider.id == "deepseek" if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
  ```
- `app/agent/client.py`
  ```python
  if reasoning == "all":
      message["reasoning_content"] = item.get("reasoning") or ""
  elif reasoning and item.get("reasoning") and index > last_user:
      message["reasoning_content"] = item["reasoning"]
  ```

**Why it costs tasks.** A reasoning model that emits `reasoning_content` in one turn needs it echoed in subsequent tool-loop turns. When it is missing, the provider rejects the request or the model loses its chain of thought.

**Change.** Default to echoing reasoning content for all providers:
```python
self.echo_reasoning = True if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
```

**Risk.** For models that never return reasoning, this only adds an empty `reasoning_content` field on turns that already have reasoning. It does not change the default empty case materially. Some strict OpenAI-compatible servers may reject unknown message fields; measure on the target providers.

**Measure.** Run `reflection/beam:free` and count:
- HTTP 400s mentioning `reasoning_content`,
- empty/unusable turns,
- task success.

---

## 4. Keep benchmark-started jobs alive on session close

**Problem.** Jobs started by the agent are stopped by default when the session closes. Terminal-Bench tasks that need a server can fail at grading time because the server died with the agent.

Evidence:
- `app/agent/session.py`
  ```python
  if self.profile.get("keep_jobs"):
      with self._usage_lock:
          for job in list(self.jobs.values()):
              if job.alive:
                  job.detach()
          self.jobs.clear()
  self.stop()
  ```
- `.github/terminal-bench/run_agent.py` creates a session but never enables `keep_jobs`:
  ```python
  session = manager.create(provider, "relay", args.model, Workspace(args.workspace, sandbox.Policy("full-access", True)), "bypass")
  ```

**Why it costs tasks.** A task can require the agent to start a web server or VM. The agent does that correctly, but the harness kills it at close, so the grader sees nothing running.

**Change.** In `run_agent.py`, enable job retention before sending:
```python
session.profile["keep_jobs"] = True
session.set_deadline(args.minutes * 60)
session.send(args.instruction)
```

**Risk.** Jobs linger after the run. In a benchmark container this is usually desirable and harmless. In normal interactive use, the default should remain opt-in.

**Measure.** Count server-dependent task passes and âjob stoppedâ failures.

---

## 5. Prevent premature final answers with a verification nudge

**Problem.** 28 of 30 reflection failures ended with the model saying it was done. The loop stops as soon as the model returns no tool calls, even if it never saw a passing check.

Evidence:
- `app/agent/session.py`
  ```python
  if not calls:
      if self._more_work():
          continue
      break
  ```
- `_edited` is set after successful edits:
  ```python
  if ok and (KIND.get(name) == "edit" or role == "edit") and name != "web_download":
      self._edited = True
  ```
- But nothing records whether a check passed.

**Why it costs tasks.** The model can declare completion after an edit without running the taskâs tests. Hidden tests then fail.

**Change.** Track a `_verified` flag and nudge once before allowing a final answer.

In `__init__`:
```python
self._verified = False
self._final_nudged = False
```

In `_new_request`:
```python
self._verified = False
self._final_nudged = False
```

In `_run_calls`, after `ok`:
```python
if ok and call["name"] == "run_command" and "[exit code 0]" in output:
    self._verified = True
if ok and call["name"] in self._root().created and ("PASS" in output or "[exit code 0]" in output):
    self._verified = True
```

In `_steps`:
```python
if not calls:
    if self._more_work():
        continue
    if self._edited and not self._verified and not self._final_nudged and not self.depth:
        self._final_nudged = True
        self.history.append({"role": "user", "content":
            "[Before you finish: files changed but no passing check was seen. Run the relevant test or check now; only say done after it passes.]"})
        continue
    break
```

**Risk.** One extra turn in cases where the model genuinely has nothing left to verify. The nudge is limited to once per request.

**Measure.** Count final answers that occur with `_edited=True` and `_verified=False`, and task success.

---

## 6. Strip Markdown fences from `create` code

**Problem.** `create` compiles code verbatim. Models often wrap code in Markdown fences, especially in text mode. This contributes to the 34 failed `create` calls.

Evidence:
- `app/agent/session.py`
  ```python
  code = str(args.get("code") or "")
  if not code.strip():
      ...
  code = _as_function(name, code)
  try:
      compile(code, f"{name}.py", "exec")
  except SyntaxError as exc:
      raise ToolError(f"Syntax error: {exc}") from exc
  ```

**Why it costs tasks.** A fence such as ` ```python ` becomes a syntax error. The model then spends a turn repairing a problem that is not in the logic.

**Change.** Strip a leading fence and trailing fence before compilation:
```python
code = str(args.get("code") or "")
code = re.sub(r"^\s*```(?:python|py)?\s*\n", "", code)
code = re.sub(r"\n```\s*$", "", code)
```

**Risk.** Very low. Code that legitimately begins or ends with triple backticks is not expected in this tool.

**Measure.** Compare `create` syntax-error rate before and after.

---

## 7. Retry dropped streamed tool calls immediately

**Problem.** When the server drops tool calls during streaming, the harness only notices after an empty reply and then disables streaming. That costs a full turn per drop.

Evidence:
- `app/agent/session.py`
  ```python
  if debug.get("finish") == "tool_calls" and not self._plain and self._base_complete is client.complete:
      self._plain = True
      self.emit("notice", text=f"The server lost the tool calls while streaming ({used} tokens); turning streaming off for this session.")
  ```

**Why it costs tasks.** The empty turn is wasted, and the model may stop or repeat work before the fallback takes effect.

**Change.** In `_turn_once`, retry the same request immediately without streaming when a streamed response reports `finish == "tool_calls"` but no calls or text were produced:
```python
turn = self._call_model(client.render(self.history, system, self.text_tools, specs, reasoning=self.echo_reasoning),
                        None if self.text_tools else specs)
if (not self.text_tools and not turn["calls"] and not turn["text"]
        and turn.get("finish") == "tool_calls" and not self._plain
        and self._base_complete is client.complete):
    self._plain = True
    self.emit("notice", text="The server dropped tool calls while streaming; retrying without streaming.")
    turn = self._call_model(client.render(self.history, system, self.text_tools, specs, reasoning=self.echo_reasoning),
                            None if self.text_tools else specs)
```

**Risk.** If a provider consistently drops calls, streaming is disabled for the rest of the session. That is the desired fallback.

**Measure.** Count streaming-drop events and turns lost to empty retries.

---

## 8. Make `skill` read-only in `KIND`

**Problem.** `skill` is not in `KIND`, so it defaults to `exec`. In `ask` mode it can require approval even though it only loads instructions.

Evidence:
- `app/agent/tools.py`
  ```python
  KIND = {... "search": "read", "glob": "read", "write_file": "edit", ...}
  ```
  There is no `"skill"` entry.
- `app/agent/session.py`
  ```python
  kind = plugin.kind if plugin else self._role(name) or KIND.get(name, "exec")
  ```

**Why it costs tasks.** Unnecessary approval cards slow down or block autonomous runs, especially in benchmark mode where approvals are auto-denied except `ask_user`.

**Change.** Add:
```python
"skill": "read",
```
to `KIND` in `app/agent/tools.py`.

**Risk.** None meaningful; `skill` only returns instructions and file listings.

**Measure.** Approval-card count and blocked `skill` calls.

---

# Dead, duplicated, or overbuilt code

## Likely removable or deferrable

1. **`app/agent/builtin_skills/mcp-builder/scripts/evaluation.py` and `connections.py`**  
   This is an MCP server evaluation harness shipped as a built-in skill. It is not used by the core agent loop. It can be moved to an example repository or removed to keep the harness thin.

2. **Branch/isolate machinery**  
   `app/agent/isolate.py`, plus `_branch`, `_adopt`, `_branch_run`, `_branch_diff`, and 3-way merge logic in `app/agent/session.py`, is complex and rarely used. It is valuable for experiments, but it is a candidate for removal if the goal is a thin core.

3. **Checkpoint auto-marking and pinned-file machinery**  
   `app/agent/checkpoint.py` is large. The core useful part is âundo last turn.â The named marks, pinned files, dropped-attempt log, and auto-marks could be simplified without harming normal coding tasks.

4. **Persistent REPL kernel**  
   `app/agent/repl.py` keeps imports and `memo` alive across created-tool calls. This is useful, but it adds a lot of process-management complexity. Running each created tool in a fresh process would be simpler, at the cost of losing `memo`.

5. **Model roles and sources**  
   `app/agent/roles.py` and the related `_role_target`, `_sources`, `_pick`, and `_price_gate` machinery are advanced features. They are not needed for Terminal-Bench-style tasks unless the profile explicitly uses model roles.

6. **Context rollover and notebook**  
   `context_notes`, `new_context`, `_roll_over`, and `_make_room` are sophisticated. They help long runs, but they are also a source of state complexity and taint-latch edge cases.

7. **Plugin tier 2**  
   The in-process `register(api)` plugin system is powerful but large. If the priority is task success per line of code, keeping `create` and MCP is likely enough; tier-2 plugins can remain optional.

## Duplicated or overbuilt areas

- **MCP client logic is duplicated in skill scripts.** `app/agent/builtin_skills/mcp-builder/scripts/connections.py` reimplements connection handling already present in `app/agent/mcp.py`.
- **Tool execution has multiple overlapping paths.** `_run_foreground`, `_run_script`, `_run_created`, `_run_call`, and `codemode.run` all wrap sandbox execution with slightly different behavior. Consolidating these would reduce maintenance and failure modes.
- **Large policy/profile surface.** `load_profile()` supports many rarely used knobs (`prewalk_model`, `stream_rules`, `notes_context`, `isolate_writers`, `max_output_tokens`, etc.). A smaller default profile would make the harness easier to reason about.

## Measurement recommendation

For each proposal, run the same Terminal-Bench subset twice and record:

- task success,
- unreadable tool calls,
- streaming drops,
- `create` failures,
- skill calls,
- premature final answers,
- token usage and runtime.

The highest-value first experiments are proposals 1, 2, 3, and 4 because they are tiny and directly target measured failure modes.