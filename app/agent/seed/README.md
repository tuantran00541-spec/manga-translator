# Seed — the tool-less harness (prototype)

An experiment in **"everything is everything"**: the harness ships with **no
domain tools at all** — no `read_file`, no shell, no web search. The model
starts with exactly one tool, `define`, and must write every capability it
needs as Python code against a tiny fixed API.

## The fixed core (the only things the model cannot rewrite)

1. **`define`** — the seed tool. Registers a capability: name, description,
   JSON-schema parameters, Python source.
2. **`registry`** — static checks at define-time (import whitelist of pure
   stdlib modules, denylist of dangerous builtins, no dunder attribute
   access), restricted `__builtins__` at call-time, per-call timeout.
3. **`harness`** — the world-access API injected into every capability:
   `read` / `write` / `ls` (workspace-contained), `run` (shell under the
   OS sandbox: Landlock/Seatbelt, no network, secrets scrubbed),
   `fetch` (size-capped HTTPS), `spawn` (sandboxed subprocess with
   line-based I/O — this is how a model-written MCP server gets launched),
   `log` (audit notes).
4. **The audit log + the approval gate.** Every define and every call is
   logged. `persist: true` on `define` asks the human before a capability
   is saved to `~/.seed/caps/` for future sessions.

Tool, plugin, skill, MCP client/server: in Seed these are **one concept** —
a *capability*. An MCP server is a file the model wrote plus a capability
that spawns it and speaks JSON-RPC through `harness.spawn`.

## Try it (mock, no API key)

```bash
python app/agent/seed/mocktest.py
```

A scripted mock model bootstraps itself (defines `read_file`/`write_file`,
does a task), then probes the guardrails: `import os`, `open()`, dunder
escapes and `/etc/passwd` reads must all be refused.

## Try it (real model)

```bash
SEED_API_KEY=... python -m app.agent.seed.demo \
    --provider gemini --model gemini-2.0-flash \
    --workspace /tmp/seed-ws \
    --task "Read notes.txt, summarize it in 3 lines, write summary.txt"
```

Watch the model's tool surface grow: it should define 2–4 capabilities
first, then use them. `--yes` auto-approves persistence.

## What's deliberately missing (next steps)

- **Capability tests before activation**: run a new capability against a
  fixture before the model may call it.
- **Capability marketplace / sharing**: export/import capability packs.
- **MCP servers as first-class citizens**: `harness.spawn` already allows
  it; a `mcp_serve` helper could expose the session's capabilities outward.
- **Richer approval**: per-capability trust tiers instead of just
  ephemeral/persisted.
- **The audit log as a capability**: deliberately NOT done — the log is
  part of the fixed core, otherwise the model could rewrite its own history.

## Honest limitations

- The restricted execution is **prototype-grade**, not a security boundary
  against a truly adversarial model: the real safety comes from the OS
  sandbox on `h.run`/`h.spawn`, workspace containment in `harness`, and
  the human approval gate. Do not expose Seed to untrusted models.
- Capability code runs in-process; a hung capability is abandoned after
  the timeout but its thread lingers.
- `h.spawn`ed processes are sandboxed without network in v0.
