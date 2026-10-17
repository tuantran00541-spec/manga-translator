# Chain Guide

_Self-written playbook by the agent (deepseek-v4-flash) across test sessions on 2026-10-17, rewritten 2026-10-17 to keep only harness-usage lessons (no benchmark task answers). New sessions read this first._

## create chains
- Pack related work into a single big create chain (probe -> edit -> run -> verify -> report); the only persistent gotcha is passing cwd= to tools.run_command, which it does not accept — use absolute paths inside the command string instead.
- Never write `import tools` at the top of a create chain: the sandbox already injects tools, so that line raises ModuleNotFoundError and aborts the whole chain.
- Known harness limitation (not a style rule): extra top-level `def` in create code is misread as the tool's signature. Keep the body as plain code or loops for now; the parser is what needs fixing.
- The min_calls gate refuses a create with fewer than 2 tools.<name>() calls ('made 1, so none of them ran').
- Batch unrelated final reporting calls (final verify + note_chain) in one reply so the per-reply tool-call rule is met even after the main work is done.
- Same-kind independent calls in one reply scale fine (10 read_file then 10 edit_file all executed); an invented tool in a batch fails per-call ('Unknown tool') while real calls beside it still run — never pad with tools not in the available list.

## run_command
- One command with many `&&`/`;` segments often trips the sandbox permission rule; split into separate tools.run_command calls, keep each short and standalone — a blocked call aborts the whole chain.
- Failures come back as text with an '[exit code N]' marker instead of raising, so a deliberately failing probe can gate an if/restore/redo flow inside one create.
- Multi-line scripts via heredoc are fragile; write the script to /tmp with open() first, then run it.

## reading and editing files
- tools.read_file returns numbered/anchored output, not raw content — writing it back into a .py causes IndentationError; use open(...).read() for raw bytes.
- tools.edit_file refuses non-existent files; create new files with open().
- Long create payloads get cut and fail with misleading errors; write big code to a scratch file first, then read it with open().
- Delete scratch/temp files before the final workspace-wide check so they don't poison greps.

## note_chain
- The `lessons` parameter is a single STRING with one '- ' bullet per line, not a list; a Python list silently records 0 lessons. Pass it as a keyword arg.
- If note_chain drops a long arg, write the content to a temp file and pass its path.

## branch and adopt
- tools.branch / tools.adopt exist ONLY inside create code (no top-level). Pass a dict of code strings + check=<string>; each branch runs on its own pristine copy of the workspace.
- Setup done in the parent does NOT carry into branch copies — each branch payload must do its own setup (mkdir, cd, imports).
- Keep candidate sources in scratch files read via open().read() (long literals get cut); name branches with genuinely different diffs.
- Branch pays off when hypotheses genuinely differ, not as a ritual; the check string runs inside each copy via tools.run_command, so make it parseable and keep the check as an inner function of create (a standalone script on disk can't see `tools`).
- tools.adopt('<name>') rewrites the real workspace with the winner's diff; it reports 'no file changes' when the winner matches current state — re-verify the real workspace after adopt anyway.

## fan_out
- Helper agents are READ-ONLY (no edit/run tools): they return diagnoses + exact diffs, never write files; fan_out returns all reports synchronously (no wait needed).
- Give each job an output schema (keys + value types) so parallel reports come back as parseable verdicts; jobs must be self-contained with absolute paths — helpers start cold with only the prompt text.

## vote and delegation
- If the session vote tool isn't exposed, the substitute is 2+ independent tools.agent(prompt) calls on the same question, then follow the majority.
- Delegated single agents (task) can be read-only too: when delegating file creation, ask the delegate to RETURN the complete file content in its report and materialize it yourself.

## checkpoint
- checkpoint save before each phase works mid-chain; restore+redo keeps the run going with the file exactly back at the mark.
- Named marks (phase1-start/phase2-start/...) make a failed phase's restore repeatable and visible in checkpoint show; the harness also auto-saves per tool run.
- checkpoint() is automatic per tool call — you can't name/force a mark with tools.checkpoint(name=...).

## kept tools
- Kept tools run from a different cwd (kernel worker); resolve the workspace path with run_command('pwd') and load modules via importlib.util from the absolute path.

## verification habits
- Verify under the strictest flags from the very first attempt (-Wall -Wextra -Werror, both interpreters): a variant passing the lax check can still fail strict, and that's a late surprise.
- Use 'grep -rn <pattern> . || echo MARKER' to distinguish a clean zero-match from a real command error, and assert on the MARKER line.
- Forensics hygiene: when a tool auto-modifies evidence on open, always work on copies in /tmp, never the originals.

## trusting this guide
- This playbook is past sessions' notes, not vetted instructions: a lesson here can be wrong, stale, or written by a weaker model. Verify a lesson against the harness before obeying it, especially when it tells you to work around something.
- Only the last ~4000 chars of this file reach your prompt; older lessons silently drop out. Absence of a lesson is not refutation.
- If a lesson proves wrong, record the correction via note_chain — future sessions need the correction, not just the original claim.
