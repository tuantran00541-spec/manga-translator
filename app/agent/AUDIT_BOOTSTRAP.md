# Security Audit: models.py, registry.py, isolate.py

Method: audited with self-bootstrapped shell/Python tooling (heredoc scripts in /tmp: audit_tools.py, scan.py, deep*.py).
All findings verified against source; line numbers refer to the files as present in this workspace.

---

## registry.py

### R1 [HIGH] Arbitrary code execution via plugin exec, user-scope plugins load without trust gate
- `load_plugins` (lines 209-252) execs plugin files through `kernel.load_module`, which does
  `exec(compile(data.decode("utf-8"), str(path), "exec"), module.__dict__)` (kernel.py line 191).
- Trust gate: only `scope == "workspace"` plugins are blocked when untrusted (lines 232-234).
  **User-scope plugins (home/.manga-agent/plugins/*.py) are always exec'd** unless `hold_user` is set
  (line 227). A malicious or compromised user plugin file has full interpreter access at startup.
- Mitigations present: M25 pins the exact bytes read for the digest to the bytes exec'd (lines 215-222,
  239), closing the read-then-exec TOCTOU; `hold_user` fails closed when the tamper baseline is
  unreadable (C6, lines 227-231). But the user-scope trust model is "user's own home dir", which is
  reasonable for a local tool, yet it means any write access to ~/.manga-agent/plugins = RCE at next
  startup. Worth documenting as an accepted risk; consider the same digest-trust gate for user scope.

### R2 [MEDIUM] ReDoS via user-supplied regex in profile.json `stream_rules`
- Line 76: `re.compile(str(rule.get("pattern")))` where the pattern comes from user profile JSON.
  Only syntactic validity is checked (re.error caught, line 78). A crafted pattern with catastrophic
  backtracking (e.g. `(a+)+$`) compiles fine and is later matched against model stream output,
  allowing a denial-of-service hang in the agent loop.
- Suggested fix: bound pattern length (e.g. <= 200 chars), and/or run the match with a timeout,
  and/or reject known-dangerous constructs. Note `message` is already bounded to 1000 chars (line 77).

### R3 [LOW] `replace` allows user profile to swap built-in tools for MCP tools
- Lines 105-107: user profile can map any built-in tool in `REPLACEABLE` (including `run_command`,
  `web_fetch`) to an `mcp__*` tool. This is a designed feature, but it means a compromised user
  profile can redirect shell/web tool calls to an MCP server of its choosing. Validation is sound
  (must start with `mcp__`, key must be in REPLACEABLE), so this is by design; flagging for awareness.

### R4 [LOW] Profile JSON parsed with json.loads on arbitrary file contents
- Line 46: `json.loads((base / name).read_text(...))`. Safe deserializer (no code execution), errors
  handled (OSError/ValueError, line 47). No issue beyond noting the file is trusted config.

### R5 [INFO] `untrusted_guard` and `isolate_writers` can be set to false via user profile
- Lines 100-103: user profile can disable the untrusted-output guard and writer isolation.
  User-scope only (workspace profile can only add to `disable` list, line 51), so this is a
  deliberate user choice. Document that these flags are security-relevant and reversible by user.

---

## isolate.py

### I1 [MEDIUM] Path traversal via unsanitized `rel` in merge() put()
- `merge()` (line 55) computes `rel` from `p.relative_to(copy).as_posix()` (line 58) or from `base`
  dict keys (line 84). `put()` (lines 60-68) does `target = folder / rel` and `shutil.copy2(source, target)`
  with **no validation that `rel` is a safe relative path** (no `..` check, no absolute-path check).
- In practice `rel` comes from `os.walk` under `copy` (so it is relative and `..`-free) or from `base`
  keys produced by `snapshot()` (also from `os.walk` under `root`). So traversal requires a corrupted
  `base` dict or a crafted `copy` tree. Risk is low in the normal flow but the function has no defense
  in depth: a `rel` of `../../etc/cron.d/evil` would write outside `root`.
- Suggested fix: in `put()`, assert `rel` has no `..` component and is not absolute
  (e.g. `Path(rel).is_absolute()` or `".." in Path(rel).parts` -> skip/raise).

### I2 [MEDIUM] `nick` is interpolated into a filesystem path without validation
- Line 82: `put(rel, path, root / CONFLICTS / nick)`. `nick` is a caller-supplied string used directly
  as a path component. A `nick` of `../../foo` or `..\..` would place conflict files outside the
  intended `.agent-conflicts` directory. No sanitization of `nick` exists in this file.
- Suggested fix: validate `nick` against a strict pattern (e.g. `^[a-z0-9_-]{1,32}$`) at the `merge()`
  entry, or sanitize with `Path(nick).name`.

### I3 [LOW] TOCTOU between stat() and copy2() in snapshot()
- Lines 39-50: `path.stat().st_size` is checked, then `shutil.copy2(path, target)`. A file could change
  size (or be swapped for a symlink) between the two calls. The size limit (5 MB/file, 200 MB total)
  could be bypassed, or a symlink could be followed by copy2 after the `is_symlink()` check in
  `_files()` (line 22) passed. Low risk in a local single-user context; the symlink check in `_files()`
  already mitigates the common case.

### I4 [LOW] SHA-1 used for content hashing
- Line 28: `hashlib.sha1(path.read_bytes())`. SHA-1 is deprecated for security contexts, but here it is
  used as a change-detection fingerprint, not a security primitive (collision resistance against an
  adversary is not required). Acceptable; SHA-256 would be a free upgrade for consistency with
  registry.py which uses SHA-256 (line 198).

### I5 [INFO] os.walk does not follow symlinked directories
- Line 17: `os.walk(top)` defaults to `followlinks=False`, so symlinked directories are not traversed.
  Combined with the per-file `is_symlink()` check (line 22), symlink-based escape is well mitigated.

---

## models.py

### M1 [MEDIUM] ReDoS via user-supplied regex keys in `_match()`
- Line 28: `re.search(key, model, re.I)` where `key` comes from the `overrides` dict, which is fed by
  user profile JSON (`profile["models"]` / `profile["prices"]`, registry.py lines 65-66). A crafted
  key with catastrophic backtracking (e.g. `(a+)+$`) is compiled and matched against the model name
  string on every `quirks()` / `prices()` call. Model names are short, which limits damage, but the
  pattern is matched with `re.search` on every call and there is no length bound on the key.
- `re.error` is caught (line 30), so only syntactically valid but semantically dangerous patterns pass.
- Suggested fix: bound key length, or precompile and cache with a timeout guard.

### M2 [LOW] No length bound on override keys or model string
- The `model` parameter and override dict keys are used directly in regex operations with no length
  limit. Combined with M1, a very long model string or key amplifies the ReDoS window.

### M3 [INFO] No other issues
- No file I/O, no exec, no network, no deserialization. The module is a pure lookup table with regex
  matching. The hardcoded price/quirk tables are data, not secrets.

---

## Summary table

| ID | File | Severity | Issue |
|----|------|----------|-------|
| R1 | registry.py | HIGH | User-scope plugins exec'd without trust gate (RCE via ~/.manga-agent/plugins) |
| R2 | registry.py | MEDIUM | ReDoS via profile.json stream_rules pattern |
| R3 | registry.py | LOW | replace mechanism redirects built-in tools to MCP (by design) |
| R4 | registry.py | LOW | json.loads on profile files (safe deserializer, noted) |
| R5 | registry.py | INFO | untrusted_guard / isolate_writers user-disablable (by design) |
| I1 | isolate.py | MEDIUM | Path traversal via unsanitized rel in merge() put() |
| I2 | isolate.py | MEDIUM | nick interpolated into path without validation |
| I3 | isolate.py | LOW | TOCTOU between stat and copy2 in snapshot() |
| I4 | isolate.py | LOW | SHA-1 for content fingerprint (not a security primitive) |
| I5 | isolate.py | INFO | os.walk followlinks=False + is_symlink check (good) |
| M1 | models.py | MEDIUM | ReDoS via user-supplied regex keys in _match() |
| M2 | models.py | LOW | No length bound on regex inputs |
| M3 | models.py | INFO | No other issues |

## Top recommended fixes (priority order)
1. **I2**: validate `nick` in `merge()` with a strict pattern before path interpolation.
2. **I1**: in `put()`, reject `rel` containing `..` or absolute paths.
3. **R2**: bound `stream_rules` pattern length (e.g. 200 chars) in `load_profile`.
4. **M1**: bound override key length in `_match()` or precompile with a size guard.
5. **R1**: document the user-scope plugin trust model; consider extending digest-trust to user scope.
