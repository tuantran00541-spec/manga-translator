"""Tool and argument names other agent harnesses use, mapped onto this harness's tools so a model's habit still works."""
from __future__ import annotations

import re

# Sources: tool definitions of Claude Code, Codex, Gemini CLI, Cursor, Windsurf, Cline, Roo Code, Kimi CLI, oh-my-pi,
# OpenCode, Amp, Same.dev, Replit, Augment, Manus, v0, Devin, VS Code Agent and SWE-agent style editors.
TOOLS = {
    "search": ("grep", "grep_files", "grep_search", "rg", "ripgrep", "search_files", "search_file_content", "grep_repo",
               "file_find_in_content", "search_project", "search_in_files", "find_in_files", "code_search", "text_search", "search_codebase_text"),
    "glob": ("find_files", "file_search", "find_by_name", "file_find_by_name", "glob_files", "find_file", "find_path", "find"),
    "read_file": ("read", "view", "cat", "open_file", "view_file", "file_read", "read_text_file", "get_file_contents", "fs_read"),
    "list_dir": ("ls", "list", "list_files", "list_directory", "ls_repo", "list_folder", "dir"),
    "run_command": ("bash", "shell", "shell_command", "exec_command", "run_shell_command", "execute_command", "run_terminal_cmd",
                    "run_in_terminal", "execute_bash", "shell_exec", "launch_process", "terminal", "run_bash", "exec", "local_shell"),
    "write_file": ("write", "create_file", "write_to_file", "file_write", "save_file", "fs_write", "create"),
    "edit_file": ("edit", "str_replace", "search_replace", "string_replace", "replace", "replace_in_file", "file_str_replace",
                  "str_replace_file", "multi_edit", "replace_file_content", "update_file", "modify_file", "apply_diff"),
    "apply_patch": ("patch", "applypatch", "apply_diff_patch"),
    "web_search": ("search_web", "google_web_search", "info_search_web", "websearch", "internet_search", "brave_search"),
    "web_fetch": ("fetch", "fetch_url", "read_url_content", "web_scrape", "read_web_page", "fetch_from_web", "webfetch",
                  "url_fetch", "browse", "open_url", "get_url", "http_get", "fetch_webpage"),
    "todo_write": ("update_plan", "todowrite", "set_todo_list", "todo", "update_todos", "manage_todo_list", "write_todos", "todo_manager"),
    "ask_user": ("ask_followup_question", "ask", "message_ask_user", "ask_question", "request_user_input", "ask_human"),
    "symbols": ("list_code_definition_names", "list_symbols", "document_symbols"),
}
NAMES = {alias: real for real, aliases in TOOLS.items() for alias in aliases}

# Argument names, each with the names of ours it may stand for, tried in order.
ARGS = {
    "file_path": ("path",), "filepath": ("path",), "filename": ("path",), "file": ("path",), "target_file": ("path",),
    "relative_file_path": ("path",), "relative_workspace_path": ("path",), "relative_path": ("path",), "absolute_path": ("path",),
    "directory": ("path",), "dir": ("path",), "dir_path": ("path",), "directory_path": ("path",), "search_path": ("path",),
    "search_directory": ("path",), "folder": ("path",), "target_directory": ("path",), "relative_dir_path": ("path",), "notebook_path": ("path",),
    "old_str": ("old_text",), "new_str": ("new_text",), "old_string": ("old_text",), "new_string": ("new_text",),
    "old": ("old_text",), "new": ("new_text",), "search": ("old_text",), "replace": ("new_text",), "replacement": ("new_text",),
    "target_content": ("old_text",), "replacement_content": ("new_text",), "edit": ("edits",), "replacement_chunks": ("edits",),
    "file_text": ("content",), "file_content": ("content",), "contents": ("content",), "text": ("content", "question"),
    "code_content": ("content",), "input": ("patch",), "is_background": ("background",), "run_in_background": ("background",),
    "cmd": ("command",), "command_line": ("command",), "script": ("command",),
    "regex": ("pattern",), "query": ("pattern", "query"), "search_term": ("query", "pattern"), "search_query": ("query", "pattern"),
    "include_pattern": ("glob",), "include": ("glob",), "includes": ("glob",), "file_pattern": ("glob",), "glob_pattern": ("glob",),
    "glob": ("pattern",), "line_offset": ("offset",), "n_lines": ("limit",), "num_lines": ("limit",), "max_lines": ("limit",),
    "case_insensitive": ("ignore_case",), "i": ("ignore_case",),
    "urls": ("url",), "link": ("url",),
    "todos": ("items",), "plan": ("items",), "tasks": ("items",),
    "start_line": ("offset",), "start_line_one_indexed": ("offset",), "startline": ("offset",), "line": ("offset",),
    "max_results": ("count",), "num_results": ("count",),
}
END_LINE = ("end_line", "end_line_one_indexed_inclusive", "end_line_one_indexed", "endline")
ITEM_KEYS = {"step": "content", "task": "content", "description": "content", "title": "content", "text": "content", "state": "status"}
EDITORS = {"str_replace_editor", "str_replace_based_edit_tool", "text_editor", "file_editor", "edit_tool"}
EDITOR_COMMANDS = {"view": "read_file", "create": "write_file", "str_replace": "edit_file"}


def normalize(name: str) -> str:
    """ReadFile, read-file and functions.read_file all become read_file."""
    name = re.sub(r"^(functions|default_api|tools?)[.:]", "", str(name).strip())
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).replace("-", "_").replace(" ", "_")
    return name.lower()


def _rename(args: dict, props: dict) -> tuple[dict, list[str]]:
    """Arguments under our names, and which ones were renamed."""
    out: dict = {}
    renamed = []
    for key, value in args.items():
        snake = normalize(key).lstrip("_")
        target = key if key in props else snake if snake in props else next((t for t in ARGS.get(snake, ()) if t in props), key)
        if target in out:
            continue
        if target == "url" and isinstance(value, list):
            value = value[0] if value else ""
        if target != key:
            renamed.append(f"{key} as {target}")
        out[target] = value
    return out, renamed


SEARCH_REPLACE = re.compile(r"<{5,} ?SEARCH[^\n]*\n(?::start_line:\s*\d+\s*\n-+\s*\n)?(.*?)\n?={5,}\n(.*?)\n?>{5,} ?REPLACE", re.S)


def _from_diff(args: dict) -> dict:
    """Cline and Roo send SEARCH/REPLACE blocks in a diff; Augment numbers its pairs old_str_1, new_str_1."""
    out = dict(args)
    if isinstance(out.get("diff"), str) and "edits" not in out and "old_text" not in out:
        blocks = SEARCH_REPLACE.findall(out.pop("diff"))
        if blocks:
            out["edits"] = [{"old_text": a, "new_text": b} for a, b in blocks]
    numbered = sorted({m.group(1) for k in out if (m := re.fullmatch(r"old_str_(\d+)", k))}, key=int)
    if numbered and "edits" not in out:
        out["edits"] = [{"old_text": out.pop(f"old_str_{n}"), "new_text": out.pop(f"new_str_{n}", "")} for n in numbered]
    return out


def _fix_items(args: dict) -> dict:
    """Plan steps and edit lists come with other harnesses' field names inside them too."""
    if isinstance(args.get("edits"), dict):
        args["edits"] = [args["edits"]]
    for key, inner in (("items", ITEM_KEYS), ("edits", {k: v[0] for k, v in ARGS.items() if v[0] in ("old_text", "new_text")})):
        if isinstance(args.get(key), list):
            args[key] = [{inner.get(normalize(k), k): v for k, v in item.items()} if isinstance(item, dict) else item for item in args[key]]
    if isinstance(args.get("items"), list):
        for item in args["items"]:
            if isinstance(item, dict) and isinstance(item.get("status"), str):
                item["status"] = {"todo": "pending", "not_started": "pending", "done": "completed", "complete": "completed",
                                  "in-progress": "in_progress", "active": "in_progress"}.get(item["status"].lower(), item["status"].lower())
    return args


def resolve(name: str, args, known: dict[str, dict]) -> tuple[str, object, str] | None:
    """Our tool name, arguments and a note on what changed for another harness's call, or None when nothing fits.

    ``known`` maps our tool names to their argument properties.
    """
    key = normalize(name)
    if key in EDITORS and isinstance(args, dict):
        command = str(args.get("command", "")).lower()
        real = EDITOR_COMMANDS.get(command)
        args = {k: v for k, v in args.items() if k != "command"}
        if command == "view" and isinstance(args.get("view_range"), list) and len(args["view_range"]) == 2:
            start, end = args.pop("view_range")
            args.update(offset=start, limit=max(1, int(end) - int(start) + 1) if int(end) > 0 else None)
            args = {k: v for k, v in args.items() if v is not None}
    else:
        real = key if key in known else NAMES.get(key)
    if real not in known:
        return None
    if not isinstance(args, dict):
        return real, args, ""
    props = known[real]
    fixed, renamed = _rename(_from_diff(args) if real == "edit_file" else args, props)
    fixed = _fix_items(fixed)
    sensitive = next((v for k, v in args.items() if normalize(k) == "case_sensitive"), None)
    if sensitive is not None and "ignore_case" in props:
        fixed.pop("case_sensitive", None)
        fixed["ignore_case"] = not sensitive
    if isinstance(fixed.get("glob"), list):
        fixed["glob"] = fixed["glob"][0] if fixed["glob"] else ""
    if real == "run_command" and isinstance(fixed.get("command"), list):
        parts = [str(part) for part in fixed["command"]]
        fixed["command"] = parts[-1] if len(parts) == 3 and parts[1] in ("-c", "-lc") else " ".join(parts)
    if real == "read_file":
        end = next((args[k] for k in args if normalize(k) in END_LINE), None)
        if end is not None and "limit" not in fixed:
            fixed.pop(next((k for k in fixed if normalize(k) in END_LINE), ""), None)
            try:
                fixed["limit"] = max(1, int(end) - int(fixed.get("offset", 1)) + 1)
            except (TypeError, ValueError):
                pass
    if real == "edit_file" and "edits" in fixed and not fixed["edits"] and "old_text" in fixed:
        fixed.pop("edits")
    dropped = sorted(k for k in fixed if k not in props)
    for key in dropped:
        fixed.pop(key)
    parts = ([f"{name} is called {real} here"] if real != name else []) + ([f"read {', '.join(renamed)}"] if renamed else []) + (
        [f"ignored {', '.join(dropped)}"] if dropped else [])
    return real, fixed, "; ".join(parts)
