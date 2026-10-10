// Agent: a coding agent on any configured model, read like a conversation with its tool calls folded into one-line groups.
(() => {
  const POLL_MS = 700;
  const HEAD = { "X-Manga-Agent": "1" };
  const VERBS = {
    "": "Unreadable call",
    list_dir: "List", read_file: "Read", search: "Search", glob: "Find files", write_file: "Write", edit_file: "Edit",
    apply_patch: "Patch", edit_lines: "Edit lines", run_command: "Run", web_fetch: "Fetch", skill: "Skill", todo_write: "Plan",
    task: "Helper", spawn_agent: "Spawn agent", wait_agent: "Wait for agents", send_input: "Message agent", close_agent: "Close agent", delegate: "External agent", memory: "Memory", ask_user: "Ask you", exit_plan_mode: "Plan", goal_done: "Goal done",
    symbols: "Symbols", web_search: "Web search", web_download: "Download", view_image: "View image", tool_search: "Find tools", run_script: "Script", fan_out: "Fan out",
    job_output: "Job output", job_input: "Type into job", job_stop: "Stop job", oracle: "Ask oracle", context_notes: "Notes", new_context: "New context",
    schedule_create: "Schedule", schedule_list: "Schedules", schedule_delete: "Cancel schedule",
  };
  // How each kind of call is counted in a group's one-line summary: verb, one, many.
  const TALLY = {
    run_command: ["ran", "command", "commands"], read_file: ["read", "file", "files"], edit_file: ["edited", "file", "files"], write_file: ["edited", "file", "files"],
    apply_patch: ["edited", "file", "files"], edit_lines: ["edited", "file", "files"], spawn_agent: ["started", "helper", "helpers"], wait_agent: ["waited on", "helper", "helpers"],
    send_input: ["messaged", "helper", "helpers"], delegate: ["delegated", "job", "jobs"], close_agent: ["closed", "helper", "helpers"], memory: ["saved", "memory", "memories"],
    list_dir: ["listed", "folder", "folders"], search: ["searched", "time", "times"], glob: ["searched", "time", "times"], symbols: ["searched", "time", "times"],
    web_fetch: ["fetched", "page", "pages"], skill: ["opened", "skill", "skills"], task: ["ran", "helper", "helpers"], mcp: ["called", "MCP tool", "MCP tools"],
    run_script: ["ran", "script", "scripts"], fan_out: ["fanned out", "batch", "batches"], job_input: ["typed into", "job", "jobs"], job_output: ["read", "job", "jobs"], job_stop: ["stopped", "job", "jobs"],
    oracle: ["asked", "oracle", "oracles"], schedule_create: ["scheduled", "run", "runs"], web_search: ["searched the web", "time", "times"],
  };
  const SANDBOX_TEXT = {
    landlock: "Lệnh chạy trong sandbox Landlock của Linux.", seatbelt: "Lệnh chạy trong sandbox Seatbelt của macOS.",
    none: "Máy này không có sandbox cho lệnh, nên lệnh luôn chờ bạn duyệt (trừ khi chọn Tự làm hết).",
  };
  const EXAMPLES = [
    "Đọc app/inpaint/clustering.py rồi viết test cho split_cluster_lines",
    "Chạy toàn bộ test và sửa những test đang hỏng",
    "/plan Thêm test cho phần chia lát ảnh dài",
    "/init",
  ];
  const $ = (id) => document.getElementById(id);
  let session = null;
  let lastSeq = 0;
  let pollTimer = null;
  let lastUsage = null;
  let loaded = false;
  let commands = [];
  let group = null;

  function store(action, key, value) {
    try {
      if (action === "get") return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (_) {}
    return null;
  }

  async function call(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { ...HEAD, ...(options.headers || {}) } });
    const data = await window.parseApiResponse(response);
    if (!response.ok) throw new Error(window.getErrorMessage(response.status, data));
    return data;
  }

  const post = (url, body, method = "POST") => call(url, {
    method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(label, cls, onClick) {
    const node = el("button", cls, label);
    node.type = "button";
    node.addEventListener("click", onClick);
    return node;
  }

  function nearBottom() {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  }

  function follow(stick) {
    if (stick) window.scrollTo({ top: document.documentElement.scrollHeight });
  }

  // `code`, **bold** and [web links](https://…) inside a line, built as nodes so model text never becomes markup.
  function inline(text) {
    const out = document.createDocumentFragment();
    String(text).split(/(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/).forEach((part) => {
      const link = /^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/.exec(part);
      if (/^`[^`]+`$/.test(part)) out.append(el("code", "", part.slice(1, -1)));
      else if (/^\*\*[^*]+\*\*$/.test(part)) out.append(el("strong", "", part.slice(2, -2)));
      else if (link) {
        const a = el("a", "", link[1]);
        a.href = link[2];
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        out.append(a);
      } else if (part) out.append(document.createTextNode(part));
    });
    return out;
  }

  // A small Markdown reading: fenced code, headings, bullet and numbered lists, paragraphs.
  function prose(text) {
    const wrap = el("div", "agent-prose");
    String(text || "").split(/```[\w-]*\n?/).forEach((part, i) => {
      if (i % 2) {
        wrap.append(el("pre", "agent-pre", part.replace(/\n$/, "")));
        return;
      }
      part.replace(/^(#{1,4} .*)\n(?=\S)/gm, "$1\n\n").split(/\n{2,}/).forEach((block) => {
        const lines = block.split("\n").filter((l) => l.trim());
        if (!lines.length) return;
        if (lines.every((l) => /^\s*[-*] /.test(l)) || lines.every((l) => /^\s*\d+[.)] /.test(l))) {
          const list = el(/^\s*\d/.test(lines[0]) ? "ol" : "ul");
          lines.forEach((l) => { const item = el("li"); item.append(inline(l.replace(/^\s*(?:[-*]|\d+[.)]) /, ""))); list.append(item); });
          wrap.append(list);
        } else if (/^#{1,4} /.test(lines[0]) && lines.length === 1) {
          const head = el("p", "agent-prose-head");
          head.append(inline(lines[0].replace(/^#+ /, "")));
          wrap.append(head);
        } else {
          const para = el("p");
          lines.forEach((l, n) => { if (n) para.append(el("br")); para.append(inline(l)); });
          wrap.append(para);
        }
      });
    });
    return wrap;
  }

  function isMcp(name) {
    return /^mcp__/.test(name || "");
  }

  function target(c) {
    const a = c.args || {};
    if (c.name === "run_command") return `${a.command || ""}${a.outside_sandbox ? "  (outside the sandbox)" : ""}`;
    if (c.name === "search" || c.name === "glob") return `${a.pattern || ""}${a.path && a.path !== "." ? `  (${a.path})` : ""}`;
    if (c.name === "web_fetch") return a.url || "";
    if (c.name === "web_search") return a.query || "";
    if (c.name === "web_download") return `${a.url || ""} to ${a.path || ""}`;
    if (c.name === "symbols") return a.name || a.path || ".";
    if (c.name === "view_image") return a.path || "";
    if (c.name === "tool_search") return a.query || "";
    if (c.name === "task") return a.description || "";
    if (c.name === "spawn_agent") return `${a.agent || "explore"}: ${String(a.message || "").slice(0, 80)}`;
    if (c.name === "wait_agent") return (a.ids || []).join(", ") || "all";
    if (c.name === "send_input") return `${a.id || ""}: ${String(a.message || "").slice(0, 60)}`;
    if (c.name === "close_agent") return a.id || "";
    if (c.name === "skill") return a.name || "";
    if (c.name === "todo_write") return `${(a.items || []).length} items`;
    if (c.name === "ask_user") return a.question || "";
    if (c.name === "delegate") return `${a.agent || ""}: ${String(a.prompt || "").slice(0, 80)}`;
    if (c.name === "memory") return `${a.action || ""} ${a.text || ""}`.trim();
    if (c.name === "goal_done") return a.summary || "";
    if (c.name === "run_script") return String(a.code || "").split("\n").find((l) => l.trim()) || "";
    if (c.name === "fan_out") return `${(a.jobs || []).length} jobs${a.schema ? " with a schema" : ""}`;
    if (c.name === "job_input") return `${a.id || ""}: ${JSON.stringify(String(a.chars || "")).slice(0, 60)}`;
    if (c.name === "job_output" || c.name === "job_stop" || c.name === "schedule_delete") return a.id || "";
    if (c.name === "oracle") return String(a.question || "").slice(0, 80);
    if (c.name === "schedule_create") return `in ${a.in_minutes || "?"} min${a.every_minutes ? `, every ${a.every_minutes} min` : ""}: ${String(a.prompt || "").slice(0, 50)}`;
    if (c.name === "apply_patch") return (String(a.patch || "").match(/^\*\*\* (?:Add|Update|Delete) File: .+$/gm) || []).map((l) => l.split(": ")[1]).join(", ");
    if (isMcp(c.name)) return c.name.replace(/^mcp__/, "").replace("__", ": ");
    return a.path || ".";
  }

  // An edit_file call as old/new pairs, whether it used old_text/new_text or an edits list.
  function editPairs(a) {
    return Array.isArray(a.edits) && a.edits.length ? a.edits : [{ old_text: a.old_text, new_text: a.new_text }];
  }

  function diffStats(c) {
    const a = c.args || {};
    const count = (s) => (String(s || "").match(/\n/g) || []).length + (s ? 1 : 0);
    if (c.name === "edit_file") return editPairs(a).reduce((t, e) => [t[0] + count((e || {}).new_text), t[1] + count((e || {}).old_text)], [0, 0]);
    if (c.name === "write_file") return [count(a.content), 0];
    if (c.name === "apply_patch") {
      const lines = String(a.patch || "").split("\n").filter((l) => !l.startsWith("***"));
      return [lines.filter((l) => l.startsWith("+")).length, lines.filter((l) => l.startsWith("-")).length];
    }
    return null;
  }

  function stats(added, removed) {
    const box = el("span", "agent-stats");
    box.append(el("span", "agent-add", `+${added}`), el("span", "agent-del", `−${removed}`));
    return box;
  }

  function newGroup() {
    const box = el("details", "agent-group");
    const summary = el("summary");
    const label = el("span", "agent-group-label");
    summary.append(label);
    const list = el("div", "agent-group-list");
    box.append(summary, list);
    $("agent-log").append(box);
    group = { box, summary, label, list, tally: {}, added: 0, removed: 0, failed: 0 };
    return group;
  }

  function refreshGroup() {
    // Tools that read the same way (search, glob, symbols) add up into one phrase.
    const merged = new Map();
    Object.entries(group.tally).forEach(([key, n]) => {
      const words = TALLY[key] || ["used", key, key];
      const id = words.join("|");
      merged.set(id, { words, n: (merged.get(id)?.n || 0) + n });
    });
    const parts = [...merged.values()].map(({ words: [verb, one, many], n }) => `${verb} ${n} ${n === 1 ? one : many}`);
    const text = parts.join(", ") || "Updated the plan";
    group.label.textContent = text.charAt(0).toUpperCase() + text.slice(1);
    group.summary.querySelector(".agent-stats")?.remove();
    group.summary.querySelector(".agent-fail")?.remove();
    if (group.added || group.removed) group.summary.append(stats(group.added, group.removed));
    if (group.failed) group.summary.append(el("span", "agent-fail", `${group.failed} failed`));
  }

  function addRow(c) {
    if (c.name === "ask_user" || c.name === "exit_plan_mode") return;
    const g = group || newGroup();
    const key = isMcp(c.name) ? "mcp" : c.name;
    if (TALLY[key]) g.tally[key] = (g.tally[key] || 0) + 1;
    const row = el("details", "agent-row");
    row.dataset.callId = c.id;
    const head = el("summary");
    head.append(el("span", "agent-verb", isMcp(c.name) ? "MCP" : VERBS[c.name] || c.name), el("code", "agent-target", target(c)));
    const diff = diffStats(c);
    if (diff) {
      head.append(stats(diff[0], diff[1]));
      g.added += diff[0];
      g.removed += diff[1];
    }
    row.append(head);
    const a = c.args || {};
    const body = el("div", "agent-row-body");
    if (c.name === "edit_file") editPairs(a).forEach((e) => body.append(el("pre", "agent-pre agent-old", (e && e.old_text) || ""), el("pre", "agent-pre agent-new", (e && e.new_text) || "")));
    else if (c.name === "write_file") body.append(el("pre", "agent-pre agent-new", a.content || ""));
    else if (c.name === "apply_patch") body.append(patchView(a.patch));
    else if (c.name === "edit_lines") body.append(editLinesView(a.edits));
    else if (c.name === "task") body.append(el("pre", "agent-pre", a.prompt || ""));
    else if (c.name === "run_script") body.append(el("pre", "agent-pre", a.code || ""));
    else if (c.name === "fan_out") body.append(el("pre", "agent-pre", (Array.isArray(a.jobs) ? a.jobs : []).map((j, n) => `${n + 1}. ${j.description || ""}\n${j.prompt || ""}`).join("\n\n") + (a.schema ? `\n\nschema: ${JSON.stringify(a.schema)}` : "")));
    else if (c.name === "oracle") body.append(el("pre", "agent-pre", a.question || ""));
    else if (c.name === "spawn_agent" || c.name === "send_input") body.append(el("pre", "agent-pre", a.message || ""));
    else if (isMcp(c.name)) body.append(el("pre", "agent-pre", JSON.stringify(a, null, 2)));
    row.append(body);
    g.list.append(row);
    refreshGroup();
  }

  function editLinesView(edits) {
    const pre = el("pre", "agent-pre");
    (Array.isArray(edits) ? edits : []).forEach((edit, i) => {
      if (i) pre.append("\n");
      pre.append(el("span", "agent-patch-file", `${edit.op} ${edit.anchor || ""}${edit.end ? `..${edit.end}` : ""}`));
      if (edit.text) pre.append(`\n${edit.text}`);
    });
    return pre;
  }

  function patchView(text) {
    const pre = el("pre", "agent-pre");
    String(text || "").split("\n").forEach((line, i) => {
      if (i) pre.append("\n");
      const cls = line.startsWith("***") ? "agent-patch-file" : line.startsWith("+") ? "agent-patch-add" : line.startsWith("-") ? "agent-patch-del" : "";
      pre.append(cls ? el("span", cls, line) : document.createTextNode(line));
    });
    return pre;
  }

  function render(event) {
    const log = $("agent-log");
    log.querySelector(".agent-empty")?.remove();
    if (event.type === "user") {
      group = null;
      const bubble = el("div", "agent-user", event.text);
      if (event.refs?.length) bubble.append(el("span", "agent-refs", `Đính kèm: ${event.refs.join(", ")}`));
      if (event.queued) bubble.append(el("span", "agent-refs", "Đã xếp hàng, agent sẽ đọc ở bước kế tiếp"));
      log.append(bubble);
    } else if (event.type === "assistant") {
      if (event.reasoning) {
        const think = el("details", "agent-think");
        think.append(el("summary", "", "Thinking"), el("pre", "agent-pre", event.reasoning));
        log.append(think);
      }
      if (event.text) {
        group = null;
        log.append(prose(event.text));
        if (lastUsage && (lastUsage.prompt_tokens || lastUsage.completion_tokens)) {
          const fmt = (n) => (n || 0).toLocaleString("en-US");
          log.append(el("p", "agent-usage", `${fmt(lastUsage.prompt_tokens)} in / ${fmt(lastUsage.completion_tokens)} out`));
        }
      }
      (event.calls || []).forEach(addRow);
    } else if (event.type === "tool") {
      const row = log.querySelector(`.agent-row[data-call-id="${CSS.escape(event.id)}"]`);
      if (row) {
        row.classList.add(event.ok ? "agent-row-ok" : "agent-row-failed");
        row.querySelector(".agent-row-body").append(el("pre", "agent-pre agent-out", event.output));
        if (!event.ok && group) {
          group.failed += 1;
          refreshGroup();
        }
      }
    } else if (event.type === "approval") {
      showApprovalCard(event.call);
    } else if (event.type === "key_switch_ask") {
      showKeySwitchedModal(event.text);
    } else if (event.type === "key_switched") {
      log.append(el("p", "agent-note", event.text));
    } else if (event.type === "subagent" && event.state === "done") {
      log.append(el("p", "agent-note", `Helper ${event.agent || ""} done: ${event.description} · ${event.tools} tool calls`));
    } else if (event.type === "notice" || event.type === "error") {
      log.append(el("p", event.type === "error" ? "agent-note agent-error" : "agent-note", event.text));
    }
  }

  // Show the approval card for a call once. The "approval" event fires only
  // once per call, so after a reload (resume()) the card is rebuilt from
  // snap.pending instead of being lost and leaving the session stuck waiting.
  function showKeySwitchedModal(text) {
    document.querySelector(".agent-key-modal")?.remove();
    const overlay = el("div", "agent-key-modal");
    const box = el("div", "agent-key-box");
    box.append(el("p", "agent-key-title", "Key đã hết hạn mức"));
    box.append(el("p", "agent-key-text", text || "Đổi sang key dự phòng?"));
    const row = el("div", "agent-key-row");
    const ok = el("button", "agent-btn agent-btn-primary", "Đồng ý đổi");
    ok.type = "button";
    ok.onclick = () => { decideKey(true); overlay.remove(); };
    const no = el("button", "agent-btn", "Từ chối");
    no.type = "button";
    no.onclick = () => { decideKey(false); overlay.remove(); };
    row.append(ok, no);
    box.append(row);
    overlay.append(box);
    document.body.append(overlay);
  }

  function decideKey(allow) {
    if (!session) return;
    post(`/api/agent/sessions/${session.id}/approval`, { decision: allow ? "allow" : "deny", note: "" }).catch(() => {});
  }

  function showApprovalCard(call) {
    if (!call || !call.id) return;
    const log = $("agent-log");
    if (log.querySelector(`.agent-approval[data-call-id="${CSS.escape(call.id)}"]`)) return;
    // The change waiting for approval is shown open, so it is read before it is allowed.
    const row = log.querySelector(`.agent-row[data-call-id="${CSS.escape(call.id)}"]`);
    if (row) {
      row.open = true;
      row.closest(".agent-group").open = true;
    }
    log.append(approvalCard(call));
  }

  function decide(card, decision, note, doneText) {
    const buttons = card.querySelectorAll("button");
    buttons.forEach((b) => { b.disabled = true; });
    post(`/api/agent/sessions/${session.id}/approval`, { decision, note }).then(() => {
      if (doneText) card.replaceWith(el("p", "agent-note", doneText));
      else card.remove();
      if (decision === "allow_all") $("agent-mode").value = "auto";
      poll();
    }).catch((error) => {
      window.showToast?.(error.message, "error");
      buttons.forEach((b) => { b.disabled = false; });
    });
  }

  // A question for the user: answer choices as buttons, or a typed answer.
  function questionCard(c) {
    const card = el("div", "agent-approval");
    card.dataset.callId = c.id || "";
    card.append(el("p", "agent-approval-title", c.args.question || ""));
    const actions = el("div", "agent-approval-actions");
    (c.args.options || []).forEach((option) => actions.append(button(option, "agent-btn", () => decide(card, "allow", option, `Bạn trả lời: ${option}`))));
    const answer = el("input", "agent-approval-note");
    answer.placeholder = "Câu trả lời của bạn";
    answer.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && answer.value.trim() && !event.isComposing) decide(card, "allow", answer.value.trim(), `Bạn trả lời: ${answer.value.trim()}`);
    });
    actions.append(answer, button("Gửi", "agent-btn agent-btn-primary", () => answer.value.trim() && decide(card, "allow", answer.value.trim(), `Bạn trả lời: ${answer.value.trim()}`)),
      button("Bỏ qua", "agent-btn", () => decide(card, "deny", "", "Đã bỏ qua câu hỏi.")));
    card.append(actions);
    return card;
  }

  function planCard(c) {
    const card = el("div", "agent-approval");
    card.dataset.callId = c.id || "";
    const note = el("input", "agent-approval-note");
    note.placeholder = "Cần sửa gì trong kế hoạch?";
    const actions = el("div", "agent-approval-actions");
    actions.append(button("Duyệt và làm", "agent-btn agent-btn-primary", () => decide(card, "allow", "", "")),
      button("Duyệt, tự làm hết", "agent-btn", () => decide(card, "allow_all", "", "")),
      button("Sửa lại", "agent-btn", () => decide(card, "deny", note.value, "Đã yêu cầu sửa kế hoạch.")), note);
    card.append(el("p", "agent-approval-title", "Kế hoạch chờ bạn duyệt"), prose(c.args.plan), actions);
    return card;
  }

  function approvalCard(c) {
    if (c.name === "ask_user") return questionCard(c);
    if (c.name === "exit_plan_mode") return planCard(c);
    const card = el("div", "agent-approval");
    card.dataset.callId = c.id || "";
    const title = el("p", "agent-approval-title");
    title.append(document.createTextNode(c.agent ? `Agent phụ ${c.agent} muốn ` : "Cho phép "), el("strong", "", (isMcp(c.name) ? "gọi MCP" : VERBS[c.name] || c.name).toLowerCase()),
      document.createTextNode(" "), el("code", "", target(c)), document.createTextNode("?"));
    const note = el("input", "agent-approval-note");
    note.placeholder = "Lý do từ chối (tùy chọn)";
    const actions = el("div", "agent-approval-actions");
    const choices = [["allow", "Cho phép", "agent-btn agent-btn-primary", "Đã cho phép."]];
    if (c.remember) choices.push(["allow_always", `Luôn cho phép ${c.remember}`, "agent-btn", `Đã lưu luật: ${c.remember}`]);
    choices.push(["allow_all", "Tự làm hết", "agent-btn", "Đã cho phép, từ giờ tự làm hết."], ["deny", "Từ chối", "agent-btn", "Đã từ chối."]);
    choices.forEach(([decision, label, cls, done]) => {
      actions.append(button(label, cls, () => decide(card, decision, note.value, done)));
    });
    actions.append(note);
    if (c.reviewer) card.append(el("p", "agent-muted", `Model duyệt không chắc: ${c.reviewer}`));
    if (c.why === "untrusted") card.append(el("p", "agent-muted", "Agent vừa đọc nội dung từ web hoặc MCP, có thể chứa lệnh giả, nên hỏi lại trước khi hành động."));
    card.append(title, actions);
    return card;
  }

  function showTodos(items) {
    const done = (items || []).filter((i) => i.status === "completed").length;
    $("agent-todos").hidden = !items?.length || done === items.length;
    $("agent-todo-list").replaceChildren(...(items || []).map((item) => el("li", `agent-todo agent-todo-${item.status}`, item.content)));
  }

  async function trust(path) {
    try {
      showInfo(await post(`/api/agent/sessions/${session.id}/trust/${path}`));
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  function showInfo(snap) {
    const box = el("div", "agent-info");
    const sb = snap.sandbox || {};
    $("agent-sandbox-info").textContent = sb.mode === "full-access"
      ? "Sandbox tắt: lệnh có toàn quyền trên máy."
      : `${SANDBOX_TEXT[sb.backend] || ""} Mạng ${sb.network ? "bật" : "tắt"}.`;
    const named = snap.agents || [];
    if (named.length) box.append(el("p", "agent-muted", `Agent phụ: ${named.map((a) => a.name).join(", ")}. Gõ /rules, /memory, /undo, /plan, /goal.`));
    const skills = snap.skills || [];
    box.append(el("p", "agent-muted", skills.length ? `Skill: ${skills.map((s) => s.name).join(", ")}` : "Chưa có skill (.agents/skills, .claude/skills, .codex/skills)."));
    (snap.mcp || []).forEach((server) => {
      const state = { running: `${server.tools} công cụ`, untrusted: "chưa cho phép", failed: `lỗi: ${server.error}`, disabled: "tắt", stopped: "đã dừng (/plugins enable mcp:" + server.name + ")" }[server.state] || server.state;
      const row = el("p", "agent-muted agent-info-row", `MCP ${server.name} · ${state}`);
      if (server.state === "untrusted") row.append(button("Cho phép", "agent-btn", () => trust(`mcp/${encodeURIComponent(server.name)}`)));
      box.append(row);
    });
    Object.entries(snap.replaced || {}).forEach(([name, target]) => box.append(el("p", "agent-muted", `${name} thay bằng ${target.replace(/^mcp__/, "MCP ").replace("__", ": ")}`)));
    const plug = snap.plugins || {};
    (plug.rows || []).forEach((row) => {
      const state = { loaded: "đã nạp", untrusted: "chưa cho phép", failed: `lỗi: ${row.error}` }[row.state] || row.state;
      const line = el("p", "agent-muted agent-info-row", `Plugin ${row.name} (${row.scope}) · ${state}`);
      if (row.state === "untrusted" && !box.querySelector(".agent-plugin-trust")) line.append(button("Cho phép", "agent-btn agent-plugin-trust", () => trust("plugins")));
      box.append(line);
    });
    if (plug.externals?.length) box.append(el("p", "agent-muted", `Agent ngoài: ${plug.externals.join(", ")} (luôn hỏi bạn trước)`));
    if (snap.disabled?.length) box.append(el("p", "agent-muted", `Đã tắt: ${snap.disabled.join(", ")}`));
    const hooks = snap.hooks || {};
    if (hooks.user || hooks.workspace) {
      const row = el("p", "agent-muted agent-info-row", `Hook · ${hooks.user} của bạn, ${hooks.workspace} của dự án${hooks.workspace && !hooks.workspace_trusted ? " (chưa cho phép)" : ""}`);
      if (hooks.workspace && !hooks.workspace_trusted) row.append(button("Cho phép", "agent-btn", () => trust("hooks")));
      box.append(row);
    }
    $("agent-tools-info").replaceChildren(box);
    commands = snap.commands || commands;
  }

  function showStatus(snap) {
    const busy = snap.status === "running" || snap.status === "waiting";
    $("agent-stop").hidden = !busy;
    $("agent-send").hidden = false;
    const usage = snap.usage || {};
    lastUsage = usage;
    const state = { running: "Working", waiting: "Waiting for you" }[snap.status];
    const parts = [snap.title || "New session"];
    if (state) parts.push(state);
    if (snap.plan_mode) parts.push("planning");
    if (snap.goal) parts.push(`goal: ${snap.goal}`);
    $("agent-status").textContent = parts.join(" · ");
    const log = $("agent-log");
    log.querySelector(".agent-working")?.remove();
    log.querySelector(".agent-live")?.remove();
    const live = snap.live;
    if (live && (live.text || live.tools.length)) {
      const tools = live.tools.length ? `\nCalling: ${live.tools.join(", ")}` : "";
      log.append(el("pre", "agent-live", live.text + tools));
    }
    if (state) log.append(el("p", "agent-working", `${state}…`));
    showTodos(snap.todos);
    showInfo(snap);
  }

  async function poll() {
    clearTimeout(pollTimer);
    if (!session) return;
    try {
      const stick = nearBottom();
      const snap = await call(`/api/agent/sessions/${session.id}?after=${lastSeq}`);
      if (snap.usage) lastUsage = snap.usage;
      snap.events.forEach((event) => { render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
      follow(stick && snap.events.length > 0);
      if (snap.status !== "idle") pollTimer = setTimeout(poll, snap.live ? POLL_MS / 2 : POLL_MS);
      else loadSessions();
    } catch (error) {
      $("agent-status").textContent = error.message;
      pollTimer = setTimeout(poll, POLL_MS * 4);
    }
  }

  async function loadModels() {
    const provider = $("agent-provider").value;
    $("agent-model").value = store("get", `manga_agent_model_${provider}`) || store("get", `manga_ai_model_${provider}`) || "";
    const list = $("agent-model-list");
    list.replaceChildren();
    if (!provider) return;
    try {
      const data = await call(`/api/visual_qc/providers/${encodeURIComponent(provider)}/models`);
      (data.models || []).forEach((id) => list.append(new Option(id, id)));
    } catch (_) {}
  }

  async function loadProviders() {
    const response = await fetch("/api/visual_qc/settings");
    const data = await window.parseApiResponse(response);
    const ready = Object.values(data.providers || {}).filter((p) => p.configured && p.id !== "manga-cloud");
    $("agent-provider").replaceChildren(...ready.map((p) => new Option(p.label || p.id, p.id)));
    if (!ready.length) {
      $("agent-status").textContent = "Chưa có dịch vụ nào có API key: mở Cài đặt, mục Dịch vụ AI để thêm (mọi API chuẩn OpenAI đều dùng được).";
      return;
    }
    const saved = store("get", "manga_agent_provider");
    if (ready.some((p) => p.id === saved)) $("agent-provider").value = saved;
    await loadModels();
  }

  async function loadSessions() {
    try {
      const data = await call("/api/agent/sessions");
      const rows = (data.sessions || []).slice(0, 30).map((row) => {
        const item = el("li", row.id === session?.id ? "agent-session-current" : "");
        const open = button(row.title || "(chưa có tin nhắn)", "agent-session-open", () => resume(row.id));
        open.title = `${row.model} · ${row.workspace}`;
        const remove = button("Xoá", "agent-session-del", async () => {
          await call(`/api/agent/sessions/${row.id}`, { method: "DELETE" }).catch(() => {});
          if (row.id === session?.id) resetLog();
          loadSessions();
        });
        item.append(open, remove);
        return item;
      });
      $("agent-sessions").replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", "Chưa có phiên nào.")]));
    } catch (_) {}
  }

  async function resume(id) {
    try {
      const snap = await post(`/api/agent/sessions/${id}/resume`);
      resetLog();
      session = snap;
      $("agent-log").replaceChildren();
      $("agent-mode").value = snap.mode;
      $("agent-sandbox").value = snap.sandbox.mode;
      $("agent-network").checked = Boolean(snap.sandbox.network);
      $("agent-workspace").value = snap.workspace;
      showFolder();
      $("agent-model").value = snap.model;
      snap.events.forEach((event) => { if (event.type !== "approval") render(event); lastSeq = Math.max(lastSeq, event.seq); });
      // A waiting session's approval card is rebuilt here: the "approval" event
      // was skipped above and only ever fires once, so without this a reload
      // loses the card and the session stays stuck on "Waiting for you".
      if (snap.status === "waiting" && snap.pending) {
        if (snap.pending.type === "key_switch") showKeySwitchedModal(`Key ${snap.pending.from_key} đã hết hạn mức (${snap.pending.reason || ""}). Đổi sang key ${snap.pending.to_key}/${snap.pending.total_keys}?`);
        else showApprovalCard(snap.pending);
      }
      showStatus(snap);
      document.querySelectorAll(".agent-menu[open]").forEach((menu) => { menu.open = false; });
      follow(true);
      loadSessions();
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  function emptyState() {
    const box = el("div", "agent-empty");
    box.append(el("h1", "", "Agent"), el("p", "agent-muted", "Đọc code, sửa file và chạy test trong sandbox bằng bất kỳ model nào bạn đã thêm. Gõ / để xem lệnh."));
    const examples = el("div", "agent-examples");
    EXAMPLES.forEach((text) => examples.append(button(text, "agent-chip", () => { $("agent-input").value = text; autosize(); $("agent-input").focus(); })));
    box.append(examples);
    return box;
  }

  function resetLog() {
    clearTimeout(pollTimer);
    session = null;
    lastSeq = 0;
    group = null;
    $("agent-log").replaceChildren(emptyState());
    $("agent-stop").hidden = true;
    $("agent-send").hidden = false;
    $("agent-status").textContent = "";
    showTodos([]);
  }

  async function ensureSession() {
    if (session) return session;
    const body = {
      provider: $("agent-provider").value, model: $("agent-model").value.trim(), mode: $("agent-mode").value,
      workspace: $("agent-workspace").value.trim(), sandbox: $("agent-sandbox").value, network: $("agent-network").checked,
    };
    session = await post("/api/agent/sessions", body);
    store("set", "manga_agent_provider", body.provider);
    store("set", `manga_agent_model_${body.provider}`, body.model);
    store("set", "manga_agent_workspace", body.workspace);
    return session;
  }

  function autosize() {
    const input = $("agent-input");
    input.style.height = "auto";
    input.style.height = `${Math.min(240, input.scrollHeight)}px`;
  }

  function showHints() {
    const hints = $("agent-hints");
    const match = /^\/(\S*)$/.exec($("agent-input").value.trim());
    const rows = match ? commands.filter((c) => c.name.startsWith(match[1])).slice(0, 16) : [];
    hints.hidden = !rows.length;
    hints.replaceChildren(...rows.map((c) => {
      const item = el("li");
      item.append(button(`/${c.name}`, "agent-hint", () => { $("agent-input").value = `/${c.name} `; $("agent-input").focus(); showHints(); }),
        el("span", "agent-muted", c.description || ""));
      return item;
    }));
  }

  async function send(event) {
    event?.preventDefault();
    const input = $("agent-input");
    const text = input.value.trim();
    if (!text) return;
    if (text === "/clear" || text === "/new") {
      input.value = "";
      resetLog();
      showHints();
      return;
    }
    if (!$("agent-provider").value || !$("agent-model").value.trim()) {
      window.showToast?.("Chọn dịch vụ và model trước.", "error");
      return;
    }
    $("agent-send").disabled = true;
    try {
      await ensureSession();
      const reply = await post(`/api/agent/sessions/${session.id}/messages`, { text });
      input.value = "";
      autosize();
      showHints();
      if (reply.message) {
        $("agent-log").querySelector(".agent-empty")?.remove();
        $("agent-log").append(el("pre", "agent-pre agent-command", reply.message));
      }
      follow(true);
      poll();
    } catch (error) {
      window.showToast?.(error.message, "error");
    } finally {
      $("agent-send").disabled = false;
    }
  }

  async function syncSession(changes) {
    if (!session) return;
    try {
      showStatus(await post(`/api/agent/sessions/${session.id}`, changes, "PATCH"));
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  // The folder `manga` was typed in, handed over once in the page URL.
  const launchFolder = (() => {
    const url = new URL(window.location.href);
    const folder = url.searchParams.get("workspace");
    if (folder === null) return "";
    url.searchParams.delete("workspace");
    history.replaceState(history.state, "", url);
    return folder.trim();
  })();

  // The composer names the folder the agent works in.
  function showFolder() {
    const folder = $("agent-workspace").value.trim();
    const name = folder.split(/[\\/]/).filter(Boolean).pop() || folder;
    $("agent-input").placeholder = name ? `Giao việc cho agent trong ${name}… (/ để xem lệnh)` : "Giao việc cho agent… (/ để xem lệnh)";
    $("agent-input").title = folder;
  }

  const SKILL_SCOPE = { builtin: "có sẵn", user: "của bạn", project: "của dự án" };
  const SKILL_TEMPLATE = "## Steps\n\n1. \n2. \n\n## Check before you finish\n\n- \n";
  let editingSkill = "";

  // The skills panel: what the agent can load here, plus loading, writing and removing your own.
  async function loadSkills() {
    const list = $("agent-skill-list");
    try {
      const data = await call(`/api/agent/skills?workspace=${encodeURIComponent($("agent-workspace").value.trim())}`);
      const rows = (data.skills || []).map((skill) => {
        const item = el("li");
        const title = el("span");
        title.append(el("b", "", skill.name), el("span", "agent-skill-tag", ` · ${SKILL_SCOPE[skill.scope] || skill.scope}`));
        const actions = el("span", "agent-skill-row");
        if (skill.scope === "user") actions.append(button("Sửa", "agent-btn", () => editSkill(skill.name)), button("Xóa", "agent-btn", () => removeSkill(skill.name)));
        const about = el("span", "agent-muted", skill.description.length > 180 ? `${skill.description.slice(0, 180)}…` : skill.description);
        about.title = skill.path;
        item.append(title, actions, about);
        return item;
      });
      list.replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", "Chưa có skill nào.")]));
    } catch (error) {
      list.replaceChildren(el("li", "agent-muted", error.message));
    }
  }

  async function skillsChanged(promise, done) {
    try {
      const data = await promise;
      window.showToast?.(done(data.skills || []), "success");
      await loadSkills();
      return true;
    } catch (error) {
      window.showToast?.(error.message, "error");
      return false;
    }
  }

  function installSkills() {
    const source = $("agent-skill-source").value.trim();
    if (!source) return;
    skillsChanged(post("/api/agent/skills/install", { source }), (names) => `Đã cài ${names.join(", ")}. Đọc SKILL.md của chúng trước khi tin.`)
      .then((ok) => { if (ok) $("agent-skill-source").value = ""; });
  }

  function uploadSkills(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    skillsChanged(call("/api/agent/skills/upload", { method: "POST", body: form }), (names) => `Đã nạp ${names.join(", ")}.`);
  }

  async function editSkill(name) {
    try {
      const { text } = await call(`/api/agent/skills/${encodeURIComponent(name)}`);
      const parts = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text) || ["", "", text];
      const head = parts[1];
      const desc = /^description:\s*(?:>-?|\|-?)?\s*\n?((?:[ \t].*\n?)*|.*)$/m.exec(head);
      editingSkill = name;
      $("agent-skill-name").value = name;
      $("agent-skill-desc").value = desc ? desc[1].split("\n").map((l) => l.trim()).join(" ").replace(/^["']|["']$/g, "").trim() : "";
      $("agent-skill-body").value = parts[2].trim();
      $("agent-skill-new").open = true;
      $("agent-skill-name").focus();
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  function removeSkill(name) {
    if (!window.confirm(`Xóa skill ${name}?`)) return;
    skillsChanged(call(`/api/agent/skills/${encodeURIComponent(name)}`, { method: "DELETE" }), () => `Đã xóa ${name}.`);
  }

  function saveSkill() {
    const name = $("agent-skill-name").value.trim();
    const body = { name, description: $("agent-skill-desc").value.trim(), body: $("agent-skill-body").value.trim(), replace: editingSkill === name };
    skillsChanged(post("/api/agent/skills", body), () => `Đã lưu skill ${name}; agent dùng được ngay.`).then((ok) => {
      if (!ok) return;
      editingSkill = "";
      $("agent-skill-name").value = $("agent-skill-desc").value = "";
      $("agent-skill-body").value = SKILL_TEMPLATE;
    });
  }

  function draftSkill() {
    const name = $("agent-skill-name").value.trim() || "ten-skill";
    const about = $("agent-skill-desc").value.trim() || "(mô tả việc skill này giúp làm)";
    $("agent-input").value = `Dùng skill writing-skills để soạn một skill tên ${name}: ${about}. `
      + `Ghi nó vào .agents/skills/${name}/SKILL.md trong thư mục làm việc, với phần đầu name và description, rồi thử nó trên một ví dụ thật.`;
    $("agent-skills-menu").open = false;
    autosize();
    $("agent-input").focus();
  }

  window.openAgentView = async () => {
    if (loaded) return;
    loaded = true;
    try {
      const config = await call("/api/agent/config");
      $("agent-workspace").value = launchFolder || store("get", "manga_agent_workspace") || config.workspace;
      if (launchFolder) store("set", "manga_agent_workspace", launchFolder);
      showFolder();
      $("agent-sandbox-info").textContent = SANDBOX_TEXT[config.sandbox_backend] || "";
      await loadProviders();
      loadSessions();
    } catch (error) {
      loaded = false;
      $("agent-status").textContent = error.message;
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("agent-view")) return;
    resetLog();
    $("agent-composer").addEventListener("submit", send);
    $("agent-input").addEventListener("keydown", (event) => {
      // Enter sends, Shift+Enter breaks the line, as in chat apps.
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) send(event);
    });
    $("agent-input").addEventListener("input", () => { autosize(); showHints(); });
    $("agent-stop").addEventListener("click", () => session && post(`/api/agent/sessions/${session.id}/stop`).then(poll).catch(() => {}));
    $("agent-new").addEventListener("click", () => { resetLog(); loadSessions(); });
    $("agent-provider").addEventListener("change", () => { resetLog(); loadModels(); });
    $("agent-workspace").addEventListener("change", () => { resetLog(); showFolder(); });
    $("agent-skill-body").value = SKILL_TEMPLATE;
    $("agent-skills-menu").addEventListener("toggle", () => { if ($("agent-skills-menu").open) loadSkills(); });
    $("agent-skill-install").addEventListener("click", installSkills);
    $("agent-skill-source").addEventListener("keydown", (event) => { if (event.key === "Enter") installSkills(); });
    $("agent-skill-file").addEventListener("change", uploadSkills);
    $("agent-skill-save").addEventListener("click", saveSkill);
    $("agent-skill-draft").addEventListener("click", draftSkill);
    $("agent-model").addEventListener("change", () => syncSession({ model: $("agent-model").value.trim() }));
    $("agent-mode").addEventListener("change", () => syncSession({ mode: $("agent-mode").value }));
    $("agent-sandbox").addEventListener("change", () => syncSession({ sandbox: $("agent-sandbox").value }));
    $("agent-network").addEventListener("change", () => syncSession({ network: $("agent-network").checked }));
    document.querySelectorAll(".agent-menu").forEach((menu) => menu.addEventListener("toggle", () => {
      if (menu.open) document.querySelectorAll(".agent-menu[open]").forEach((other) => { if (other !== menu) other.open = false; });
    }));
  });
})();
