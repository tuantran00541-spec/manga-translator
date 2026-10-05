// Agent: a coding agent on any configured model, read like a conversation with its tool calls folded into one-line groups.
(() => {
  const POLL_MS = 700;
  const HEAD = { "X-Manga-Agent": "1" };
  const VERBS = {
    list_dir: "Xem", read_file: "Đọc", search: "Tìm", glob: "Tìm file", write_file: "Tạo", edit_file: "Sửa",
    apply_patch: "Vá", edit_lines: "Sửa dòng", run_command: "Chạy", web_fetch: "Mở trang", skill: "Mở skill", todo_write: "Kế hoạch",
    task: "Agent phụ", spawn_agent: "Giao việc", wait_agent: "Đợi agent", send_input: "Nhắn agent", close_agent: "Đóng agent", memory: "Ghi nhớ", ask_user: "Hỏi bạn", exit_plan_mode: "Kế hoạch", goal_done: "Xong mục tiêu",
  };
  // How each kind of call is counted in a group's one-line summary.
  const TALLY = {
    run_command: ["chạy", "lệnh"], read_file: ["đọc", "file"], edit_file: ["sửa", "file"], write_file: ["sửa", "file"],
    apply_patch: ["sửa", "file"], edit_lines: ["sửa", "file"], spawn_agent: ["giao", "việc cho agent phụ"], wait_agent: ["đợi", "agent phụ"],
    send_input: ["nhắn", "agent phụ"], close_agent: ["đóng", "agent phụ"], memory: ["ghi", "nhớ"], list_dir: ["tìm", "lần"], search: ["tìm", "lần"], glob: ["tìm", "lần"],
    web_fetch: ["đọc", "trang web"], skill: ["mở", "skill"], task: ["giao", "việc cho agent phụ"], mcp: ["gọi", "công cụ MCP"],
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

  // `code` and **bold** inside a line, built as nodes so model text never becomes markup.
  function inline(text) {
    const out = document.createDocumentFragment();
    String(text).split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/).forEach((part) => {
      if (/^`[^`]+`$/.test(part)) out.append(el("code", "", part.slice(1, -1)));
      else if (/^\*\*[^*]+\*\*$/.test(part)) out.append(el("strong", "", part.slice(2, -2)));
      else if (part) out.append(document.createTextNode(part));
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
    if (c.name === "run_command") return `${a.command || ""}${a.outside_sandbox ? "  (ngoài sandbox)" : ""}`;
    if (c.name === "search" || c.name === "glob") return `${a.pattern || ""}${a.path && a.path !== "." ? `  (${a.path})` : ""}`;
    if (c.name === "web_fetch") return a.url || "";
    if (c.name === "task") return a.description || "";
    if (c.name === "spawn_agent") return `${a.agent || "explore"}: ${String(a.message || "").slice(0, 80)}`;
    if (c.name === "wait_agent") return (a.ids || []).join(", ") || "tất cả";
    if (c.name === "send_input") return `${a.id || ""}: ${String(a.message || "").slice(0, 60)}`;
    if (c.name === "close_agent") return a.id || "";
    if (c.name === "skill") return a.name || "";
    if (c.name === "todo_write") return `${(a.items || []).length} việc`;
    if (c.name === "ask_user") return a.question || "";
    if (c.name === "memory") return `${a.action || ""} ${a.text || ""}`.trim();
    if (c.name === "goal_done") return a.summary || "";
    if (c.name === "apply_patch") return (String(a.patch || "").match(/^\*\*\* (?:Add|Update|Delete) File: .+$/gm) || []).map((l) => l.split(": ")[1]).join(", ");
    if (isMcp(c.name)) return c.name.replace(/^mcp__/, "").replace("__", ": ");
    return a.path || ".";
  }

  function diffStats(c) {
    const a = c.args || {};
    const count = (s) => (String(s || "").match(/\n/g) || []).length + (s ? 1 : 0);
    if (c.name === "edit_file") return [count(a.new_text), count(a.old_text)];
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
    const parts = Object.entries(group.tally).map(([key, n]) => {
      const [verb, noun] = TALLY[key] || ["dùng", key];
      return `${verb} ${n} ${noun}`;
    });
    const text = parts.join(", ") || "Cập nhật kế hoạch";
    group.label.textContent = text.charAt(0).toUpperCase() + text.slice(1);
    group.summary.querySelector(".agent-stats")?.remove();
    group.summary.querySelector(".agent-fail")?.remove();
    if (group.added || group.removed) group.summary.append(stats(group.added, group.removed));
    if (group.failed) group.summary.append(el("span", "agent-fail", `${group.failed} lỗi`));
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
    if (c.name === "edit_file") body.append(el("pre", "agent-pre agent-old", a.old_text || ""), el("pre", "agent-pre agent-new", a.new_text || ""));
    else if (c.name === "write_file") body.append(el("pre", "agent-pre agent-new", a.content || ""));
    else if (c.name === "apply_patch") body.append(patchView(a.patch));
    else if (c.name === "edit_lines") body.append(editLinesView(a.edits));
    else if (c.name === "task") body.append(el("pre", "agent-pre", a.prompt || ""));
    else if (c.name === "spawn_agent" || c.name === "send_input") body.append(el("pre", "agent-pre", a.message || ""));
    else if (isMcp(c.name)) body.append(el("pre", "agent-pre", JSON.stringify(a, null, 2)));
    row.append(body);
    g.list.append(row);
    refreshGroup();
  }

  function editLinesView(edits) {
    const pre = el("pre", "agent-pre");
    (edits || []).forEach((edit, i) => {
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
        think.append(el("summary", "", "Suy nghĩ"), el("pre", "agent-pre", event.reasoning));
        log.append(think);
      }
      if (event.text) {
        group = null;
        log.append(prose(event.text));
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
      // The change waiting for approval is shown open, so it is read before it is allowed.
      const row = log.querySelector(`.agent-row[data-call-id="${CSS.escape(event.call.id)}"]`);
      if (row) {
        row.open = true;
        row.closest(".agent-group").open = true;
      }
      log.append(approvalCard(event.call));
    } else if (event.type === "subagent" && event.state === "done") {
      log.append(el("p", "agent-note", `Agent phụ ${event.agent || ""} xong: ${event.description} · ${event.tools} lần gọi công cụ`));
    } else if (event.type === "notice" || event.type === "error") {
      log.append(el("p", event.type === "error" ? "agent-note agent-error" : "agent-note", event.text));
    }
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
    const title = el("p", "agent-approval-title");
    title.append(document.createTextNode(c.agent ? `Agent phụ ${c.agent} muốn ` : "Cho phép "), el("strong", "", (isMcp(c.name) ? "gọi MCP" : VERBS[c.name] || c.name).toLowerCase()),
      document.createTextNode(" "), el("code", "", target(c)), document.createTextNode("?"));
    const note = el("input", "agent-approval-note");
    note.placeholder = "Lý do từ chối (tùy chọn)";
    const actions = el("div", "agent-approval-actions");
    [["allow", "Cho phép", "agent-btn agent-btn-primary", "Đã cho phép."], ["allow_all", "Luôn cho phép", "agent-btn", "Đã cho phép, từ giờ tự làm hết."],
      ["deny", "Từ chối", "agent-btn", "Đã từ chối."]].forEach(([decision, label, cls, done]) => {
      actions.append(button(label, cls, () => decide(card, decision, note.value, done)));
    });
    actions.append(note);
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
      const state = { running: `${server.tools} công cụ`, untrusted: "chưa cho phép", failed: `lỗi: ${server.error}`, disabled: "tắt" }[server.state] || server.state;
      const row = el("p", "agent-muted agent-info-row", `MCP ${server.name} · ${state}`);
      if (server.state === "untrusted") row.append(button("Cho phép", "agent-btn", () => trust(`mcp/${encodeURIComponent(server.name)}`)));
      box.append(row);
    });
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
    const state = { running: "Đang làm", waiting: "Chờ bạn duyệt" }[snap.status];
    const parts = [snap.title || "Phiên mới", `${usage.prompt_tokens || 0} token vào, ${usage.completion_tokens || 0} ra`];
    if (snap.text_tools) parts.push("gọi công cụ bằng văn bản");
    if (snap.plan_mode) parts.push("đang lập kế hoạch");
    if (snap.goal) parts.push(`mục tiêu: ${snap.goal}`);
    $("agent-status").textContent = parts.join(" · ");
    const log = $("agent-log");
    log.querySelector(".agent-working")?.remove();
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
      snap.events.forEach((event) => { render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
      follow(stick && snap.events.length > 0);
      if (snap.status !== "idle") pollTimer = setTimeout(poll, POLL_MS);
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
      $("agent-model").value = snap.model;
      snap.events.forEach((event) => { if (event.type !== "approval") render(event); lastSeq = Math.max(lastSeq, event.seq); });
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

  window.openAgentView = async () => {
    if (loaded) return;
    loaded = true;
    try {
      const config = await call("/api/agent/config");
      $("agent-workspace").value = store("get", "manga_agent_workspace") || config.workspace;
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
    $("agent-workspace").addEventListener("change", resetLog);
    $("agent-model").addEventListener("change", () => syncSession({ model: $("agent-model").value.trim() }));
    $("agent-mode").addEventListener("change", () => syncSession({ mode: $("agent-mode").value }));
    $("agent-sandbox").addEventListener("change", () => syncSession({ sandbox: $("agent-sandbox").value }));
    $("agent-network").addEventListener("change", () => syncSession({ network: $("agent-network").checked }));
    document.querySelectorAll(".agent-menu").forEach((menu) => menu.addEventListener("toggle", () => {
      if (menu.open) document.querySelectorAll(".agent-menu[open]").forEach((other) => { if (other !== menu) other.open = false; });
    }));
  });
})();
