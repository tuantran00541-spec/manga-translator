// Agent: a coding agent on any configured model, read like a conversation with its tool calls folded into one-line groups.
(() => {
  const POLL_MS = 700;
  const HEAD = { "X-Manga-Agent": "1" };
  const VERBS = {
    "": "Lệnh lỗi",
    list_dir: "Xem thư mục", read_file: "Đọc", search: "Tìm", glob: "Tìm file", write_file: "Ghi", edit_file: "Sửa",
    run_command: "Chạy", web_fetch: "Tải trang", skill: "Skill", todo_write: "Kế hoạch",
    plugin_write: "Ghi plugin", task: "Agent con", spawn_agent: "Tạo agent con", wait_agent: "Chờ agent con", send_input: "Nhắn agent con", close_agent: "Đóng agent con", memory: "Ghi nhớ", ask_user: "Hỏi bạn", exit_plan_mode: "Kế hoạch", goal_done: "Xong mục tiêu",
    symbols: "Tìm ký hiệu", web_search: "Tìm web", web_download: "Tải về", view_image: "Xem ảnh", tool_search: "Tìm công cụ", run_script: "Chạy script", fan_out: "Chạy song song",
    job_output: "Đọc tác vụ", job_input: "Nhập vào tác vụ", job_stop: "Dừng tác vụ", context_notes: "Ghi chú", new_context: "Ngữ cảnh mới",
    schedule_create: "Hẹn giờ", schedule_list: "Lịch hẹn", schedule_delete: "Hủy hẹn",
    use_model: "Dùng model trả phí",
  };
  // How each kind of call is counted in a group's one-line summary: verb, noun.
  const TALLY = {
    run_command: ["chạy", "lệnh"], read_file: ["đọc", "file"], edit_file: ["sửa", "file"], write_file: ["sửa", "file"],
    spawn_agent: ["tạo", "agent con"], wait_agent: ["chờ", "agent con"],
    send_input: ["nhắn", "agent con"], close_agent: ["đóng", "agent con"], memory: ["ghi nhớ", "mục"],
    list_dir: ["xem", "thư mục"], search: ["tìm", "lần"], glob: ["tìm", "lần"], symbols: ["tìm", "lần"],
    web_fetch: ["tải", "trang"], skill: ["mở", "skill"], task: ["chạy", "agent con"], mcp: ["gọi", "công cụ MCP"],
    run_script: ["chạy", "script"], fan_out: ["chạy", "lô song song"], job_input: ["nhập vào", "tác vụ"], job_output: ["đọc", "tác vụ"], job_stop: ["dừng", "tác vụ"],
    schedule_create: ["hẹn", "lượt"], web_search: ["tìm web", "lần"],
  };
  const SANDBOX_TEXT = {
    landlock: "Sandbox Landlock (Linux).", seatbelt: "Sandbox Seatbelt (macOS).",
    docker: "Container Docker, chỉ thấy /workspace.",
    "low-integrity": "Windows integrity thấp: chỉ ghi trong thư mục làm việc, vẫn đọc file và dùng mạng nên cần duyệt.",
    none: "Không có sandbox: mọi lệnh cần duyệt, trừ chế độ Tự động.",
  };
  const $ = (id) => document.getElementById(id);
  let session = null;
  let lastSeq = 0;
  let pollTimer = null;
  let loaded = false;
  let commands = [];
  let infoRev = "";
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

  function showEngine(engine) {
    $("agent-engine").value = engine.engine || "auto";
    $("agent-engine-image").value = engine.image || "";
    $("agent-engine-image").closest("label").hidden = $("agent-engine").value !== "docker";
  }

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
    if (c.name === "run_command") return `${a.command || ""}${a.outside_sandbox ? "  (ngoài sandbox)" : ""}`;
    if (c.name === "search" || c.name === "glob") return `${a.pattern || ""}${a.path && a.path !== "." ? `  (${a.path})` : ""}`;
    if (c.name === "use_model") return `${a.model || ""}  ($${a.per_million_in ?? "?"} vào / $${a.per_million_out ?? "?"} ra mỗi 1M token)`;
    if (c.name === "web_fetch") return a.url || "";
    if (c.name === "web_search") return a.query || "";
    if (c.name === "web_download") return `${a.url || ""} vào ${a.path || ""}`;
    if (c.name === "symbols") return a.name || a.path || ".";
    if (c.name === "view_image") return a.path || "";
    if (c.name === "tool_search") return a.query || "";
    if (c.name === "task") return a.description || "";
    if (c.name === "spawn_agent") return `${a.agent || "explore"}: ${String(a.message || "").slice(0, 80)}`;
    if (c.name === "wait_agent") return (a.ids || []).join(", ") || "tất cả";
    if (c.name === "send_input") return `${a.id || ""}: ${String(a.message || "").slice(0, 60)}`;
    if (c.name === "close_agent") return a.id || "";
    if (c.name === "plugin_write") return `${a.scope === "workspace" ? ".agents/plugins" : "~/.manga-agent/plugins"}/${a.name || ""}.py${a.remove ? " (xóa)" : ""}`;
    if (c.name === "skill") return a.name || "";
    if (c.name === "todo_write") return `${(a.items || []).length} mục`;
    if (c.name === "ask_user") return a.question || "";
    if (c.name === "memory") return `${a.action || ""} ${a.text || ""}`.trim();
    if (c.name === "goal_done") return a.summary || "";
    if (c.name === "run_script") return String(a.code || "").split("\n").find((l) => l.trim()) || "";
    if (c.name === "fan_out") return `${(a.jobs || []).length} tác vụ${a.schema ? " có schema" : ""}`;
    if (c.name === "job_input") return `${a.id || ""}: ${JSON.stringify(String(a.chars || "")).slice(0, 60)}`;
    if (c.name === "job_output" || c.name === "job_stop" || c.name === "schedule_delete") return a.id || "";
    if (c.name === "schedule_create") return `sau ${a.in_minutes || "?"} phút${a.every_minutes ? `, mỗi ${a.every_minutes} phút` : ""}: ${String(a.prompt || "").slice(0, 50)}`;
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
      const words = TALLY[key] || ["dùng", key];
      const id = words.join("|");
      merged.set(id, { words, n: (merged.get(id)?.n || 0) + n });
    });
    const parts = [...merged.values()].map(({ words: [verb, noun], n }) => `${verb} ${n} ${noun}`);
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
    if (c.name === "edit_file") editPairs(a).forEach((e) => body.append(el("pre", "agent-pre agent-old", (e && e.old_text) || ""), el("pre", "agent-pre agent-new", (e && e.new_text) || "")));
    else if (c.name === "write_file") body.append(el("pre", "agent-pre agent-new", a.content || ""));
    else if (c.name === "task") body.append(el("pre", "agent-pre", a.prompt || ""));
    else if (c.name === "run_script") body.append(el("pre", "agent-pre", a.code || ""));
    else if (c.name === "fan_out") body.append(el("pre", "agent-pre", (Array.isArray(a.jobs) ? a.jobs : []).map((j, n) => `${n + 1}. ${j.description || ""}\n${j.prompt || ""}`).join("\n\n") + (a.schema ? `\n\nschema: ${JSON.stringify(a.schema)}` : "")));
    else if (c.name === "spawn_agent" || c.name === "send_input") body.append(el("pre", "agent-pre", a.message || ""));
    else if (isMcp(c.name)) body.append(el("pre", "agent-pre", JSON.stringify(a, null, 2)));
    row.append(body);
    g.list.append(row);
    refreshGroup();
  }

  function render(event) {
    const log = $("agent-log");
    log.querySelector(".agent-empty")?.remove();
    if (event.type === "user") {
      group = null;
      const bubble = el("div", "agent-user", event.text);
      if (event.refs?.length) bubble.append(el("span", "agent-refs", `Đính kèm: ${event.refs.join(", ")}`));
      if (event.queued) bubble.append(el("span", "agent-refs", "Đã xếp hàng, đọc ở bước sau"));
      log.append(bubble);
    } else if (event.type === "assistant") {
      if (event.reasoning) {
        const think = el("details", "agent-think");
        think.append(el("summary", "", "Suy luận"), el("pre", "agent-pre", event.reasoning));
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
      log.append(el("p", "agent-note", `Agent con ${event.agent || ""} xong: ${event.description} · ${event.tools} lệnh gọi`));
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
      if (decision === "allow_all") {
        $("agent-mode").value = "edits";
        showPicks();
      }
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
    note.placeholder = "Cần sửa gì?";
    const actions = el("div", "agent-approval-actions");
    actions.append(button("Duyệt", "agent-btn agent-btn-primary", () => decide(card, "allow", "", "")),
      button("Duyệt, tự động", "agent-btn", () => decide(card, "allow_all", "", "")),
      button("Sửa lại", "agent-btn", () => decide(card, "deny", note.value, "Đã yêu cầu sửa.")), note);
    card.append(el("p", "agent-approval-title", "Kế hoạch chờ duyệt"), prose(c.args.plan), actions);
    return card;
  }

  function approvalCard(c) {
    if (c.name === "ask_user") return questionCard(c);
    if (c.name === "exit_plan_mode") return planCard(c);
    const card = el("div", "agent-approval");
    const title = el("p", "agent-approval-title");
    title.append(document.createTextNode(c.agent ? `Agent con ${c.agent} muốn ` : "Cho phép "), el("strong", "", (isMcp(c.name) ? "gọi MCP" : VERBS[c.name] || c.name).toLowerCase()),
      document.createTextNode(" "), el("code", "", target(c)), document.createTextNode("?"));
    const note = el("input", "agent-approval-note");
    note.placeholder = "Lý do từ chối (tùy chọn)";
    const actions = el("div", "agent-approval-actions");
    const choices = [["allow", "Cho phép", "agent-btn agent-btn-primary", "Đã cho phép."]];
    if (c.remember) choices.push(["allow_always", `Luôn cho phép ${c.remember}`, "agent-btn", `Đã lưu luật: ${c.remember}`]);
    // A plugin asks every time, so "do it all" would promise something it cannot keep.
    if (c.name !== "plugin_write") choices.push(["allow_all", "Tự động", "agent-btn", "Đã chuyển sang tự sửa và chạy."]);
    choices.push(["deny", "Từ chối", "agent-btn", "Đã từ chối."]);
    choices.forEach(([decision, label, cls, done]) => {
      actions.append(button(label, cls, () => decide(card, decision, note.value, done)));
    });
    actions.append(note);
    if (c.why === "untrusted") card.append(el("p", "agent-muted", "Vừa đọc nội dung từ web hoặc MCP, có thể chứa lệnh giả."));
    card.append(title);
    if (c.name === "plugin_write" && !c.args?.remove) {
      // A plugin runs inside the harness with its full rights, so its whole code is shown before it is allowed.
      card.append(el("p", "agent-muted", "Plugin chạy với toàn quyền, không qua sandbox. Đọc kỹ code trước khi cho phép."),
        el("pre", "agent-pre agent-plugin-code", String(c.args?.code || "")));
    }
    card.append(actions);
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
    // A poll whose info has not changed carries only its revision; the panel is left as it is.
    if (!("skills" in snap)) return;
    infoRev = snap.info_rev || "";
    const box = el("div", "agent-info");
    const sb = snap.sandbox || {};
    $("agent-sandbox-info").textContent = sb.mode === "full-access"
      ? "Sandbox tắt: lệnh có toàn quyền."
      : `${SANDBOX_TEXT[sb.backend] || ""} Mạng ${sb.network ? "bật" : "tắt"}.`;
    const named = snap.agents || [];
    if (named.length) box.append(el("p", "agent-muted", `Agent con: ${named.map((a) => a.name).join(", ")}. Lệnh: /rules, /memory, /undo, /plan, /goal.`));
    const skills = snap.skills || [];
    box.append(el("p", "agent-muted", skills.length ? `Skill: ${skills.map((s) => s.name).join(", ")}` : "Chưa có skill."));
    (snap.mcp || []).forEach((server) => {
      const state = { running: `${server.tools} công cụ`, untrusted: "chưa cho phép", failed: `lỗi: ${server.error}`, disabled: "tắt", stopped: "đã dừng" }[server.state] || server.state;
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
    const made = snap.created || [];
    if (made.length) box.append(el("p", "agent-muted", `Công cụ đã tạo: ${made.map((t) => `${t.name}${t.kind === "read" ? " (chỉ đọc)" : ""}${t.scope === "workspace" ? " · dự án" : ""}`).join(", ")}. Xem: /tools.`));
    if ((snap.untrusted_tools || []).length) {
      const line = el("p", "agent-muted agent-info-row", `Công cụ dự án chưa cho phép: ${snap.untrusted_tools.join(", ")}`);
      line.append(button("Cho phép", "agent-btn", () => trust("tools")));
      box.append(line);
    }
    if (snap.disabled?.length) box.append(el("p", "agent-muted", `Đã tắt: ${snap.disabled.join(", ")}`));
    const hooks = snap.hooks || {};
    if (hooks.user || hooks.workspace) {
      const row = el("p", "agent-muted agent-info-row", `Hook: ${hooks.user} của bạn, ${hooks.workspace} của dự án${hooks.workspace && !hooks.workspace_trusted ? " (chưa cho phép)" : ""}`);
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
    const state = { running: "Đang làm", waiting: "Chờ bạn" }[snap.status];
    const parts = [snap.title || "Phiên mới", `${usage.prompt_tokens || 0} token vào, ${usage.completion_tokens || 0} ra`];
    const stats = snap.stats || {};
    if (usage.prompt_tokens) parts.push(`cache ${stats.cache_pct || 0}%`);
    if (stats.cost != null) parts.push(`$${stats.cost}`);
    if (stats.model_seconds || stats.tool_seconds) parts.push(`model ${stats.model_seconds}s, công cụ ${stats.tool_seconds}s`);
    if (snap.text_tools) parts.push("gọi công cụ dạng văn bản");
    if (snap.plan_mode) parts.push("lập kế hoạch");
    if (snap.goal) parts.push(`mục tiêu: ${snap.goal}`);
    $("agent-status").textContent = parts.join(" · ");
    // A /effort typed in the chat moves the picker too.
    if (typeof snap.effort === "string" && $("agent-effort").value !== snap.effort) {
      $("agent-effort").value = snap.effort;
      showPicks();
    }
    const log = $("agent-log");
    log.querySelector(".agent-working")?.remove();
    log.querySelector(".agent-live")?.remove();
    const live = snap.live;
    if (live && (live.text || live.tools.length)) {
      const tools = live.tools.length ? `\nĐang gọi: ${live.tools.join(", ")}` : "";
      log.append(el("pre", "agent-live", live.text + tools));
    }
    if (state) log.append(el("p", "agent-working", `${state}…`));
    showTodos(snap.todos);
    showInfo(snap);
    showKeyPrompt(snap.key_prompt);
    showPicks();
  }

  // The composer's two menus name what is picked inside them.
  const MODE_SHORT = { ask: "Hỏi trước", edits: "Tự sửa", bypass: "Tự động" };
  const SANDBOX_SHORT = { "read-only": "chỉ đọc", "workspace-write": "ghi thư mục", "full-access": "không sandbox" };
  const EFFORT_SHORT = { low: "suy luận thấp", medium: "suy luận vừa", high: "suy luận cao", xhigh: "suy luận tối đa" };
  function showPicks() {
    const network = $("agent-network").checked ? " · mạng" : "";
    $("agent-perm-label").textContent = `${MODE_SHORT[$("agent-mode").value] || ""} · ${SANDBOX_SHORT[$("agent-sandbox").value] || ""}${network}`;
    const provider = $("agent-provider").selectedOptions[0]?.textContent || "";
    const model = $("agent-model").value.trim();
    const effort = EFFORT_SHORT[$("agent-effort").value] ? ` · ${EFFORT_SHORT[$("agent-effort").value]}` : "";
    $("agent-model-label").textContent = (model ? (provider ? `${model} · ${provider}` : model) : (provider ? `${provider} · chọn model` : "Chọn model")) + effort;
  }

  // The effort picked for a model comes back when that model is chosen again.
  const effortKey = () => `manga_agent_effort_${$("agent-provider").value}_${$("agent-model").value.trim()}`;
  function loadEffort() {
    $("agent-effort").value = store("get", effortKey()) || "";
  }

  // The session list sits in a side column that folds away; on a narrow screen it opens over the chat.
  const narrow = () => window.matchMedia("(max-width: 900px)").matches;
  function showSide(open) {
    $("agent-view").classList.toggle("agent-side-closed", !open);
    $("agent-side-toggle").setAttribute("aria-expanded", String(open));
    if (!narrow()) store("set", "manga_agent_side", open ? "open" : "closed");
  }

  // A key that hit a limit with no wait named: ask in the middle of the screen whether to move on to the next stored key.
  let keyDialog = null;
  function showKeyPrompt(prompt) {
    if (!prompt) {
      if (keyDialog?.open) keyDialog.close();
      return;
    }
    const shown = `${prompt.index}>${prompt.next}`;
    if (keyDialog?.open && keyDialog.dataset.shown === shown) return;
    if (!keyDialog) {
      keyDialog = el("dialog", "agent-key-dialog");
      keyDialog.addEventListener("cancel", (event) => event.preventDefault());
      document.body.append(keyDialog);
    }
    keyDialog.dataset.shown = shown;
    const answer = (decision) => {
      keyDialog.querySelectorAll("button").forEach((b) => { b.disabled = true; });
      post(`/api/agent/sessions/${session.id}/key`, { decision })
        .then(() => keyDialog.close())
        .catch((error) => {
          keyDialog.querySelectorAll("button").forEach((b) => { b.disabled = false; });
          window.showToast?.(error.message, "error");
        });
    };
    const next = el("button", "agent-btn agent-btn-primary", `Đổi sang key ${prompt.next}`);
    next.type = "button";
    next.addEventListener("click", () => answer("next"));
    const stop = el("button", "agent-btn", "Dừng");
    stop.type = "button";
    stop.addEventListener("click", () => answer("stop"));
    const actions = el("div", "agent-approval-actions");
    actions.append(stop, next);
    keyDialog.replaceChildren(
      el("p", "agent-key-title", `Key ${prompt.index}/${prompt.count} (${prompt.provider}) bị giới hạn`),
      el("p", "agent-key-detail", prompt.detail || "Không rõ thời gian chờ."),
      el("p", "agent-key-ask", `Đổi sang key ${prompt.next}/${prompt.count} và chạy tiếp?`),
      actions,
    );
    if (!keyDialog.open) keyDialog.showModal();
    next.focus();
  }

  async function poll() {
    clearTimeout(pollTimer);
    if (!session) return;
    try {
      const stick = nearBottom();
      const snap = await call(`/api/agent/sessions/${session.id}?after=${lastSeq}&rev=${encodeURIComponent(infoRev)}`);
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
    loadEffort();
    showPicks();
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
      $("agent-status").textContent = "Chưa có API key. Thêm trong Cài đặt, mục Dịch vụ AI.";
      return;
    }
    const saved = store("get", "manga_agent_provider");
    if (ready.some((p) => p.id === saved)) $("agent-provider").value = saved;
    await loadModels();
  }

  async function loadSessions() {
    try {
      const data = await call("/api/agent/sessions");
      let folder = null;
      const rows = (data.sessions || []).slice(0, 30).flatMap((row) => {
        const item = el("li", row.id === session?.id ? "agent-session-current" : "");
        const open = button(row.title || "(chưa có tin nhắn)", "agent-session-open", () => {
          if (narrow()) showSide(false);
          resume(row.id);
        });
        open.title = `${row.model} · ${row.workspace}`;
        const remove = button("Xóa", "agent-session-del", async () => {
          await call(`/api/agent/sessions/${row.id}`, { method: "DELETE" }).catch(() => {});
          if (row.id === session?.id) resetLog();
          loadSessions();
        });
        item.append(open, remove);
        const place = (row.workspace || "").split(/[\\/]/).filter(Boolean).pop() || row.workspace || "";
        if (place === folder) return [item];
        folder = place;
        const head = el("li", "agent-side-head", place);
        head.title = row.workspace || "";
        return [head, item];
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
      $("agent-effort").value = snap.effort || "";
      showPicks();
      snap.events.forEach((event) => { if (event.type !== "approval") render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
      document.querySelectorAll(".agent-menu[open]").forEach((menu) => { menu.open = false; });
      follow(true);
      loadSessions();
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  // The Chaos title drawn like a webtoon logo: blade-tipped letters, a magic circle and a red eye for the O (the one colour, kept in the art, not the layout CSS).
  const SVG_NS = "http://www.w3.org/2000/svg";
  const GLYPHS = [
    "M152 58L148 76L96 80L84 98L80 148L94 162L146 160L140 174L82 178L64 152L68 94L86 68Z",
    "M168 72L186 64L185 178L176 204L167 176Z", "M226 62L244 68L242 180L224 174Z", "M186 114L226 108L226 122L186 128Z",
    "M252 182L270 182L304 96L350 2L292 70Z", "M306 84L322 72L354 180L336 184Z", "M282 136L336 132L338 144L280 148Z",
    "M562 58L556 74L500 80L492 94L498 104L546 116L558 138L544 170L452 184L470 166L528 160L538 144L530 132L480 122L468 102L478 78L498 66Z",
  ];
  const EYE = "M370 104L392 72L428 70L448 102L440 146L414 170L380 166L366 136Z";

  function svg(tag, attrs = {}, children = []) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    node.append(...children);
    return node;
  }

  function chaosMark() {
    const at = (r, a) => `${(300 + Math.cos(a) * r).toFixed(1)} ${(130 + Math.sin(a) * r).toFixed(1)}`;
    const ticks = Array.from({ length: 72 }, (_, i) => {
      const a = (i / 72) * Math.PI * 2;
      return svg("path", { d: `M${at(i % 6 ? 122 : 114, a)}L${at(128, a)}` });
    });
    const star = Array.from({ length: 7 }, (_, i) => `${i ? "L" : "M"}${at(104, -Math.PI / 2 + (i * 6 * Math.PI) / 7)}`).join("") + "Z";
    const runes = svg("textPath", { href: "#chaos-ring" });
    runes.textContent = "· CHAOS · カオス · 混沌 · CHAOS · カオス · 混沌 · CHAOS · カオス ·";
    const kana = [..."カオス"].map((ch, i) => { const t = svg("text", { x: 572, y: 60 + i * 26, class: "chaos-kana" }); t.textContent = ch; return t; });
    return svg("svg", { class: "chaos-mark", viewBox: "0 0 600 260", role: "img", "aria-label": "Chaos" }, [
      svg("defs", {}, [
        svg("filter", { id: "chaos-glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, [svg("feGaussianBlur", { stdDeviation: 5 })]),
        svg("path", { id: "chaos-ring", d: "M300 130m-138 0a138 138 0 1 1 276 0a138 138 0 1 1 -276 0" }),
        svg("g", { id: "chaos-glyphs" }, GLYPHS.map((d) => svg("path", { d }))),
      ]),
      svg("g", { class: "chaos-circle" }, [
        ...[128, 112, 60].map((r) => svg("circle", { cx: 300, cy: 130, r })),
        svg("g", { class: "chaos-ticks" }, ticks), svg("path", { d: star }), svg("text", { class: "chaos-runes" }, [runes]),
      ]),
      svg("g", { transform: "skewX(-6) translate(14 0)" }, [
        svg("use", { href: "#chaos-glyphs", class: "chaos-shade", x: 3, y: 4 }),
        svg("use", { href: "#chaos-glyphs", class: "chaos-letters" }),
        svg("g", { class: "chaos-eye" }, [
          svg("path", { d: EYE, class: "chaos-eye-glow", stroke: "#ff2a36", filter: "url(#chaos-glow)" }),
          svg("path", { d: EYE, class: "chaos-eye-lid", stroke: "#d0121b" }),
          svg("path", { d: "M408 88L418 120L408 152L398 120Z", class: "chaos-pupil", fill: "#d0121b" }),
        ]),
      ]),
      ...kana,
    ]);
  }

  function emptyState() {
    const box = el("div", "agent-empty");
    box.append(chaosMark());
    return box;
  }

  function resetLog() {
    clearTimeout(pollTimer);
    session = null;
    lastSeq = 0;
    infoRev = "";
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
      provider: $("agent-provider").value, model: $("agent-model").value.trim(), mode: $("agent-mode").value, effort: $("agent-effort").value,
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
      window.showToast?.("Chọn dịch vụ và model.", "error");
      $("agent-model-menu").open = true;
      $("agent-model").focus();
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
    $("agent-input").placeholder = name ? `Việc cần làm trong ${name}… (/ xem lệnh)` : "Việc cần làm… (/ xem lệnh)";
    $("agent-input").title = folder;
  }

  // The MCP panel: every server the agent would start here, and adding, testing, switching and removing your own.
  const MCP_STATE = { running: "đang chạy", untrusted: "chờ cho phép", failed: "lỗi", disabled: "đã tắt", stopped: "đã dừng", waiting: "chờ" };
  const mcpQuery = () => `?workspace=${encodeURIComponent($("agent-workspace").value.trim())}`;

  function mcpKind() {
    return document.querySelector('input[name="agent-mcp-kind"]:checked')?.value || "stdio";
  }

  function showMcpKind() {
    const kind = mcpKind();
    $("agent-mcp-stdio").hidden = kind !== "stdio";
    $("agent-mcp-http").hidden = kind !== "http";
    $("agent-mcp-json").hidden = kind !== "json";
    $("agent-mcp-name").placeholder = kind === "json" ? "chỉ cần khi JSON là một server" : "context7";
  }

  function pairs(text, separator) {
    const out = {};
    text.split("\n").map((line) => line.trim()).filter(Boolean).forEach((line) => {
      const at = line.indexOf(separator);
      if (at > 0) out[line.slice(0, at).trim()] = line.slice(at + 1).trim();
    });
    return out;
  }

  function showMcp(data) {
    $("agent-mcp-file").textContent = data.file ? `Lưu ở ${data.file} (agent không đọc được).` : "";
    const rows = (data.servers || []).map((server) => {
      const item = el("li");
      const title = el("span");
      const where = server.scope === "workspace" ? "của dự án" : server.editable ? "của bạn" : server.source;
      title.append(el("b", "", server.name), el("span", "agent-skill-tag", ` · ${server.type} · ${where}`));
      const actions = el("span", "agent-skill-row");
      if (server.scope !== "workspace") actions.append(button("Thử", "agent-btn", () => testMcp(server.name, state)));
      if (server.editable) {
        actions.append(button(server.disabled ? "Bật" : "Tắt", "agent-btn", () => switchMcp(server.name, server.disabled)),
          button("Xóa", "agent-btn", () => removeMcp(server.name)));
      }
      const secrets = [...Object.keys(server.env || {}), ...Object.keys(server.headers || {})];
      const about = el("span", "agent-muted", server.target + (secrets.length ? ` · ${secrets.join(", ")}` : ""));
      const label = server.disabled ? "đã tắt" : server.state ? `${MCP_STATE[server.state] || server.state}${server.state === "running" ? `, ${server.tools} công cụ` : ""}` : "";
      const state = el("span", `agent-mcp-state${server.state === "running" ? " ok" : server.state === "failed" ? " bad" : ""}`, label + (server.error ? `: ${server.error}` : ""));
      item.append(title, actions, about, state);
      if ((server.tool_list || []).length) item.append(mcpTools(server));
      return item;
    });
    $("agent-mcp-list").replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", "Chưa có MCP server nào.")]));
  }

  function mcpTools(server) {
    const box = el("details", "agent-mcp-tools");
    const on = server.tool_list.filter((t) => t.on).length;
    box.append(el("summary", "agent-muted", `Công cụ bật: ${on}/${server.tool_list.length}`));
    server.tool_list.forEach((tool) => {
      const row = el("label", "agent-mcp-tool");
      const box_ = document.createElement("input");
      box_.type = "checkbox";
      box_.checked = tool.on;
      box_.disabled = !server.editable;
      box_.addEventListener("change", () => switchMcpTool(server.name, tool.name, box_.checked));
      row.title = tool.description;
      row.append(box_, el("span", "", tool.name));
      box.append(row);
    });
    return box;
  }

  async function switchMcpTool(name, tool, enabled) {
    try {
      showMcp(await post(`/api/agent/mcp/${encodeURIComponent(name)}/tools/${encodeURIComponent(tool)}${mcpQuery()}`, { enabled }));
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  async function loadMcp() {
    try {
      showMcp(await call(`/api/agent/mcp${mcpQuery()}`));
    } catch (error) {
      $("agent-mcp-list").replaceChildren(el("li", "agent-muted", error.message));
    }
  }

  async function testMcp(name, label) {
    label.textContent = "đang thử…";
    label.className = "agent-mcp-state";
    try {
      const data = await post(`/api/agent/mcp/${encodeURIComponent(name)}/test${mcpQuery()}`);
      label.textContent = data.ok ? `Đã kết nối, ${data.tools.length} công cụ: ${data.tools.slice(0, 8).join(", ")}${data.tools.length > 8 ? "…" : ""}` : `Lỗi kết nối: ${data.error}`;
      label.className = `agent-mcp-state ${data.ok ? "ok" : "bad"}`;
      return data.ok;
    } catch (error) {
      label.textContent = error.message;
      label.className = "agent-mcp-state bad";
      return false;
    }
  }

  async function saveMcp() {
    const kind = mcpKind();
    const name = $("agent-mcp-name").value.trim();
    const body = { name };
    if (kind === "json") body.pasted = $("agent-mcp-pasted").value;
    else if (kind === "http") body.config = { type: "http", url: $("agent-mcp-url").value.trim(), headers: pairs($("agent-mcp-headers").value, ":") };
    else body.config = { command: $("agent-mcp-command").value.trim(), env: pairs($("agent-mcp-env").value, "=") };
    const save = $("agent-mcp-save");
    save.disabled = true;
    try {
      const data = await post(`/api/agent/mcp${mcpQuery()}`, body);
      showMcp(data);
      if (data.warning) window.showToast?.(`Đã lưu, nhưng ${data.warning}.`, "error");
      ["agent-mcp-name", "agent-mcp-command", "agent-mcp-env", "agent-mcp-url", "agent-mcp-headers", "agent-mcp-pasted"].forEach((id) => { $(id).value = ""; });
      for (const added of data.added || []) {
        const row = [...$("agent-mcp-list").children].find((item) => item.querySelector("b")?.textContent === added);
        const label = row?.querySelector(".agent-mcp-state");
        if (label) window.showToast?.(await testMcp(added, label) ? `Đã thêm ${added}, dùng từ lượt sau.` : `Đã lưu ${added}, chưa kết nối được.`, "info");
      }
    } catch (error) {
      window.showToast?.(error.message, "error");
    } finally {
      save.disabled = false;
    }
  }

  async function switchMcp(name, enable) {
    try {
      showMcp(await post(`/api/agent/mcp/${encodeURIComponent(name)}/enabled${mcpQuery()}`, { enabled: enable }));
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  async function removeMcp(name) {
    if (!window.confirm(`Xóa MCP server ${name}?`)) return;
    try {
      showMcp(await call(`/api/agent/mcp/${encodeURIComponent(name)}${mcpQuery()}`, { method: "DELETE" }));
      window.showToast?.(`Đã xóa ${name}.`, "success");
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
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
    skillsChanged(post("/api/agent/skills/install", { source }), (names) => `Đã cài ${names.join(", ")}. Nên đọc SKILL.md trước khi dùng.`)
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
    skillsChanged(post("/api/agent/skills", body), () => `Đã lưu skill ${name}.`).then((ok) => {
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
    $("agent-more").open = false;
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
      showEngine(config.engine || {});
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
    $("agent-new").addEventListener("click", () => {
      if (narrow()) showSide(false);
      resetLog();
      loadSessions();
      $("agent-input").focus();
    });
    showSide(!narrow() && store("get", "manga_agent_side") !== "closed");
    $("agent-side-toggle").addEventListener("click", () => showSide($("agent-view").classList.contains("agent-side-closed")));
    // On a narrow screen the column covers the chat, so a tap outside it folds it away.
    document.addEventListener("click", (event) => {
      if (!narrow() || $("agent-view").classList.contains("agent-side-closed")) return;
      if (!event.target.closest("#agent-side, #agent-side-toggle")) showSide(false);
    });
    $("agent-provider").addEventListener("change", () => { resetLog(); loadModels(); });
    $("agent-workspace").addEventListener("change", () => { resetLog(); showFolder(); });
    $("agent-skill-body").value = SKILL_TEMPLATE;
    $("agent-skills-menu").addEventListener("toggle", () => { if ($("agent-skills-menu").open) loadSkills(); });
    $("agent-mcp-menu").addEventListener("toggle", () => { if ($("agent-mcp-menu").open) loadMcp(); });
    document.querySelectorAll('input[name="agent-mcp-kind"]').forEach((radio) => radio.addEventListener("change", showMcpKind));
    $("agent-mcp-save").addEventListener("click", saveMcp);
    $("agent-skill-install").addEventListener("click", installSkills);
    $("agent-skill-source").addEventListener("keydown", (event) => { if (event.key === "Enter") installSkills(); });
    $("agent-skill-file").addEventListener("change", uploadSkills);
    $("agent-skill-save").addEventListener("click", saveSkill);
    $("agent-skill-draft").addEventListener("click", draftSkill);
    $("agent-model").addEventListener("change", () => {
      loadEffort();
      showPicks();
      syncSession({ model: $("agent-model").value.trim(), effort: $("agent-effort").value });
    });
    $("agent-effort").addEventListener("change", () => {
      store("set", effortKey(), $("agent-effort").value);
      showPicks();
      syncSession({ effort: $("agent-effort").value });
    });
    $("agent-model").addEventListener("input", showPicks);
    $("agent-mode").addEventListener("change", () => syncSession({ mode: $("agent-mode").value }));
    const saveEngine = async () => {
      try {
        const engine = await post("/api/agent/sandbox/engine", { engine: $("agent-engine").value, image: $("agent-engine-image").value.trim() });
        if (engine.engine === "docker" && !engine.docker) window.showToast?.("Không thấy Docker, dùng sandbox hệ điều hành.", "error");
        showEngine(engine);
        $("agent-sandbox-info").textContent = SANDBOX_TEXT[engine.backend] || "";
      } catch (error) {
        window.showToast?.(error.message, "error");
      }
    };
    $("agent-engine").addEventListener("change", saveEngine);
    $("agent-engine-image").addEventListener("change", saveEngine);
    $("agent-sandbox").addEventListener("change", () => syncSession({ sandbox: $("agent-sandbox").value }));
    $("agent-network").addEventListener("change", () => syncSession({ network: $("agent-network").checked }));
    ["agent-mode", "agent-sandbox", "agent-network"].forEach((id) => $(id).addEventListener("change", showPicks));
    document.querySelectorAll(".agent-menu").forEach((menu) => menu.addEventListener("toggle", () => {
      if (menu.open) document.querySelectorAll(".agent-menu[open]").forEach((other) => { if (other !== menu) other.open = false; });
    }));
  });
})();
