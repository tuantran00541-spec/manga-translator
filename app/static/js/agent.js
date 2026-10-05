// Agent: a coding agent on any configured model, with tool calls, approvals, skills, MCP and hooks shown in the page.
(() => {
  const POLL_MS = 700;
  const HEAD = { "X-Manga-Agent": "1" };
  const TOOL_LABELS = {
    list_dir: "Xem thư mục", read_file: "Đọc file", search: "Tìm", glob: "Tìm file", write_file: "Ghi file",
    edit_file: "Sửa file", apply_patch: "Vá file", run_command: "Chạy lệnh", web_fetch: "Đọc trang web",
    skill: "Mở skill", todo_write: "Lập kế hoạch", task: "Giao agent phụ",
  };
  const SANDBOX_TEXT = {
    landlock: "Lệnh chạy trong sandbox Landlock của Linux.", seatbelt: "Lệnh chạy trong sandbox Seatbelt của macOS.",
    none: "Máy này không có sandbox cho lệnh, nên lệnh luôn chờ bạn duyệt (trừ khi chọn Tự làm hết).",
  };
  const $ = (id) => document.getElementById(id);
  let session = null;
  let lastSeq = 0;
  let pollTimer = null;
  let loaded = false;
  let commands = [];

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

  function scrollLog() {
    const log = $("agent-log");
    log.scrollTop = log.scrollHeight;
  }

  // Plain text with fenced code blocks kept as code.
  function richText(text) {
    const wrap = el("div", "agent-text");
    String(text || "").split(/```[\w-]*\n?/).forEach((part, i) => {
      if (!part) return;
      wrap.append(i % 2 ? el("pre", "agent-code", part.replace(/\n$/, "")) : el("p", "", part.trim()));
    });
    return wrap;
  }

  function toolLabel(name) {
    if (TOOL_LABELS[name]) return TOOL_LABELS[name];
    const mcp = /^mcp__(.+?)__(.+)$/.exec(name || "");
    return mcp ? `MCP ${mcp[1]}: ${mcp[2]}` : name || "Công cụ";
  }

  function callSummary(callInfo) {
    const args = callInfo.args || {};
    if (callInfo.name === "run_command") return `${args.command || ""}${args.outside_sandbox ? "  (ngoài sandbox)" : ""}`;
    if (callInfo.name === "search" || callInfo.name === "glob") return `${args.pattern || ""}${args.path ? ` trong ${args.path}` : ""}`;
    if (callInfo.name === "web_fetch") return args.url || "";
    if (callInfo.name === "task") return args.description || "";
    if (callInfo.name === "skill") return args.name || "";
    if (callInfo.name === "apply_patch") return (String(args.patch || "").match(/^\*\*\* (?:Add|Update|Delete) File: .+$/gm) || []).map((l) => l.split(": ")[1]).join(", ");
    if (callInfo.name === "todo_write") return `${(args.items || []).length} việc`;
    return args.path || (String(callInfo.name).startsWith("mcp__") ? JSON.stringify(args).slice(0, 120) : ".");
  }

  function toolCard(callInfo) {
    const card = el("details", "agent-tool");
    card.dataset.callId = callInfo.id;
    const summary = el("summary");
    summary.append(el("strong", "", toolLabel(callInfo.name)), el("code", "", callSummary(callInfo)));
    card.append(summary);
    const args = callInfo.args || {};
    if (callInfo.name === "edit_file") {
      card.append(el("pre", "agent-diff agent-diff-old", args.old_text || ""), el("pre", "agent-diff agent-diff-new", args.new_text || ""));
    } else if (callInfo.name === "write_file") {
      card.append(el("pre", "agent-code", args.content || ""));
    } else if (callInfo.name === "apply_patch") {
      card.append(el("pre", "agent-code agent-patch", args.patch || ""));
    } else if (callInfo.name === "task") {
      card.append(el("pre", "agent-code", args.prompt || ""));
    } else if (String(callInfo.name).startsWith("mcp__")) {
      card.append(el("pre", "agent-code", JSON.stringify(args, null, 2)));
    }
    return card;
  }

  function render(event) {
    const log = $("agent-log");
    log.querySelector(".agent-empty")?.remove();
    if (event.type === "user") {
      log.append(el("div", "agent-msg agent-user", event.text));
    } else if (event.type === "assistant") {
      const msg = el("div", "agent-msg agent-assistant");
      if (event.reasoning) {
        const think = el("details", "agent-reasoning");
        think.append(el("summary", "", "Suy nghĩ"), el("pre", "", event.reasoning));
        msg.append(think);
      }
      if (event.text) msg.append(richText(event.text));
      (event.calls || []).forEach((c) => msg.append(toolCard(c)));
      log.append(msg);
    } else if (event.type === "tool") {
      const card = log.querySelector(`.agent-tool[data-call-id="${CSS.escape(event.id)}"]`);
      const output = el("pre", "agent-output", event.output);
      if (card) {
        card.classList.add(event.ok ? "agent-tool-ok" : "agent-tool-failed");
        card.append(output);
      } else {
        log.append(output);
      }
    } else if (event.type === "approval") {
      // The change waiting for approval is shown open, so it is read before it is allowed.
      const card = log.querySelector(`.agent-tool[data-call-id="${CSS.escape(event.call.id)}"]`);
      if (card) card.open = true;
      log.append(approvalCard(event.call));
    } else if (event.type === "subagent") {
      const text = event.state === "started" ? `Agent phụ bắt đầu: ${event.description}` : `Agent phụ xong: ${event.description} (${event.tools} lần gọi công cụ)`;
      log.append(el("p", "agent-notice", text));
    } else if (event.type === "notice" || event.type === "error") {
      log.append(el("p", `agent-${event.type}`, event.text));
    }
    scrollLog();
  }

  function approvalCard(callInfo) {
    const card = el("div", "agent-approval");
    card.append(el("p", "", `Agent muốn ${toolLabel(callInfo.name).toLowerCase()}: ${callSummary(callInfo)}`));
    const note = el("input", "ui-input");
    note.placeholder = "Ghi chú khi từ chối (tùy chọn)";
    const actions = el("div", "agent-approval-actions");
    const choices = [["allow", "Cho phép", "ui-btn ui-btn-primary"], ["allow_all", "Cho phép hết từ giờ", "ui-btn"], ["deny", "Từ chối", "ui-btn ui-btn-ghost"]];
    choices.forEach(([decision, label, cls]) => {
      actions.append(button(label, cls, async () => {
        actions.querySelectorAll("button").forEach((b) => { b.disabled = true; });
        try {
          await post(`/api/agent/sessions/${session.id}/approval`, { decision, note: note.value });
          card.replaceWith(el("p", "agent-notice", `${label}.`));
          if (decision === "allow_all") $("agent-mode").value = "auto";
          poll();
        } catch (error) {
          window.showToast?.(error.message, "error");
          actions.querySelectorAll("button").forEach((b) => { b.disabled = false; });
        }
      }));
    });
    card.append(note, actions);
    return card;
  }

  function showTodos(items) {
    $("agent-todos").hidden = !items?.length;
    $("agent-todo-list").replaceChildren(...(items || []).map((item) => {
      const row = el("li", `agent-todo agent-todo-${item.status}`, item.content);
      row.title = { pending: "Chưa làm", in_progress: "Đang làm", completed: "Xong" }[item.status] || "";
      return row;
    }));
  }

  async function trust(path) {
    try {
      showInfo(await post(`/api/agent/sessions/${session.id}/trust/${path}`));
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  function showInfo(snap) {
    const box = el("div");
    const sandboxMode = snap.sandbox || {};
    $("agent-sandbox-info").textContent = sandboxMode.mode === "full-access"
      ? "Sandbox tắt: lệnh có toàn quyền trên máy."
      : `${SANDBOX_TEXT[sandboxMode.backend] || ""} Mạng ${sandboxMode.network ? "bật" : "tắt"}.`;
    const skills = snap.skills || [];
    box.append(el("p", "agent-note", skills.length ? `Skill: ${skills.map((s) => s.name).join(", ")}` : "Chưa có skill (thư mục .agents/skills, .claude/skills, .codex/skills)."));
    (snap.mcp || []).forEach((server) => {
      const state = { running: `chạy, ${server.tools} công cụ`, untrusted: "chưa cho phép", failed: `lỗi ${server.error}`, disabled: "tắt" }[server.state] || server.state;
      const row = el("p", "agent-note agent-mcp-row", `MCP ${server.name}: ${state}`);
      if (server.state === "untrusted") row.append(button("Cho phép", "ui-btn ui-btn-compact", () => trust(`mcp/${encodeURIComponent(server.name)}`)));
      box.append(row);
    });
    const hooks = snap.hooks || {};
    if (hooks.user || hooks.workspace) {
      const row = el("p", "agent-note", `Hook: ${hooks.user} của bạn, ${hooks.workspace} của dự án${hooks.workspace && !hooks.workspace_trusted ? " (chưa cho phép)" : ""}`);
      if (hooks.workspace && !hooks.workspace_trusted) row.append(button("Cho phép", "ui-btn ui-btn-compact", () => trust("hooks")));
      box.append(row);
    }
    $("agent-tools-info").replaceChildren(box);
    commands = snap.commands || commands;
  }

  function showStatus(snap) {
    const busy = snap.status === "running" || snap.status === "waiting";
    $("agent-stop").hidden = !busy;
    $("agent-send").disabled = busy;
    const usage = snap.usage || {};
    const parts = [`${snap.provider_label} · ${snap.model}`,
      { idle: "Sẵn sàng", running: "Đang làm…", waiting: "Chờ bạn duyệt" }[snap.status] || snap.status,
      `${usage.prompt_tokens || 0} token vào · ${usage.completion_tokens || 0} token ra`];
    if (snap.text_tools) parts.push("gọi công cụ bằng văn bản");
    $("agent-status").textContent = parts.join(" · ");
    showTodos(snap.todos);
    showInfo(snap);
  }

  async function poll() {
    clearTimeout(pollTimer);
    if (!session) return;
    try {
      const snap = await call(`/api/agent/sessions/${session.id}?after=${lastSeq}`);
      snap.events.forEach((event) => { render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
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
    const select = $("agent-provider");
    const ready = Object.values(data.providers || {}).filter((p) => p.configured && p.id !== "manga-cloud");
    select.replaceChildren(...ready.map((p) => new Option(p.label || p.id, p.id)));
    if (!ready.length) {
      $("agent-status").textContent = "Chưa có dịch vụ nào có API key. Mở Cài đặt, mục Dịch vụ AI để thêm (mọi API chuẩn OpenAI đều dùng được).";
      return;
    }
    const saved = store("get", "manga_agent_provider");
    if (ready.some((p) => p.id === saved)) select.value = saved;
    await loadModels();
  }

  async function loadSessions() {
    try {
      const data = await call("/api/agent/sessions");
      const rows = (data.sessions || []).slice(0, 20).map((row) => {
        const item = el("li", row.id === session?.id ? "agent-session-current" : "");
        const open = button(row.title || "(chưa có tin nhắn)", "agent-session-open", () => resume(row.id));
        open.title = `${row.model} · ${row.workspace}`;
        const remove = button("Xoá", "ui-btn ui-btn-ghost ui-btn-compact", async () => {
          await call(`/api/agent/sessions/${row.id}`, { method: "DELETE" }).catch(() => {});
          if (row.id === session?.id) resetLog();
          loadSessions();
        });
        item.append(open, remove);
        return item;
      });
      $("agent-sessions").replaceChildren(...(rows.length ? rows : [el("li", "agent-note", "Chưa có phiên nào.")]));
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
      loadSessions();
    } catch (error) {
      window.showToast?.(error.message, "error");
    }
  }

  function resetLog() {
    clearTimeout(pollTimer);
    session = null;
    lastSeq = 0;
    $("agent-log").replaceChildren(el("p", "agent-empty", "Phiên mới. Nhắn việc cần làm cho agent, gõ / để xem lệnh."));
    $("agent-stop").hidden = true;
    $("agent-send").disabled = false;
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
    showStatus(session);
    return session;
  }

  function showHints() {
    const hints = $("agent-hints");
    const match = /^\/(\S*)$/.exec($("agent-input").value.trim());
    const rows = match ? commands.filter((c) => c.name.startsWith(match[1])).slice(0, 16) : [];
    hints.hidden = !rows.length;
    hints.replaceChildren(...rows.map((c) => {
      const item = el("li");
      item.append(button(`/${c.name}`, "agent-hint", () => { $("agent-input").value = `/${c.name} `; $("agent-input").focus(); showHints(); }),
        el("span", "", c.description || ""));
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
      showHints();
      if (reply.message) {
        $("agent-log").querySelector(".agent-empty")?.remove();
        $("agent-log").append(el("pre", "agent-output agent-command", reply.message));
        scrollLog();
      }
      poll();
    } catch (error) {
      window.showToast?.(error.message, "error");
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
    $("agent-composer").addEventListener("submit", send);
    $("agent-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) send(event);
    });
    $("agent-input").addEventListener("input", showHints);
    $("agent-stop").addEventListener("click", () => session && post(`/api/agent/sessions/${session.id}/stop`).then(poll).catch(() => {}));
    $("agent-new").addEventListener("click", () => { resetLog(); loadSessions(); });
    $("agent-provider").addEventListener("change", () => { resetLog(); loadModels(); });
    $("agent-workspace").addEventListener("change", resetLog);
    $("agent-model").addEventListener("change", () => syncSession({ model: $("agent-model").value.trim() }));
    $("agent-mode").addEventListener("change", () => syncSession({ mode: $("agent-mode").value }));
    $("agent-sandbox").addEventListener("change", () => syncSession({ sandbox: $("agent-sandbox").value }));
    $("agent-network").addEventListener("change", () => syncSession({ network: $("agent-network").checked }));
    $("agent-setup").addEventListener("submit", (event) => event.preventDefault());
  });
})();
