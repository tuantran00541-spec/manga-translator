// Chaos: a chat with a coding agent that reads, edits and runs commands in one folder.
(() => {
  const $ = (id) => document.getElementById(id);
  const TOOL_LABELS = { read_file: "đọc file", write_file: "ghi file", edit_file: "sửa file", run_command: "chạy lệnh" };
  const STATUS = { idle: "Sẵn sàng", running: "Đang làm…", waiting: "Chờ bạn duyệt", error: "Lỗi" };
  let config = null;
  let current = null;
  let timer = null;

  async function call(path, options = {}) {
    const response = await fetch(`/api/agent${path}`, {
      ...options,
      headers: { "X-Manga-Agent": "1", "Content-Type": "application/json" },
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }

  function fail(error) {
    showToast(error.message || String(error), "error");
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function argsText(name, args) {
    if (name === "run_command") return args.command || "";
    if (name === "edit_file") return `${args.path}\n--- cũ\n${args.old_text || ""}\n+++ mới\n${args.new_text || ""}`;
    if (name === "write_file") return `${args.path}\n${args.content || ""}`;
    return JSON.stringify(args, null, 2);
  }

  function parseArgs(raw) {
    try { return JSON.parse(raw || "{}"); } catch { return {}; }
  }

  function renderLog(messages) {
    const log = $("agent-log");
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    const results = new Map(messages.filter((m) => m.role === "tool").map((m) => [m.tool_call_id, m.content]));
    log.replaceChildren();
    for (const message of messages) {
      if (message.role === "user") log.append(el("div", "agent-msg agent-user", message.content));
      if (message.role !== "assistant") continue;
      if (message.content) log.append(el("div", "agent-msg agent-reply", message.content));
      for (const tool of message.tool_calls || []) {
        const args = parseArgs(tool.function.arguments);
        const card = el("details", "agent-tool");
        const head = args.command || args.path || "";
        card.append(el("summary", "", `${TOOL_LABELS[tool.function.name] || tool.function.name} ${head}`.trim()));
        card.append(el("pre", "", argsText(tool.function.name, args)));
        card.append(el("pre", "agent-result", results.has(tool.id) ? results.get(tool.id) : "…"));
        log.append(card);
      }
    }
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function renderSession(view) {
    current = view;
    const busy = view.status === "running" || view.status === "waiting";
    $("agent-setup").hidden = true;
    $("agent-mode").value = view.mode;
    $("agent-status").textContent = `${view.model} · ${view.workspace} · ${view.error ? `Lỗi: ${view.error}` : STATUS[view.status] || view.status}`;
    renderLog(view.messages);
    const pending = view.pending;
    $("agent-approval").hidden = !pending;
    if (pending) {
      $("agent-approval-name").textContent = TOOL_LABELS[pending.name] || pending.name;
      $("agent-approval-args").textContent = argsText(pending.name, pending.args);
    }
    $("agent-stop").hidden = !busy;
    $("agent-send").disabled = busy;
    clearTimeout(timer);
    if (busy) timer = setTimeout(() => load(view.id), 1000);
  }

  async function load(id) {
    try {
      renderSession(await call(`/sessions/${id}`));
    } catch (error) {
      fail(error);
    }
  }

  async function listSessions() {
    const list = $("agent-sessions");
    const sessions = await call("/sessions");
    list.replaceChildren();
    for (const session of sessions) {
      const item = el("li");
      const open = el("button", `agent-session${current?.id === session.id ? " active" : ""}`, session.title || "(chưa có tin nhắn)");
      open.type = "button";
      open.addEventListener("click", () => load(session.id).then(listSessions));
      const remove = el("button", "agent-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", "Xoá phiên");
      remove.addEventListener("click", async () => {
        try {
          await call(`/sessions/${session.id}`, { method: "DELETE" });
          if (current?.id === session.id) newSession();
          await listSessions();
        } catch (error) {
          fail(error);
        }
      });
      item.append(open, remove);
      list.append(item);
    }
  }

  function newSession() {
    current = null;
    clearTimeout(timer);
    $("agent-setup").hidden = false;
    $("agent-approval").hidden = true;
    $("agent-stop").hidden = true;
    $("agent-send").disabled = false;
    $("agent-status").textContent = "Chọn model và thư mục, rồi giao việc.";
    $("agent-log").replaceChildren();
  }

  async function send(event) {
    event.preventDefault();
    const text = $("agent-input").value.trim();
    if (!text) return;
    try {
      if (!current) {
        if (!$("agent-setup").reportValidity()) return;
        current = await call("/sessions", { method: "POST", body: {
          provider: $("agent-provider").value, model: $("agent-model").value.trim(),
          workspace: $("agent-workspace").value.trim(), mode: $("agent-mode").value, effort: $("agent-effort").value,
        } });
      }
      await call(`/sessions/${current.id}/messages`, { method: "POST", body: { text } });
      $("agent-input").value = "";
      await load(current.id);
      await listSessions();
    } catch (error) {
      fail(error);
    }
  }

  async function decide(allow) {
    if (!current) return;
    try {
      await call(`/sessions/${current.id}/approval`, { method: "POST", body: { allow } });
      await load(current.id);
    } catch (error) {
      fail(error);
    }
  }

  async function openAgentView() {
    try {
      if (!config) {
        config = await call("/config");
        const select = $("agent-provider");
        select.replaceChildren(...config.providers.map((p) => Object.assign(el("option", "", p.label), { value: p.id })));
        const pick = () => { $("agent-model").value = config.providers.find((p) => p.id === select.value)?.model || ""; };
        select.addEventListener("change", pick);
        pick();
        const launched = new URLSearchParams(location.search).get("workspace");
        $("agent-workspace").value = launched || config.workspace;
        if (!config.providers.length) showToast("Thêm API key trong Cài đặt trước khi dùng agent.", "error");
        newSession();
      }
      await listSessions();
    } catch (error) {
      fail(error);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("agent-view")) return;
    $("agent-new").addEventListener("click", () => { newSession(); listSessions().catch(fail); });
    $("agent-composer").addEventListener("submit", send);
    $("agent-input").addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
      event.preventDefault();
      $("agent-composer").requestSubmit();
    });
    $("agent-allow").addEventListener("click", () => decide(true));
    $("agent-deny").addEventListener("click", () => decide(false));
    $("agent-stop").addEventListener("click", () => current && call(`/sessions/${current.id}/stop`, { method: "POST" }).catch(fail));
    $("agent-mode").addEventListener("change", () => {
      if (current) call(`/sessions/${current.id}`, { method: "PATCH", body: { mode: $("agent-mode").value } }).catch(fail);
    });
  });

  window.openAgentView = openAgentView;
})();
