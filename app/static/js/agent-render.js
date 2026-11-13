(() => {
  const HEAD = { "X-Manga-Agent": "1" };
  const VERBS = {
    "": "Lệnh lỗi",
    list_dir: "Xem thư mục", read_file: "Đọc", search: "Tìm", glob: "Tìm file", write_file: "Ghi", edit_file: "Sửa",
    run_command: "Chạy", web_fetch: "Tải trang", skill: "Skill", todo_write: "Kế hoạch",
    plugin_write: "Ghi plugin", task: "Agent con", spawn_agent: "Tạo agent con", wait_agent: "Chờ agent con",
    send_input: "Nhắn agent con", close_agent: "Đóng agent con", memory: "Ghi nhớ", ask_user: "Hỏi bạn",
    exit_plan_mode: "Kế hoạch", goal_done: "Xong mục tiêu",
    symbols: "Tìm ký hiệu", web_search: "Tìm web", web_download: "Tải về", view_image: "Xem ảnh",
    tool_search: "Tìm công cụ", run_script: "Chạy script", fan_out: "Chạy song song",
    job_output: "Đọc tác vụ", job_input: "Nhập vào tác vụ", job_stop: "Dừng tác vụ",
    context_notes: "Ghi chú", new_context: "Ngữ cảnh mới",
    schedule_create: "Hẹn giờ", schedule_list: "Lịch hẹn", schedule_delete: "Hủy hẹn",
    use_model: "Dùng model trả phí",
  };

  const TALLY = {
    run_command: ["chạy", "lệnh"], read_file: ["đọc", "file"], edit_file: ["sửa", "file"], write_file: ["sửa", "file"],
    spawn_agent: ["tạo", "agent con"], wait_agent: ["chờ", "agent con"],
    send_input: ["nhắn", "agent con"], close_agent: ["đóng", "agent con"], memory: ["ghi nhớ", "mục"],
    list_dir: ["xem", "thư mục"], search: ["tìm", "lần"], glob: ["tìm", "lần"], symbols: ["tìm", "lần"],
    web_fetch: ["tải", "trang"], skill: ["mở", "skill"], task: ["chạy", "agent con"], mcp: ["gọi", "công cụ MCP"],
    run_script: ["chạy", "script"], fan_out: ["chạy", "lô song song"], job_input: ["nhập vào", "tác vụ"],
    job_output: ["đọc", "tác vụ"], job_stop: ["dừng", "tác vụ"],
    schedule_create: ["hẹn", "lượt"], web_search: ["tìm web", "lần"],
  };

  const SANDBOX_TEXT = {
    landlock: "Sandbox Landlock (Linux).", seatbelt: "Sandbox Seatbelt (macOS).",
    docker: "Container Docker, chỉ thấy /workspace.",
    "low-integrity": "Windows integrity thấp: chỉ ghi trong thư mục làm việc, vẫn đọc file và dùng mạng nên cần duyệt.",
    none: "Không có sandbox: mọi lệnh cần duyệt, trừ chế độ Tự động.",
  };

  const $ = (id) => document.getElementById(id);

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

  function codeCard(code, label = "code", extraCls = "") {
    const clean = String(code || "").replace(/\n$/, "");
    const card = el("div", `agent-code-card${extraCls ? ` ${extraCls}` : ""}`);
    const bar = el("div", "agent-code-bar");
    bar.append(el("span", "agent-code-lang", label || "code"));
    const copyBtn = button("Sao chép", "agent-code-copy", () => {
      navigator.clipboard?.writeText(clean).then(() => {
        copyBtn.textContent = "Đã chép";
        setTimeout(() => { copyBtn.textContent = "Sao chép"; }, 1400);
      }).catch(() => {});
    });
    bar.append(copyBtn);
    card.append(bar, el("pre", "agent-pre", clean));
    return card;
  }

  function prose(text) {
    const wrap = el("div", "agent-prose");
    const raw = String(text || "");
    const parts = raw.split(/```([\w-]*)\n?/);
    for (let i = 0; i < parts.length; i += 2) {
      const chunk = parts[i];
      if (chunk) {
        chunk.replace(/^(#{1,4} .*)\n(?=\S)/gm, "$1\n\n").split(/\n{2,}/).forEach((block) => {
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
      }
      if (i + 2 < parts.length) {
        const lang = parts[i + 1] || "code";
        const body = parts[i + 2] || "";
        wrap.append(codeCard(body, lang));
        i += 1;
      }
    }
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

  function formatTokens(n) {
    const val = Number(n || 0);
    return val >= 1000 ? `${(val / 1000).toFixed(1)}k` : String(val);
  }

  window._AgentUI = {
    VERBS,
    TALLY,
    SANDBOX_TEXT,
    $,
    store,
    call,
    post,
    showEngine,
    el,
    button,
    nearBottom,
    follow,
    inline,
    codeCard,
    prose,
    isMcp,
    target,
    editPairs,
    diffStats,
    stats,
    formatTokens,
  };
})();
