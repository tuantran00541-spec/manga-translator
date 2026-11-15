(() => {
  const HEAD = { "X-Manga-Agent": "1" };
  const VERBS = {
    "": "Lệnh lỗi",
    list_dir: "Xem thư mục", read_file: "Đọc", search: "Tìm", glob: "Tìm tệp", write_file: "Ghi", edit_file: "Sửa",
    run_command: "Chạy", web_fetch: "Tải trang", skill: "Kỹ năng", todo_write: "Kế hoạch",
    plugin_write: "Ghi phần mở rộng", task: "Tác vụ con", spawn_agent: "Tạo tác vụ con", wait_agent: "Chờ tác vụ con",
    send_input: "Nhắn tác vụ con", close_agent: "Đóng tác vụ con", memory: "Ghi nhớ", ask_user: "Hỏi người dùng",
    exit_plan_mode: "Kế hoạch", goal_done: "Hoàn tất mục tiêu",
    symbols: "Tìm ký hiệu", web_search: "Tìm kiếm web", web_download: "Tải về", view_image: "Xem ảnh",
    tool_search: "Tìm công cụ", create: "Tạo công cụ", run_script: "Chạy tập lệnh", fan_out: "Chạy song song",
    job_output: "Đọc tác vụ", job_input: "Nhập vào tác vụ", job_stop: "Dừng tác vụ",
    context_notes: "Ghi chú", new_context: "Ngữ cảnh mới",
    schedule_create: "Hẹn giờ", schedule_list: "Lịch hẹn", schedule_delete: "Hủy hẹn",
    use_model: "Dùng mô hình trả phí",
  };

  const TALLY = {
    run_command: ["chạy", "lệnh"], read_file: ["đọc", "tệp"], edit_file: ["sửa", "tệp"], write_file: ["sửa", "tệp"],
    create: ["chạy", "công cụ"],
    spawn_agent: ["tạo", "tác vụ con"], wait_agent: ["chờ", "tác vụ con"],
    send_input: ["nhắn", "tác vụ con"], close_agent: ["đóng", "tác vụ con"], memory: ["ghi nhớ", "mục"],
    list_dir: ["xem", "thư mục"], search: ["tìm", "lần"], glob: ["tìm", "lần"], symbols: ["tìm", "lần"],
    web_fetch: ["tải", "trang"], skill: ["mở", "kỹ năng"], task: ["chạy", "tác vụ con"], mcp: ["gọi", "công cụ MCP"],
    run_script: ["chạy", "tập lệnh"], fan_out: ["chạy", "lô song song"], job_input: ["nhập vào", "tác vụ"],
    job_output: ["đọc", "tác vụ"], job_stop: ["dừng", "tác vụ"],
    schedule_create: ["hẹn", "lượt"], web_search: ["tìm web", "lần"],
  };

  const SANDBOX_TEXT = {
    landlock: "Cơ chế cô lập Landlock (Linux).", seatbelt: "Cơ chế cô lập Seatbelt (macOS).",
    docker: "Container Docker, chỉ truy cập được /workspace.",
    "low-integrity": "Windows mức toàn vẹn thấp: chỉ được ghi trong thư mục làm việc; thao tác đọc tệp và truy cập mạng vẫn cần phê duyệt.",
    none: "Không bật cơ chế cô lập: mọi lệnh đều cần phê duyệt, trừ khi bật chế độ Tự động.",
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
    bar.append(el("span", "agent-code-lang", label || "mã nguồn"));
    const copyBtn = button("Sao chép", "agent-code-copy", () => {
      navigator.clipboard?.writeText(clean).then(() => {
        copyBtn.textContent = "Đã sao chép";
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
    if (c.name === "create") return a.name ? `${a.name}${a.keep === false ? " (chạy một lần)" : ""}` : ".";
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

  function shortPath(path) {
    const raw = String(path || "").trim();
    if (!raw || raw === ".") return raw || ".";
    const parts = raw.replace(/[\\/]+$/, "").split(/[\\/]/).filter(Boolean);
    return parts.pop() || raw;
  }

  function svgIcon(paths, vb = "0 0 16 16") {
    const ns = "http://www.w3.org/2000/svg";
    const icon = document.createElementNS(ns, "svg");
    icon.setAttribute("class", "agent-row-svg");
    icon.setAttribute("viewBox", vb);
    icon.setAttribute("aria-hidden", "true");
    paths.forEach((d) => {
      const p = document.createElementNS(ns, "path");
      p.setAttribute("d", d);
      icon.append(p);
    });
    return icon;
  }

  function toolBadge(c) {
    const name = c.name || "";
    const path = String((c.args || {}).path || "");
    const dot = path.lastIndexOf(".");
    const ext = dot > -1 ? path.slice(dot + 1).toLowerCase() : "";
    const wrap = el("span", "agent-row-icon");
    if ((name === "edit_file" || name === "write_file" || name === "read_file" || name === "plugin_write") && ext && /^[a-z0-9]{1,4}$/.test(ext)) {
      wrap.append(el("span", "agent-ext-badge", ext.toUpperCase()));
      return wrap;
    }
    if (name === "run_command" || name === "run_script" || /^job_/.test(name)) {
      wrap.append(svgIcon(["M3 4.5L6.5 8L3 11.5", "M8 11.5H13"]));
    } else if (name === "create") {
      wrap.append(svgIcon(["M5.5 4.5L2 8L5.5 11.5", "M10.5 4.5L14 8L10.5 11.5", "M9 3.5L7 12.5"]));
    } else if (name === "edit_file" || name === "write_file" || name === "plugin_write") {
      wrap.append(svgIcon(["M10.8 2.8L13.2 5.2L5.8 12.6L2.8 13.2L3.4 10.2Z"]));
    } else if (name === "read_file" || name === "view_image") {
      wrap.append(svgIcon(["M4 2.5H9.5L12.5 5.5V13.5H4Z", "M9.5 2.5V5.5H12.5"]));
    } else if (name === "search" || name === "glob" || name === "symbols" || name === "tool_search" || name === "web_search") {
      wrap.append(svgIcon(["M7 3A4 4 0 1 0 7 11A4 4 0 1 0 7 3", "M10.2 10.2L13.5 13.5"]));
    } else {
      wrap.append(svgIcon(["M8 3.5A4.5 4.5 0 1 0 8 12.5A4.5 4.5 0 1 0 8 3.5"]));
    }
    return wrap;
  }

  function rowTarget(c) {
    const a = c.args || {};
    if (c.name === "read_file" || c.name === "edit_file" || c.name === "write_file" || c.name === "view_image") {
      const base = shortPath(a.path);
      if (c.name === "read_file" && a.start_line && a.end_line) return `${base} #L${a.start_line}-${a.end_line}`;
      if (c.name === "read_file" && a.offset && a.limit) return `${base} #L${a.offset}-${Number(a.offset) + Number(a.limit) - 1}`;
      return base;
    }
    if (c.name === "run_command") {
      const clean = String(a.command || "")
        .replace(/^(?:cd(?:\s+\/d)?\s+(?:"[^"]+"|[^\s&]+)\s*&&\s*)+/i, "")
        .replace(/\s+/g, " ")
        .trim();
      return `${clean || String(a.command || "").trim()}${a.outside_sandbox ? " (ngoài sandbox)" : ""}`;
    }
    return String(target(c)).replace(/\s+/g, " ").trim();
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
    rowTarget,
    toolBadge,
    editPairs,
    diffStats,
    stats,
    formatTokens,
  };
})();
