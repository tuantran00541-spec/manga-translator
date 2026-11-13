(() => {
  const POLL_MS = 700;
  const UI = window._AgentUI;
  const {
    VERBS, TALLY, SANDBOX_TEXT, $, store, call, post, showEngine, el, button,
    nearBottom, follow, prose, isMcp, target, editPairs, diffStats, stats, formatTokens,
    SKILL_TEMPLATE, showMcpKind, saveMcp, installSkills, uploadSkills,
    saveSkill, draftSkill, selectHubTab, showHub,
  } = UI;
  let session = null;
  let lastSeq = 0;
  let pollTimer = null;
  let pollGeneration = 0;
  let loaded = false;
  let commands = [];
  let allSessions = [];
  let infoRev = "";
  let group = null;

  function newGroup() {
    const box = el("details", "agent-group");
    const summary = el("summary");
    const countBadge = el("span", "agent-group-count", "0");
    const label = el("span", "agent-group-label");
    summary.append(countBadge, label);
    const list = el("div", "agent-group-list");
    box.append(summary, list);
    $("agent-log").append(box);
    group = { box, summary, countBadge, label, list, tally: {}, count: 0, added: 0, removed: 0, failed: 0 };
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
    group.countBadge.textContent = `${group.count} bước`;
    group.label.textContent = text.charAt(0).toUpperCase() + text.slice(1);
    group.summary.querySelector(".agent-stats")?.remove();
    group.summary.querySelector(".agent-fail")?.remove();
    if (group.added || group.removed) group.summary.append(stats(group.added, group.removed));
    if (group.failed) group.summary.append(el("span", "agent-fail", `${group.failed} lỗi`));
  }

  function addRow(c) {
    if (c.name === "ask_user" || c.name === "exit_plan_mode") return;
    const g = group || newGroup();
    g.count += 1;
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
    const list = items || [];
    const done = list.filter((i) => i.status === "completed").length;
    $("agent-todos").hidden = !list.length || done === list.length;
    const head = $("agent-todos-head");
    if (head) head.textContent = `Tiến độ kế hoạch · ${done}/${list.length} hoàn tất`;
    const progress = $("agent-todos-progress");
    if (progress) {
      progress.max = Math.max(1, list.length);
      progress.value = Math.min(list.length, done);
    }
    $("agent-todo-list").replaceChildren(...list.map((item) => el("li", `agent-todo agent-todo-${item.status}`, item.content)));
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
    const state = { running: "Đang chạy", waiting: "Chờ duyệt" }[snap.status];
    const statsData = snap.stats || {};
    const bar = el("div", "agent-telemetry");

    const statePill = el("span", `agent-badge ${busy ? "agent-badge-active" : "agent-badge-idle"}`);
    statePill.append(el("span", "agent-dot"), el("span", "", state || "Sẵn sàng"));
    bar.append(statePill);

    const titlePill = el("span", "agent-badge agent-badge-title", snap.title || "Phiên mới");
    bar.append(titlePill);

    const detailedPerf = (store("get", "manga_agent_perf_detail") || "detailed") !== "compact";
    if (detailedPerf) {
      const tokenText = `${formatTokens(usage.prompt_tokens)} vào · ${formatTokens(usage.completion_tokens)} ra${usage.prompt_tokens ? ` · cache ${statsData.cache_pct || 0}%` : ""}`;
      bar.append(el("span", "agent-badge", tokenText));

      if (statsData.cost != null) {
        bar.append(el("span", "agent-badge agent-badge-mono", `$${statsData.cost}`));
      }
      if (statsData.model_seconds || statsData.tool_seconds) {
        bar.append(el("span", "agent-badge agent-badge-mono", `AI ${statsData.model_seconds || 0}s · Tool ${statsData.tool_seconds || 0}s`));
      }
    }
    if (snap.plan_mode) bar.append(el("span", "agent-badge agent-badge-mode", "Chế độ /plan"));
    if (snap.goal) bar.append(el("span", "agent-badge agent-badge-mode", `Mục tiêu: ${snap.goal}`));
    if (snap.text_tools) bar.append(el("span", "agent-badge", "Text tools"));

    $("agent-status").replaceChildren(bar);
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
    if (state) {
      const working = el("p", "agent-working");
      working.append(el("span", "agent-dot"), el("span", "", `${state}…`));
      log.append(working);
    }
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

  async function poll(generation) {
    clearTimeout(pollTimer);
    if (!session) return;
    // Only the latest poll loop may apply responses; a fresh poll() (send,
    // approval, session open) supersedes any loop still awaiting a reply.
    if (generation === undefined) generation = ++pollGeneration;
    try {
      const stick = nearBottom();
      const sid = session.id;
      const snap = await call(`/api/agent/sessions/${session.id}?after=${lastSeq}&rev=${encodeURIComponent(infoRev)}`);
      if (!session || session.id !== sid || generation !== pollGeneration) return;
      snap.events.forEach((event) => { render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
      follow(stick && snap.events.length > 0);
      if (snap.status !== "idle") pollTimer = setTimeout(() => poll(generation), snap.live ? POLL_MS / 2 : POLL_MS);
      else loadSessions();
    } catch (error) {
      if (generation !== pollGeneration) return;
      $("agent-status").textContent = error.message;
      pollTimer = setTimeout(() => poll(generation), POLL_MS * 4);
    }
  }

  let providerModels = {};

  async function loadModels() {
    const provider = $("agent-provider").value;
    const select = $("agent-model");
    const saved = store("get", `manga_agent_model_${provider}`) || store("get", `manga_ai_model_${provider}`) || providerModels[provider] || "";
    select.replaceChildren(new Option(saved || "Đang tải model…", saved));
    if (saved) select.value = saved;
    loadEffort();
    showPicks();
    if (!provider) {
      select.replaceChildren(new Option("Chưa có nhà cung cấp", ""));
      showPicks();
      return;
    }
    try {
      const data = await call(`/api/visual_qc/providers/${encodeURIComponent(provider)}/models`);
      const models = Array.isArray(data.models) ? [...data.models] : [];
      if (saved && !models.includes(saved)) models.unshift(saved);
      if (models.length) {
        select.replaceChildren(...models.map((id) => new Option(id, id)));
        select.value = saved && models.includes(saved) ? saved : models[0];
        store("set", `manga_agent_model_${provider}`, select.value);
      } else {
        select.replaceChildren(new Option("Không có model", ""));
      }
    } catch (_) {
      if (!saved) select.replaceChildren(new Option("Không tải được model", ""));
    }
    loadEffort();
    showPicks();
  }

  async function loadProviders() {
    const response = await fetch("/api/visual_qc/settings");
    const data = await window.parseApiResponse(response);
    const ready = Object.values(data.providers || {}).filter((p) => p.configured && p.id !== "manga-cloud");
    providerModels = Object.fromEntries(ready.map((p) => [p.id, p.model || ""]));
    $("agent-provider").replaceChildren(...ready.map((p) => new Option(p.label || p.id, p.id)));
    if (!ready.length) {
      $("agent-model").replaceChildren(new Option("Chưa có model", ""));
      showPicks();
      $("agent-status").textContent = "Chưa có nhà cung cấp nào. Thêm trong Cài đặt, mục Nhà cung cấp AI.";
      return;
    }
    const saved = store("get", "manga_agent_provider");
    if (ready.some((p) => p.id === saved)) $("agent-provider").value = saved;
    await loadModels();
  }

  function renderSessions() {
    const list = $("agent-sessions");
    if (!list) return;
    const query = ($("agent-session-filter")?.value || "").trim().toLowerCase();
    const filtered = allSessions.slice(0, 40).filter((row) => {
      if (!query) return true;
      const hay = `${row.title || ""} ${row.model || ""} ${row.workspace || ""}`.toLowerCase();
      return hay.includes(query);
    });
    let folder = null;
    const rows = filtered.flatMap((row) => {
      const item = el("li", row.id === session?.id ? "agent-session-current" : "");
      const open = button("", "agent-session-open", () => {
        if (narrow()) showSide(false);
        resume(row.id);
      });
      const shortModel = (row.model || "").split("/").pop() || row.provider || "";
      open.append(
        el("span", "agent-session-title", row.title || "(chưa có tin nhắn)"),
        el("span", "agent-session-meta", shortModel || "Phiên làm việc")
      );
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
    const emptyMsg = query ? "Không tìm thấy phiên phù hợp." : "Chưa có phiên nào.";
    list.replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", emptyMsg)]));
  }

  async function loadSessions() {
    try {
      const data = await call("/api/agent/sessions");
      allSessions = data.sessions || [];
      renderSessions();
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
      if (snap.provider && [...$("agent-provider").options].some((o) => o.value === snap.provider)) {
        $("agent-provider").value = snap.provider;
      }
      if (snap.model && ![...$("agent-model").options].some((o) => o.value === snap.model)) {
        $("agent-model").append(new Option(snap.model, snap.model));
      }
      $("agent-model").value = snap.model;
      $("agent-effort").value = snap.effort || "";
      showPicks();
      snap.events.forEach((event) => { if (event.type !== "approval") render(event); lastSeq = Math.max(lastSeq, event.seq); });
      showStatus(snap);
      showHub(false);
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

  function fillPrompt(text) {
    const input = $("agent-input");
    if (!input) return;
    input.value = text;
    autosize();
    showHints();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  function emptyState() {
    const box = el("div", "agent-empty");
    const sub = el("p", "agent-hero-sub", "Harness tự hành cho đọc hiểu mã nguồn, lập kế hoạch, chạy kiểm thử và tự động hóa tác vụ dài hạn.");
    const startersData = [
      {
        title: "Khảo sát & Lập kế hoạch",
        tag: "/plan",
        desc: "Đọc kiến trúc & đề xuất từng bước",
        prompt: "/plan Khảo sát cấu trúc dự án và lập kế hoạch thực hiện: ",
        icon: ["M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3V6z", "M9 3v15", "M15 6v15"],
      },
      {
        title: "Kiểm toán & Sửa lỗi",
        tag: "/tools",
        desc: "Rà soát bug, luồng dữ liệu & đề xuất bản vá",
        prompt: "Kiểm toán các module chính trong thư mục làm việc, liệt kê lỗi tiềm ẩn và đề xuất bản vá chi tiết.",
        icon: ["M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z", "M21 21l-4.35-4.35", "M11 8v6", "M8 11h6"],
      },
      {
        title: "Mục tiêu tự hành",
        tag: "/goal",
        desc: "Tự thực thi & chạy test đến khi xong",
        prompt: "/goal Hoàn thành mục tiêu sau và chạy kiểm thử xác nhận: ",
        icon: ["M13 2L3 14h9l-1 8 10-12h-9l1-8z"],
      },
      {
        title: "Tạo Skill & Công cụ",
        tag: "/rules",
        desc: "Mở rộng khả năng chuyên biệt cho Chaos",
        prompt: "Giúp tôi thiết kế một SKILL.md hoặc công cụ plugin mới cho dự án này: ",
        icon: ["M12 2L2 7l10 5 10-5-10-5z", "M2 17l10 5 10-5", "M2 12l10 5 10-5"],
      },
    ];
    const starters = el("div", "agent-starters");
    startersData.forEach((item) => {
      const card = button("", "agent-starter-card", () => fillPrompt(item.prompt));
      const iconWrap = el("span", "agent-starter-icon");
      iconWrap.append(svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, item.icon.map((d) => svg("path", { d }))));
      const copy = el("span", "agent-starter-copy");
      const headRow = el("span", "agent-starter-head");
      headRow.append(el("span", "agent-starter-title", item.title), el("span", "agent-starter-tag", item.tag));
      copy.append(headRow, el("span", "agent-starter-desc", item.desc));
      card.append(iconWrap, copy);
      starters.append(card);
    });
    box.append(chaosMark(), sub, starters);
    return box;
  }

  function resetLog() {
    clearTimeout(pollTimer);
    pollGeneration++;
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

  UI.onDraftSkill = autosize;

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
      window.showToast?.("Chọn nhà cung cấp và model.", "error");
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
    const savedDefaultMode = store("get", "manga_agent_default_mode");
    if (savedDefaultMode && [...$("agent-mode").options].some((o) => o.value === savedDefaultMode)) {
      $("agent-mode").value = savedDefaultMode;
    }
    const savedDefaultSandbox = store("get", "manga_agent_default_sandbox");
    if (savedDefaultSandbox && [...$("agent-sandbox").options].some((o) => o.value === savedDefaultSandbox)) {
      $("agent-sandbox").value = savedDefaultSandbox;
    }
    showPicks();
    resetLog();
    $("agent-composer").addEventListener("submit", send);
    $("agent-input").addEventListener("keydown", (event) => {
      // Enter sends, Shift+Enter breaks the line, as in chat apps.
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) send(event);
    });
    $("agent-input").addEventListener("input", () => { autosize(); showHints(); });
    $("agent-stop").addEventListener("click", () => session && post(`/api/agent/sessions/${session.id}/stop`).then(poll).catch(() => {}));
    const startNewSession = () => {
      if (narrow()) showSide(false);
      showHub(false);
      resetLog();
      loadSessions();
      $("agent-input").focus();
    };
    $("agent-new").addEventListener("click", startNewSession);
    $("agent-top-new")?.addEventListener("click", startNewSession);
    const toggleHub = () => {
      if (narrow()) showSide(false);
      showHub($("agent-hub")?.hidden !== false);
    };
    $("agent-more")?.addEventListener("click", toggleHub);
    $("agent-top-more")?.addEventListener("click", toggleHub);
    $("agent-hub-close")?.addEventListener("click", () => {
      showHub(false);
      $("agent-input").focus();
    });
    document.querySelectorAll("[data-agent-tab]").forEach((btn) => {
      btn.addEventListener("click", () => selectHubTab(btn.dataset.agentTab));
    });
    $("agent-session-filter")?.addEventListener("input", renderSessions);
    showSide(!narrow() && store("get", "manga_agent_side") !== "closed");
    $("agent-side-toggle").addEventListener("click", () => showSide($("agent-view").classList.contains("agent-side-closed")));
    $("agent-side-close")?.addEventListener("click", () => showSide(false));
    // On a narrow screen the column covers the chat, so a tap outside it folds it away.
    document.addEventListener("click", (event) => {
      if (!narrow() || $("agent-view").classList.contains("agent-side-closed")) return;
      if (!event.target.closest("#agent-side, #agent-side-toggle, #agent-side-close")) showSide(false);
    });
    $("agent-provider").addEventListener("change", () => {
      store("set", "manga_agent_provider", $("agent-provider").value);
      resetLog();
      loadModels();
    });
    document.addEventListener("ai-providers-updated", () => {
      if (loaded) loadProviders();
    });
    $("agent-workspace").addEventListener("change", () => { resetLog(); showFolder(); });
    $("agent-skill-body").value = SKILL_TEMPLATE;
    document.querySelectorAll('input[name="agent-mcp-kind"]').forEach((radio) => radio.addEventListener("change", showMcpKind));
    $("agent-mcp-save").addEventListener("click", saveMcp);
    $("agent-skill-install").addEventListener("click", installSkills);
    $("agent-skill-source").addEventListener("keydown", (event) => { if (event.key === "Enter") installSkills(); });
    $("agent-skill-file").addEventListener("change", uploadSkills);
    $("agent-skill-save").addEventListener("click", saveSkill);
    $("agent-skill-draft").addEventListener("click", draftSkill);
    $("agent-model").addEventListener("change", () => {
      const provider = $("agent-provider").value;
      if (provider && $("agent-model").value.trim()) {
        store("set", `manga_agent_model_${provider}`, $("agent-model").value.trim());
      }
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
