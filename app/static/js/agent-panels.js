(() => {
  const UI = window._AgentUI;
  const { $, call, post, el, button } = UI;

  // The MCP panel: every server the agent would start here, and adding, testing, switching and removing your own.
  const MCP_STATE = { running: "đang chạy", untrusted: "chờ cho phép", failed: "lỗi", disabled: "đã tắt", stopped: "đã dừng", waiting: "chờ" };
  const mcpQuery = () => `?workspace=${encodeURIComponent($("agent-workspace").value.trim())}`;

  function mcpKind() {
    return document.querySelector('input[name="agent-mcp-kind"]:checked')?.value || "stdio";
  }

  function showMcpKind() {
    const kind = mcpKind();
    const nameField = $("agent-mcp-name")?.closest(".agent-field");
    if (nameField && kind === "http") $("agent-mcp-http")?.prepend(nameField);
    else if (nameField && kind === "stdio") $("agent-mcp-stdio")?.querySelector(".agent-mcp-grid")?.prepend(nameField);
    if (nameField) nameField.hidden = kind === "json";
    $("agent-mcp-stdio").hidden = kind !== "stdio";
    $("agent-mcp-http").hidden = kind !== "http";
    $("agent-mcp-json").hidden = kind !== "json";
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
    $("agent-mcp-file").textContent = data.file ? `Lưu ở ${data.file}` : "";
    const servers = data.servers || [];
    const count = $("agent-mcp-count");
    if (count) count.textContent = String(servers.length);
    const rows = servers.map((server) => {
      const item = el("li");
      const title = el("span", "agent-mcp-head");
      const where = server.scope === "workspace" ? "của dự án" : server.editable ? "của bạn" : server.source;
      title.append(el("b", "", server.name), el("span", "agent-skill-tag", server.type), el("span", "agent-skill-tag", where));
      const actions = el("span", "agent-skill-row");
      const label = server.disabled ? "đã tắt" : server.state ? `${MCP_STATE[server.state] || server.state}${server.state === "running" ? `, ${server.tools} công cụ` : ""}` : "";
      const state = el("span", `agent-mcp-state${server.state === "running" ? " ok" : server.state === "failed" ? " bad" : ""}`, label + (server.error ? `: ${server.error}` : ""));
      if (server.scope !== "workspace") actions.append(button("Thử", "agent-btn agent-skill-act", () => testMcp(server.name, state)));
      if (server.editable) {
        actions.append(
          button(server.disabled ? "Bật" : "Tắt", "agent-btn agent-skill-act", () => switchMcp(server.name, server.disabled)),
          button("Xóa", "agent-btn agent-skill-act", () => removeMcp(server.name)),
        );
      }
      const secrets = [...Object.keys(server.env || {}), ...Object.keys(server.headers || {})];
      const about = el("span", "agent-mcp-target", server.target + (secrets.length ? ` · ${secrets.length} biến` : ""));
      about.title = server.target + (secrets.length ? `\n${secrets.join(", ")}` : "");
      item.append(title, actions, about, state);
      if ((server.tool_list || []).length) item.append(mcpTools(server));
      return item;
    });
    $("agent-mcp-list").replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", "Chưa có máy chủ MCP nào.")]));
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
    label.textContent = "đang kiểm tra…";
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
        if (label) window.showToast?.(await testMcp(added, label) ? `Đã thêm ${added}, có hiệu lực từ lượt tiếp theo.` : `Đã lưu ${added}, hiện chưa thể kết nối.`, "info");
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
    if (!window.confirm(`Xóa máy chủ MCP ${name}?`)) return;
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
    const count = $("agent-skill-count");
    try {
      const data = await call(`/api/agent/skills?workspace=${encodeURIComponent($("agent-workspace").value.trim())}`);
      const skills = data.skills || [];
      if (count) count.textContent = String(skills.length);
      const rows = skills.map((skill) => {
        const item = el("li");
        item.title = [skill.description, skill.path].filter(Boolean).join("\n");
        const name = el("b", "agent-skill-name", `/${skill.name}`);
        const badge = el("span", "agent-skill-tag", SKILL_SCOPE[skill.scope] || skill.scope);
        const about = el("span", "agent-skill-desc", skill.description || "");
        item.append(name, badge, about);
        if (skill.scope === "user") {
          const actions = el("span", "agent-skill-actions");
          actions.append(button("Sửa", "agent-btn agent-skill-act", () => editSkill(skill.name)), button("Xóa", "agent-btn agent-skill-act", () => removeSkill(skill.name)));
          item.append(actions);
        }
        return item;
      });
      list.replaceChildren(...(rows.length ? rows : [el("li", "agent-muted", "Chưa có kỹ năng nào.")]));
    } catch (error) {
      if (count) count.textContent = "0";
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
    skillsChanged(post("/api/agent/skills/install", { source }), (names) => `Đã cài đặt ${names.join(", ")}. Vui lòng đọc SKILL.md trước khi sử dụng.`)
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
    if (!window.confirm(`Xóa kỹ năng ${name}?`)) return;
    skillsChanged(call(`/api/agent/skills/${encodeURIComponent(name)}`, { method: "DELETE" }), () => `Đã xóa ${name}.`);
  }

  function saveSkill() {
    const name = $("agent-skill-name").value.trim();
    const body = { name, description: $("agent-skill-desc").value.trim(), body: $("agent-skill-body").value.trim(), replace: editingSkill === name };
    skillsChanged(post("/api/agent/skills", body), () => `Đã lưu kỹ năng ${name}.`).then((ok) => {
      if (!ok) return;
      editingSkill = "";
      $("agent-skill-name").value = $("agent-skill-desc").value = "";
      $("agent-skill-body").value = SKILL_TEMPLATE;
    });
  }

  function draftSkill() {
    const name = $("agent-skill-name").value.trim() || "ten-ky-nang";
    const about = $("agent-skill-desc").value.trim() || "(mô tả chức năng của kỹ năng này)";
    $("agent-input").value = `Sử dụng kỹ năng writing-skills để biên soạn một kỹ năng có tên ${name}: ${about}. `
      + `Lưu tệp tại .agents/skills/${name}/SKILL.md trong thư mục làm việc, kèm phần mở đầu (frontmatter) name và description, sau đó kiểm chứng trên một ví dụ thực tế.`;
    showHub(false);
    UI.onDraftSkill?.();
    $("agent-input").focus();
  }

  let activeHubTab = "skills";

  function selectHubTab(tab) {
    activeHubTab = tab || "skills";
    const panels = {
      skills: $("agent-skills-menu"),
      mcp: $("agent-mcp-menu"),
      settings: $("agent-settings-menu"),
    };
    document.querySelectorAll("[data-agent-tab]").forEach((btn) => {
      const active = btn.dataset.agentTab === activeHubTab;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    Object.entries(panels).forEach(([key, panel]) => {
      if (panel) panel.hidden = key !== activeHubTab;
    });
    if (activeHubTab === "skills") loadSkills();
    else if (activeHubTab === "mcp") loadMcp();
  }

  function showHub(open, tab) {
    const hub = $("agent-hub");
    const log = $("agent-log");
    const dock = document.querySelector(".agent-dock");
    const more = $("agent-more");
    if (!hub) return;
    hub.hidden = !open;
    if (log) log.hidden = Boolean(open);
    if (dock) dock.hidden = Boolean(open);
    if (more) {
      more.classList.toggle("is-active", Boolean(open));
      more.setAttribute("aria-pressed", String(Boolean(open)));
    }
    $("agent-top-more")?.classList.toggle("is-active", Boolean(open));
    if (open) selectHubTab(tab || activeHubTab);
  }

  UI.SKILL_TEMPLATE = SKILL_TEMPLATE;
  UI.showMcpKind = showMcpKind;
  UI.saveMcp = saveMcp;
  UI.installSkills = installSkills;
  UI.uploadSkills = uploadSkills;
  UI.saveSkill = saveSkill;
  UI.draftSkill = draftSkill;
  UI.selectHubTab = selectHubTab;
  UI.showHub = showHub;
})();
