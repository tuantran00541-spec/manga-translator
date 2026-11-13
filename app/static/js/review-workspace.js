(() => {
  "use strict";

  let reviewLastChapterId = null;
  let aiSettingsInstance = null;

  function syncAIProviderSelects() {
    const providers = window.aiProviderSettings || {};
    const configured = Object.values(providers).filter((info) => !info.builtin || info.configured);
    document.querySelectorAll(".ai-active-provider, .ai-mode-provider").forEach((select) => {
      const selected = select.value;
      const capability = select.classList.contains("ai-mode-provider") ? "visual_qc" : null;
      const available = configured.filter((info) => !capability || info.capabilities?.[capability]);
      const options = available.map((info) => new Option(info.label || info.id, info.id));
      select.replaceChildren(...options);
      const savedActive = localStorage.getItem("manga_ai_active_provider");
      if ([...select.options].some((option) => option.value === selected)) select.value = selected;
      else if (savedActive && [...select.options].some((option) => option.value === savedActive)) select.value = savedActive;
      else if (select.options.length) select.value = select.options[0].value;
      select.dispatchEvent(new Event("ai-providers-updated", { bubbles: true }));
    });
    document.dispatchEvent(new Event("ai-providers-updated"));
  }
  window.syncAIProviderSelects = syncAIProviderSelects;

  function createAIProviderSettings() {
    if (aiSettingsInstance?.config?.isConnected) return aiSettingsInstance.status;

    const SVG_NS = "http://www.w3.org/2000/svg";
    const builtinProviders = ["gemini", "deepseek", "openai", "openrouter"];

    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const svgIcon = (paths) => {
      const node = document.createElementNS(SVG_NS, "svg");
      node.setAttribute("viewBox", "0 0 24 24");
      node.setAttribute("aria-hidden", "true");
      paths.forEach((d) => {
        const p = document.createElementNS(SVG_NS, "path");
        p.setAttribute("d", d);
        node.append(p);
      });
      return node;
    };
    const button = (text, className = "ui-btn ui-btn-ghost ui-btn-compact") => {
      const node = el("button", className, text);
      node.type = "button";
      return node;
    };
    const input = (placeholder, label, type = "text", className = "ui-input") => {
      const node = el("input", className);
      node.type = type;
      node.placeholder = placeholder;
      node.setAttribute("aria-label", label);
      return node;
    };
    const fieldBlock = (labelText, control, helpText) => {
      const wrap = el("div", "settings-field");
      const label = el("label", "settings-field-label", labelText);
      wrap.append(label, control);
      if (helpText) wrap.append(el("p", "settings-field-help", helpText));
      return wrap;
    };
    const settingRow = (titleText, descText, control) => {
      const row = el("div", "settings-row");
      const copy = el("div", "settings-row-copy");
      copy.append(el("strong", "settings-row-title", titleText));
      if (descText) copy.append(el("p", "settings-row-desc", descText));
      row.append(copy, control);
      return row;
    };

    const config = el("div", "settings-layout ai-provider-config");
    const nav = el("nav", "settings-nav");
    nav.setAttribute("aria-label", "Danh mục cài đặt");
    const contentHost = el("div", "settings-content");

    const tabGeneralBtn = button("", "settings-nav-item");
    tabGeneralBtn.append(
      svgIcon(["M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z", "M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.6.85 1 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"]),
      el("span", "", "Chung"),
    );
    const tabModelsBtn = button("", "settings-nav-item is-active");
    tabModelsBtn.append(
      svgIcon(["M4 6c0 1.66 3.58 3 8 3s8-1.34 8-3-3.58-3-8-3-8 1.34-8 3Z", "M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6", "M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"]),
      el("span", "", "Models"),
    );
    nav.append(tabGeneralBtn, tabModelsBtn);

    const generalPanel = el("section", "settings-panel");
    generalPanel.hidden = true;
    const modelsPanel = el("section", "settings-panel");

    const switchSettingsTab = (tab) => {
      const isGeneral = tab === "general";
      tabGeneralBtn.classList.toggle("is-active", isGeneral);
      tabModelsBtn.classList.toggle("is-active", !isGeneral);
      generalPanel.hidden = !isGeneral;
      modelsPanel.hidden = isGeneral;
    };
    tabGeneralBtn.addEventListener("click", () => switchSettingsTab("general"));
    tabModelsBtn.addEventListener("click", () => switchSettingsTab("models"));

    const permSelect = el("select", "ui-select settings-row-select");
    [["ask", "Hỏi trước"], ["edits", "Tự sửa và chạy"], ["bypass", "Tự động"]].forEach(([v, l]) => permSelect.add(new Option(l, v)));
    permSelect.value = localStorage.getItem("manga_agent_default_mode") || document.getElementById("agent-mode")?.value || "edits";
    permSelect.addEventListener("change", () => {
      localStorage.setItem("manga_agent_default_mode", permSelect.value);
      const agentMode = document.getElementById("agent-mode");
      if (agentMode) {
        agentMode.value = permSelect.value;
        agentMode.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });

    const sandboxSelect = el("select", "ui-select settings-row-select");
    [["read-only", "Chỉ đọc"], ["workspace-write", "Ghi trong thư mục"], ["full-access", "Không sandbox"]].forEach(([v, l]) => sandboxSelect.add(new Option(l, v)));
    sandboxSelect.value = localStorage.getItem("manga_agent_default_sandbox") || document.getElementById("agent-sandbox")?.value || "workspace-write";
    sandboxSelect.addEventListener("change", () => {
      localStorage.setItem("manga_agent_default_sandbox", sandboxSelect.value);
      const agentSandbox = document.getElementById("agent-sandbox");
      if (agentSandbox) {
        agentSandbox.value = sandboxSelect.value;
        agentSandbox.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });

    const themeSection = el("div", "settings-block");
    themeSection.append(el("strong", "settings-row-title", "Giao diện"));
    const themeGrid = el("div", "settings-theme-grid");
    const themeCards = {};
    const themeDefs = [
      ["light", "Sáng", false, ["M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"]],
      ["dark", "Tối", false, ["M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"]],
      ["system", "Hệ thống", true, ["M3 4h18a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z", "M8 20h8m-4-3v3"]],
    ];
    const syncThemeCards = (mode) => {
      const current = mode || document.documentElement.dataset.theme || "system";
      Object.entries(themeCards).forEach(([key, card]) => {
        card.classList.toggle("is-active", key === current);
        card.setAttribute("aria-pressed", String(key === current));
      });
    };
    themeDefs.forEach(([key, label, wide, paths]) => {
      const card = button("", `settings-theme-card${wide ? " settings-theme-card-wide" : ""}`);
      card.append(svgIcon(paths), el("span", "", label));
      card.addEventListener("click", () => {
        window.setAppTheme?.(key);
        syncThemeCards(key);
      });
      themeCards[key] = card;
      themeGrid.append(card);
    });
    window.addEventListener("app-theme-change", (e) => syncThemeCards(e.detail?.mode));
    syncThemeCards();
    themeSection.append(themeGrid);

    const collapseSelect = el("select", "ui-select settings-row-select");
    [["on_complete", "Khi hoàn tất"], ["keep_open", "Giữ nguyên"]].forEach(([v, l]) => collapseSelect.add(new Option(l, v)));
    collapseSelect.value = localStorage.getItem("manga_agent_collapse_tools") || "on_complete";
    collapseSelect.addEventListener("change", () => localStorage.setItem("manga_agent_collapse_tools", collapseSelect.value));

    const perfSelect = el("select", "ui-select settings-row-select");
    [["detailed", "Chi tiết"], ["compact", "Gọn"]].forEach(([v, l]) => perfSelect.add(new Option(l, v)));
    perfSelect.value = localStorage.getItem("manga_agent_perf_detail") || "detailed";
    perfSelect.addEventListener("change", () => localStorage.setItem("manga_agent_perf_detail", perfSelect.value));

    generalPanel.append(
      settingRow("Quyền thực thi", "Chọn chế độ duyệt mặc định cho phiên làm việc mới", permSelect),
      settingRow("Phạm vi truy cập tệp (Sandbox)", "Giới hạn thư mục mà các lệnh được phép ghi", sandboxSelect),
      themeSection,
      settingRow("Tự động thu gọn bước chạy", "Chọn thời điểm tự đóng chi tiết công cụ trong Chaos", collapseSelect),
      settingRow("Hiệu năng & mức dùng", "Mức độ hiển thị thông tin token, cache và thời gian chạy", perfSelect),
    );

    const status = el("span", "ai-provider-status ai-settings-summary", "Đang kiểm tra cấu hình…");
    const active = el("select", "ui-select ai-active-provider");
    active.setAttribute("aria-label", "Nhà cung cấp AI mặc định");
    active.addEventListener("change", () => {
      localStorage.setItem("manga_ai_active_provider", active.value);
      window.syncAIProviderSelects?.();
    });
    const defaultRow = el("label", "ai-default-row");
    defaultRow.append(el("span", "", "Mặc định"), active);

    const providerTabsBar = el("div", "settings-provider-tabs");
    const topBar = el("div", "settings-models-topbar");
    topBar.append(providerTabsBar, defaultRow);

    const customNote = el("p", "settings-intro-note", "Kết nối relay, server tự host hoặc bất kỳ endpoint tương thích OpenAI / Gemini nào qua Base URL, giao thức và danh sách model.");
    const customId = input("acme-gateway", "Mã nhà cung cấp (Provider ID)");
    customId.pattern = "[a-z0-9][a-z0-9_-]{0,63}";
    const customLabel = input("Tên hiển thị (VD: MiniRouter)", "Tên hiển thị");
    const customBase = input("https://gateway.example/v1", "Base URL");
    const customProtocol = el("select", "ui-select");
    customProtocol.setAttribute("aria-label", "Giao thức API");
    customProtocol.add(new Option("OpenAI Chat Completions", "openai"));
    customProtocol.add(new Option("Gemini API", "gemini"));
    const customKey = input("Nhập API key của bạn", "API key", "password", "ui-input api-key-input");
    customKey.autocomplete = "new-password";

    const customStatus = el("span", "ai-provider-status");
    const customSave = button("Lưu nhà cung cấp", "ui-btn ui-btn-primary ui-btn-compact");
    const customAdd = button("Thêm key");
    customAdd.title = "Giữ các key đã lưu và thêm key này vào sau";
    const customClear = button("Xóa nhà cung cấp", "ai-link-btn");
    customClear.hidden = true;
    const customActions = el("div", "ai-provider-actions");
    const leftActions = el("div", "settings-action-group");
    leftActions.append(customSave, customAdd, customClear);
    customActions.append(leftActions, customStatus);

    const modelsHeader = el("div", "settings-models-head");
    const modelsTitle = el("strong", "settings-field-label", "Models");
    const fetchModelsBtn = button("Tải danh sách model", "settings-fetch-btn");
    modelsHeader.append(modelsTitle, fetchModelsBtn);

    const modelsFilter = input("Tìm nhanh model…", "Tìm model trong danh sách");
    modelsFilter.className = "ui-input settings-models-filter";
    modelsFilter.hidden = true;
    const modelsBox = el("div", "settings-models-box");

    modelsPanel.append(
      topBar,
      customNote,
      fieldBlock("Provider ID", customId, "Định danh chữ thường (a-z, 0-9, -), dùng để định danh nhà cung cấp trong request và kho khóa bảo mật."),
      fieldBlock("Tên hiển thị", customLabel),
      fieldBlock("Base URL", customBase),
      fieldBlock("Giao thức API", customProtocol),
      fieldBlock("API key", customKey),
      customActions,
      modelsHeader,
      modelsFilter,
      modelsBox,
    );

    contentHost.append(generalPanel, modelsPanel);
    config.append(nav, contentHost);

    let selectedProviderId = "";
    let idManuallyEdited = false;
    let currentModelsList = [];

    const slugifyId = (text) => text
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[đĐ]/g, "d")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 56);

    customId.addEventListener("input", () => {
      idManuallyEdited = Boolean(customId.value.trim());
    });
    customLabel.addEventListener("input", () => {
      if (!selectedProviderId && !idManuallyEdited) {
        let derived = slugifyId(customLabel.value.trim());
        if (builtinProviders.includes(derived)) derived = `${derived}-custom`;
        customId.value = derived;
      }
    });

    const customProviderId = () => {
      if (selectedProviderId) return selectedProviderId;
      let id = customId.value.trim().toLowerCase() || slugifyId(customLabel.value.trim());
      if (builtinProviders.includes(id)) id = `${id}-custom`;
      return id;
    };

    const renderModelsBox = (providerId, models) => {
      currentModelsList = Array.isArray(models) ? models : [];
      modelsFilter.hidden = currentModelsList.length <= 6;
      const query = modelsFilter.value.trim().toLowerCase();
      const filtered = query ? currentModelsList.filter((m) => m.toLowerCase().includes(query)) : currentModelsList;
      if (!filtered.length) {
        modelsBox.className = "settings-models-box is-empty";
        modelsBox.replaceChildren(
          el("p", "settings-models-empty", currentModelsList.length ? "Không tìm thấy model phù hợp." : "Chưa có model nào được tải. Lưu nhà cung cấp hoặc bấm 'Tải danh sách model' để lấy danh sách từ endpoint."),
        );
        return;
      }
      modelsBox.className = "settings-models-box";
      const activeModel = localStorage.getItem(`manga_agent_model_${providerId}`) || localStorage.getItem(`manga_ai_model_${providerId}`) || currentModelsList[0];
      const items = filtered.slice(0, 120).map((modelId) => {
        const item = el("details", "settings-model-item");
        const sum = el("summary", "settings-model-summary");
        const nameCode = el("code", "settings-model-id", modelId);
        const isDefault = modelId === activeModel;
        const pickBtn = button(isDefault ? "Đang chọn" : "Chọn mặc định", `ui-btn ui-btn-compact ${isDefault ? "ui-btn-primary" : "ui-btn-ghost"}`);
        pickBtn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          localStorage.setItem(`manga_ai_model_${providerId}`, modelId);
          localStorage.setItem(`manga_agent_model_${providerId}`, modelId);
          localStorage.setItem(`manga_translation_vision_model_${providerId}`, modelId);
          renderModelsBox(providerId, currentModelsList);
          window.syncAIProviderSelects?.();
        });
        sum.append(nameCode, pickBtn);

        const body = el("div", "settings-model-body");
        const typesRow = el("div", "settings-model-row");
        typesRow.append(el("span", "settings-field-help", "Loại đầu vào hỗ trợ:"));
        const checks = el("div", "settings-model-checks");
        ["Text", "Image"].forEach((t) => {
          const lb = el("label", "settings-check-label");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = true;
          lb.append(cb, el("span", "", t));
          checks.append(lb);
        });
        typesRow.append(checks);

        const effortRow = el("div", "settings-model-row");
        effortRow.append(el("span", "settings-field-help", "Mức suy luận (Reasoning level):"));
        const effortSelect = el("select", "ui-select settings-row-select");
        [
          ["", "off — Mặc định của model"],
          ["low", "low — Thấp"],
          ["medium", "medium — Vừa"],
          ["high", "high — Cao"],
          ["xhigh", "xhigh — Tối đa"],
        ].forEach(([val, text]) => effortSelect.add(new Option(text, val)));
        const effortKey = `manga_agent_effort_${providerId}_${modelId}`;
        effortSelect.value = localStorage.getItem(effortKey) || "";
        effortSelect.addEventListener("change", () => {
          localStorage.setItem(effortKey, effortSelect.value);
          window.syncAIProviderSelects?.();
        });
        effortRow.append(effortSelect);

        body.append(typesRow, effortRow);
        item.append(sum, body);
        return item;
      });
      modelsBox.replaceChildren(...items);
    };

    modelsFilter.addEventListener("input", () => {
      renderModelsBox(customProviderId(), currentModelsList);
    });

    const fetchProviderModels = async (providerId, silent = false) => {
      if (!providerId) return window.showToast?.("Hãy chọn hoặc lưu nhà cung cấp trước.", "error");
      fetchModelsBtn.disabled = true;
      fetchModelsBtn.textContent = "Đang tải…";
      try {
        const res = await fetch(`/api/visual_qc/providers/${encodeURIComponent(providerId)}/models`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        const models = data.models || [];
        if (models.length && !localStorage.getItem(`manga_ai_model_${providerId}`)) {
          localStorage.setItem(`manga_ai_model_${providerId}`, models[0]);
          localStorage.setItem(`manga_agent_model_${providerId}`, models[0]);
          localStorage.setItem(`manga_translation_vision_model_${providerId}`, models[0]);
        }
        renderModelsBox(providerId, models);
        if (!silent) window.showToast?.(`Đã tải ${models.length} model.`, "success");
      } catch (err) {
        renderModelsBox(providerId, []);
        if (!silent) window.showToast?.("Không thể tải model: " + err.message, "error");
      } finally {
        fetchModelsBtn.disabled = false;
        fetchModelsBtn.textContent = "Tải danh sách model";
      }
    };
    fetchModelsBtn.addEventListener("click", () => fetchProviderModels(customProviderId(), false));

    const selectProviderTab = (info) => {
      if (!info) {
        selectedProviderId = "";
        idManuallyEdited = false;
        customId.disabled = false;
        customId.value = "";
        customLabel.value = "";
        customBase.value = "";
        customProtocol.value = "openai";
        customKey.value = "";
        customKey.placeholder = "Nhập API key của bạn";
        customStatus.textContent = "";
        customClear.hidden = true;
        modelsFilter.value = "";
        renderModelsBox("", []);
      } else {
        selectedProviderId = info.id;
        idManuallyEdited = true;
        customId.disabled = true;
        customId.value = info.id || "";
        customLabel.value = info.label || info.id || "";
        customBase.value = info.api_base || "";
        customProtocol.value = info.protocol || "openai";
        customKey.value = "";
        customKey.placeholder = info.configured ? "Đã lưu key — nhập key mới nếu muốn thay hoặc thêm" : "Nhập API key của bạn";
        customStatus.textContent = info.configured ? (info.count > 1 ? `Đã kết nối · ${info.count} key` : "Đã kết nối") : "Chưa có key";
        customStatus.classList.toggle("configured", Boolean(info.configured));
        customClear.hidden = false;
        modelsFilter.value = "";
        void fetchProviderModels(info.id, true);
      }
      providerTabsBar.querySelectorAll(".settings-provider-pill").forEach((pill) => {
        pill.classList.toggle("is-active", pill.dataset.providerId === (info ? info.id : "__new__"));
      });
    };

    const saveCustom = async (append) => {
      const id = customProviderId();
      if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(id)) return window.showToast?.("Provider ID dùng chữ thường, số, dấu gạch ngang hoặc gạch dưới.", "error");
      if (!customLabel.value.trim() || !customBase.value.trim() || !customKey.value.trim()) {
        return window.showToast?.("Nhập Tên hiển thị, Base URL và API key.", "error");
      }
      customSave.disabled = customAdd.disabled = true;
      try {
        const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(id)}/key`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            api_key: customKey.value.trim(), provider_label: customLabel.value.trim(),
            provider_protocol: customProtocol.value || "openai", provider_api_base: customBase.value.trim(), append,
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        selectedProviderId = id;
        window.showToast?.(`Đã lưu nhà cung cấp ${customLabel.value.trim()} (${data.api_base || ""}, ${data.count || 1} key).`, "success");
        await refresh();
      } catch (err) {
        window.showToast?.("Không thể lưu nhà cung cấp: " + err.message, "error");
      } finally {
        customSave.disabled = customAdd.disabled = false;
      }
    };
    customSave.addEventListener("click", () => saveCustom(false));
    customAdd.addEventListener("click", () => saveCustom(true));
    customClear.addEventListener("click", async () => {
      const id = customProviderId();
      if (!id) return;
      customClear.disabled = true;
      try {
        const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(id)}/key?remove_config=true&provider_label=${encodeURIComponent(customLabel.value.trim() || id)}`, { method: "DELETE" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        localStorage.removeItem(`manga_ai_model_${id}`);
        localStorage.removeItem(`manga_agent_model_${id}`);
        localStorage.removeItem(`manga_translation_vision_model_${id}`);
        window.showToast?.(`Đã xóa nhà cung cấp ${customLabel.value.trim() || id}.`, "success");
        selectedProviderId = "";
        await refresh();
      } catch (err) {
        window.showToast?.("Không thể xóa nhà cung cấp: " + err.message, "error");
      } finally {
        customClear.disabled = false;
      }
    });

    const setStatus = (node, configured, text) => {
      node.textContent = text;
      node.classList.toggle("configured", configured);
    };

    async function refresh() {
      try {
        const response = await fetch("/api/visual_qc/settings");
        const data = await response.json();
        window.aiProviderSettings = data.providers || {};
        const userInfos = Object.values(data.providers || {}).filter((info) => !info.builtin || info.configured);
        const ready = userInfos.filter((info) => info.configured).length;

        providerTabsBar.replaceChildren();
        userInfos.forEach((info) => {
          const pill = button(info.label || info.id, "settings-provider-pill");
          pill.dataset.providerId = info.id;
          pill.addEventListener("click", () => selectProviderTab(info));
          providerTabsBar.append(pill);
        });
        const addPill = button("+ Thêm nhà cung cấp", "settings-provider-pill settings-provider-pill-add");
        addPill.dataset.providerId = "__new__";
        addPill.addEventListener("click", () => selectProviderTab(null));
        providerTabsBar.append(addPill);

        defaultRow.hidden = !userInfos.length;
        active.replaceChildren(...userInfos.map((info) => new Option(info.label || info.id, info.id)));
        const savedDefault = localStorage.getItem("manga_ai_active_provider");
        if (savedDefault && [...active.options].some((option) => option.value === savedDefault)) {
          active.value = savedDefault;
        } else if (active.options.length) {
          active.value = active.options[0].value;
          localStorage.setItem("manga_ai_active_provider", active.value);
        }

        const currentInfo = userInfos.find((info) => info.id === selectedProviderId) || userInfos[0] || null;
        selectProviderTab(currentInfo);

        window.syncAIProviderSelects?.();
        setStatus(status, ready > 0, ready ? `${ready} nhà cung cấp sẵn sàng` : "Chưa có nhà cung cấp nào");
      } catch (_) {
        setStatus(status, false, "Không đọc được cấu hình AI");
      }
    }

    void refresh();
    window.mountAISettings?.(config);
    aiSettingsInstance = { config, status };
    return status;
  }
  window.createAIProviderSettings = createAIProviderSettings;

  function captureReviewWorkspaceState() {
    const workspace = document.querySelector("#page-view .review-workspace-shell");
    if (!workspace) return null;
    const viewport = workspace.querySelector(".review-document-viewport");
    const active = document.activeElement;
    let draft = null;
    if (active instanceof HTMLTextAreaElement) {
      const overlay = active.closest?.(".review-text-object-overlay");
      const objectId = overlay?.dataset?.objectId || active.dataset?.textObjectId || window.editorState?.selectedTextObjectId || null;
      const pageIndex = Number.parseInt(overlay?.dataset?.pageIndex || active.dataset?.pageIndex || window.editorState?.activePageIndex, 10);
      if (objectId && Number.isInteger(pageIndex)) {
        const field = active.classList.contains("ocr-textarea") ? "ocr_text" : "translation";
        const surface = active.classList.contains("review-inline-translation")
          ? "inline"
          : active.classList.contains("ocr-textarea")
            ? "panel-ocr"
            : "panel-translation";
        draft = {
          pageIndex,
          objectId: String(objectId),
          field,
          surface,
          value: active.value,
          selectionStart: active.selectionStart,
          selectionEnd: active.selectionEnd,
        };
      }
    }
    return {
      chapterId: window.currentChapterId || "",
      scrollLeft: viewport?.scrollLeft || 0,
      scrollTop: viewport?.scrollTop || 0,
      selectedTextObjectId: window.editorState?.selectedTextObjectId || null,
      activePageIndex: Number(window.editorState?.activePageIndex || 0),
      draft,
    };
  }

  function reapplyFocusedDraft(state) {
    const draft = state?.draft;
    if (!draft || state.chapterId !== (window.currentChapterId || "")) return;
    const page = window.currentManifest?.pages?.[draft.pageIndex];
    const obj = (page?.text_objects || []).find((item) => String(item?.id) === draft.objectId);
    if (!obj) return;
    const field = draft.field === "ocr_text" ? "ocr_text" : "translation";
    if (obj[field] === draft.value) return;
    obj[field] = draft.value;
    window.scheduleTextObjectPersist?.(draft.pageIndex, draft.objectId);
  }

  function setupReviewWorkspace() {
    const previousState = captureReviewWorkspaceState();
    const requestedPageIndex = window.initialReviewCanonicalPageIndex;
    window.initialReviewCanonicalPageIndex = null;
    window.cleanupReviewWorkspace?.();

    const container = document.getElementById("page-view");
    if (!container) return;
    window.setAppStage?.("review");

    container.replaceChildren();
    container.className = "review-mode review-canvas-mode";

    const chapterId = window.currentChapterId || "";
    if (reviewLastChapterId !== chapterId) reviewLastChapterId = chapterId;

    const slices = (window.currentManifest?.pages || []).filter(Boolean);
    if (!slices.length) return;

    reapplyFocusedDraft(previousState);

    const workspace = document.createElement("div");
    workspace.className = "review-workspace-shell review-canvas-only review-canvas-fullbleed";
    workspace._initialCanonicalPageIndex = requestedPageIndex;
    if (previousState?.chapterId === chapterId) {
      workspace._reviewRestoreState = previousState;
    }

    const layout = document.createElement("div");
    layout.className = "review-workbench-grid review-canvas-only-layout";

    const canvasHost = document.createElement("div");
    canvasHost.className = "review-canvas-host review-canvas-only-host";
    layout.appendChild(canvasHost);

    workspace.appendChild(layout);
    container.appendChild(workspace);

    createAIProviderSettings();

    window.cleanupReviewWorkspace = () => {
      window._reviewStitchAbort?.abort();
    };

    window.mountStitchInspector?.();
  }

  window.renderReview = setupReviewWorkspace;
})();
