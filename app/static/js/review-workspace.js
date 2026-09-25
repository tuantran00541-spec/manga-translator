(() => {
  "use strict";

  let reviewLastChapterId = null;
  let aiSettingsInstance = null;

  function syncAIProviderSelects() {
    const providers = window.aiProviderSettings || {};
    const fixedLabels = {
      gemini: "Google Gemini", deepseek: "DeepSeek", openai: "OpenAI",
      openrouter: "OpenRouter", experiential: "Experiential Labs",
    };
    document.querySelectorAll(".chapter-translate-provider, .chapter-qc-provider, .ai-active-provider, .ai-mode-provider").forEach((select) => {
      const selected = select.value;
      const capability = select.classList.contains("chapter-translate-provider") ? "translation"
        : select.classList.contains("chapter-qc-provider") || select.classList.contains("ai-mode-provider") ? "visual_qc" : null;
      const available = Object.values(providers).filter((info) => !capability || info.capabilities?.[capability]);
      const options = available.map((info) => new Option(info.label || fixedLabels[info.id] || info.id, info.id));
      if (options.length) select.replaceChildren(...options);
      if ([...select.options].some((option) => option.value === selected)) select.value = selected;
      else if (select.classList.contains("chapter-translate-provider") || select.classList.contains("ai-mode-provider")) select.value = "deepseek";
      else if (select.classList.contains("chapter-qc-provider") || select.classList.contains("ai-active-provider")) select.value = "gemini";
      select.dispatchEvent(new Event("ai-providers-updated", { bubbles: true }));
    });
  }
  window.syncAIProviderSelects = syncAIProviderSelects;

  function createAIProviderSettings() {
    if (aiSettingsInstance?.config?.isConnected) return aiSettingsInstance.status;

    const builtinProviders = ["gemini", "deepseek", "openai", "openrouter", "experiential"];
    const labels = { gemini: "Google Gemini", deepseek: "DeepSeek", openai: "OpenAI", openrouter: "OpenRouter", experiential: "Experiential Labs" };

    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const button = (text, className = "ui-btn ui-btn-ghost ui-btn-compact") => {
      const node = el("button", className, text);
      node.type = "button";
      return node;
    };
    const inlineField = (input, action) => {
      const row = el("div", "ai-inline-field");
      row.append(input, action);
      return row;
    };
    const exclusive = (details) => {
      details.name = "ai-provider";
      details.addEventListener("toggle", () => {
        if (!details.open) return;
        details.parentElement?.querySelectorAll(":scope > details[open]").forEach((other) => {
          if (other !== details) other.open = false;
        });
      });
      return details;
    };
    const providerRow = (name) => {
      const details = exclusive(el("details", "ai-provider-card ai-provider-row"));
      const summary = el("summary", "ai-provider-summary");
      const title = el("span", "ai-provider-name", name);
      const providerStatus = el("span", "ai-provider-status");
      summary.append(title, providerStatus);
      const body = el("div", "ai-provider-body");
      details.append(summary, body);
      return { details, title, providerStatus, body };
    };

    const config = el("div", "ai-provider-config");
    const status = el("span", "ai-provider-status ai-settings-summary", "Đang kiểm tra cấu hình…");
    const active = el("select", "ui-select ai-active-provider");
    active.setAttribute("aria-label", "Dịch vụ AI mặc định");
    builtinProviders.forEach((id) => active.add(new Option(labels[id], id)));
    active.value = localStorage.getItem("manga_ai_active_provider") || "gemini";
    active.addEventListener("change", () => {
      localStorage.setItem("manga_ai_active_provider", active.value);
      window.syncAIProviderSelects?.();
    });
    const defaultRow = el("label", "ai-default-row");
    defaultRow.append(el("span", "", "Mặc định"), active);
    const head = el("div", "ai-settings-head");
    head.append(status, defaultRow);

    const cards = {};
    const cardsHost = el("div", "ai-provider-cards");
    builtinProviders.forEach((id) => {
      const { details, providerStatus, body } = providerRow(labels[id]);
      const key = el("input", "api-key-input");
      key.type = "password";
      key.placeholder = "API key";
      key.autocomplete = "new-password";
      key.setAttribute("aria-label", `${labels[id]} API key`);
      const model = el("input", "ui-input ai-model-input");
      model.placeholder = "Model";
      model.setAttribute("aria-label", `Model ${labels[id]}`);
      model.value = localStorage.getItem(`manga_ai_model_${id}`) || "";
      model.addEventListener("change", () => localStorage.setItem(`manga_ai_model_${id}`, model.value.trim()));
      const listId = `ai-models-${id}`;
      model.setAttribute("list", listId);
      const datalist = el("datalist");
      datalist.id = listId;

      const save = button("Lưu", "ui-btn ui-btn-primary ui-btn-compact");
      const load = button("Tải");
      load.title = "Tải danh sách model";
      const clear = button("Xóa key", "ai-link-btn");
      clear.hidden = true;

      save.addEventListener("click", async () => {
        if (!key.value.trim()) return window.showToast?.(`Nhập ${labels[id]} API key trước.`, "error");
        save.disabled = true;
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/key`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key.value.trim() }),
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          key.value = "";
          window.showToast?.(`Đã lưu key ${labels[id]}.`, "success");
          await refresh();
        } catch (err) {
          window.showToast?.("Không thể lưu key: " + err.message, "error");
        } finally {
          save.disabled = false;
        }
      });

      clear.addEventListener("click", async () => {
        clear.disabled = true;
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/key`, { method: "DELETE" });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          window.showToast?.(
            data.source === "environment"
              ? "Key đến từ biến môi trường; hãy xóa ở môi trường chạy."
              : `Đã xóa key ${labels[id]}.`,
            "info",
          );
          await refresh();
        } catch (err) {
          window.showToast?.("Không thể xóa key: " + err.message, "error");
        } finally {
          clear.disabled = false;
        }
      });

      load.addEventListener("click", async () => {
        load.disabled = true;
        load.textContent = "…";
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/models`);
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          datalist.replaceChildren(...(data.models || []).map((name) => {
            const option = document.createElement("option");
            option.value = name;
            return option;
          }));
          if (!model.value && data.models?.length) model.value = data.models[0];
          window.showToast?.(`Đã tải ${data.models?.length || 0} model từ ${labels[id]}.`, "success");
        } catch (err) {
          window.showToast?.("Không thể tải model: " + err.message, "error");
        } finally {
          load.disabled = false;
          load.textContent = "Tải";
        }
      });

      body.append(inlineField(key, save), inlineField(model, load), datalist, clear);
      cardsHost.append(details);
      cards[id] = { providerStatus, clear, model, key };
    });

    const customForm = exclusive(el("details", "ai-provider-card ai-provider-row ai-custom-provider-form"));
    const customSummary = el("summary", "ai-provider-summary");
    customSummary.append(el("span", "ai-provider-name", "+ Thêm provider tùy chỉnh"));
    const customBody = el("div", "ai-provider-body");
    const customNote = el("p", "ai-provider-note", "OpenAI-compatible, API root HTTPS, model nhận ảnh.");
    const input = (placeholder, label, type = "text", className = "ui-input") => {
      const node = el("input", className);
      node.type = type;
      node.placeholder = placeholder;
      node.setAttribute("aria-label", label);
      return node;
    };
    const customLabel = input("Tên hiển thị", "Tên provider tùy chỉnh");
    const customId = input("Mã, ví dụ local-vision", "Mã provider tùy chỉnh");
    customId.pattern = "[a-z0-9][a-z0-9_-]{0,63}";
    const customBase = input("https://api.example.com/v1", "API root HTTPS của provider", "url");
    const customModel = input("Model vision", "Model vision của provider tùy chỉnh");
    const customKey = input("API key", "API key của provider tùy chỉnh", "password", "api-key-input");
    customKey.autocomplete = "new-password";
    const customStatus = el("span", "ai-provider-status");
    const customSave = button("Lưu provider", "ui-btn ui-btn-primary ui-btn-compact");
    const customClear = button("Xóa provider", "ai-link-btn");
    customClear.disabled = true;
    const namePair = el("div", "ai-field-pair");
    namePair.append(customLabel, customId);
    const customActions = el("div", "ai-provider-actions");
    customActions.append(customSave, customClear);
    customBody.append(customNote, customStatus, namePair, customBase, customModel, customKey, customActions);
    customForm.append(customSummary, customBody);
    config.append(head, cardsHost);

    const customProviderId = () => customId.value.trim().toLowerCase();
    customSave.addEventListener("click", async () => {
      const id = customProviderId();
      if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(id)) return window.showToast?.("Mã provider dùng chữ thường, số, dấu gạch ngang hoặc gạch dưới.", "error");
      if (builtinProviders.includes(id)) return window.showToast?.("Mã này đã dành riêng cho provider có sẵn. Hãy chọn mã khác.", "error");
      if (!customLabel.value.trim() || !customBase.value.trim() || !customModel.value.trim() || !customKey.value.trim()) {
        return window.showToast?.("Nhập tên, mã, API root, model vision và API key.", "error");
      }
      customSave.disabled = true;
      try {
        const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(id)}/key`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            api_key: customKey.value.trim(), provider_label: customLabel.value.trim(),
            provider_protocol: "openai", provider_api_base: customBase.value.trim(),
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        localStorage.setItem(`manga_ai_model_${id}`, customModel.value.trim());
        localStorage.setItem(`manga_translation_vision_model_${id}`, customModel.value.trim());
        customKey.value = "";
        window.showToast?.(`Đã lưu provider ${customLabel.value.trim()}.`, "success");
        await refresh();
      } catch (err) {
        window.showToast?.("Không thể lưu provider: " + err.message, "error");
      } finally {
        customSave.disabled = false;
      }
    });
    customClear.addEventListener("click", async () => {
      const id = customProviderId();
      if (!id) return;
      customClear.disabled = true;
      try {
        const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(id)}/key?remove_config=true&provider_label=${encodeURIComponent(customLabel.value.trim() || id)}`, { method: "DELETE" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        localStorage.removeItem(`manga_ai_model_${id}`);
        localStorage.removeItem(`manga_translation_vision_model_${id}`);
        window.showToast?.(`Đã xóa provider ${id}.`, "success");
        customLabel.value = customId.value = customBase.value = customModel.value = "";
        customStatus.textContent = "";
        await refresh();
      } catch (err) {
        window.showToast?.("Không thể xóa provider: " + err.message, "error");
      } finally {
        customClear.disabled = !customProviderId();
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
        let ready = 0;
        builtinProviders.forEach((id) => {
          const info = data.providers?.[id] || {};
          const refs = cards[id];
          if (!refs.model.value) refs.model.value = info.model || "";
          setStatus(refs.providerStatus, Boolean(info.configured), info.configured ? "Đã kết nối" : "Chưa có key");
          refs.clear.hidden = !info.configured || info.source === "environment";
          refs.key.placeholder = info.configured ? "Nhập key mới để thay" : "API key";
          if (info.configured) ready += 1;
        });
        const customInfos = Object.values(data.providers || {}).filter((info) => !info.builtin);
        ready += customInfos.filter((info) => info.configured).length;
        cardsHost.querySelectorAll(".ai-custom-provider-card").forEach((card) => card.remove());
        customInfos.forEach((info) => {
          const { details, providerStatus, body } = providerRow(info.label || info.id);
          details.classList.add("ai-custom-provider-card");
          setStatus(providerStatus, Boolean(info.configured), info.configured ? "Đã kết nối" : "Chưa có key");
          const detailsText = el("p", "ai-provider-note", `${info.api_base} · ${localStorage.getItem(`manga_ai_model_${info.id}`) || "chưa chọn model"}`);
          const edit = button("Chỉnh sửa");
          edit.addEventListener("click", () => {
            customLabel.value = info.label || ""; customId.value = info.id || "";
            customBase.value = info.api_base || "";
            customModel.value = localStorage.getItem(`manga_ai_model_${info.id}`) || "";
            customStatus.textContent = info.configured ? "Đã kết nối · nhập key mới để thay" : "Chưa có API key";
            customClear.disabled = false;
            customForm.open = true;
            customLabel.focus();
          });
          body.append(detailsText, edit);
          cardsHost.insertBefore(details, customForm.isConnected ? customForm : null);
        });
        if (!customForm.isConnected) cardsHost.append(customForm);
        active.replaceChildren(...Object.values(data.providers || {}).map((info) => new Option(info.label || info.id, info.id)));
        if ([...active.options].some((option) => option.value === localStorage.getItem("manga_ai_active_provider"))) active.value = localStorage.getItem("manga_ai_active_provider");
        else active.value = "gemini";
        const editingProvider = customInfos.find((info) => info.id === customProviderId());
        if (editingProvider) {
          customClear.disabled = false;
          customStatus.textContent = editingProvider.configured ? "Đã kết nối · nhập key mới để thay" : "Chưa có API key";
        }
        window.syncAIProviderSelects?.();
        const total = builtinProviders.length + customInfos.length;
        setStatus(status, ready > 0, ready ? `${ready}/${total} dịch vụ sẵn sàng` : "Chưa có dịch vụ nào được kết nối");
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

    window.setupWorkbenchPanels?.("review");
    window.mountStitchInspector?.();
  }

  window.renderReview = setupReviewWorkspace;
})();
