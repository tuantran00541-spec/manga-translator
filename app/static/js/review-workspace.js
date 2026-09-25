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

    const config = document.createElement("div");
    config.className = "ai-provider-config";
    const status = document.createElement("span");
    status.className = "ai-provider-status";
    status.textContent = "Kiểm tra AI: Đang kiểm tra cấu hình…";

    const active = document.createElement("select");
    active.className = "ui-select ai-active-provider";
    active.setAttribute("aria-label", "Dịch vụ AI mặc định");
    builtinProviders.forEach((id) => active.add(new Option(labels[id], id)));
    active.value = localStorage.getItem("manga_ai_active_provider") || "gemini";
    active.addEventListener("change", () => {
      localStorage.setItem("manga_ai_active_provider", active.value);
      window.syncAIProviderSelects?.();
    });
    config.append(status, active);

    const cards = {};
    const cardsHost = document.createElement("div");
    cardsHost.className = "ai-provider-cards";
    builtinProviders.forEach((id) => {
      const card = document.createElement("section");
      card.className = "ai-provider-card";
      const title = document.createElement("strong");
      title.textContent = labels[id];
      const providerStatus = document.createElement("span");
      providerStatus.className = "ai-provider-status";
      const key = document.createElement("input");
      key.type = "password";
      key.className = "api-key-input";
      key.placeholder = `${labels[id]} API key`;
      key.autocomplete = "new-password";
      key.setAttribute("aria-label", `${labels[id]} API key`);
      const model = document.createElement("input");
      model.className = "ui-input ai-model-input";
      model.placeholder = "Tên model";
      model.setAttribute("aria-label", `Model ${labels[id]}`);
      model.value = localStorage.getItem(`manga_ai_model_${id}`) || "";
      model.addEventListener("change", () => localStorage.setItem(`manga_ai_model_${id}`, model.value.trim()));
      const listId = `ai-models-${id}`;
      model.setAttribute("list", listId);
      const datalist = document.createElement("datalist");
      datalist.id = listId;

      const save = document.createElement("button");
      save.type = "button";
      save.className = "ui-btn ui-btn-primary";
      save.textContent = "Lưu key";
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "ui-btn ui-btn-ghost";
      clear.textContent = "Xóa key";
      const load = document.createElement("button");
      load.type = "button";
      load.className = "ui-btn ui-btn-ghost";
      load.textContent = "Tải model";

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
        load.textContent = "Đang tải…";
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
          load.textContent = "Tải model";
        }
      });

      card.append(title, providerStatus, key, model, datalist, save, clear, load);
      cardsHost.append(card);
      cards[id] = { providerStatus, clear, model };
    });

    const customHeading = document.createElement("strong");
    customHeading.textContent = "Provider OpenAI-compatible tùy chỉnh";
    const customNote = document.createElement("p");
    customNote.className = "ui-note";
    customNote.textContent = "Dùng API root HTTPS có /models và /chat/completions; model cần hỗ trợ ảnh để dịch từ ảnh inpaint.";
    const customForm = document.createElement("section");
    customForm.className = "ai-provider-card ai-custom-provider-form";
    const customLabel = document.createElement("input");
    customLabel.className = "ui-input";
    customLabel.placeholder = "Tên hiển thị, ví dụ: Local Vision";
    customLabel.setAttribute("aria-label", "Tên provider tùy chỉnh");
    const customId = document.createElement("input");
    customId.className = "ui-input";
    customId.placeholder = "Mã provider, ví dụ: local-vision";
    customId.pattern = "[a-z0-9][a-z0-9_-]{0,63}";
    customId.setAttribute("aria-label", "Mã provider tùy chỉnh");
    const customBase = document.createElement("input");
    customBase.className = "ui-input";
    customBase.type = "url";
    customBase.placeholder = "https://api.example.com/v1";
    customBase.setAttribute("aria-label", "API root HTTPS của provider");
    const customModel = document.createElement("input");
    customModel.className = "ui-input";
    customModel.placeholder = "Tên model hỗ trợ vision";
    customModel.setAttribute("aria-label", "Model vision của provider tùy chỉnh");
    const customKey = document.createElement("input");
    customKey.className = "api-key-input";
    customKey.type = "password";
    customKey.autocomplete = "new-password";
    customKey.placeholder = "API key";
    customKey.setAttribute("aria-label", "API key của provider tùy chỉnh");
    const customStatus = document.createElement("span");
    customStatus.className = "ai-provider-status";
    const customSave = document.createElement("button");
    customSave.type = "button";
    customSave.className = "ui-btn ui-btn-primary";
    customSave.textContent = "Lưu provider";
    const customClear = document.createElement("button");
    customClear.type = "button";
    customClear.className = "ui-btn ui-btn-ghost";
    customClear.textContent = "Xóa provider";
    customClear.disabled = true;
    const customFields = [
      ["Tên hiển thị", customLabel], ["Mã provider", customId], ["API root HTTPS", customBase],
      ["Model vision", customModel], ["API key", customKey],
    ].map(([label, input]) => {
      const field = document.createElement("label");
      field.className = "ui-field";
      field.append(document.createTextNode(label), input);
      return field;
    });
    const customActions = document.createElement("div");
    customActions.className = "ui-control-row";
    customActions.append(customSave, customClear);
    customForm.append(customHeading, customNote, customStatus, ...customFields, customActions);
    config.append(cardsHost, customForm);

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
        await refresh();
      } catch (err) {
        window.showToast?.("Không thể xóa provider: " + err.message, "error");
      } finally {
        customClear.disabled = false;
      }
    });

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
          refs.providerStatus.textContent = info.configured ? `Sẵn sàng · ${info.model || "chọn model"}` : "Chưa cấu hình";
          refs.providerStatus.classList.toggle("configured", Boolean(info.configured));
          refs.clear.disabled = !info.configured || info.source === "environment";
          if (info.configured) ready += 1;
        });
        const customInfos = Object.values(data.providers || {}).filter((info) => !info.builtin);
        ready += customInfos.filter((info) => info.configured).length;
        cardsHost.querySelectorAll(".ai-custom-provider-card").forEach((card) => card.remove());
        customInfos.forEach((info) => {
          const card = document.createElement("section");
          card.className = "ai-provider-card ai-custom-provider-card";
          const title = document.createElement("strong");
          title.textContent = info.label || info.id;
          const details = document.createElement("span");
          details.className = "ui-value";
          details.textContent = `${info.api_base} · ${localStorage.getItem(`manga_ai_model_${info.id}`) || "chưa chọn model"}`;
          const edit = document.createElement("button");
          edit.type = "button";
          edit.className = "ui-btn ui-btn-ghost";
          edit.textContent = "Chỉnh sửa";
          edit.addEventListener("click", () => {
            customLabel.value = info.label || ""; customId.value = info.id || "";
            customBase.value = info.api_base || "";
            customModel.value = localStorage.getItem(`manga_ai_model_${info.id}`) || "";
            customStatus.textContent = info.configured ? "Đã cấu hình · model nhận ảnh OpenAI-compatible" : "Chưa có API key";
            customClear.disabled = false;
            customLabel.focus();
          });
          card.append(title, details, edit);
          cardsHost.append(card);
        });
        active.replaceChildren(...Object.values(data.providers || {}).map((info) => new Option(info.label || info.id, info.id)));
        if ([...active.options].some((option) => option.value === localStorage.getItem("manga_ai_active_provider"))) active.value = localStorage.getItem("manga_ai_active_provider");
        else active.value = "gemini";
        if (customInfos.length && customId.value && !customInfos.some((info) => info.id === customId.value)) customStatus.textContent = "Sẵn sàng thêm provider tùy chỉnh.";
        const editingProvider = customInfos.find((info) => info.id === customProviderId());
        if (editingProvider) {
          customClear.disabled = false;
          customStatus.textContent = editingProvider.configured
            ? "Đã cấu hình · model vision gửi ảnh qua OpenAI-compatible API"
            : "Provider đã lưu · chưa có API key";
        }
        window.syncAIProviderSelects?.();
        status.textContent = ready ? `Kiểm tra AI: ${ready} dịch vụ sẵn sàng` : "Kiểm tra AI: Chưa cấu hình";
        status.classList.toggle("configured", ready > 0);
      } catch (_) {
        status.textContent = "Kiểm tra AI: Lỗi cấu hình";
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
