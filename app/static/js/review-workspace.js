(() => {
  if (typeof window.createReviewCard !== "function") return;

  let activeReviewIndex = 0;
  let reviewLastChapterId = null;
  const maskSnapshots = new Map();
  let aiSettingsInstance = null;
  let aiProviderRegistry = new Map();
  let aiSelectSyncQueued = false;

  const CUSTOM_PROVIDER_STORAGE = "manga_ai_custom_providers_v1";
  const ACTIVE_PROVIDER_STORAGE = "manga_ai_active_provider";

  function stopCardBrush(card) {
    if (!card) return;
    const canvas = card.querySelector("canvas.brush-canvas");
    if (canvas && typeof canvas._stopBrush === "function") canvas._stopBrush();
  }

  function cleanupCard(card) {
    if (!card) return;
    const canvas = card.querySelector("canvas.brush-canvas");
    if (canvas && typeof canvas._cleanupBrush === "function") canvas._cleanupBrush();
    else if (canvas?._brushAbort) canvas._brushAbort.abort();
    card.remove();
  }

  function updateAIStatus(source, target) {
    if (!target) return;
    const raw = source?.textContent || "";
    target.classList.toggle("ready", /sẵn sàng/i.test(raw));
    if (/sẵn sàng/i.test(raw)) {
      target.textContent = raw.replace(/^(?:Gemini QC|Kiểm tra AI):\s*/i, "AI ");
    } else if (/chưa (?:có key|cấu hình)/i.test(raw)) {
      target.textContent = "AI chưa cấu hình";
    } else if (/(?:secure storage|kho bí mật)/i.test(raw)) {
      target.textContent = "Kho bí mật chưa sẵn sàng";
    } else if (/lỗi cấu hình/i.test(raw)) {
      target.textContent = "Lỗi cấu hình AI";
    } else {
      target.textContent = "Đang kiểm tra cấu hình AI…";
    }
  }

  function bindAIStatus(source, target) {
    updateAIStatus(source, target);
    if (!source) return;
    const observer = new MutationObserver(() => updateAIStatus(source, target));
    observer.observe(source, { childList: true, characterData: true, subtree: true, attributes: true });
    return () => observer.disconnect();
  }

  function readCustomProviders() {
    try {
      const raw = JSON.parse(localStorage.getItem(CUSTOM_PROVIDER_STORAGE) || "[]");
      if (!Array.isArray(raw)) return [];
      return raw.filter((item) => {
        if (!item || typeof item !== "object") return false;
        if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(String(item.id || ""))) return false;
        if (item.protocol !== "openai") return false;
        return /^https:\/\//i.test(String(item.api_base || ""));
      }).map((item) => ({
        id: String(item.id),
        label: String(item.label || item.id).slice(0, 80),
        protocol: "openai",
        api_base: String(item.api_base).replace(/\/+$/, ""),
        builtin: false,
        configured: Boolean(item.configured),
        source: item.configured ? "os_secure_storage" : "none",
        model: String(item.model || ""),
        translation_model: String(item.translation_model || item.model || ""),
      }));
    } catch (_) {
      return [];
    }
  }

  function writeCustomProviders(items) {
    localStorage.setItem(CUSTOM_PROVIDER_STORAGE, JSON.stringify(items));
  }

  function updateCustomProvider(id, patch) {
    const items = readCustomProviders();
    const index = items.findIndex((item) => item.id === id);
    if (index < 0) return null;
    items[index] = { ...items[index], ...patch };
    writeCustomProviders(items);
    return items[index];
  }

  function providerModel(provider) {
    return localStorage.getItem(`manga_ai_model_${provider.id}`)
      || provider.model
      || provider.translation_model
      || "";
  }

  function setProviderModel(provider, value) {
    const model = String(value || "").trim();
    localStorage.setItem(`manga_ai_model_${provider.id}`, model);
    if (!provider.builtin) updateCustomProvider(provider.id, { model, translation_model: model });
    provider.model = model;
    return model;
  }

  function providerRowsFor(task) {
    const rows = [...aiProviderRegistry.values()];
    if (task === "translation") return rows.filter((item) => item.protocol === "openai");
    return rows;
  }

  function syncProviderSelect(select, task) {
    if (!select || !aiProviderRegistry.size) return;
    const rows = providerRowsFor(task);
    const storageKey = task === "translation" ? "manga_translation_provider" : ACTIVE_PROVIDER_STORAGE;
    const requested = localStorage.getItem(storageKey) || select.value;
    const previous = select.value;
    select.replaceChildren(...rows.map((provider) => new Option(provider.label, provider.id)));
    const fallback = rows.find((item) => item.configured)?.id || rows[0]?.id || "";
    select.value = rows.some((item) => item.id === requested) ? requested : fallback;
    if (select.value) localStorage.setItem(storageKey, select.value);
    if (select.value !== previous) select.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function syncDynamicProviderSelects() {
    aiSelectSyncQueued = false;
    document.querySelectorAll(".chapter-qc-provider").forEach((select) => syncProviderSelect(select, "qc"));
    document.querySelectorAll(".chapter-translate-provider").forEach((select) => syncProviderSelect(select, "translation"));
  }

  function scheduleDynamicProviderSelectSync() {
    if (aiSelectSyncQueued) return;
    aiSelectSyncQueued = true;
    queueMicrotask(syncDynamicProviderSelects);
  }

  function publishProviderRegistry() {
    const settings = {};
    aiProviderRegistry.forEach((provider, id) => {
      settings[id] = { ...provider, model: providerModel(provider) };
    });
    window.aiProviderSettings = settings;
    window.aiProviderRegistry = [...aiProviderRegistry.values()].map((provider) => ({ ...provider }));
    scheduleDynamicProviderSelectSync();
  }

  function customProviderQuery(provider) {
    const params = new URLSearchParams();
    if (!provider.builtin) {
      params.set("provider_label", provider.label);
      params.set("provider_protocol", provider.protocol);
      params.set("provider_api_base", provider.api_base);
    }
    const query = params.toString();
    return query ? `?${query}` : "";
  }

  function makeCustomProviderId() {
    return `custom-${Date.now().toString(36)}`.slice(0, 64);
  }

  function createAIProviderSettings() {
    if (aiSettingsInstance?.config?.isConnected) return aiSettingsInstance.status;

    const config = document.createElement("div");
    config.className = "ai-provider-config ai-provider-registry";

    const top = document.createElement("div");
    top.className = "ai-registry-top";
    const status = document.createElement("span");
    status.className = "ai-provider-status";
    status.textContent = "Kiểm tra AI: Đang tải cấu hình…";
    const addToggle = document.createElement("button");
    addToggle.type = "button";
    addToggle.className = "ui-btn ui-btn-ghost ui-btn-compact";
    addToggle.textContent = "+ Thêm API";
    top.append(status, addToggle);

    const defaultField = document.createElement("label");
    defaultField.className = "ui-field ai-default-provider-field";
    const defaultLabel = document.createElement("span");
    defaultLabel.textContent = "Dịch vụ mặc định";
    const active = document.createElement("select");
    active.className = "ui-select ai-active-provider";
    active.setAttribute("aria-label", "Dịch vụ AI mặc định");
    active.addEventListener("change", () => {
      localStorage.setItem(ACTIVE_PROVIDER_STORAGE, active.value);
      publishProviderRegistry();
    });
    defaultField.append(defaultLabel, active);

    const addPanel = document.createElement("div");
    addPanel.className = "ai-add-provider-panel";
    addPanel.hidden = true;
    const addHint = document.createElement("p");
    addHint.className = "ai-provider-hint";
    addHint.textContent = "API tùy chỉnh phải tương thích OpenAI: HTTPS API root có /models và /chat/completions.";
    const addName = document.createElement("input");
    addName.className = "ui-input";
    addName.placeholder = "Tên dịch vụ, ví dụ: My API";
    addName.setAttribute("aria-label", "Tên dịch vụ AI tùy chỉnh");
    const addBase = document.createElement("input");
    addBase.className = "ui-input";
    addBase.type = "url";
    addBase.placeholder = "https://api.example.com/v1";
    addBase.autocomplete = "url";
    addBase.setAttribute("aria-label", "OpenAI-compatible API root");
    const addModel = document.createElement("input");
    addModel.className = "ui-input";
    addModel.placeholder = "Model mặc định (có thể nhập sau)";
    addModel.setAttribute("aria-label", "Model mặc định của API tùy chỉnh");
    const addActions = document.createElement("div");
    addActions.className = "ai-provider-actions";
    const addConfirm = document.createElement("button");
    addConfirm.type = "button";
    addConfirm.className = "ui-btn ui-btn-primary";
    addConfirm.textContent = "Thêm dịch vụ";
    const addCancel = document.createElement("button");
    addCancel.type = "button";
    addCancel.className = "ui-btn ui-btn-ghost";
    addCancel.textContent = "Hủy";
    addActions.append(addConfirm, addCancel);
    addPanel.append(addHint, addName, addBase, addModel, addActions);

    const list = document.createElement("div");
    list.className = "ai-provider-list";
    config.append(top, defaultField, addPanel, list);

    addToggle.addEventListener("click", () => {
      addPanel.hidden = !addPanel.hidden;
      if (!addPanel.hidden) addName.focus();
    });
    addCancel.addEventListener("click", () => {
      addPanel.hidden = true;
      addName.value = "";
      addBase.value = "";
      addModel.value = "";
    });
    addConfirm.addEventListener("click", () => {
      const label = addName.value.trim();
      const apiBase = addBase.value.trim().replace(/\/+$/, "");
      const model = addModel.value.trim();
      if (!label) return window.showToast?.("Nhập tên dịch vụ.", "error");
      if (!/^https:\/\/[^\s]+$/i.test(apiBase)) return window.showToast?.("API root phải là URL HTTPS hợp lệ.", "error");
      const items = readCustomProviders();
      const provider = {
        id: makeCustomProviderId(),
        label,
        protocol: "openai",
        api_base: apiBase,
        builtin: false,
        configured: false,
        source: "none",
        model,
        translation_model: model,
      };
      items.push(provider);
      writeCustomProviders(items);
      if (model) localStorage.setItem(`manga_ai_model_${provider.id}`, model);
      localStorage.setItem(ACTIVE_PROVIDER_STORAGE, provider.id);
      addPanel.hidden = true;
      addName.value = "";
      addBase.value = "";
      addModel.value = "";
      refresh();
    });

    function makeProviderCard(provider) {
      const card = document.createElement("details");
      card.className = "ai-provider-card";
      card.dataset.providerId = provider.id;

      const summary = document.createElement("summary");
      summary.className = "ai-provider-summary";
      const identity = document.createElement("span");
      identity.className = "ai-provider-identity";
      const name = document.createElement("strong");
      name.textContent = provider.label;
      const meta = document.createElement("span");
      meta.className = "ai-provider-summary-meta";
      const currentModel = providerModel(provider);
      meta.textContent = provider.configured
        ? `Sẵn sàng${currentModel ? ` · ${currentModel}` : " · chọn model"}`
        : "Chưa có API key";
      identity.append(name, meta);
      const badge = document.createElement("span");
      badge.className = "ui-badge ai-provider-protocol";
      badge.textContent = provider.protocol === "gemini" ? "Gemini" : "OpenAI-compatible";
      summary.append(identity, badge);

      const body = document.createElement("div");
      body.className = "ai-provider-body";
      if (!provider.builtin) {
        const endpoint = document.createElement("div");
        endpoint.className = "ai-provider-endpoint";
        const endpointLabel = document.createElement("span");
        endpointLabel.textContent = "API";
        const endpointValue = document.createElement("code");
        endpointValue.textContent = provider.api_base;
        endpoint.append(endpointLabel, endpointValue);
        body.appendChild(endpoint);
      }

      const key = document.createElement("input");
      key.type = "password";
      key.className = "api-key-input";
      key.placeholder = "API key";
      key.autocomplete = "new-password";
      key.setAttribute("aria-label", `${provider.label} API key`);

      const modelWrap = document.createElement("div");
      modelWrap.className = "ai-model-row";
      const model = document.createElement("input");
      model.className = "ui-input ai-model-input";
      model.placeholder = "Tên model";
      model.setAttribute("aria-label", `Model ${provider.label}`);
      model.value = currentModel;
      const listId = `ai-models-${provider.id}`;
      model.setAttribute("list", listId);
      const datalist = document.createElement("datalist");
      datalist.id = listId;
      model.addEventListener("change", () => {
        setProviderModel(provider, model.value);
        meta.textContent = provider.configured
          ? `Sẵn sàng${model.value.trim() ? ` · ${model.value.trim()}` : " · chọn model"}`
          : "Chưa có API key";
        publishProviderRegistry();
      });
      const load = document.createElement("button");
      load.type = "button";
      load.className = "ui-btn ui-btn-ghost";
      load.textContent = "Kiểm tra & tải model";
      modelWrap.append(model, load, datalist);

      const actions = document.createElement("div");
      actions.className = "ai-provider-actions";
      const save = document.createElement("button");
      save.type = "button";
      save.className = "ui-btn ui-btn-primary";
      save.textContent = "Lưu key";
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "ui-btn ui-btn-ghost";
      clear.textContent = "Xóa key";
      clear.disabled = !provider.configured || provider.source === "environment";
      actions.append(save, clear);

      if (!provider.builtin) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "ui-btn ui-btn-danger";
        remove.textContent = "Xóa dịch vụ";
        remove.addEventListener("click", async () => {
          remove.disabled = true;
          try {
            const query = new URLSearchParams({ provider_label: provider.label, remove_config: "true" });
            await fetch(`/api/visual_qc/providers/${encodeURIComponent(provider.id)}/key?${query}`, { method: "DELETE" });
          } catch (_) {}
          const remaining = readCustomProviders().filter((item) => item.id !== provider.id);
          writeCustomProviders(remaining);
          localStorage.removeItem(`manga_ai_model_${provider.id}`);
          if (localStorage.getItem(ACTIVE_PROVIDER_STORAGE) === provider.id) localStorage.removeItem(ACTIVE_PROVIDER_STORAGE);
          if (localStorage.getItem("manga_translation_provider") === provider.id) localStorage.removeItem("manga_translation_provider");
          refresh();
        });
        actions.appendChild(remove);
      }

      save.addEventListener("click", async () => {
        if (!key.value.trim()) return window.showToast?.(`Nhập API key cho ${provider.label}.`, "error");
        save.disabled = true;
        try {
          const payload = { api_key: key.value.trim(), provider_label: provider.label };
          if (!provider.builtin) {
            payload.provider_protocol = provider.protocol;
            payload.provider_api_base = provider.api_base;
          }
          const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(provider.id)}/key`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          key.value = "";
          if (!provider.builtin) updateCustomProvider(provider.id, { configured: true });
          window.showToast?.(`Đã lưu key ${provider.label}.`, "success");
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
          const query = new URLSearchParams({ provider_label: provider.label });
          const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(provider.id)}/key?${query}`, { method: "DELETE" });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          if (!provider.builtin) updateCustomProvider(provider.id, { configured: false });
          window.showToast?.(data.source === "environment" ? "Key đến từ biến môi trường; hãy xóa ở môi trường chạy." : `Đã xóa key ${provider.label}.`, "info");
          await refresh();
        } catch (err) {
          window.showToast?.("Không thể xóa key: " + err.message, "error");
        } finally {
          clear.disabled = false;
        }
      });

      load.addEventListener("click", async () => {
        load.disabled = true;
        load.textContent = "Đang kiểm tra…";
        try {
          const response = await fetch(`/api/visual_qc/providers/${encodeURIComponent(provider.id)}/models${customProviderQuery(provider)}`);
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          const models = Array.isArray(data.models) ? data.models : [];
          datalist.replaceChildren(...models.map((modelName) => {
            const option = document.createElement("option");
            option.value = modelName;
            return option;
          }));
          if (!model.value && models.length) {
            model.value = models[0];
            setProviderModel(provider, model.value);
          }
          if (!provider.builtin) updateCustomProvider(provider.id, { configured: true });
          window.showToast?.(`Kết nối ${provider.label} hợp lệ · ${models.length} model.`, "success");
          await refresh();
        } catch (err) {
          if (!provider.builtin && /API key|409/i.test(err.message)) updateCustomProvider(provider.id, { configured: false });
          window.showToast?.("Không thể tải model: " + err.message, "error");
        } finally {
          load.disabled = false;
          load.textContent = "Kiểm tra & tải model";
        }
      });

      body.append(key, modelWrap, actions);
      card.append(summary, body);
      return card;
    }

    async function refresh() {
      try {
        const response = await fetch("/api/visual_qc/settings");
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        const merged = new Map();
        Object.values(data.providers || {}).forEach((provider) => {
          if (!provider?.id) return;
          merged.set(provider.id, { ...provider, builtin: true });
        });
        readCustomProviders().forEach((provider) => merged.set(provider.id, provider));
        aiProviderRegistry = merged;
        publishProviderRegistry();

        const rows = [...merged.values()];
        active.replaceChildren(...rows.map((provider) => new Option(provider.label, provider.id)));
        const requested = localStorage.getItem(ACTIVE_PROVIDER_STORAGE);
        const fallback = rows.find((provider) => provider.configured)?.id || rows[0]?.id || "";
        active.value = rows.some((provider) => provider.id === requested) ? requested : fallback;
        if (active.value) localStorage.setItem(ACTIVE_PROVIDER_STORAGE, active.value);

        list.replaceChildren(...rows.map(makeProviderCard));
        const ready = rows.filter((provider) => provider.configured).length;
        status.textContent = ready ? `Kiểm tra AI: ${ready} dịch vụ sẵn sàng` : "Kiểm tra AI: Chưa cấu hình";
        status.classList.toggle("configured", ready > 0);
      } catch (err) {
        status.textContent = "Kiểm tra AI: Lỗi cấu hình";
        status.classList.remove("configured");
      }
    }

    refresh();
    if (typeof window.mountAISettings === "function") window.mountAISettings(config);
    aiSettingsInstance = { config, status, refresh };
    return status;
  }
  window.createAIProviderSettings = createAIProviderSettings;
  window.syncAIProviderSelects = syncDynamicProviderSelects;
  window.getAIProviderRegistry = () => [...aiProviderRegistry.values()].map((provider) => ({ ...provider }));

  function captureMaskSnapshot(card) {
    if (!card) return;
    const canonicalIndex = parseInt(card.dataset.pageIndex, 10);
    const canvas = card.querySelector("canvas.brush-canvas");
    if (!Number.isFinite(canonicalIndex) || !canvas) return;
    if (!canvas._reviewDirty || !canvas.width || !canvas.height) {
      maskSnapshots.delete(canonicalIndex);
      return;
    }
    try {
      maskSnapshots.set(canonicalIndex, canvas.toDataURL("image/png"));
    } catch (err) {
      console.warn("Could not preserve review mask snapshot:", err);
    }
  }

  function setupReviewWorkspace() {
    window.cleanupReviewWorkspace?.();
    const container = document.getElementById("page-view");
    if (!container) return;
    window.setAppStage?.("review");

    if (window._reviewKeyDownHandler) {
      window.removeEventListener("keydown", window._reviewKeyDownHandler);
      window._reviewKeyDownHandler = null;
    }

    container.querySelectorAll(".review-card").forEach(captureMaskSnapshot);
    container.querySelectorAll(".brush-canvas").forEach((canvas) => {
      if (typeof canvas._cleanupBrush === "function") canvas._cleanupBrush();
      else if (canvas?._brushAbort) canvas._brushAbort.abort();
    });
    container.replaceChildren();
    container.className = "review-mode";

    if (window.currentChapterId && reviewLastChapterId !== window.currentChapterId) {
      reviewLastChapterId = window.currentChapterId;
      activeReviewIndex = 0;
      maskSnapshots.clear();
    }

    const aiSettingsStatus = createAIProviderSettings();
    const pageIndices = (window.currentManifest?.pages || [])
      .map((page, index) => ({ page, index }))
      .filter(({ page }) => !page.skipped)
      .map(({ index }) => index);

    if (!pageIndices.length) return;

    const targetCanonical = window.initialReviewCanonicalPageIndex !== undefined && window.initialReviewCanonicalPageIndex !== null
      ? window.initialReviewCanonicalPageIndex
      : (window.currentManifest?.workflow?.stage === "review" && window.currentManifest?.workflow?.page_index !== undefined
          ? window.currentManifest.workflow.page_index
          : null);

    if (targetCanonical !== null) {
      const foundIdx = pageIndices.indexOf(Number(targetCanonical));
      if (foundIdx !== -1) activeReviewIndex = foundIdx;
      window.initialReviewCanonicalPageIndex = null;
    }
    activeReviewIndex = Math.max(0, Math.min(activeReviewIndex, pageIndices.length - 1));

    const workspace = document.createElement("div");
    workspace.className = "review-workspace-shell";

    const toolbar = document.createElement("div");
    toolbar.className = "review-sticky-toolbar";
    const title = document.createElement("div");
    title.className = "review-toolbar-title";
    title.textContent = `${pageIndices.length} lát đã xử lý`;
    const controlsSlot = document.createElement("div");
    controlsSlot.className = "review-controls-slot";
    const actions = document.createElement("div");
    actions.className = "review-actions-group";
    const aiStatus = document.createElement("span");
    aiStatus.className = "review-ai-status";
    const cleanupAIStatus = bindAIStatus(aiSettingsStatus, aiStatus);

    const help = document.createElement("details");
    help.className = "ui-disclosure review-help";
    const helpSummary = document.createElement("summary");
    helpSummary.className = "ui-btn ui-btn-ghost";
    helpSummary.textContent = "Hướng dẫn";
    const helpPanel = document.createElement("div");
    helpPanel.className = "review-help-panel";
    helpPanel.innerHTML = '<p>Chọn <strong>Đánh dấu vùng lỗi</strong> để tô thủ công. Nhấp đúp vào vùng nền đồng màu để chọn nhanh toàn bộ vùng liên thông. Sau khi kiểm tra vùng đánh dấu, chọn <strong>Xử lý vùng đánh dấu</strong>.</p>';
    help.append(helpSummary, helpPanel);

    const continueBtn = document.createElement("button");
    continueBtn.className = "ui-btn ui-btn-primary review-primary-action";
    continueBtn.textContent = "Mở biên tập";
    continueBtn.addEventListener("click", () => {
      const activeCard = container.querySelector(".review-canvas-host .review-card");
      captureMaskSnapshot(activeCard);
      if (maskSnapshots.size > 0) {
        const pendingCanonical = maskSnapshots.keys().next().value;
        const pendingIndex = pageIndices.indexOf(pendingCanonical);
        if (pendingIndex >= 0 && pendingIndex !== activeReviewIndex) {
          activeReviewIndex = pendingIndex;
          navigator.setActive(pendingIndex);
          renderActive();
        }
        showToast(
          `${maskSnapshots.size} trang còn vùng đánh dấu chưa được xử lý. `
            + "Hãy xử lý hoặc xóa các nét đánh dấu trước khi mở trình biên tập.",
          "error",
        );
        return;
      }
      if (typeof window.hasUnsavedStitchedMarks === "function" && window.hasUnsavedStitchedMarks()) {
        showToast("Ảnh ghép còn vùng đánh dấu chưa được xử lý. Hãy xử lý hoặc xóa các nét đánh dấu trước khi mở trình biên tập.", "error");
        return;
      }
      if (window._reviewKeyDownHandler) {
        window.removeEventListener("keydown", window._reviewKeyDownHandler);
        window._reviewKeyDownHandler = null;
      }
      const canonicalIndex = activeCard ? (parseInt(activeCard.dataset.pageIndex, 10) || 0) : 0;
      window.cleanupReviewWorkspace?.();
      if (window.editorState) window.editorState.activePageIndex = canonicalIndex;
      if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("editor", canonicalIndex);
      if (typeof window.renderEditor === "function") window.renderEditor();
    });
    actions.appendChild(continueBtn);
    toolbar.append(title, actions);

    const layout = document.createElement("div");
    layout.className = "workbench-stage-grid review-workbench-grid";

    const navItems = pageIndices.map((canonicalIndex) => {
      const item = window.currentManifest.pages[canonicalIndex];
      const needsReview = Boolean(item.needs_review || item.detection_state === "needs_review" || (item.detection_issues || []).length);
      return {
        key: canonicalIndex,
        label: pageLabel(window.currentManifest.pages, canonicalIndex),
        image: item.clean || item.original,
        state: needsReview ? "review" : "ready",
        stateLabel: needsReview ? "Cần kiểm tra" : "Đã xác minh",
      };
    });
    let navigator = null;

    const canvasHost = document.createElement("div");
    canvasHost.className = "review-canvas-host workbench-canvas-column";

    const inspector = document.createElement("aside");
    inspector.className = "context-inspector review-inspector";
    inspector.setAttribute("aria-label", "Công cụ và kiểm tra chất lượng");
    const inspectorHeading = document.createElement("div");
    inspectorHeading.className = "context-inspector-heading";
    inspectorHeading.innerHTML = '<strong>Hiệu chỉnh trang</strong>';
    const gestureSection = document.createElement("section");
    gestureSection.className = "inspector-section review-gesture-section";
    const stitchedControlsSlot = document.createElement("div");
    stitchedControlsSlot.className = "review-stitched-controls-slot";
    gestureSection.append(controlsSlot, stitchedControlsSlot);
    workspace._stitchedControlsSlot = stitchedControlsSlot;
    const helpSection = document.createElement("section");
    helpSection.className = "inspector-section review-help-section";
    helpSection.append(aiStatus, help);
    inspector.append(inspectorHeading, gestureSection, helpSection);

    navigator = window.createPageNavigator({
      items: navItems,
      activeIndex: activeReviewIndex,
      title: "Trang kiểm tra",
      ariaLabel: "Điều hướng trang kiểm tra chất lượng",
      onSelect: (index) => {
        if (workspace.classList.contains("review-busy")) {
          if (typeof window.showToast === "function") {
            window.showToast("Đang bận xử lý (OCR/AI/Lưu), vui lòng đợi giây lát...", "warning");
          }
          navigator.setActive(activeReviewIndex);
          return;
        }
        activeReviewIndex = index;
        navigator.setActive(index);
        renderActive();
        if (typeof window.onReviewSliceSelect === "function") {
          window.onReviewSliceSelect(pageIndices[index]);
        }
      },
    });

    workspace._pageNavigator = navigator;
    layout.append(navigator.element, canvasHost, inspector);
    workspace.append(toolbar, layout);
    container.appendChild(workspace);

    let mountedCard = null;
    const restoreMounted = () => {
      if (!mountedCard) return;
      stopCardBrush(mountedCard);
      captureMaskSnapshot(mountedCard);
      cleanupCard(mountedCard);
      mountedCard = null;
    };

    window.cleanupReviewWorkspace = () => {
      cleanupAIStatus?.();
      window._reviewStitchAbort?.abort();
      restoreMounted();
      if (window._reviewKeyDownHandler) {
        window.removeEventListener("keydown", window._reviewKeyDownHandler);
        window._reviewKeyDownHandler = null;
      }
    };

    const renderActive = () => {
      restoreMounted();
      controlsSlot.replaceChildren();

      const canonicalIndex = pageIndices[activeReviewIndex];
      const card = window.createReviewCard(canonicalIndex, maskSnapshots.get(canonicalIndex) || null);
      if (!card) return;
      mountedCard = card;
      const stitchedView = canvasHost.querySelector(".review-stitched-shell");
      canvasHost.replaceChildren(card, ...(stitchedView ? [stitchedView] : []));
      if (typeof card._mountReview === "function") card._mountReview();

      const controls = card.querySelector(".review-controls");
      if (controls) controlsSlot.appendChild(controls);

      navigator.setActive(activeReviewIndex);
      if (typeof window.setWorkflowCheckpoint === "function") {
        window.setWorkflowCheckpoint("review", canonicalIndex);
      }

      const canvas = card.querySelector("canvas.brush-canvas");
      const brushBtn = controls?.querySelector(".brush-toggle-btn") || null;
      const clearBtn = controls?.querySelector(".clear-brush-btn") || null;
      const repaintBtn = controls?.querySelector(".repaint-btn") || null;
      const resetBtn = controls?.querySelector(".reset-manual-btn") || null;
      const brushSize = controls?.querySelector(".brush-size-slider") || null;
      const aiQcBtn = controls?.querySelector(".ai-qc-btn") || null;
      let isSyncing = false;
      let lastBusy = null;
      const syncBusy = () => {
        if (isSyncing) return;
        isSyncing = true;
        try {
          const busy = Boolean(card._reviewBusy);
          if (busy === lastBusy) return;
          lastBusy = busy;

          workspace.classList.toggle("review-busy", busy);
          if (busy && canvas && typeof canvas._stopBrush === "function") canvas._stopBrush();
          if (brushBtn && brushBtn.disabled !== busy) brushBtn.disabled = busy;
          if (clearBtn && clearBtn.disabled !== busy) clearBtn.disabled = busy;
          if (repaintBtn && repaintBtn.disabled !== busy) repaintBtn.disabled = busy;
          if (resetBtn && resetBtn.disabled !== busy) resetBtn.disabled = busy;
          if (brushSize && brushSize.disabled !== busy) brushSize.disabled = busy;
          if (aiQcBtn && aiQcBtn.disabled !== busy) aiQcBtn.disabled = busy;
          navigator.setBusy(busy);
          if (continueBtn && continueBtn.disabled !== busy) continueBtn.disabled = busy;
          window.syncChapterQCWorkspace?.(workspace);
        } finally {
          isSyncing = false;
        }
      };
      card._syncReviewBusy = syncBusy;
      syncBusy();
    };

    const onReviewKeyDown = (e) => {
      const pageView = document.getElementById("page-view");
      if (!pageView || !pageView.classList.contains("review-mode")) return;

      const activeEl = document.activeElement;
      const tag = activeEl ? activeEl.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || (activeEl && activeEl.isContentEditable)) {
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "PageUp") {
        if (activeReviewIndex > 0) {
          e.preventDefault();
          if (workspace.classList.contains("review-busy")) {
            if (typeof window.showToast === "function") {
              window.showToast("Đang bận xử lý (OCR/AI/Lưu), vui lòng đợi giây lát...", "warning");
            }
            return;
          }
          navigator.select(activeReviewIndex - 1);
        }
      } else if (e.key === "ArrowRight" || e.key === "PageDown") {
        if (activeReviewIndex < pageIndices.length - 1) {
          e.preventDefault();
          if (workspace.classList.contains("review-busy")) {
            if (typeof window.showToast === "function") {
              window.showToast("Đang bận xử lý (OCR/AI/Lưu), vui lòng đợi giây lát...", "warning");
            }
            return;
          }
          navigator.select(activeReviewIndex + 1);
        }
      }
    };

    window._reviewKeyDownHandler = onReviewKeyDown;
    window.addEventListener("keydown", onReviewKeyDown);

    renderActive();
    window.setupWorkbenchPanels?.("review");
    window.mountChapterOCR?.();
    window.mountChapterQC?.();
    if (typeof window.mountStitchInspector === "function") {
      window.mountStitchInspector();
    }
  }

  const aiSelectObserver = new MutationObserver(scheduleDynamicProviderSelectSync);
  document.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("page-view");
    if (root) aiSelectObserver.observe(root, { childList: true, subtree: true });
  });

  window.renderReview = setupReviewWorkspace;
})();