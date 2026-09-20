(() => {
  "use strict";

  let reviewLastChapterId = null;
  let aiSettingsInstance = null;

  function cleanSourcePage(page, fallbackIndex) {
    return Number.isInteger(page?.source_page) ? page.source_page : fallbackIndex;
  }

  function sourceGroups() {
    const groups = new Map();
    (window.currentManifest?.pages || []).forEach((page, canonicalIndex) => {
      if (!page) return;
      const sourcePage = cleanSourcePage(page, canonicalIndex);
      if (!groups.has(sourcePage)) groups.set(sourcePage, []);
      groups.get(sourcePage).push({ page, canonicalIndex });
    });
    for (const items of groups.values()) {
      items.sort((a, b) => Number(a.page?.slice_index || 0) - Number(b.page?.slice_index || 0));
    }
    return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
  }

  function createAIProviderSettings() {
    if (aiSettingsInstance?.config?.isConnected) return aiSettingsInstance.status;

    const providers = ["gemini", "deepseek", "openai", "openrouter", "experiential"];
    const labels = {
      gemini: "Google Gemini",
      deepseek: "DeepSeek",
      openai: "OpenAI",
      openrouter: "OpenRouter",
      experiential: "Experiential Labs",
    };

    const config = document.createElement("div");
    config.className = "ai-provider-config";
    const status = document.createElement("span");
    status.className = "ai-provider-status";
    status.textContent = "Kiểm tra AI: Đang kiểm tra cấu hình…";

    const active = document.createElement("select");
    active.className = "ui-select ai-active-provider";
    active.setAttribute("aria-label", "Dịch vụ AI mặc định");
    providers.forEach((id) => active.add(new Option(labels[id], id)));
    active.value = localStorage.getItem("manga_ai_active_provider") || "gemini";
    active.addEventListener("change", () => localStorage.setItem("manga_ai_active_provider", active.value));
    config.append(status, active);

    const cards = {};
    providers.forEach((id) => {
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
      config.append(card);
      cards[id] = { providerStatus, clear, model };
    });

    async function refresh() {
      try {
        const response = await fetch("/api/visual_qc/settings");
        const data = await response.json();
        window.aiProviderSettings = data.providers || {};
        let ready = 0;
        providers.forEach((id) => {
          const info = data.providers?.[id] || {};
          const refs = cards[id];
          if (!refs.model.value) refs.model.value = info.model || "";
          refs.providerStatus.textContent = info.configured ? `Sẵn sàng · ${info.model || "chọn model"}` : "Chưa cấu hình";
          refs.providerStatus.classList.toggle("configured", Boolean(info.configured));
          refs.clear.disabled = !info.configured || info.source === "environment";
          if (info.configured) ready += 1;
        });
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
    const rawSource = Number.parseInt(workspace.dataset.pendingSourcePage || "", 10);
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
      sourcePage: Number.isInteger(rawSource) ? rawSource : null,
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
    window.cleanupReviewWorkspace?.();

    const container = document.getElementById("page-view");
    if (!container) return;
    window.setAppStage?.("review");

    if (window._reviewKeyDownHandler) {
      window.removeEventListener("keydown", window._reviewKeyDownHandler);
      window._reviewKeyDownHandler = null;
    }

    container.replaceChildren();
    container.className = "review-mode review-canvas-mode";

    const chapterId = window.currentChapterId || "";
    if (reviewLastChapterId !== chapterId) reviewLastChapterId = chapterId;

    const groups = sourceGroups();
    const sourcePages = [...groups.keys()];
    if (!sourcePages.length) return;

    const requestedCanonical = window.initialReviewCanonicalPageIndex
      ?? window.currentManifest?.workflow?.page_index
      ?? null;
    const requestedPage = Number.isFinite(Number(requestedCanonical))
      ? window.currentManifest?.pages?.[Number(requestedCanonical)]
      : null;
    const requestedSource = requestedPage
      ? cleanSourcePage(requestedPage, Number(requestedCanonical))
      : sourcePages[0];
    const previousSource = previousState?.chapterId === chapterId
      ? previousState.sourcePage
      : null;
    const activeSource = sourcePages.includes(previousSource)
      ? previousSource
      : sourcePages.includes(requestedSource)
        ? requestedSource
        : sourcePages[0];

    window.initialReviewCanonicalPageIndex = null;
    reapplyFocusedDraft(previousState);

    const workspace = document.createElement("div");
    workspace.className = "review-workspace-shell review-canvas-only review-canvas-fullbleed";
    workspace.dataset.pendingSourcePage = String(activeSource);
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
      if (window._reviewKeyDownHandler) {
        window.removeEventListener("keydown", window._reviewKeyDownHandler);
        window._reviewKeyDownHandler = null;
      }
    };

    window.setupWorkbenchPanels?.("review");
    window.mountStitchInspector?.();
  }

  window.renderReview = setupReviewWorkspace;
})();
