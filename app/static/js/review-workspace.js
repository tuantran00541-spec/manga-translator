(() => {
  if (typeof window.createReviewCard !== "function") return;

  let activeReviewIndex = 0;
  let reviewLastChapterId = null;
  const maskSnapshots = new Map();

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
  }

  function mountGeminiSettings() {
    const config = document.createElement("div");
    config.className = "gemini-qc-config ai-provider-config";

    const status = document.createElement("span");
    status.className = "gemini-qc-status";
    status.textContent = "Kiểm tra AI: Đang kiểm tra cấu hình…";
    const active = document.createElement("select");
    active.className = "ui-select ai-active-provider";
    active.setAttribute("aria-label", "Dịch vụ AI mặc định");
    const providers = ["gemini", "deepseek", "openai", "openrouter", "experiential"];
    const labels = { gemini: "Google Gemini", deepseek: "DeepSeek", openai: "OpenAI", openrouter: "OpenRouter", experiential: "Experiential Labs" };
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
      providerStatus.className = "gemini-qc-status";
      const key = document.createElement("input");
      key.type = "password";
      key.className = "gemini-key-input";
      key.placeholder = `${labels[id]} API key`;
      key.autocomplete = "off";
      const model = document.createElement("input");
      model.className = "ui-input ai-model-input";
      model.placeholder = "Tên model";
      model.value = localStorage.getItem(`manga_ai_model_${id}`) || "";
      model.addEventListener("change", () => localStorage.setItem(`manga_ai_model_${id}`, model.value.trim()));
      const listId = `ai-models-${id}`;
      model.setAttribute("list", listId);
      const datalist = document.createElement("datalist");
      datalist.id = listId;
      const save = document.createElement("button");
      save.type = "button"; save.className = "ui-btn ui-btn-primary"; save.textContent = "Lưu key";
      const clear = document.createElement("button");
      clear.type = "button"; clear.className = "ui-btn ui-btn-ghost"; clear.textContent = "Xóa key";
      const load = document.createElement("button");
      load.type = "button"; load.className = "ui-btn ui-btn-ghost"; load.textContent = "Tải model";
      save.addEventListener("click", async () => {
        if (!key.value.trim()) return window.showToast?.(`Nhập ${labels[id]} API key trước.`, "error");
        save.disabled = true;
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/key`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: key.value.trim() }) });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          key.value = "";
          window.showToast?.(`Đã lưu key ${labels[id]}.`, "success");
          await refresh();
        } catch (err) { window.showToast?.("Không thể lưu key: " + err.message, "error"); }
        finally { save.disabled = false; }
      });
      clear.addEventListener("click", async () => {
        clear.disabled = true;
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/key`, { method: "DELETE" });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          window.showToast?.(data.source === "environment" ? "Key đến từ biến môi trường; hãy xóa ở môi trường chạy." : `Đã xóa key ${labels[id]}.`, "info");
          await refresh();
        } catch (err) { window.showToast?.("Không thể xóa key: " + err.message, "error"); }
        finally { clear.disabled = false; }
      });
      load.addEventListener("click", async () => {
        load.disabled = true; load.textContent = "Đang tải…";
        try {
          const response = await fetch(`/api/visual_qc/providers/${id}/models`);
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          datalist.replaceChildren(...(data.models || []).map((name) => { const option = document.createElement("option"); option.value = name; return option; }));
          if (!model.value && data.models?.length) model.value = data.models[0];
          window.showToast?.(`Đã tải ${data.models?.length || 0} model từ ${labels[id]}.`, "success");
        } catch (err) { window.showToast?.("Không thể tải model: " + err.message, "error"); }
        finally { load.disabled = false; load.textContent = "Tải model"; }
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
      } catch (err) { status.textContent = "Kiểm tra AI: Lỗi cấu hình"; }
    }
    refresh();
    if (typeof window.mountAISettings === "function") {
      window.mountAISettings(config);
    }
    return status;
  }
  window.createAIProviderSettings = mountGeminiSettings;

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
    const container = document.getElementById("page-view");
    if (!container) return;

    if (window._reviewKeyDownHandler) {
      window.removeEventListener("keydown", window._reviewKeyDownHandler);
      window._reviewKeyDownHandler = null;
    }

    container.querySelectorAll(".review-card").forEach(captureMaskSnapshot);
    container.querySelectorAll(".brush-canvas").forEach((canvas) => {
      if (typeof canvas._cleanupBrush === "function") canvas._cleanupBrush();
      else if (canvas._brushAbort) canvas._brushAbort.abort();
    });
    container.replaceChildren();
    container.className = "review-mode";

    if (window.currentChapterId && reviewLastChapterId !== window.currentChapterId) {
      reviewLastChapterId = window.currentChapterId;
      activeReviewIndex = 0;
      maskSnapshots.clear();
    }

    const geminiStatus = mountGeminiSettings();
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
    bindAIStatus(geminiStatus, aiStatus);

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
    let busyObserver = null;

    const restoreMounted = () => {
      if (busyObserver) {
        busyObserver.disconnect();
        busyObserver = null;
      }
      if (!mountedCard) return;
      stopCardBrush(mountedCard);
      captureMaskSnapshot(mountedCard);
      cleanupCard(mountedCard);
      mountedCard = null;
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
          const aiBusy = Boolean(aiQcBtn?.disabled && /đang/i.test(aiQcBtn.textContent));
          const busy = Boolean(card._reviewBusy || aiBusy);
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
        } finally {
          isSyncing = false;
        }
      };
      card._syncReviewBusy = syncBusy;
      if (aiQcBtn) {
        busyObserver = new MutationObserver(syncBusy);
        busyObserver.observe(aiQcBtn, { childList: true, characterData: true, subtree: true });
      }
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
    if (typeof window.mountStitchInspector === "function") {
      window.mountStitchInspector();
    }
  }

  window.renderReview = setupReviewWorkspace;
})();
