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

  function bindDeepSeekControls(config) {
    const heading = document.createElement("strong");
    heading.className = "gemini-qc-privacy-note deepseek-qc-heading";
    heading.textContent = "DeepSeek Vision · kiểm tra toàn chương";

    const status = document.createElement("span");
    status.className = "gemini-qc-status deepseek-qc-status";
    status.textContent = "DeepSeek: Đang kiểm tra cấu hình…";

    const input = document.createElement("input");
    input.type = "password";
    input.className = "gemini-key-input deepseek-key-input";
    input.placeholder = "DeepSeek API key";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.setAttribute("aria-label", "DeepSeek API key");

    const save = document.createElement("button");
    save.type = "button";
    save.className = "ui-btn ui-btn-primary gemini-key-save-btn deepseek-key-save-btn";
    save.textContent = "Lưu khóa DeepSeek";

    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "ui-btn ui-btn-ghost gemini-key-clear-btn deepseek-key-clear-btn";
    clear.textContent = "Xóa khóa DeepSeek";

    const privacy = document.createElement("span");
    privacy.className = "gemini-qc-privacy-note deepseek-qc-privacy-note";
    privacy.textContent = "Chỉ khi chọn DeepSeek cho kiểm tra toàn chương, contact sheet sẽ được gửi đến DeepSeek.";

    config.append(heading, status, input, save, clear, privacy);

    async function refresh() {
      try {
        const resp = await fetch("/api/visual_qc/settings");
        const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
        const settings = await parse(resp);
        const provider = settings?.providers?.deepseek || {};
        window.deepseekVisualQCConfigured = Boolean(provider.configured);
        if (provider?.configured) {
          status.textContent = `DeepSeek: Sẵn sàng · ${provider.model || "Vision Exp"}`;
          status.classList.add("configured");
        } else if (provider?.source === "unavailable") {
          status.textContent = "DeepSeek: Kho bí mật chưa sẵn sàng";
          status.classList.remove("configured");
        } else {
          status.textContent = "DeepSeek: Chưa cấu hình";
          status.classList.remove("configured");
        }
        clear.disabled = !provider.configured || provider.source === "environment";
      } catch (err) {
        window.deepseekVisualQCConfigured = false;
        status.textContent = "DeepSeek: Lỗi cấu hình";
        status.classList.remove("configured");
        console.warn("DeepSeek QC settings check failed:", err);
      }
    }

    save.addEventListener("click", async () => {
      const apiKey = input.value.trim();
      if (!apiKey) {
        window.showToast?.("Nhập DeepSeek API key trước khi lưu.", "error");
        return;
      }
      save.disabled = true;
      save.textContent = "Đang lưu…";
      try {
        const resp = await fetch("/api/visual_qc/deepseek/key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: apiKey }),
        });
        const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
        const data = await parse(resp);
        if (!resp.ok) {
          const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d?.detail || `HTTP ${s}`;
          throw new Error(getErr(resp.status, data));
        }
        input.value = "";
        window.showToast?.("Đã lưu DeepSeek API key trong kho bí mật của hệ điều hành.", "success");
      } catch (err) {
        window.showToast?.("Không thể lưu DeepSeek API key: " + err.message, "error");
      } finally {
        save.disabled = false;
        save.textContent = "Lưu khóa DeepSeek";
        await refresh();
      }
    });

    clear.addEventListener("click", async () => {
      clear.disabled = true;
      try {
        const resp = await fetch("/api/visual_qc/deepseek/key", { method: "DELETE" });
        const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
        const data = await parse(resp);
        const message = data?.source === "environment"
          ? "DeepSeek API key đang đến từ biến môi trường; hãy xóa tại môi trường chạy ứng dụng."
          : "Đã xóa DeepSeek API key khỏi kho bí mật.";
        window.showToast?.(message, data?.source === "environment" ? "info" : "success");
      } catch (err) {
        window.showToast?.("Không thể xóa DeepSeek API key: " + err.message, "error");
      } finally {
        await refresh();
      }
    });

    refresh();
  }

  function mountGeminiSettings() {
    const config = document.createElement("div");
    config.className = "gemini-qc-config";

    const status = document.createElement("span");
    status.className = "gemini-qc-status";
    status.textContent = "Kiểm tra AI: Đang kiểm tra cấu hình…";

    const keyInput = document.createElement("input");
    keyInput.type = "password";
    keyInput.className = "gemini-key-input";
    keyInput.placeholder = "Gemini API key";
    keyInput.autocomplete = "off";
    keyInput.spellcheck = false;
    keyInput.setAttribute("aria-label", "Gemini API key");

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "ui-btn ui-btn-primary gemini-key-save-btn";
    saveBtn.textContent = "Lưu khóa API";

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "ui-btn ui-btn-ghost gemini-key-clear-btn";
    clearBtn.textContent = "Xóa khóa API";

    const privacyNote = document.createElement("span");
    privacyNote.className = "gemini-qc-privacy-note";
    privacyNote.textContent = "Ảnh gốc và ảnh đã xử lý sẽ được gửi đến Gemini để kiểm tra chất lượng.";

    config.append(status, keyInput, saveBtn, clearBtn, privacyNote);
    if (typeof window.setupGeminiQCSettings === "function") {
      window.setupGeminiQCSettings(status, keyInput, saveBtn, clearBtn);
    } else if (typeof setupGeminiQCSettings === "function") {
      setupGeminiQCSettings(status, keyInput, saveBtn, clearBtn);
    }
    bindDeepSeekControls(config);
    if (typeof window.mountAISettings === "function") {
      window.mountAISettings(config);
    }
    return status;
  }

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
