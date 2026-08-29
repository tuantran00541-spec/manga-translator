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
    saveBtn.className = "gemini-key-save-btn";
    saveBtn.textContent = "Lưu khóa API";

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "gemini-key-clear-btn";
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
    title.innerHTML = '<span class="ui-eyebrow">Kiểm tra chất lượng</span><strong>Hiệu chỉnh ảnh đã xử lý</strong><span>Đánh dấu vùng còn lỗi và xử lý lại khi cần.</span>';
    const controlsSlot = document.createElement("div");
    controlsSlot.className = "review-controls-slot";
    const actions = document.createElement("div");
    actions.className = "review-actions-group";
    const aiStatus = document.createElement("span");
    aiStatus.className = "review-ai-status";
    bindAIStatus(geminiStatus, aiStatus);

    const help = document.createElement("details");
    help.className = "review-help";
    const helpSummary = document.createElement("summary");
    helpSummary.className = "ui-btn ui-btn-ghost";
    helpSummary.textContent = "Hướng dẫn";
    const helpPanel = document.createElement("div");
    helpPanel.className = "review-help-panel";
    helpPanel.innerHTML = '<p>Chọn <strong>Đánh dấu vùng lỗi</strong> để tô thủ công. Nhấp đúp vào vùng nền đồng màu để chọn nhanh toàn bộ vùng liên thông. Sau khi kiểm tra vùng đánh dấu, chọn <strong>Xử lý vùng đánh dấu</strong>.</p>';
    help.append(helpSummary, helpPanel);

    const continueBtn = document.createElement("button");
    continueBtn.className = "ui-btn ui-btn-primary review-primary-action";
    continueBtn.textContent = "Mở trình biên tập bản dịch";
    continueBtn.addEventListener("click", () => {
      const activeCard = container.querySelector(".review-canvas-host .review-card");
      const canonicalIndex = activeCard ? (parseInt(activeCard.dataset.pageIndex, 10) || 0) : 0;
      if (window.editorState) window.editorState.activePageIndex = canonicalIndex;
      if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("editor", canonicalIndex);
      if (typeof window.renderEditor === "function") window.renderEditor();
    });
    actions.append(aiStatus, continueBtn);
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
    inspectorHeading.innerHTML = '<span class="ui-eyebrow">Trang đang chọn</span><strong>Hiệu chỉnh & kiểm tra</strong>';
    const gestureSection = document.createElement("section");
    gestureSection.className = "inspector-section review-gesture-section";
    const gestureTitle = document.createElement("h3");
    gestureTitle.textContent = "Công cụ chỉnh sửa";
    gestureSection.append(gestureTitle, controlsSlot);
    const helpSection = document.createElement("section");
    helpSection.className = "inspector-section review-help-section";
    helpSection.appendChild(help);
    inspector.append(inspectorHeading, gestureSection, helpSection);

    navigator = window.createPageNavigator({
      items: navItems,
      activeIndex: activeReviewIndex,
      title: "Trang kiểm tra",
      ariaLabel: "Điều hướng trang kiểm tra chất lượng",
      onSelect: (index) => {
        if (workspace.classList.contains("review-busy")) return;
        activeReviewIndex = index;
        navigator.setActive(index);
        renderActive();
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
      canvasHost.replaceChildren(card);
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
      const brushSize = controls?.querySelector(".brush-size-slider") || null;
      const aiQcBtn = controls?.querySelector(".ai-qc-btn") || null;
      const syncBusy = () => {
        const busy = Boolean(aiQcBtn?.disabled && /đang/i.test(aiQcBtn.textContent));
        workspace.classList.toggle("review-busy", busy);
        if (busy && canvas && typeof canvas._stopBrush === "function") canvas._stopBrush();
        if (brushBtn) brushBtn.disabled = busy;
        if (clearBtn) clearBtn.disabled = busy;
        if (brushSize) brushSize.disabled = busy;
        navigator.setBusy(busy);
        continueBtn.disabled = busy;
      };
      if (aiQcBtn) {
        busyObserver = new MutationObserver(syncBusy);
        busyObserver.observe(aiQcBtn, { attributes: true, childList: true, characterData: true, subtree: true });
      }
      syncBusy();
    };

    renderActive();
  }

  window.renderReview = setupReviewWorkspace;
})();
