(() => {
  const legacyRenderReview = window.renderReview;
  if (typeof legacyRenderReview !== "function" || typeof window.createReviewCard !== "function") return;

  window.REVIEW_VIRTUALIZED = true;

  let activeReviewIndex = 0;
  let reviewLastChapterId = null;
  const maskSnapshots = new Map();

  const CONTROL_COPY = new Map([
    ["Tô lỗi", "Đánh dấu vùng lỗi"],
    ["Đang tô (bấm để tắt)", "Đang đánh dấu · Chọn để kết thúc"],
    ["Xóa nét vẽ", "Xóa nét đánh dấu"],
    ["Xử lý lại vùng đã tô", "Xử lý vùng đánh dấu"],
    ["Đang xử lý lại...", "Đang xử lý…"],
    ["Xóa vùng tô tay", "Xóa vùng chỉnh sửa"],
    ["Đang xóa...", "Đang xóa…"],
    ["AI rà lỗi", "Kiểm tra bằng AI"],
    ["AI đang rà…", "AI đang kiểm tra…"],
  ]);

  function normalizeControlText(control) {
    if (!control) return;
    const next = CONTROL_COPY.get(control.textContent.trim());
    if (next && control.textContent !== next) control.textContent = next;
  }

  function observeControlText(control) {
    if (!control || control._uiCopyObserver) return;
    normalizeControlText(control);
    const observer = new MutationObserver(() => normalizeControlText(control));
    observer.observe(control, { childList: true, characterData: true, subtree: true });
    control._uiCopyObserver = observer;
  }

  function normalizeReviewControls(controls) {
    if (!controls) return;
    controls.querySelectorAll("button").forEach(observeControlText);
    const size = controls.querySelector(".brush-size-control");
    if (size && size.firstChild && size.firstChild.nodeType === Node.TEXT_NODE) {
      size.firstChild.nodeValue = "Kích thước cọ ";
    }
  }

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
    container.className = "review-mode";

    if (window.currentChapterId && reviewLastChapterId !== window.currentChapterId) {
      reviewLastChapterId = window.currentChapterId;
      activeReviewIndex = 0;
      maskSnapshots.clear();
    }

    const legacyToolbar = container.querySelector("#preview-toolbar");
    const geminiConfig = legacyToolbar?.querySelector(".gemini-qc-config") || null;
    const geminiStatus = geminiConfig?.querySelector(".gemini-qc-status") || null;
    if (geminiConfig && typeof window.mountAISettings === "function") {
      window.mountAISettings(geminiConfig);
    }

    let continueBtn = null;
    if (legacyToolbar) {
      continueBtn = [...legacyToolbar.children].find((el) => el.tagName === "BUTTON") || null;
    }

    const pageIndices = (window.currentManifest?.pages || [])
      .map((page, index) => ({ page, index }))
      .filter(({ page }) => !page.skipped)
      .map(({ index }) => index);

    if (!pageIndices.length) {
      legacyToolbar?.remove();
      return;
    }

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
    legacyToolbar?.remove();

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

    if (!continueBtn) {
      continueBtn = document.createElement("button");
      continueBtn.addEventListener("click", () => {
        const activeCard = container.querySelector(".review-canvas-host .review-card");
        const canonicalIndex = activeCard ? (parseInt(activeCard.dataset.pageIndex, 10) || 0) : 0;
        if (window.editorState) window.editorState.activePageIndex = canonicalIndex;
        if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("editor", canonicalIndex);
        if (typeof window.renderEditor === "function") window.renderEditor();
      });
    }
    continueBtn.className = "ui-btn ui-btn-primary review-primary-action";
    continueBtn.textContent = "Mở trình biên tập bản dịch";
    actions.append(aiStatus, help, continueBtn);
    toolbar.append(title, controlsSlot, actions);

    const nav = document.createElement("nav");
    nav.className = "review-page-nav workspace-nav-bar";
    nav.setAttribute("aria-label", "Điều hướng trang kiểm tra chất lượng");
    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "review-nav-btn workspace-nav-btn";
    prev.textContent = "← Trang trước";
    prev.setAttribute("aria-label", "Trang trước");
    const position = document.createElement("div");
    position.className = "review-position workspace-nav-position";
    position.setAttribute("aria-live", "polite");
    const next = document.createElement("button");
    next.type = "button";
    next.className = "review-nav-btn workspace-nav-btn";
    next.textContent = "Trang sau →";
    next.setAttribute("aria-label", "Trang sau");
    nav.append(prev, position, next);

    const canvasHost = document.createElement("div");
    canvasHost.className = "review-canvas-host";
    workspace.append(toolbar, nav, canvasHost);
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
      if (controls) {
        normalizeReviewControls(controls);
        controlsSlot.appendChild(controls);
      }

      position.replaceChildren();
      const jumpWrap = document.createElement("label");
      jumpWrap.className = "workspace-nav-jump-wrap";
      jumpWrap.textContent = "Trang ";
      const jumpInput = document.createElement("input");
      jumpInput.type = "number";
      jumpInput.min = "1";
      jumpInput.max = String(pageIndices.length);
      jumpInput.value = String(activeReviewIndex + 1);
      jumpInput.className = "workspace-nav-jump-input";
      jumpInput.setAttribute("aria-label", "Nhảy đến trang");
      const doJump = () => {
        const targetIndex = parsePageNumber(jumpInput.value, pageIndices.length);
        if (targetIndex === null) {
          jumpInput.value = String(activeReviewIndex + 1);
          return;
        }
        if (targetIndex !== activeReviewIndex) {
          activeReviewIndex = targetIndex;
          renderActive();
        }
      };
      jumpInput.addEventListener("change", doJump);
      jumpInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          doJump();
        }
      });
      jumpWrap.appendChild(jumpInput);
      const totalText = document.createElement("span");
      totalText.textContent = ` / ${pageIndices.length}`;
      const labelSpan = document.createElement("span");
      labelSpan.textContent = ` · ${pageLabel(window.currentManifest.pages, canonicalIndex)}`;
      position.append(jumpWrap, totalText, labelSpan);

      prev.disabled = activeReviewIndex === 0;
      next.disabled = activeReviewIndex === pageIndices.length - 1;
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
        prev.disabled = busy || activeReviewIndex === 0;
        next.disabled = busy || activeReviewIndex === pageIndices.length - 1;
        jumpInput.disabled = busy;
        continueBtn.disabled = busy;
      };
      if (aiQcBtn) {
        busyObserver = new MutationObserver(syncBusy);
        busyObserver.observe(aiQcBtn, { attributes: true, childList: true, characterData: true, subtree: true });
      }
      syncBusy();
    };

    prev.addEventListener("click", () => {
      if (workspace.classList.contains("review-busy") || activeReviewIndex <= 0) return;
      activeReviewIndex -= 1;
      renderActive();
    });
    next.addEventListener("click", () => {
      if (workspace.classList.contains("review-busy") || activeReviewIndex >= pageIndices.length - 1) return;
      activeReviewIndex += 1;
      renderActive();
    });

    renderActive();
  }

  window.renderReview = function reviewWorkspaceRender() {
    legacyRenderReview();
    setupReviewWorkspace();
  };
})();
