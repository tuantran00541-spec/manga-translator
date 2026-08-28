(() => {
  const STAGES = ["landing", "preview", "review", "script", "editor", "final_qc"];
  const STAGE_LABELS = {
    landing: "Nhập nội dung",
    preview: "Xử lý ảnh",
    review: "Kiểm tra cleaning",
    script: "Dịch & soát Script",
    editor: "Typeset",
    final_qc: "Final QC",
  };

  let trackedChapterId = null;
  let maxReachedIndex = 0;
  let navigationBusy = false;

  function setAppContext(text) {
    const el = document.getElementById("app-context");
    if (el) el.textContent = text || "Chưa mở chương";
  }

  function syncWorkflowSteps(activeIndex) {
    document.querySelectorAll(".app-rail-item[data-stage]").forEach((step) => {
      const idx = STAGES.indexOf(step.dataset.stage);
      const available = idx >= 0 && idx <= maxReachedIndex;
      step.classList.toggle("active", idx === activeIndex);
      step.classList.toggle("complete", idx >= 0 && idx < activeIndex);
      step.classList.toggle("available", available && idx !== activeIndex);
      step.disabled = navigationBusy || !available;
      step.setAttribute("aria-disabled", step.disabled ? "true" : "false");
      if (idx === activeIndex) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
  }

  function setAppStage(stage) {
    const resolved = STAGES.includes(stage) ? stage : "landing";
    const chapterId = window.currentChapterId || null;
    const activeIndex = STAGES.indexOf(resolved);

    if (chapterId !== trackedChapterId) {
      trackedChapterId = chapterId;
      maxReachedIndex = chapterId ? Math.max(1, activeIndex) : 0;
    } else if (chapterId) {
      maxReachedIndex = Math.max(maxReachedIndex, activeIndex);
    } else {
      maxReachedIndex = 0;
    }

    document.body.dataset.appStage = resolved;
    const stageTitle = document.getElementById("stage-title");
    if (stageTitle) stageTitle.textContent = STAGE_LABELS[resolved];

    const landing = document.getElementById("landing-view");
    const workspace = document.getElementById("page-view");
    if (landing) {
      landing.hidden = resolved !== "landing";
      landing.classList.toggle("app-stage-active", resolved === "landing");
    }
    if (workspace) workspace.hidden = resolved === "landing";

    syncWorkflowSteps(activeIndex);

    if (chapterId) {
      setAppContext(`Chương ${chapterId}`);
    } else if (resolved === "landing") {
      setAppContext("Chưa mở chương");
    } else {
      setAppContext(STAGE_LABELS[resolved]);
    }
  }

  function currentCanonicalPageIndex() {
    const stage = document.body.dataset.appStage;
    if (stage === "editor" && window.editorState) {
      return Math.max(0, parseInt(window.editorState.activePageIndex, 10) || 0);
    }

    if (stage === "review") {
      const activeCard = document.querySelector(".review-canvas-host .review-card");
      if (activeCard) return Math.max(0, parseInt(activeCard.dataset.pageIndex, 10) || 0);
    }

    if (stage === "preview") {
      const activeCard = document.querySelector(".preview-card-active[data-page-index]");
      if (activeCard) return Math.max(0, parseInt(activeCard.dataset.pageIndex, 10) || 0);
    }

    const workflowIndex = window.currentManifest?.workflow?.page_index;
    return Math.max(0, parseInt(workflowIndex, 10) || 0);
  }

  function prepareTargetPage(stage, pageIndex) {
    const pages = window.currentManifest?.pages || [];
    const lastIndex = Math.max(0, pages.length - 1);
    const canonicalIndex = Math.max(0, Math.min(parseInt(pageIndex, 10) || 0, lastIndex));

    if (stage === "preview") {
      window.initialPreviewCanonicalPageIndex = canonicalIndex;
    } else if (stage === "review") {
      window.initialReviewCanonicalPageIndex = canonicalIndex;
    } else if (stage === "script") {
      window.initialScriptCanonicalPageIndex = canonicalIndex;
    } else if (stage === "editor" && window.editorState) {
      window.editorState.activePageIndex = canonicalIndex;
    } else if (stage === "final_qc") {
      window.initialFinalQCCanonicalPageIndex = canonicalIndex;
    }
  }

  function showNavigationMessage(message, type = "info") {
    if (typeof window.showToast === "function") window.showToast(message, type);
  }

  async function navigateAppStage(stage) {
    if (!STAGES.includes(stage) || navigationBusy) return false;

    const targetIndex = STAGES.indexOf(stage);
    const currentStage = document.body.dataset.appStage || "landing";
    const currentIndex = STAGES.indexOf(currentStage);
    if (targetIndex === currentIndex) return true;

    if (stage !== "landing" && (!window.currentChapterId || !window.currentManifest?.pages?.length)) {
      showNavigationMessage("Chưa có chương đang mở để chuyển tới bước này.");
      return false;
    }

    if (targetIndex > maxReachedIndex) {
      showNavigationMessage("Hãy hoàn tất bước hiện tại trước khi chuyển sang bước tiếp theo.");
      return false;
    }

    if (currentStage === "review" && document.querySelector(".review-workspace-shell.review-busy")) {
      showNavigationMessage("Đang xử lý kiểm tra chất lượng. Vui lòng chờ thao tác hiện tại hoàn tất.");
      return false;
    }

    navigationBusy = true;
    syncWorkflowSteps(currentIndex);

    try {
      if (currentStage === "editor" && stage !== "editor" && typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      }

      const pageIndex = currentCanonicalPageIndex();
      prepareTargetPage(stage, pageIndex);

      if (stage === "landing") {
        if (typeof window.cleanupPreviewDrawListeners === "function") window.cleanupPreviewDrawListeners();
        if (typeof window._editorDrawCleanup === "function") {
          window._editorDrawCleanup();
          window._editorDrawCleanup = null;
        }
        setAppStage("landing");
        return true;
      }

      const rendererName = {
        preview: "renderPreview",
        review: "renderReview",
        script: "renderScript",
        editor: "renderEditor",
        final_qc: "renderFinalQC",
      }[stage];
      const renderer = window[rendererName];
      if (typeof renderer !== "function") {
        throw new Error(`Không tìm thấy trình hiển thị cho bước ${STAGE_LABELS[stage]}.`);
      }
      renderer();
      return true;
    } catch (err) {
      showNavigationMessage("Không thể chuyển bước: " + (err?.message || String(err)), "error");
      return false;
    } finally {
      navigationBusy = false;
      const activeStage = STAGES.indexOf(document.body.dataset.appStage || currentStage);
      syncWorkflowSteps(activeStage);
    }
  }

  function mountAISettings(configEl) {
    const host = document.getElementById("ai-settings-host");
    if (!host || !configEl) return;
    host.innerHTML = "";
    host.appendChild(configEl);
  }

  function openSettings() {
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    if (!drawer || !backdrop) return;
    drawer.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add("settings-open");
    const closeBtn = document.getElementById("settings-close");
    if (closeBtn) closeBtn.focus();
  }

  function closeSettings() {
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    if (!drawer || !backdrop) return;
    drawer.hidden = true;
    backdrop.hidden = true;
    document.body.classList.remove("settings-open");
    const toggle = document.getElementById("settings-toggle");
    if (toggle) toggle.focus();
  }

  function wrapRenderer(name, stage) {
    const original = window[name];
    if (typeof original !== "function" || original._uiSystemWrapped) return;
    const wrapped = function uiSystemStageRenderer(...args) {
      setAppStage(stage);
      const result = original.apply(this, args);
      if (window.currentChapterId) setAppContext(`Chương ${window.currentChapterId}`);
      return result;
    };
    wrapped._uiSystemWrapped = true;
    wrapped._uiSystemOriginal = original;
    window[name] = wrapped;
  }

  function setupShellEvents() {
    const toggle = document.getElementById("settings-toggle");
    const close = document.getElementById("settings-close");
    const backdrop = document.getElementById("settings-backdrop");
    if (toggle) toggle.addEventListener("click", openSettings);
    if (close) close.addEventListener("click", closeSettings);
    if (backdrop) backdrop.addEventListener("click", closeSettings);

    document.querySelectorAll(".app-rail-item[data-stage]").forEach((step) => {
      step.addEventListener("click", () => navigateAppStage(step.dataset.stage));
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("settings-open")) {
        closeSettings();
      }
    });
  }

  window.setAppStage = setAppStage;
  window.setAppContext = setAppContext;
  window.navigateAppStage = navigateAppStage;
  window.mountAISettings = mountAISettings;
  window.openAppSettings = openSettings;
  window.closeAppSettings = closeSettings;

  wrapRenderer("renderPreview", "preview");
  wrapRenderer("renderReview", "review");
  wrapRenderer("renderScript", "script");
  wrapRenderer("renderEditor", "editor");
  wrapRenderer("renderFinalQC", "final_qc");

  document.addEventListener("DOMContentLoaded", () => {
    setupShellEvents();
    if (!window.currentChapterId) setAppStage("landing");
  });
})();
