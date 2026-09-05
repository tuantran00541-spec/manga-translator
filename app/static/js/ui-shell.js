(() => {
  const STAGES = ["landing", "preview", "review", "editor"];
  const STAGE_LABELS = {
    landing: "Nhập nội dung",
    preview: "Xử lý ảnh",
    review: "Kiểm tra chất lượng",
    editor: "Biên tập bản dịch",
  };

  let trackedChapterId = null;
  let maxReachedIndex = 0;
  let navigationBusy = false;
  const panelPreferences = new Map();
  let activePanels = null;

  function panelMode() {
    return window.matchMedia("(max-width: 1000px)").matches ? "compact" : "wide";
  }

  function syncWorkbenchPanels() {
    const controls = document.getElementById("workbench-panel-controls");
    const pageView = document.getElementById("page-view");
    const visible = activePanels?.grid.isConnected && !pageView?.hidden
      && !pageView?.classList.contains("review-show-stitched");
    if (controls) controls.hidden = !visible;
    if (!activePanels?.grid.isConnected) return;
    const { stage, grid, nav, inspector } = activePanels;
    const compact = panelMode() === "compact";
    const key = `${stage}:${panelMode()}`;
    const state = panelPreferences.get(key) || { nav: !compact, inspector: !compact };
    nav.hidden = !state.nav;
    inspector.hidden = !state.inspector;
    grid.dataset.navOpen = String(state.nav);
    grid.dataset.inspectorOpen = String(state.inspector);
    [["toggle-page-panel", nav, state.nav], ["toggle-inspector-panel", inspector, state.inspector]].forEach(([id, panel, open]) => {
      const button = document.getElementById(id);
      if (!button) return;
      button.setAttribute("aria-controls", panel.id);
      button.setAttribute("aria-expanded", String(open));
      button.title = `${open ? "Ẩn" : "Hiện"} ${id === "toggle-page-panel" ? "danh sách trang" : "bảng công cụ"}`;
    });
  }

  function setPanelOpen(name, open) {
    if (!activePanels?.grid.isConnected) return;
    const compact = panelMode() === "compact";
    const key = `${activePanels.stage}:${panelMode()}`;
    const state = { ...(panelPreferences.get(key) || { nav: !compact, inspector: !compact }), [name]: open };
    // A narrow screen shows one auxiliary panel above the image at a time.
    if (compact && open) state[name === "nav" ? "inspector" : "nav"] = false;
    panelPreferences.set(key, state);
    syncWorkbenchPanels();
  }

  function setupWorkbenchPanels(stage) {
    const grid = document.querySelector("#page-view .workbench-stage-grid");
    const nav = grid?.querySelector(":scope > .page-navigator");
    const inspector = grid?.querySelector(":scope > .context-inspector");
    if (!grid || !nav || !inspector) {
      activePanels = null;
      syncWorkbenchPanels();
      return;
    }
    nav.id = `${stage}-page-panel`;
    inspector.id = `${stage}-inspector-panel`;
    activePanels = { stage, grid, nav, inspector };
    syncWorkbenchPanels();
  }

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
    syncWorkbenchPanels();

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
    } else if (stage === "editor" && window.editorState) {
      window.editorState.activePageIndex = canonicalIndex;
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
        editor: "renderEditor",
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
    document.getElementById("app").inert = true;
    document.getElementById("settings-toggle")?.setAttribute("aria-expanded", "true");
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
    document.getElementById("app").inert = false;
    document.getElementById("settings-toggle")?.setAttribute("aria-expanded", "false");
    const toggle = document.getElementById("settings-toggle");
    if (toggle) toggle.focus();
  }

  function wrapRenderer(name, stage) {
    const original = window[name];
    if (typeof original !== "function" || original._uiSystemWrapped) return;
    const wrapped = function uiSystemStageRenderer(...args) {
      setAppStage(stage);
      const result = original.apply(this, args);
      setupWorkbenchPanels(stage);
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
    [["toggle-page-panel", "nav"], ["toggle-inspector-panel", "inspector"]].forEach(([id, name]) => {
      document.getElementById(id)?.addEventListener("click", (event) => {
        setPanelOpen(name, event.currentTarget.getAttribute("aria-expanded") !== "true");
      });
    });
    window.matchMedia("(max-width: 1000px)").addEventListener("change", syncWorkbenchPanels);

    // Native disclosures keep configuration out of the command bar until needed.
    document.addEventListener("click", (event) => {
      document.querySelectorAll(".command-disclosure[open]").forEach((details) => {
        if (!details.contains(event.target)) details.open = false;
      });
    });

    document.querySelectorAll(".app-rail-item[data-stage]").forEach((step) => {
      step.addEventListener("click", () => navigateAppStage(step.dataset.stage));
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("settings-open")) {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeSettings();
      } else if (event.key === "Escape") {
        const details = event.target.closest?.(".command-disclosure[open]");
        if (details) {
          event.preventDefault();
          event.stopImmediatePropagation();
          details.open = false;
          details.querySelector("summary")?.focus();
        }
      } else if (event.key === "Tab" && document.body.classList.contains("settings-open")) {
        const drawer = document.getElementById("settings-drawer");
        const focusable = [...drawer.querySelectorAll("button, input, select, textarea, a[href], [tabindex='0']")]
          .filter((el) => !el.disabled && !el.closest("[hidden]") && el.getClientRects().length);
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    }, true);
  }

  window.setAppStage = setAppStage;
  window.setAppContext = setAppContext;
  window.navigateAppStage = navigateAppStage;
  window.mountAISettings = mountAISettings;
  window.openAppSettings = openSettings;
  window.closeAppSettings = closeSettings;
  window.setupWorkbenchPanels = setupWorkbenchPanels;
  window.syncWorkbenchPanels = syncWorkbenchPanels;
  window.showWorkbenchInspector = () => setPanelOpen("inspector", true);

  wrapRenderer("renderPreview", "preview");
  wrapRenderer("renderReview", "review");
  wrapRenderer("renderEditor", "editor");

  document.addEventListener("DOMContentLoaded", () => {
    setupShellEvents();
    if (!window.currentChapterId) setAppStage("landing");
  });
})();
