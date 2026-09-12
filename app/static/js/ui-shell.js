(() => {
  const STAGES = ["landing", "preview", "review", "editor"];
  const STAGE_LABELS = {
    landing: "Trang chủ",
    preview: "Xem cắt lát",
    review: "Xem inpaint",
    editor: "Biên tập",
  };
  const PANEL_QUERY = "(max-width: 1000px)";

  let trackedChapterId = null;
  let maxReachedIndex = 0;
  let navigationBusy = false;
  let activePanels = null;
  let focusModeActive = false;
  let shellMounted = false;
  const panelPreferences = new Map();

  function inferredReachedIndex(activeStage) {
    const pages = window.currentManifest?.pages || [];
    const workflowStage = window.currentManifest?.workflow?.stage;
    let reached = Math.max(1, STAGES.indexOf(activeStage), STAGES.indexOf(workflowStage));
    if (pages.some((page) => page?.clean)) reached = Math.max(reached, STAGES.indexOf("review"));
    if (pages.some((page) => page?.rendered || (page?.text_objects || []).length)) reached = Math.max(reached, STAGES.indexOf("editor"));
    return Math.min(STAGES.length - 1, reached);
  }

  function setAppContext(text) {
    const el = document.getElementById("app-context");
    if (el) el.textContent = text || "Chưa mở chương";
  }

  function setPageTitle(text) {
    const el = document.getElementById("app-page-title");
    if (el) el.textContent = text || "Manga Translator";
  }

  function closeSidebar() {
    document.body.classList.remove("sidebar-open");
    const toggle = document.getElementById("sidebar-toggle");
    const backdrop = document.getElementById("sidebar-backdrop");
    toggle?.setAttribute("aria-expanded", "false");
    if (backdrop) backdrop.hidden = true;
  }

  function openSidebar() {
    document.body.classList.add("sidebar-open");
    const toggle = document.getElementById("sidebar-toggle");
    const backdrop = document.getElementById("sidebar-backdrop");
    toggle?.setAttribute("aria-expanded", "true");
    if (backdrop) backdrop.hidden = false;
  }

  function syncSidebar(activeStage) {
    const mode = document.body.dataset.landingMode || "home";
    document.querySelectorAll(".sidebar-link[data-route]").forEach((button) => {
      const active = activeStage === "landing" && button.dataset.route === mode;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    document.querySelectorAll(".sidebar-link[data-stage]").forEach((button) => {
      const index = STAGES.indexOf(button.dataset.stage);
      const available = Boolean(window.currentChapterId) && index <= maxReachedIndex;
      const active = button.dataset.stage === activeStage;
      button.disabled = navigationBusy || !available;
      button.classList.toggle("active", active);
      button.classList.toggle("complete", available && index < STAGES.indexOf(activeStage));
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
      button.setAttribute("aria-disabled", String(button.disabled));
    });
  }

  function setLandingMode(mode = "home") {
    const resolved = mode === "import" ? "import" : "home";
    document.body.dataset.landingMode = resolved;
    const home = document.getElementById("home-view");
    const importView = document.getElementById("import-view");
    if (home) home.hidden = resolved !== "home";
    if (importView) importView.hidden = resolved !== "import";
    setPageTitle(resolved === "home" ? "Trang chủ" : "Nhập nội dung");
    const start = document.getElementById("start-action");
    if (start) start.textContent = resolved === "home" ? "Bắt đầu xử lý" : "Tải chương";
    if ((document.body.dataset.appStage || "landing") !== "landing") return;
    syncSidebar("landing");
    closeSidebar();
    if (resolved === "import") queueMicrotask(() => document.getElementById("chapter-url")?.focus());
  }

  function panelMode() {
    return window.matchMedia(PANEL_QUERY).matches ? "compact" : "wide";
  }

  function syncWorkbenchPanels() {
    const controls = document.getElementById("workbench-panel-controls");
    const pageView = document.getElementById("page-view");
    const visible = Boolean(activePanels?.grid?.isConnected && !pageView?.hidden);
    if (controls) controls.hidden = !visible;
    if (!activePanels?.grid?.isConnected) return;

    const { stage, grid, nav, inspector } = activePanels;
    const pageBtn = document.getElementById("toggle-page-panel");
    const inspectorBtn = document.getElementById("toggle-inspector-panel");
    const focusBtn = document.getElementById("toggle-focus-mode");
    if (focusBtn) {
      focusBtn.setAttribute("aria-pressed", String(focusModeActive));
      focusBtn.classList.toggle("ui-btn-primary", focusModeActive);
      focusBtn.classList.toggle("ui-btn-ghost", !focusModeActive);
    }

    if (focusModeActive) {
      nav.hidden = true;
      inspector.hidden = true;
      grid.dataset.navOpen = "false";
      grid.dataset.inspectorOpen = "false";
      grid.classList.add("focus-mode");
      pageBtn?.setAttribute("aria-expanded", "false");
      inspectorBtn?.setAttribute("aria-expanded", "false");
      return;
    }

    grid.classList.remove("focus-mode");
    const compact = panelMode() === "compact";
    const key = `${stage}:${panelMode()}`;
    const state = panelPreferences.get(key) || { nav: !compact, inspector: !compact };
    nav.hidden = !state.nav;
    inspector.hidden = !state.inspector;
    grid.dataset.navOpen = String(state.nav);
    grid.dataset.inspectorOpen = String(state.inspector);
    [[pageBtn, nav, state.nav], [inspectorBtn, inspector, state.inspector]].forEach(([button, panel, open]) => {
      if (!button) return;
      button.setAttribute("aria-controls", panel.id);
      button.setAttribute("aria-expanded", String(open));
    });
  }

  function setPanelOpen(name, open) {
    if (!activePanels?.grid?.isConnected) return;
    if (focusModeActive && open) {
      focusModeActive = false;
      document.body.classList.remove("focus-mode");
    }
    const compact = panelMode() === "compact";
    const key = `${activePanels.stage}:${panelMode()}`;
    const state = { ...(panelPreferences.get(key) || { nav: !compact, inspector: !compact }), [name]: open };
    if (compact && open) state[name === "nav" ? "inspector" : "nav"] = false;
    panelPreferences.set(key, state);
    syncWorkbenchPanels();
  }

  function toggleFocusMode() {
    focusModeActive = !focusModeActive;
    document.body.classList.toggle("focus-mode", focusModeActive);
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

  function setAppStage(stage) {
    const resolved = STAGES.includes(stage) ? stage : "landing";
    const chapterId = window.currentChapterId || null;
    if (chapterId !== trackedChapterId) {
      trackedChapterId = chapterId;
      maxReachedIndex = chapterId ? inferredReachedIndex(resolved) : 0;
    } else if (chapterId) {
      maxReachedIndex = Math.max(maxReachedIndex, inferredReachedIndex(resolved));
    } else {
      maxReachedIndex = 0;
    }

    document.body.dataset.appStage = resolved;
    const landing = document.getElementById("landing-view");
    const workspace = document.getElementById("page-view");
    const start = document.getElementById("start-action");
    if (landing) {
      landing.hidden = resolved !== "landing";
      landing.classList.toggle("app-stage-active", resolved === "landing");
    }
    if (workspace) workspace.hidden = resolved === "landing";
    if (start) {
      const isPreview = resolved === "preview";
      start.hidden = resolved !== "landing" && !isPreview;
      start.classList.toggle("preview-primary-action", isPreview);
      if (isPreview) start.textContent = "Bắt đầu xử lý";
    }
    if (resolved === "landing") {
      activePanels = null;
      setLandingMode(document.body.dataset.landingMode || "home");
    } else {
      setPageTitle(STAGE_LABELS[resolved]);
    }
    setAppContext(chapterId ? `Chương ${chapterId}` : "Chưa mở chương");
    syncSidebar(resolved);
    syncWorkbenchPanels();
    closeSidebar();
  }

  function currentCanonicalPageIndex() {
    const stage = document.body.dataset.appStage;
    if (stage === "editor" && window.editorState) return Math.max(0, parseInt(window.editorState.activePageIndex, 10) || 0);
    if (stage === "review") {
      const card = document.querySelector(".review-canvas-host .review-card");
      if (card) return Math.max(0, parseInt(card.dataset.pageIndex, 10) || 0);
    }
    if (stage === "preview") {
      const card = document.querySelector(".preview-card-active[data-page-index]");
      if (card) return Math.max(0, parseInt(card.dataset.pageIndex, 10) || 0);
    }
    return Math.max(0, parseInt(window.currentManifest?.workflow?.page_index, 10) || 0);
  }

  function prepareTargetPage(stage, pageIndex) {
    const lastIndex = Math.max(0, (window.currentManifest?.pages || []).length - 1);
    const canonicalIndex = Math.max(0, Math.min(parseInt(pageIndex, 10) || 0, lastIndex));
    if (stage === "preview") window.initialPreviewCanonicalPageIndex = canonicalIndex;
    else if (stage === "review") window.initialReviewCanonicalPageIndex = canonicalIndex;
    else if (stage === "editor" && window.editorState) window.editorState.activePageIndex = canonicalIndex;
  }

  function showNavigationMessage(message, type = "info") {
    window.showToast?.(message, type);
  }

  async function leaveActiveWorkspace(targetStage) {
    const currentStage = document.body.dataset.appStage || "landing";
    if (currentStage === "editor" && targetStage !== "editor" && typeof window.flushAllPendingPersists === "function") await window.flushAllPendingPersists();
    if (currentStage === "review" && targetStage !== "review") window.cleanupReviewWorkspace?.();
    if (currentStage === "preview" && targetStage !== "preview") window.cleanupPreviewDrawListeners?.();
    if (currentStage === "editor" && targetStage !== "editor" && typeof window._editorDrawCleanup === "function") {
      window._editorDrawCleanup();
      window._editorDrawCleanup = null;
    }
  }

  async function navigateAppStage(stage, landingMode = "home") {
    if (!STAGES.includes(stage) || navigationBusy) return false;
    const currentStage = document.body.dataset.appStage || "landing";
    if (stage === "landing" && currentStage === "landing") {
      setLandingMode(landingMode);
      return true;
    }
    if (stage !== "landing" && (!window.currentChapterId || !window.currentManifest?.pages?.length)) {
      showNavigationMessage("Chưa có chương đang mở để chuyển tới màn hình này.");
      return false;
    }
    const targetIndex = STAGES.indexOf(stage);
    if (targetIndex > maxReachedIndex) {
      showNavigationMessage("Hãy hoàn tất bước hiện tại trước khi mở màn hình này.");
      return false;
    }
    if (currentStage === "review" && document.querySelector(".review-workspace-shell.review-busy")) {
      showNavigationMessage("Đang xử lý kiểm tra chất lượng. Vui lòng chờ hoàn tất.");
      return false;
    }

    navigationBusy = true;
    syncSidebar(currentStage);
    try {
      const pageIndex = currentCanonicalPageIndex();
      await leaveActiveWorkspace(stage);
      prepareTargetPage(stage, pageIndex);
      if (stage === "landing") {
        try {
          const cleanUrl = `${window.location.pathname}${window.location.search}`;
          window.history.replaceState(null, "", cleanUrl);
        } catch (_) {}
        setLandingMode(landingMode);
        setAppStage("landing");
        return true;
      }
      const renderer = window[{ preview: "renderPreview", review: "renderReview", editor: "renderEditor" }[stage]];
      if (typeof renderer !== "function") throw new Error(`Không tìm thấy màn hình ${STAGE_LABELS[stage]}.`);
      renderer();
      return true;
    } catch (error) {
      showNavigationMessage(`Không thể chuyển màn hình: ${error?.message || String(error)}`, "error");
      return false;
    } finally {
      navigationBusy = false;
      syncSidebar(document.body.dataset.appStage || currentStage);
    }
  }

  function mountAISettings(configEl) {
    const host = document.getElementById("ai-settings-host");
    if (!host || !configEl) return;
    host.replaceChildren(configEl);
  }

  function openSettings() {
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    if (!drawer || !backdrop) return;
    closeSidebar();
    drawer.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add("settings-open");
    document.getElementById("app").inert = true;
    document.getElementById("settings-toggle")?.setAttribute("aria-expanded", "true");
    document.getElementById("settings-close")?.focus();
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
    document.getElementById("settings-toggle")?.focus();
  }

  function onShellKeydown(event) {
    if (event.key === "Escape" && document.body.classList.contains("settings-open")) {
      event.preventDefault();
      event.stopImmediatePropagation();
      closeSettings();
      return;
    }
    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
      event.preventDefault();
      closeSidebar();
      document.getElementById("sidebar-toggle")?.focus();
      return;
    }
    if (event.key === "Escape") {
      const pageOpen = document.getElementById("toggle-page-panel")?.getAttribute("aria-expanded") === "true";
      const inspectorOpen = document.getElementById("toggle-inspector-panel")?.getAttribute("aria-expanded") === "true";
      if (panelMode() === "compact" && activePanels && (pageOpen || inspectorOpen)) {
        event.preventDefault();
        setPanelOpen(pageOpen ? "nav" : "inspector", false);
      }
      return;
    }
    if (event.key === "Tab" && document.body.classList.contains("settings-open")) {
      const drawer = document.getElementById("settings-drawer");
      const focusable = [...drawer.querySelectorAll("button, input, select, textarea, a[href], [tabindex='0']")].filter((el) => !el.disabled && !el.closest("[hidden]") && el.getClientRects().length);
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      return;
    }
    const tag = event.target?.tagName?.toLowerCase();
    const editable = event.target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select";
    if ((event.key === "f" || event.key === "F") && !event.ctrlKey && !event.altKey && !event.metaKey && !editable && activePanels?.grid?.isConnected) {
      event.preventDefault();
      toggleFocusMode();
    } else if (event.key === "\\" && !editable && activePanels?.grid?.isConnected) {
      event.preventDefault();
      const open = document.getElementById("toggle-inspector-panel")?.getAttribute("aria-expanded") === "true";
      setPanelOpen("inspector", !open);
    }
  }

  function setupShellEvents() {
    if (shellMounted) return;
    shellMounted = true;
    document.getElementById("settings-toggle")?.addEventListener("click", openSettings);
    document.getElementById("settings-close")?.addEventListener("click", closeSettings);
    document.getElementById("settings-backdrop")?.addEventListener("click", closeSettings);
    document.getElementById("sidebar-toggle")?.addEventListener("click", () => document.body.classList.contains("sidebar-open") ? closeSidebar() : openSidebar());
    document.getElementById("sidebar-backdrop")?.addEventListener("click", closeSidebar);
    document.getElementById("toggle-focus-mode")?.addEventListener("click", toggleFocusMode);
    document.getElementById("toggle-page-panel")?.addEventListener("click", (event) => setPanelOpen("nav", event.currentTarget.getAttribute("aria-expanded") !== "true"));
    document.getElementById("toggle-inspector-panel")?.addEventListener("click", (event) => setPanelOpen("inspector", event.currentTarget.getAttribute("aria-expanded") !== "true"));
    window.matchMedia(PANEL_QUERY).addEventListener("change", syncWorkbenchPanels);
    document.getElementById("app-home")?.addEventListener("click", () => navigateAppStage("landing", "home"));
    document.querySelectorAll("[data-open-import]").forEach((button) => button.addEventListener("click", () => navigateAppStage("landing", "import")));
    document.querySelectorAll(".sidebar-link[data-route]").forEach((button) => button.addEventListener("click", () => navigateAppStage("landing", button.dataset.route)));
    document.querySelectorAll(".sidebar-link[data-stage]").forEach((button) => button.addEventListener("click", () => navigateAppStage(button.dataset.stage)));
    document.getElementById("start-action")?.addEventListener("click", () => {
      if ((document.body.dataset.appStage || "landing") === "preview") {
        if (typeof window.processSelectedPages === "function") window.processSelectedPages();
        return;
      }
      if ((document.body.dataset.landingMode || "home") === "home") {
        setLandingMode("import");
        document.getElementById("chapter-url")?.focus();
        return;
      }
      const url = document.getElementById("chapter-url")?.value?.trim();
      if (url && typeof window.loadChapter === "function") window.loadChapter();
      else {
        document.getElementById("chapter-url")?.focus();
        showNavigationMessage("Dán liên kết chương hoặc chọn tệp từ thiết bị.");
      }
    });
    document.addEventListener("keydown", onShellKeydown, true);
  }

  window.setAppStage = setAppStage;
  window.setAppContext = setAppContext;
  window.setLandingMode = setLandingMode;
  window.navigateAppStage = navigateAppStage;
  window.mountAISettings = mountAISettings;
  window.openAppSettings = openSettings;
  window.closeAppSettings = closeSettings;
  window.setupWorkbenchPanels = setupWorkbenchPanels;
  window.syncWorkbenchPanels = syncWorkbenchPanels;
  window.showWorkbenchInspector = () => setPanelOpen("inspector", true);
  window.toggleFocusMode = toggleFocusMode;

  document.addEventListener("DOMContentLoaded", () => {
    setupShellEvents();
    window.createAIProviderSettings?.();
    if (!window.currentChapterId) setAppStage("landing");
  });
})();
