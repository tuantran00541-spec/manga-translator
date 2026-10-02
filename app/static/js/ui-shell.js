(() => {
  const STAGES = ["landing", "preview", "review"];
  const STAGE_LABELS = {
    landing: "Trang chủ",
    preview: "Xem cắt lát",
    review: "Xử lý & Biên tập",
  };

  let trackedChapterId = null;
  let maxReachedIndex = 0;
  let navigationBusy = false;
  let pendingNavigation = null;
  let shellMounted = false;

  function inferredReachedIndex(activeStage) {
    const pages = window.currentManifest?.pages || [];
    const workflowStage = window.currentManifest?.workflow?.stage;
    let reached = Math.max(1, STAGES.indexOf(activeStage), STAGES.indexOf(workflowStage));
    if (pages.some((page) => page?.clean)) reached = Math.max(reached, STAGES.indexOf("review"));
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
      button.disabled = !available;
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
    if (start) start.textContent = resolved === "home" ? "Dự án mới" : "Tải chương";
    if ((document.body.dataset.appStage || "landing") !== "landing") return;
    syncSidebar("landing");
    closeSidebar();
    if (resolved === "import") queueMicrotask(() => document.getElementById("chapter-url")?.focus());
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
      setLandingMode(document.body.dataset.landingMode || "home");
    } else {
      setPageTitle(STAGE_LABELS[resolved]);
    }
    setAppContext(chapterId ? `Chương ${chapterId}` : "Chưa mở chương");
    syncSidebar(resolved);
    closeSidebar();
  }

  function currentCanonicalPageIndex() {
    const stage = document.body.dataset.appStage;
    if (stage === "review") {
      const reviewWorkspace = document.querySelector("#page-view .review-workspace-shell");
      const visiblePageIndex = Number(reviewWorkspace?.dataset.reviewCanonicalIndex);
      if (Number.isInteger(visiblePageIndex) && visiblePageIndex >= 0) return visiblePageIndex;
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
  }

  function showNavigationMessage(message, type = "info") {
    window.showToast?.(message, type);
  }

  async function leaveActiveWorkspace(targetStage) {
    const currentStage = document.body.dataset.appStage || "landing";
    if (currentStage === "review" && targetStage !== "review") window.cleanupReviewWorkspace?.();
    if (currentStage === "preview" && targetStage !== "preview") window.cleanupPreviewDrawListeners?.();
  }

  function syncChapterHash(chapterId) {
    if (!chapterId) return;
    const expected = `#${chapterId}`;
    if (window.location.hash === expected) return;
    try {
      window.history.replaceState(
        null,
        "",
        `${window.location.pathname}${window.location.search}${expected}`,
      );
    } catch (_) {}
  }

  async function navigateAppStage(stage, landingMode = "home") {
    if (!STAGES.includes(stage)) return false;
    if (navigationBusy) {
      pendingNavigation = { stage, landingMode };
      return true;
    }
    const currentStage = document.body.dataset.appStage || "landing";
    if (stage === "landing" && currentStage === "landing") {
      setLandingMode(landingMode);
      return true;
    }
    if (stage !== "landing" && (!window.currentChapterId || !window.currentManifest?.pages?.length)) {
      showNavigationMessage("Chưa có chương đang mở để chuyển tới màn hình này.");
      return false;
    }
    const chapterId = window.currentChapterId || null;
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
      const renderer = window[{ preview: "renderPreview", review: "renderReview" }[stage]];
      if (typeof renderer !== "function") throw new Error(`Không tìm thấy màn hình ${STAGE_LABELS[stage]}.`);
      syncChapterHash(chapterId);
      setAppStage(stage);
      renderer();
      return true;
    } catch (error) {
      showNavigationMessage(`Không thể chuyển màn hình: ${error?.message || String(error)}`, "error");
      return false;
    } finally {
      navigationBusy = false;
      syncSidebar(document.body.dataset.appStage || currentStage);
      const pending = pendingNavigation;
      pendingNavigation = null;
      if (pending) {
        queueMicrotask(() => navigateAppStage(pending.stage, pending.landingMode));
      }
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
    if (event.key === "Tab" && document.body.classList.contains("settings-open")) {
      const drawer = document.getElementById("settings-drawer");
      const focusable = [...drawer.querySelectorAll("button, input, select, textarea, a[href], [tabindex='0']")].filter((el) => !el.disabled && !el.closest("[hidden]") && el.getClientRects().length);
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      return;
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
  window.setLandingMode = setLandingMode;
  window.navigateAppStage = navigateAppStage;
  window.mountAISettings = mountAISettings;

  document.addEventListener("DOMContentLoaded", () => {
    setupShellEvents();
    window.createAIProviderSettings?.();
    if (!window.currentChapterId) setAppStage("landing");
  });
})();
