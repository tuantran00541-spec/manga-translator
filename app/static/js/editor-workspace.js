// editor-workspace.js - UI-04 focused translation workspace
(() => {
  const legacyRenderEditor = window.renderEditor;
  if (typeof legacyRenderEditor !== "function") return;

  let activeIndex = 0;
  let activeLastChapterId = null;

  function setupTranslationWorkspace() {
    const container = document.getElementById("page-view");
    if (!container) return;

    if (window.currentChapterId && activeLastChapterId !== window.currentChapterId) {
      activeLastChapterId = window.currentChapterId;
      activeIndex = 0;
    }

    const rawWrappers = [...container.querySelectorAll(".page-block-wrapper")];
    if (!rawWrappers.length) return;

    const visibleCount = rawWrappers.length;
    activeIndex = Math.max(0, Math.min(activeIndex, visibleCount - 1));

    // Keep only active page DOM in memory, destroy inactive page wrappers
    const activeWrapper = rawWrappers[activeIndex];
    rawWrappers.forEach((el, idx) => {
      if (idx !== activeIndex) {
        el.remove();
      }
    });

    const shell = document.createElement("div");
    shell.className = "translation-workspace";

    const toolbar = document.createElement("div");
    toolbar.className = "translation-sticky-toolbar";

    const title = document.createElement("div");
    title.className = "translation-toolbar-title";
    title.innerHTML = '<strong>Biên tập bản dịch</strong><span>Kéo chữ trên ảnh sang vị trí phù hợp — công cụ tinh chỉnh sẽ đặt ở panel thuộc tính.</span>';

    const position = document.createElement("span");
    position.className = "translation-position";

    toolbar.append(title, position);

    const workspace = document.createElement("div");
    workspace.className = "translation-workspace-body";

    const canvasHost = document.createElement("main");
    canvasHost.className = "translation-canvas-host";

    const panelHost = document.createElement("aside");
    panelHost.className = "translation-panel-host";
    panelHost.setAttribute("aria-label", "Bảng biên tập bản dịch");

    workspace.append(canvasHost, panelHost);

    const nav = document.createElement("nav");
    nav.className = "translation-page-nav";
    const prev = document.createElement("button");
    prev.className = "translation-nav-btn";
    prev.textContent = "← Trước";
    const next = document.createElement("button");
    next.className = "translation-nav-btn";
    next.textContent = "Sau →";
    nav.append(prev, next);

    shell.append(toolbar, workspace, nav);
    container.innerHTML = "";
    container.appendChild(shell);

    // Mount active wrapper into workspace DOM
    activeWrapper.style.display = "block";
    canvasHost.appendChild(activeWrapper);

    const block = activeWrapper.querySelector(".page-block");
    const panel = activeWrapper.querySelector(".box-panel");
    if (block) block.classList.add("translation-active-page");
    if (panel) panelHost.appendChild(panel);

    const page = currentManifest?.pages?.[Number(block?.dataset.pageIndex)];
    const label = activeWrapper.querySelector(".page-block-label");
    position.textContent = `${activeIndex + 1} / ${visibleCount}${label ? " · " + label.textContent : ""}`;
    prev.disabled = activeIndex === 0;
    next.disabled = activeIndex === visibleCount - 1;

    if (page && block) {
      panelHost.dataset.pageIndex = String(Number(block.dataset.pageIndex));
    }

    const switchPage = (newIndex) => {
      if (newIndex < 0 || newIndex >= visibleCount) return;
      if (typeof window.saveDraftNow === "function") window.saveDraftNow();
      if (typeof window.cancelPendingPersist === "function") window.cancelPendingPersist();
      if (typeof window.clearBoxSelection === "function") window.clearBoxSelection();

      canvasHost.innerHTML = "";
      panelHost.innerHTML = "";

      activeIndex = newIndex;
      window.renderEditor();
    };

    prev.addEventListener("click", () => switchPage(activeIndex - 1));
    next.addEventListener("click", () => switchPage(activeIndex + 1));
  }

  window.renderEditor = function translationWorkspaceRender() {
    legacyRenderEditor();
    setupTranslationWorkspace();
  };
})();
