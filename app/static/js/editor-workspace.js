// editor-workspace.js - UI-04 focused translation workspace
(() => {
  const legacyRenderEditor = window.renderEditor;
  if (typeof legacyRenderEditor !== "function") return;

  let activeIndex = 0;

  function setupTranslationWorkspace() {
    const container = document.getElementById("page-view");
    if (!container) return;

    const wrappers = [...container.querySelectorAll(".page-block-wrapper")];
    if (!wrappers.length) return;

    activeIndex = Math.max(0, Math.min(activeIndex, wrappers.length - 1));

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

    const renderActive = () => {
      canvasHost.innerHTML = "";
      panelHost.innerHTML = "";

      wrappers.forEach((wrapper) => {
        wrapper.style.display = "none";
      });

      const wrapper = wrappers[activeIndex];
      wrapper.style.display = "block";
      canvasHost.appendChild(wrapper);

      const block = wrapper.querySelector(".page-block");
      const panel = wrapper.querySelector(".box-panel");
      if (block) block.classList.add("translation-active-page");
      if (panel) panelHost.appendChild(panel);

      const page = currentManifest?.pages?.[Number(block?.dataset.pageIndex)];
      const label = wrapper.querySelector(".page-block-label");
      const visibleCount = wrappers.length;
      position.textContent = `${activeIndex + 1} / ${visibleCount}${label ? " · " + label.textContent : ""}`;
      prev.disabled = activeIndex === 0;
      next.disabled = activeIndex === visibleCount - 1;

      if (page) {
        panelHost.dataset.pageIndex = String(Number(block.dataset.pageIndex));
      }
    };

    prev.addEventListener("click", () => {
      if (activeIndex > 0) {
        activeIndex -= 1;
        renderActive();
      }
    });
    next.addEventListener("click", () => {
      if (activeIndex < wrappers.length - 1) {
        activeIndex += 1;
        renderActive();
      }
    });

    renderActive();
  }

  window.renderEditor = function translationWorkspaceRender() {
    legacyRenderEditor();
    setupTranslationWorkspace();
  };
})();
