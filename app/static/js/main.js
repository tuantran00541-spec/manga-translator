let editorActivePageIndex = 0;

function renderEditor() {
  const container = document.getElementById("page-view");
  if (!container || !currentManifest?.pages?.length) return;

  if (window.currentChapterId && editorState.lastChapterId !== window.currentChapterId) {
    editorState.lastChapterId = window.currentChapterId;
    editorActivePageIndex = 0;
    editorState.activePageIndex = 0;
    editorState.selectedTextObjectId = null;
  }

  if (typeof window.cancelPendingPersist === "function") window.cancelPendingPersist();
  const pages = currentManifest.pages;
  editorActivePageIndex = Math.max(0, Math.min(editorActivePageIndex, pages.length - 1));
  editorState.activePageIndex = editorActivePageIndex;

  container.innerHTML = "";
  container.className = "";

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";
  toolbar.className = "editor-top-toolbar";

  const toolGroup = document.createElement("div");
  toolGroup.className = "editor-tool-group";

  const selectBtn = document.createElement("button");
  selectBtn.type = "button";
  selectBtn.className = "editor-tool-btn" + (editorState.tool === "select" ? " active" : "");
  selectBtn.dataset.tool = "select";
  selectBtn.textContent = "Chọn vùng";
  selectBtn.title = "Bấm để chọn / kéo vùng chữ đã có";
  selectBtn.addEventListener("click", () => setEditorTool("select"));

  const rectBtn = document.createElement("button");
  rectBtn.type = "button";
  rectBtn.className = "editor-tool-btn" + (editorState.tool === "rectangle" ? " active" : "");
  rectBtn.dataset.tool = "rectangle";
  rectBtn.textContent = "Vẽ ô vuông";
  rectBtn.title = "Kéo trên ảnh để tạo ô vuông mới";
  rectBtn.addEventListener("click", () => setEditorTool("rectangle"));

  const ellipseBtn = document.createElement("button");
  ellipseBtn.type = "button";
  ellipseBtn.className = "editor-tool-btn" + (editorState.tool === "ellipse" ? " active" : "");
  ellipseBtn.dataset.tool = "ellipse";
  ellipseBtn.textContent = "Vẽ ô tròn";
  ellipseBtn.title = "Kéo trên ảnh để tạo ô tròn mới";
  ellipseBtn.addEventListener("click", () => setEditorTool("ellipse"));

  toolGroup.append(selectBtn, rectBtn, ellipseBtn);

  const renderBtn = document.createElement("button");
  renderBtn.className = "editor-render-btn";
  renderBtn.textContent = "Chèn chữ vào ảnh";
  renderBtn.addEventListener("click", () => renderTranslations(editorActivePageIndex));

  toolbar.append(toolGroup, renderBtn);
  container.appendChild(toolbar);

  pages.forEach((page, pageIndex) => {
    if (page.skipped) return;
    const wrapper = document.createElement("div");
    wrapper.className = "page-block-wrapper";
    wrapper.dataset.pageIndex = pageIndex;
    if (pageIndex !== editorActivePageIndex) wrapper.style.display = "none";

    const label = document.createElement("div");
    label.className = "page-block-label";
    label.textContent = pageLabel(pages, pageIndex);
    wrapper.appendChild(label);

    const block = document.createElement("div");
    block.className = "page-block";
    block.dataset.pageIndex = pageIndex;

    const imgWrap = document.createElement("div");
    imgWrap.className = "page-image-wrap" + (editorState.tool !== "select" ? " draw-mode" : "");
    const img = document.createElement("img");
    img.src = page.clean ? page.clean + "?t=" + Date.now() : page.original;
    imgWrap.appendChild(img);
    block.appendChild(imgWrap);
    wrapper.appendChild(block);

    const panel = document.createElement("div");
    panel.className = "box-panel";
    wrapper.appendChild(panel);

    container.appendChild(wrapper);

    setupEditorDraw(wrapper, pageIndex);
    renderTextObjectOverlays(pageIndex, page);
  });

  renderEditorPanel(editorActivePageIndex);
}
window.renderEditor = renderEditor;

function setupEditorDraw(wrapper, pageIndex) {
  const imgWrap = wrapper.querySelector(".page-image-wrap");
  const img = imgWrap.querySelector("img");
  let drawing = false;
  let start = null;
  let temp = null;

  const updateTemp = (x, y) => {
    if (!temp || !start) return;
    const left = Math.min(start.x, x);
    const top = Math.min(start.y, y);
    const width = Math.abs(x - start.x);
    const height = Math.abs(y - start.y);
    temp.style.left = left + "px";
    temp.style.top = top + "px";
    temp.style.width = width + "px";
    temp.style.height = height + "px";
  };

  const onDown = (e) => {
    if (editorState.tool === "select") return;
    if (e.button !== 0) return;
    if (e.target.closest(".text-object-overlay")) return;
    e.preventDefault();

    const rect = imgWrap.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    drawing = true;
    start = { x, y };

    temp = document.createElement("div");
    temp.className = "text-object-overlay drawing" + (editorState.tool === "ellipse" ? " ellipse" : "");
    imgWrap.appendChild(temp);
    updateTemp(x, y);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const onMove = (e) => {
    if (!drawing || !start || !temp) return;
    const rect = imgWrap.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    updateTemp(x, y);
  };

  const onUp = (e) => {
    if (!drawing || !start || !temp) return;
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    drawing = false;

    const rect = imgWrap.getBoundingClientRect();
    const endX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const endY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    const left = Math.min(start.x, endX);
    const top = Math.min(start.y, endY);
    const w = Math.abs(endX - start.x);
    const h = Math.abs(endY - start.y);

    temp.remove();
    temp = null;
    const shape = editorState.tool;
    start = null;

    if (w < 10 || h < 10) return;

    if (!img.naturalWidth || !img.clientWidth) return;
    const sx = img.naturalWidth / img.clientWidth;
    const sy = img.naturalHeight / img.clientHeight;

    const region = {
      x1: Math.round(left * sx),
      y1: Math.round(top * sy),
      x2: Math.round((left + w) * sx),
      y2: Math.round((top + h) * sy),
    };

    createTextObject(pageIndex, shape, region).catch((err) => {
      showToast("Không tạo được text object: " + err.message, "error");
    });
  };

  imgWrap.addEventListener("mousedown", onDown);
}

function showRenderResult(pageIndex, outputUrl) {
  const host = document.querySelector(".translation-canvas-host");
  const wrapper = host ? host.querySelector(".page-block-wrapper") : null;
  if (!wrapper) return;
  const block = wrapper.querySelector(".page-block");
  const imgWrap = block ? block.querySelector(".page-image-wrap") : null;
  if (!imgWrap) return;
  let resultImg = imgWrap.querySelector(".render-result-img");
  if (!resultImg) {
    resultImg = document.createElement("img");
    resultImg.className = "render-result-img";
    imgWrap.appendChild(resultImg);
  }
  resultImg.src = outputUrl + "?t=" + Date.now();
  imgWrap.classList.add("rendered");
}

// App bootstrap was lost when main.js was replaced by the editor renderer.
// Keep the current renderer intact and restore only the application startup wiring.
document.addEventListener("DOMContentLoaded", () => {
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn && typeof loadChapter === "function") {
    loadBtn.addEventListener("click", loadChapter);
  }

  const workersEl = document.getElementById("workers-select");
  if (workersEl) {
    const saved = localStorage.getItem("mt_workers");
    if (saved && [...workersEl.options].some((o) => o.value === saved)) {
      workersEl.value = saved;
    }
    workersEl.addEventListener("change", () => {
      localStorage.setItem("mt_workers", workersEl.value);
    });
  }

  if (typeof initUpload === "function") initUpload();
  if (typeof loadRecentChapters === "function") loadRecentChapters();
  if (typeof loadFonts === "function") loadFonts();
});
