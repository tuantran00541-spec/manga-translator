const editorState = {
  activePageIndex: 0,
  selectedTextObjectId: null,
  tool: "select",
  lastChapterId: null,
};
window.editorState = editorState;
let editorOverlayResizeObserver = null;

const DEFAULT_TEXT_OBJECT_STYLE = {
  color: "auto",
  font: "default",
  fontSize: "auto",
  bold: false,
  strokeWidth: "auto",
  strokeColor: "auto",
  bgColor: "transparent",
  cornerRadius: "0",
  horizontalAlign: "center",
  verticalAlign: "middle",
};
window.DEFAULT_TEXT_OBJECT_STYLE = DEFAULT_TEXT_OBJECT_STYLE;

function findTextObject(pageIndex, id) {
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  if (!page) return null;
  const list = page.text_objects || [];
  return list.find((o) => o && o.id === id) || null;
}
window.findTextObject = findTextObject;

async function apiTextObject(action, payload) {
  const resp = await fetch(`/api/text_object/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const parse = typeof window.parseApiResponse === "function"
    ? window.parseApiResponse
    : async (r) => (await r.json().catch(() => ({})));
  const getErr = typeof window.getErrorMessage === "function"
    ? window.getErrorMessage
    : (s, d) => (d && d.detail) || `lỗi ${s}`;
  const data = await parse(resp);
  if (!resp.ok) throw new Error(getErr(resp.status, data));
  return data;
}

function collectPanelState(pageIndex, id) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return null;
  const state = { ocr_text: null, translation: null, style: null };
  panelHost.querySelectorAll(`textarea[data-text-object-id="${id}"]`).forEach((ta) => {
    if (ta.classList.contains("ocr-textarea")) state.ocr_text = ta.value;
    if (ta.classList.contains("translation-textarea")) state.translation = ta.value;
  });
  const panel = panelHost.querySelector(`.text-editor-panel[data-page-index="${pageIndex}"]`);
  if (panel && panel.dataset.objectId === String(id) && panel.dataset.font !== undefined) {
    state.style = {
      color: panel.dataset.color,
      font: panel.dataset.font,
      fontSize: panel.dataset.fontSize,
      bold: panel.dataset.bold === "true",
      strokeWidth: panel.dataset.strokeWidth,
      strokeColor: panel.dataset.strokeColor,
      bgColor: panel.dataset.bgColor,
      cornerRadius: panel.dataset.cornerRadius,
      horizontalAlign: panel.dataset.horizontalAlign || "center",
      verticalAlign: panel.dataset.verticalAlign || "middle",
    };
  }
  return state;
}

function _styleEqual(a, b) {
  return JSON.stringify(a || null) === JSON.stringify(b || null);
}

function _applyStateDiff(obj, current, snapshot) {
  if (current.ocr_text !== null && (snapshot === null || current.ocr_text !== snapshot.ocr_text)) {
    obj.ocr_text = current.ocr_text;
  }
  if (current.translation !== null && (snapshot === null || current.translation !== snapshot.translation)) {
    obj.translation = current.translation;
  }
  if (
    current.style
    && (snapshot === null || !snapshot.style || !_styleEqual(current.style, snapshot.style))
  ) {
    obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE, obj.style || {}, current.style);
  }
}

function applyManifestResponse(manifest, pageIndex, opts = {}) {
  if (!manifest || (manifest.chapter_id && manifest.chapter_id !== currentChapterId)) {
    return false;
  }
  const skipOverlays = opts.skipOverlays === true;
  const snapshot = opts.snapshot || null;
  const targetId = opts.id || editorState.selectedTextObjectId;
  const currentState = targetId ? collectPanelState(pageIndex, targetId) : null;
  if (typeof window.capturePendingGeom === "function") window.capturePendingGeom();
  currentManifest = manifest;
  if (currentState) {
    const obj = findTextObject(pageIndex, targetId);
    if (obj) _applyStateDiff(obj, currentState, snapshot);
  }
  _textDirty.forEach((entry) => {
    if (entry.pageIndex === pageIndex && entry.id === targetId) return;
    const obj = findTextObject(entry.pageIndex, entry.id);
    if (!obj) return;
    if (entry.ocr_text != null) obj.ocr_text = entry.ocr_text;
    if (entry.translation != null) obj.translation = entry.translation;
    if (entry.style) obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE, obj.style || {}, entry.style);
  });
  if (typeof window.reapplyPendingGeom === "function") window.reapplyPendingGeom();
  const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
  const page = currentManifest ? currentManifest.pages[pageIndex] : null;
  if (!skipOverlays && wrapper && page) renderTextObjectOverlays(pageIndex, page);
  renderEditorPanel(pageIndex);
  return true;
}

async function createTextObject(pageIndex, shape, region) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const manifest = await apiTextObject("create", {
    chapter_id: chapterId,
    page_index: pageIndex,
    shape,
    region,
  });
  if (chapterId !== currentChapterId) return;
  const page = manifest.pages[pageIndex];
  const objs = page.text_objects || [];
  const obj = objs[objs.length - 1];
  if (!obj) throw new Error("Không tạo được vùng chữ");
  editorState.selectedTextObjectId = obj.id;
  applyManifestResponse(manifest, pageIndex, { id: obj.id });
  associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
    showToast("Không thể nhóm OCR tự động. Bạn vẫn có thể nhập nội dung thủ công: " + err.message, "info");
  });
}

async function deleteTextObject(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const manifest = await apiTextObject("delete", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id,
  });
  if (chapterId !== currentChapterId) return;
  if (editorState.selectedTextObjectId === id) editorState.selectedTextObjectId = null;
  if (typeof window.removePendingPersist === "function") window.removePendingPersist(pageIndex, id);
  applyManifestResponse(manifest, pageIndex, { id });
}
window.deleteTextObject = deleteTextObject;

const DUPLICATE_OFFSET = 24;

async function duplicateTextObject(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const obj = findTextObject(pageIndex, id);
  if (!obj) throw new Error("Không tìm thấy vùng chữ");
  const source = {
    shape: obj.shape,
    region: Object.assign({}, obj.region),
    ocrText: obj.ocr_text || "",
    translation: obj.translation || "",
    style: JSON.parse(JSON.stringify(obj.style || DEFAULT_TEXT_OBJECT_STYLE)),
  };
  const { w: W, h: H } = getPageImageSize(pageIndex);
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const bw = source.region.x2 - source.region.x1;
  const bh = source.region.y2 - source.region.y1;
  const x1 = clamp(source.region.x1 + DUPLICATE_OFFSET, 0, Math.max(0, W - bw));
  const y1 = clamp(source.region.y1 + DUPLICATE_OFFSET, 0, Math.max(0, H - bh));

  const manifest = await apiTextObject("create", {
    chapter_id: chapterId,
    page_index: pageIndex,
    shape: source.shape,
    region: { x1, y1, x2: x1 + bw, y2: y1 + bh },
  });
  const objs = manifest.pages[pageIndex].text_objects || [];
  const created = objs[objs.length - 1];
  if (!created) throw new Error("Không thể nhân đôi vùng chữ");

  const updated = await apiTextObject("update", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id: created.id,
    ocr_text: source.ocrText,
    translation: source.translation,
    style: source.style,
  });
  if (chapterId !== currentChapterId) return;
  editorState.selectedTextObjectId = created.id;
  applyManifestResponse(updated, pageIndex, { id: created.id });
}
window.duplicateTextObject = duplicateTextObject;

async function addTextObjectBox() {
  const pageIndex = editorState.activePageIndex;
  const { w: W, h: H } = getPageImageSize(pageIndex);
  if (!Number.isFinite(W) || !Number.isFinite(H)) {
    throw new Error("Chưa xác định được kích thước trang");
  }
  const bw = Math.max(TEXT_OBJECT_MIN_SIZE, Math.min(200, Math.round(W * 0.3)));
  const bh = Math.max(TEXT_OBJECT_MIN_SIZE, Math.min(64, Math.round(H * 0.12)));
  const x1 = Math.round((W - bw) / 2);
  const y1 = Math.round((H - bh) / 2);
  await createTextObject(pageIndex, "rectangle", { x1, y1, x2: x1 + bw, y2: y1 + bh });
}

async function associateTextObjectOcr(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const lang = await window.resolveSourceLang();
  const snapshot = collectPanelState(pageIndex, id);
  const manifest = await apiTextObject("ocr", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id,
    lang,
  });
  if (chapterId !== currentChapterId) return;
  if (!findTextObject(pageIndex, id)) return;
  applyManifestResponse(manifest, pageIndex, { skipOverlays: true, snapshot, id });
}
window.associateTextObjectOcr = associateTextObjectOcr;

function setEditorTool(tool) {
  editorState.tool = tool;
  document.querySelectorAll(".editor-tool-btn").forEach((btn) => {
    const active = btn.dataset.tool === tool;
    btn.classList.toggle("active", active);
    btn.classList.toggle("ui-btn-primary", active);
    btn.classList.toggle("ui-btn-ghost", !active);
    if (btn.dataset.tool) btn.setAttribute("aria-pressed", String(active));
  });
  const imgWrap = document.querySelector(".translation-canvas-host .page-image-wrap");
  if (imgWrap) imgWrap.classList.toggle("draw-mode", tool !== "select");
}
window.setEditorTool = setEditorTool;

function setSelectedTextObject(pageIndex, id) {
  editorState.selectedTextObjectId = id;
  document.querySelectorAll(".text-object-overlay").forEach((el) => {
    el.classList.toggle("selected", el.dataset.objectId === id);
  });
  renderEditorPanel(pageIndex);
  if (id) window.showWorkbenchInspector?.();
}
window.setSelectedTextObject = setSelectedTextObject;

function clearSelectedTextObject() {
  editorState.selectedTextObjectId = null;
  document.querySelectorAll(".text-object-overlay").forEach((el) => el.classList.remove("selected"));
  renderEditorPanel(editorState.activePageIndex);
}
window.clearSelectedTextObject = clearSelectedTextObject;

function editorImageMetrics(img) {
  if (!img || !img.naturalWidth || !img.naturalHeight || !img.clientWidth || !img.clientHeight) {
    return null;
  }
  return {
    offsetX: img.offsetLeft,
    offsetY: img.offsetTop,
    width: img.clientWidth,
    height: img.clientHeight,
    sx: img.clientWidth / img.naturalWidth,
    sy: img.clientHeight / img.naturalHeight,
  };
}
window.editorImageMetrics = editorImageMetrics;

function renderTextObjectOverlays(pageIndex, page) {
  editorOverlayResizeObserver?.disconnect();
  editorOverlayResizeObserver = null;
  const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
  if (!wrapper) return;
  const imgWrap = wrapper.querySelector(".page-image-wrap");
  if (!imgWrap) return;
  const img = imgWrap.querySelector("img");
  const render = () => {
    const metrics = editorImageMetrics(img);
    if (!metrics) return;
    imgWrap.querySelectorAll(".text-object-overlay:not(.drawing)").forEach((el) => el.remove());
    (page.text_objects || []).forEach((obj) => {
      if (!obj || !obj.region) return;
      const overlay = document.createElement("div");
      overlay.className = "text-object-overlay" + (obj.shape === "ellipse" ? " ellipse" : "");
      overlay.dataset.pageIndex = String(pageIndex);
      overlay.dataset.objectId = obj.id;
      overlay.style.left = metrics.offsetX + obj.region.x1 * metrics.sx + "px";
      overlay.style.top = metrics.offsetY + obj.region.y1 * metrics.sy + "px";
      overlay.style.width = (obj.region.x2 - obj.region.x1) * metrics.sx + "px";
      overlay.style.height = (obj.region.y2 - obj.region.y1) * metrics.sy + "px";
      if (editorState.selectedTextObjectId === obj.id) overlay.classList.add("selected");
      overlay.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const editor = document.querySelector(".translation-panel-host .translation-textarea");
        if (editor) editor.focus();
      });
      imgWrap.appendChild(overlay);
    });
    window.installEditorBoxTransforms?.(imgWrap);
  };

  if (img.complete && img.naturalWidth > 0) render();
  else img.onload = render;
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => {
      if (!img.isConnected) observer.disconnect();
      else render();
    });
    observer.observe(img);
    editorOverlayResizeObserver = observer;
  }
}

function setupEditorDraw(wrapper, pageIndex) {
  const imgWrap = wrapper.querySelector(".page-image-wrap");
  const img = imgWrap.querySelector("img");
  let drawing = false;
  let start = null;
  let last = null;
  let temp = null;

  const pointInImage = (e) => {
    const metrics = editorImageMetrics(img);
    if (!metrics) return null;
    const rect = img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const rawX = (e.clientX - rect.left) * (metrics.width / rect.width);
    const rawY = (e.clientY - rect.top) * (metrics.height / rect.height);
    return {
      x: Math.max(0, Math.min(metrics.width, rawX)),
      y: Math.max(0, Math.min(metrics.height, rawY)),
      inside: rawX >= 0 && rawX <= metrics.width && rawY >= 0 && rawY <= metrics.height,
      metrics,
    };
  };

  const updateTemp = (x, y) => {
    if (!temp || !start) return;
    const metrics = editorImageMetrics(img);
    if (!metrics) return;
    temp.style.left = metrics.offsetX + Math.min(start.x, x) + "px";
    temp.style.top = metrics.offsetY + Math.min(start.y, y) + "px";
    temp.style.width = Math.abs(x - start.x) + "px";
    temp.style.height = Math.abs(y - start.y) + "px";
  };

  const onDown = (e) => {
    if (editorState.tool === "select") return;
    if (e.button !== 0) return;
    if (e.target.closest(".text-object-overlay")) return;
    e.preventDefault();
    const point = pointInImage(e);
    if (!point || !point.inside) return;
    drawing = true;
    start = { x: point.x, y: point.y };
    last = start;
    temp = document.createElement("div");
    temp.className = "text-object-overlay drawing" + (editorState.tool === "ellipse" ? " ellipse" : "");
    imgWrap.appendChild(temp);
    updateTemp(start.x, start.y);
  };

  const onMove = (e) => {
    if (!drawing) return;
    const point = pointInImage(e);
    if (!point) return;
    last = { x: point.x, y: point.y };
    updateTemp(last.x, last.y);
  };

  const onUp = () => {
    if (!drawing) return;
    drawing = false;
    if (temp) { temp.remove(); temp = null; }
    if (!start || !last) { start = null; last = null; return; }
    const metrics = editorImageMetrics(img);
    if (!metrics) { start = null; last = null; return; }
    const sx = img.naturalWidth / metrics.width;
    const sy = img.naturalHeight / metrics.height;
    const x1 = Math.round(Math.min(start.x, last.x) * sx);
    const y1 = Math.round(Math.min(start.y, last.y) * sy);
    const x2 = Math.round(Math.max(start.x, last.x) * sx);
    const y2 = Math.round(Math.max(start.y, last.y) * sy);
    const shape = editorState.tool;
    start = null;
    last = null;
    if (x2 - x1 < 10 || y2 - y1 < 10) return;
    createTextObject(pageIndex, shape, { x1, y1, x2, y2 }).catch((err) => {
      showToast("Không thể tạo vùng chữ: " + err.message, "error");
    });
  };

  const onClick = (e) => {
    if (editorState.tool !== "select") return;
    if (e.target.closest(".text-object-overlay")) return;
    clearSelectedTextObject();
  };

  const drawAbort = new AbortController();
  const signal = drawAbort.signal;
  imgWrap.style.touchAction = editorState.tool === "select" ? "pan-x pan-y" : "none";
  imgWrap.addEventListener("pointerdown", (event) => {
    onDown(event);
    if (drawing) imgWrap.setPointerCapture?.(event.pointerId);
  }, { signal });
  imgWrap.addEventListener("pointermove", onMove, { signal });
  window.addEventListener("pointerup", onUp, { signal });
  window.addEventListener("pointercancel", onUp, { signal });
  imgWrap.addEventListener("click", onClick);

  window._editorDrawCleanup = function cleanupEditorDraw() {
    drawAbort.abort();
  };
}

function buildPageWrapper(page, pageIndex, pages) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-block-wrapper";

  const label = document.createElement("div");
  label.className = "page-block-label";
  label.textContent = pageLabel(pages, pageIndex);
  wrapper.appendChild(label);

  const block = document.createElement("div");
  block.className = "page-block";
  block.dataset.pageIndex = pageIndex;

  const imgWrap = document.createElement("div");
  imgWrap.className = "page-image-wrap";
  const img = document.createElement("img");
  img.src = typeof window.pageImageUrl === "function"
    ? window.pageImageUrl(page)
    : (page.clean || page.original);
  img.draggable = false;
  imgWrap.appendChild(img);
  block.appendChild(imgWrap);
  wrapper.appendChild(block);
  return wrapper;
}

function switchEditorPage(newIndex) {
  const pages = currentManifest ? currentManifest.pages : null;
  if (!pages || newIndex < 0 || newIndex >= pages.length) return;
  const target = Math.max(0, Math.min(Number(newIndex) || 0, pages.length - 1));
  if (target === editorState.activePageIndex) return true;

  void flushAllPendingPersists().catch((err) => {
    console.warn("Không thể lưu bản nháp trước khi chuyển trang:", err);
  });
  editorState.activePageIndex = target;
  editorState.selectedTextObjectId = null;
  renderEditor();
  return true;
}

window.switchEditorPage = switchEditorPage;

function showRenderResult(pageIndex, outputPath, renderRevision = null) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return;
  let resultBox = panelHost.querySelector(".render-result");
  if (!resultBox) {
    resultBox = document.createElement("div");
    resultBox.className = "render-result";
    panelHost.appendChild(resultBox);
  }
  const cacheBust = renderRevision === null || renderRevision === undefined
    ? "?t=" + Date.now()
    : `?r=${encodeURIComponent(renderRevision)}`;
  resultBox.innerHTML = "";

  const label = document.createElement("div");
  label.className = "render-result-label";
  label.textContent = "Kết quả kết xuất";
  resultBox.appendChild(label);

  const img = document.createElement("img");
  img.src = outputPath + cacheBust;
  resultBox.appendChild(img);

  const link = document.createElement("a");
  link.href = window.currentChapterId ? `/api/download/${encodeURIComponent(window.currentChapterId)}/${pageIndex}` : outputPath + cacheBust;
  link.download = `page_${pageIndex + 1}_rendered.png`;
  link.className = "download-link";
  link.textContent = "Tải ảnh đã kết xuất";
  resultBox.appendChild(link);
}

const _pendingAutoSync = new Map();

let _autoSyncingPageIndex = null;

function _sourceBoxSet(obj) {
  return new Set(Array.isArray(obj?.source_boxes) ? obj.source_boxes.map(String) : []);
}

function _sameRegion(region, box) {
  if (!region || !box) return false;
  return ["x1", "y1", "x2", "y2"].every((key) => Number(region[key]) === Number(box[key]));
}

function _autoObjectNeedsSync(obj, box) {
  if (!obj?.auto_generated) return false;
  const boxText = String(box?.ocr_text || "");
  const objectText = String(obj.ocr_text || "");
  const previousAutoText = String(obj.auto_ocr_text || "");
  const machineTextCanMove = objectText === previousAutoText;
  if (machineTextCanMove && boxText !== objectText) return true;

  const currentRegion = obj.region || null;
  const previousAutoRegion = obj.auto_geometry || null;
  const machineGeometryCanMove = !previousAutoRegion || _sameRegion(currentRegion, previousAutoRegion);
  return machineGeometryCanMove && !_sameRegion(currentRegion, box);
}

function _isAutoSyncEligibleBox(box) {
  if (!box || box.removed || !box.id || box.ocr_eligible === false) return false;
  const [x1, y1, x2, y2] = [box.x1, box.y1, box.x2, box.y2].map(Number);
  return [x1, y1, x2, y2].every(Number.isFinite) && x1 < x2 && y1 < y2;
}

function _pageNeedsAutoSync(page) {
  if (!page || page.skipped) return false;
  const activeBoxes = (page.boxes || []).filter(_isAutoSyncEligibleBox);
  if (!activeBoxes.length) return false;
  const objects = page.text_objects || [];
  return activeBoxes.some((box) => {
    const linked = objects.find((obj) => _sourceBoxSet(obj).has(String(box.id)));
    return !linked || _autoObjectNeedsSync(linked, box);
  });
}

function _autoSyncChangedPage(manifest, pageIndex) {
  return Array.isArray(manifest?.auto_text_objects?.changed_pages)
    && manifest.auto_text_objects.changed_pages.some((index) => Number(index) === Number(pageIndex));
}

async function ensureAutoTextObjects(pageIndex) {
  const chapterId = window.currentChapterId;
  if (!chapterId) return null;
  const key = `${chapterId}:${pageIndex}`;
  if (_pendingAutoSync.has(key)) return _pendingAutoSync.get(key);

  const job = (async () => {
    const response = await fetch("/api/text_objects/ensure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: chapterId, page_indices: [pageIndex] }),
    });
    const parse = typeof window.parseApiResponse === "function"
      ? window.parseApiResponse
      : async (r) => r.json().catch(() => ({}));
    const data = await parse(response);
    if (!response.ok) {
      const getError = typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, payload) => payload?.detail || `HTTP ${status}`;
      throw new Error(getError(response.status, data));
    }
    if (chapterId !== window.currentChapterId) return null;
    window.currentManifest = data;
    return data;
  })();

  _pendingAutoSync.set(key, job);
  try {
    return await job;
  } finally {
    _pendingAutoSync.delete(key);
  }
}
window.ensureAutoTextObjects = ensureAutoTextObjects;

function buildChapterTranslateControls() {
  const controls = document.createElement("details");
  controls.className = "ui-disclosure chapter-translate-controls command-disclosure";
  const summary = document.createElement("summary");
  summary.className = "ui-btn ui-btn-ghost";
  summary.textContent = "Dịch tự động";
  const options = document.createElement("div");
  options.className = "ui-disclosure-panel command-disclosure-panel chapter-translate-options";

  const note = document.createElement("p");
  note.className = "ui-note";
  note.textContent = "Mỗi lát gửi ảnh gốc + ảnh sau inpaint đến model vision. Chỉ nội dung chữ được thay đổi; bố cục và màu đã lưu được dùng để render tự động.";

  const provider = document.createElement("select");
  provider.className = "chapter-translate-provider";
  [["deepseek", "DeepSeek"], ["gemini", "Google Gemini"], ["openai", "OpenAI"], ["openrouter", "OpenRouter"], ["experiential", "Experiential Labs"]]
    .forEach(([value, label]) => provider.add(new Option(label, value)));
  const storedProvider = localStorage.getItem("manga_translation_provider");
  provider.value = [...provider.options].some((o) => o.value === storedProvider)
    ? storedProvider : "deepseek";

  const model = document.createElement("input");
  model.className = "ui-input chapter-translate-model";
  model.placeholder = "Model vision mặc định của dịch vụ";
  model.setAttribute("aria-label", "Model vision");
  const target = document.createElement("select");
  target.className = "chapter-translate-target";
  target.setAttribute("aria-label", "Ngôn ngữ bản dịch");
  [["vi", "Tiếng Việt"], ["en", "English"]].forEach(([value, label]) => target.add(new Option(label, value)));

  const budget = document.createElement("input");
  budget.type = "number";
  budget.min = "0.001";
  budget.max = "0.25";
  budget.step = "0.001";
  budget.value = "0.25";
  budget.className = "chapter-translate-budget";
  budget.title = "Giới hạn ước tính cho toàn chương; giá ảnh tùy API";
  budget.setAttribute("aria-label", "Ngân sách ước tính dịch toàn chương bằng USD");

  const forceLabel = document.createElement("label");
  forceLabel.className = "ui-field";
  const force = document.createElement("input");
  force.type = "checkbox";
  force.className = "chapter-translate-force";
  forceLabel.append(force, document.createTextNode(" Dịch lại vùng đã có bản dịch"));

  const field = (title, input) => {
    const label = document.createElement("label");
    label.className = "ui-field";
    label.append(document.createTextNode(title), input);
    return label;
  };
  const budgetLabel = field("Giới hạn DeepSeek (USD, ước tính)", budget);
  const syncModel = () => {
    model.value = localStorage.getItem("manga_translation_vision_model_" + provider.value) || "";
    localStorage.setItem("manga_translation_provider", provider.value);
    budgetLabel.hidden = provider.value !== "deepseek";
    budget.disabled = provider.value !== "deepseek";
  };
  provider.addEventListener("change", syncModel);
  provider.addEventListener("ai-providers-updated", syncModel);
  model.addEventListener("change", () => {
    localStorage.setItem("manga_translation_vision_model_" + provider.value, model.value.trim());
  });

  const run = document.createElement("button");
  run.type = "button";
  run.className = "ui-btn ui-btn-primary chapter-translate-run";
  run.textContent = "Dịch và render";

  run.addEventListener("click", async () => {
    const chapterId = window.currentChapterId;
    if (!chapterId || run.disabled) return;
    run.disabled = true;
    const originalSummary = "Dịch tự động";
    let done = 0;
    let translatedCount = 0;
    let unreadable = 0;
    let staleCount = 0;
    let actualCost = 0;
    let costKnown = provider.value === "deepseek";
    const renderErrors = [];
    const budgetTotal = Number(budget.value || 0.25);
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      } else if (typeof window.flushTextObjectPersist === "function") {
        await window.flushTextObjectPersist();
      }
      if (chapterId !== window.currentChapterId) return;
      const indices = (window.currentManifest?.pages || [])
        .map((page, index) => ({ page, index }))
        .filter(({ page }) => page && !page.skipped && !page.process_required &&
          page.original && page.clean &&
          (page.text_objects?.some((obj) => obj && !obj.source_missing &&
            (force.checked || !String(obj.translation || "").trim())) ||
           page.boxes?.some((box) => box && !box.removed && box.ocr_eligible !== false)))
        .map(({ index }) => index);
      if (!indices.length) {
        window.showToast?.("Không có vùng chữ nào cần dịch trên các lát đã inpaint.", "info");
        return;
      }
      const sourceLang = await window.resolveSourceLang();
      if (chapterId !== window.currentChapterId) return;
      for (const pageIndex of indices) {
        if (chapterId !== window.currentChapterId) return;
        if (provider.value === "deepseek" && actualCost >= budgetTotal) {
          window.showToast?.("Đã chạm giới hạn chi phí ước tính. Các lát đã dịch được lưu.", "info");
          break;
        }
        run.textContent = "Đang dịch " + (done + 1) + "/" + indices.length;
        summary.textContent = run.textContent;
        const response = await fetch("/api/translate/page/vision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter_id: chapterId,
            page_index: pageIndex,
            source_lang: sourceLang,
            target_lang: target.value,
            provider: provider.value,
            model: model.value.trim() || null,
            budget_usd: provider.value === "deepseek"
              ? Math.max(0.001, budgetTotal - actualCost) : 0.25,
            force: force.checked,
          }),
        });
        const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
        const data = await parse(response);
        if (!response.ok) {
          const getErr = window.getErrorMessage || ((s, d) => d?.detail || "HTTP " + s);
          throw new Error("Lát " + (pageIndex + 1) + ": " + getErr(response.status, data));
        }
        if (chapterId !== window.currentChapterId) return;
        window.currentManifest = data;
        (data.pages || []).forEach((page, index) => {
          if (page?.rendered) {
            page._reviewRenderedUrl = "/api/image/" + encodeURIComponent(chapterId)
              + "/" + index + "/rendered";
          }
        });
        const info = data.translation_run || {};
        translatedCount += Number(info.translated || 0);
        unreadable += Number(info.unreadable || 0);
        staleCount += Number(info.stale || 0);
        if (info.estimated_cost_usd == null) costKnown = false;
        else actualCost += Number(info.estimated_cost_usd);
        if (info.render_error) renderErrors.push("Lát " + (pageIndex + 1) + ": " + info.render_error);
        done++;
        if (info.budget_exceeded) break;
      }
      if (chapterId !== window.currentChapterId) return;
      const shell = document.querySelector("#page-view.review-mode .review-document-shell");
      if (shell && typeof shell._showRendered === "function") shell._showRendered();
      else if (document.body.dataset.appStage === "editor" && typeof window.renderEditor === "function") window.renderEditor();
      const price = costKnown ? " · ~$" + actualCost.toFixed(4) : " · phí ảnh theo provider";
      const warning = renderErrors.length ? " · " + renderErrors.length + " lát chưa render" : "";
      window.showToast?.(
        "Đã dịch " + translatedCount + " vùng / " + done + " lát" +
        (unreadable ? " · không đọc được " + unreadable : "") +
        (staleCount ? " · thay đổi khi dịch " + staleCount : "") + price + warning,
        renderErrors.length ? "error" : "success"
      );
    } catch (err) {
      window.showToast?.(
        "Dịch dừng sau " + done + " lát đã lưu: " + err.message, "error"
      );
    } finally {
      run.disabled = false;
      run.textContent = "Dịch và render";
      summary.textContent = originalSummary;
    }
  });

  options.append(
    note, field("Dịch vụ AI", provider), field("Model vision", model),
    field("Dịch sang", target), budgetLabel, forceLabel, run,
  );
  syncModel();
  controls.append(summary, options);
  window.syncAIProviderSelects?.();
  return controls;
}

function buildChapterExportButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-btn ui-btn-primary chapter-export-run";
  button.textContent = "Xuất chương (.zip)";

  button.addEventListener("click", async () => {
    const chapterId = window.currentChapterId;
    if (!chapterId) return;
    button.disabled = true;
    button.textContent = "Đang kết xuất chương…";
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      }
      if (chapterId !== window.currentChapterId) return;
      const response = await fetch(`/api/render/chapter?chapter_id=${encodeURIComponent(chapterId)}`, {
        method: "POST",
      });
      const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
      const data = await parse(response);
      if (!response.ok) {
        const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d?.detail || `HTTP ${s}`;
        throw new Error(getErr(response.status, data));
      }
      if (chapterId !== window.currentChapterId) return;
      window.currentManifest = data;

      const href = data.chapter_render?.download_url || `/api/export/${encodeURIComponent(chapterId)}.zip`;
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = `manga-translator-${chapterId}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      if (typeof window.showToast === "function") {
        window.showToast(`Đã kết xuất ${data.chapter_render?.rendered || 0} trang.`, "info");
      }
    } catch (err) {
      if (chapterId === window.currentChapterId && typeof window.showToast === "function") {
        window.showToast("Xuất chương thất bại: " + err.message, "error");
      }
    } finally {
      button.disabled = false;
      button.textContent = "Xuất chương (.zip)";
    }
  });

  return button;
}

window.buildChapterTranslateControls = buildChapterTranslateControls;
window.buildChapterExportButton = buildChapterExportButton;

function renderEditor() {
  const container = document.getElementById("page-view");
  if (!container) return;
  if (!currentManifest || !currentManifest.pages || currentManifest.pages.length === 0) return;

  window.setAppStage?.("editor");

  if (currentChapterId && editorState.lastChapterId !== currentChapterId) {
    if (
      editorState.lastChapterId
      && typeof window.cancelPendingPersist === "function"
    ) {
      window.cancelPendingPersist();
    }
    editorState.lastChapterId = currentChapterId;
    editorState.activePageIndex = 0;
    editorState.selectedTextObjectId = null;
  }

  const pages = currentManifest.pages;
  editorState.activePageIndex = Math.max(0, Math.min(editorState.activePageIndex, pages.length - 1));
  const pageIndex = editorState.activePageIndex;
  const page = pages[pageIndex];

  if (_autoSyncingPageIndex !== pageIndex && _pageNeedsAutoSync(page)) {
    _autoSyncingPageIndex = pageIndex;
    ensureAutoTextObjects(pageIndex)
      .then((manifest) => {
        if (Number(editorState.activePageIndex || 0) !== pageIndex) {
          _autoSyncingPageIndex = null;
          return;
        }
        if (!manifest || !_autoSyncChangedPage(manifest, pageIndex)) {
          _autoSyncingPageIndex = null;
          return;
        }
        queueMicrotask(() => {
          _autoSyncingPageIndex = null;
          renderEditor();
        });
      })
      .catch((err) => {
        _autoSyncingPageIndex = null;
        if (typeof window.showToast === "function") {
          window.showToast("Không thể đồng bộ vùng chữ từ nhận diện: " + err.message, "error");
        }
      });
  }

  if (typeof window.setWorkflowCheckpoint === "function") {
    window.setWorkflowCheckpoint("editor", pageIndex);
  }

  if (typeof window._editorDrawCleanup === "function") {
    window._editorDrawCleanup();
    window._editorDrawCleanup = null;
  }
  editorOverlayResizeObserver?.disconnect();
  editorOverlayResizeObserver = null;
  container.innerHTML = "";
  container.className = "editor-mode";

  const shell = document.createElement("div");
  shell.className = "translation-workspace";

  const toolbar = document.createElement("div");
  toolbar.className = "translation-sticky-toolbar";

  const tools = document.createElement("div");
  tools.className = "editor-tools";
  tools.setAttribute("role", "group");
  tools.setAttribute("aria-label", "Công cụ vùng chữ");
  [
    { key: "select", label: "Chọn" },
    { key: "rectangle", label: "Khung chữ nhật" },
    { key: "ellipse", label: "Khung elip" },
  ].forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-btn editor-tool-btn" + (editorState.tool === t.key ? " ui-btn-primary active" : " ui-btn-ghost");
    btn.dataset.tool = t.key;
    btn.textContent = t.label;
    btn.addEventListener("click", () => setEditorTool(t.key));
    tools.appendChild(btn);
  });

  const addBoxBtn = document.createElement("button");
  addBoxBtn.type = "button";
  addBoxBtn.className = "ui-btn ui-btn-ghost editor-tool-btn";
  addBoxBtn.textContent = "Thêm vùng chữ";
  addBoxBtn.title = "Tạo một vùng chữ mới ở giữa trang";
  addBoxBtn.addEventListener("click", () => {
    addTextObjectBox().catch((err) => {
      showToast("Không thể thêm vùng chữ: " + err.message, "error");
    });
  });
  tools.appendChild(addBoxBtn);

  const translateControls = buildChapterTranslateControls();

  const renderBtn = document.createElement("button");
  renderBtn.type = "button";
  renderBtn.className = "ui-btn ui-btn-primary render-btn editor-render-btn";
  renderBtn.textContent = "Kết xuất trang";
  renderBtn.addEventListener("click", () => renderTranslations(pageIndex));

  const exportBtn = buildChapterExportButton();

  const saveStatus = document.createElement("div");
  const initialStatus = refreshSaveStatus();
  saveStatus.className = `editor-save-status save-status-${initialStatus}`;
  setSaveStatusContent(saveStatus, initialStatus);

  saveStatus.setAttribute("role", "status");
  toolbar.append(tools, translateControls, renderBtn, exportBtn, saveStatus);

  const navItems = pages.map((item, index) => ({
    key: index,
    label: typeof pageLabel === "function" ? pageLabel(pages, index) : `Trang ${index + 1}`,
    image: item.rendered || item.clean || item.original,
    state: item.skipped ? "skipped" : (item.rendered ? "rendered" : "ready"),
    stateLabel: item.skipped ? "Bỏ qua" : (item.rendered ? "Đã kết xuất" : "Đang biên tập"),
  }));
  const navigator = window.createPageNavigator({
    items: navItems,
    activeIndex: pageIndex,
    title: "Trang biên tập",
    ariaLabel: "Điều hướng trang biên tập",
    onSelect: (index) => switchEditorPage(index),
  });

  const body = document.createElement("div");
  body.className = "translation-workspace-body workbench-stage-grid editor-workbench-grid";

  const canvasHost = document.createElement("main");
  canvasHost.className = "translation-canvas-host";

  const panelHost = document.createElement("aside");
  panelHost.className = "translation-panel-host context-inspector editor-inspector";
  panelHost.setAttribute("aria-label", "Bảng biên tập vùng chữ");

  body.append(navigator.element, canvasHost, panelHost);

  shell.append(toolbar, body);
  container.appendChild(shell);

  const wrapper = buildPageWrapper(page, pageIndex, pages);
  canvasHost.appendChild(wrapper);

  if (editorState.selectedTextObjectId && !findTextObject(pageIndex, editorState.selectedTextObjectId)) {
    editorState.selectedTextObjectId = null;
  }

  setupEditorDraw(wrapper, pageIndex);
  renderTextObjectOverlays(pageIndex, page);
  renderEditorPanel(pageIndex);
  setEditorTool(editorState.tool);
  window.setupWorkbenchPanels?.("editor");
}
