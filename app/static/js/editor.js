const editorState = {
  activePageIndex: 0,
  selectedTextObjectId: null,
  tool: "select",
  lastChapterId: null,
};
window.editorState = editorState;

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

function currentTextObject() {
  if (!editorState.selectedTextObjectId) return null;
  return findTextObject(editorState.activePageIndex, editorState.selectedTextObjectId);
}
window.currentTextObject = currentTextObject;

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
window.addTextObjectBox = addTextObjectBox;

async function associateTextObjectOcr(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const langEl = document.getElementById("lang-select");
  const lang = langEl ? langEl.value : "ja";
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
  window._editorOverlayResizeObserver?.disconnect();
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
  };

  if (img.complete && img.naturalWidth > 0) render();
  else img.onload = render;
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => {
      if (!img.isConnected) observer.disconnect();
      else render();
    });
    observer.observe(img);
    window._editorOverlayResizeObserver = observer;
  }
}

function renderEditorPanel(pageIndex) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return;
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  panelHost.innerHTML = "";
  const panel = document.createElement("div");
  panel.className = "text-editor-panel";
  panel.dataset.pageIndex = String(pageIndex);

  const obj = findTextObject(pageIndex, editorState.selectedTextObjectId);
  if (!obj) {
    const section = document.createElement("section");
    section.className = "inspector-section editor-page-summary";
    const title = document.createElement("h3");
    title.textContent = "Trạng thái trang";
    const pageStatus = document.createElement("p");
    pageStatus.className = "editor-page-status";
    const objects = page?.text_objects || [];
    const translated = objects.filter((item) => item?.translation?.trim()).length;
    pageStatus.textContent = `${objects.length} vùng chữ · ${translated}/${objects.length} đã dịch`;
    const empty = document.createElement("div");
    empty.className = "ui-empty-state text-editor-empty";
    empty.textContent = "Chọn một vùng chữ trên ảnh để chỉnh sửa.";
    section.append(title, pageStatus, empty);
    panel.appendChild(section);
    panelHost.appendChild(panel);
    return;
  }
  panel.dataset.objectId = obj.id;

  const ocrLabel = document.createElement("label");
  ocrLabel.className = "ui-field text-editor-field";
  ocrLabel.textContent = "Nội dung gốc / OCR";

  const ocrTa = document.createElement("textarea");
  ocrTa.className = "ui-textarea text-editor-textarea ocr-textarea";
  ocrTa.dataset.textObjectId = obj.id;
  ocrTa.rows = 4;
  ocrTa.placeholder = "Nhập hoặc hiệu chỉnh nội dung gốc…";
  ocrTa.value = obj.ocr_text || "";
  ocrTa.addEventListener("input", () => {
    obj.ocr_text = ocrTa.value;
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  const trLabel = document.createElement("label");
  trLabel.className = "ui-field text-editor-field";
  trLabel.textContent = "Bản dịch";

  const trTa = document.createElement("textarea");
  trTa.className = "ui-textarea text-editor-textarea translation-textarea";
  trTa.dataset.textObjectId = obj.id;
  trTa.rows = 4;
  trTa.placeholder = "Nhập nội dung bản dịch…";
  trTa.value = obj.translation || "";
  trTa.addEventListener("input", () => {
    obj.translation = trTa.value;
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  ocrLabel.appendChild(ocrTa);
  trLabel.appendChild(trTa);

  const textBody = buildPanelSection(panel, "Text", true);
  textBody.append(ocrLabel, trLabel);

  const typographyBody = buildPanelSection(panel, "Typography", false);
  buildTextSection(typographyBody, panel, obj, pageIndex);

  const appearanceBody = buildPanelSection(panel, "Appearance", false);
  buildAppearanceSection(appearanceBody, panel, obj, pageIndex);

  buildBackgroundSection(appearanceBody, panel, obj, pageIndex);

  const geometryBody = buildPanelSection(panel, "Geometry", false);
  buildGeometryControls(geometryBody, obj, pageIndex);

  const actions = buildPanelSection(panel, "Actions", false);
  actions.classList.add("text-object-actions");

  const ocrBtn = document.createElement("button");
  ocrBtn.type = "button";
  ocrBtn.className = "ui-btn ui-btn-ghost text-object-action-btn";
  ocrBtn.textContent = "Nhận dạng lại bằng OCR";
  ocrBtn.addEventListener("click", () => {
    associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
      showToast("Không thể nhận dạng lại bằng OCR: " + err.message, "info");
    });
  });

  const dupBtn = document.createElement("button");
  dupBtn.type = "button";
  dupBtn.className = "ui-btn ui-btn-ghost text-object-action-btn";
  dupBtn.textContent = "Nhân đôi";
  dupBtn.title = "Tạo bản sao vùng chữ này";
  dupBtn.addEventListener("click", () => {
    duplicateTextObject(pageIndex, obj.id).catch((err) => {
      showToast("Không thể nhân đôi vùng chữ: " + err.message, "error");
    });
  });

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "ui-btn ui-btn-danger text-object-action-btn";
  delBtn.textContent = "Xóa";
  delBtn.title = "Xóa vùng chữ";
  delBtn.addEventListener("click", () => {
    if (delBtn.dataset.armed !== "1") {
      delBtn.dataset.armed = "1";
      delBtn.textContent = "Xác nhận xóa";
      return;
    }
    deleteTextObject(pageIndex, obj.id).catch((err) => {
      showToast("Không thể xóa vùng chữ: " + err.message, "error");
    });
  });

  actions.append(ocrBtn, dupBtn, delBtn);
  panel.appendChild(actions);

  syncStyleDataset(panel, obj.style);
  panelHost.appendChild(panel);
}

function buildPanelSection(panel, title, open) {
  const section = document.createElement("details");
  section.className = "ui-disclosure inspector-section text-editor-section";
  section.open = open;

  const header = document.createElement("summary");
  header.textContent = title;

  const body = document.createElement("div");
  body.className = "text-editor-section-body";

  section.append(header, body);
  panel.appendChild(section);
  return body;
}

function syncStyleDataset(panel, style) {
  panel.dataset.font = style.font || "default";
  panel.dataset.fontSize = style.fontSize || "auto";
  panel.dataset.bold = String(style.bold === true);
  panel.dataset.color = style.color || "auto";
  panel.dataset.strokeWidth = style.strokeWidth || "auto";
  panel.dataset.strokeColor = style.strokeColor || "auto";
  panel.dataset.bgColor = style.bgColor || "transparent";
  panel.dataset.cornerRadius = style.cornerRadius || "0";
  panel.dataset.horizontalAlign = style.horizontalAlign || "center";
  panel.dataset.verticalAlign = style.verticalAlign || "middle";
}

function buildTextSection(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const fontToolbar = document.createElement("div");
  fontToolbar.className = "ui-control-row font-style-toolbar";

  const fontSelect = document.createElement("select");
  fontSelect.className = "ui-select";
  fontSelect.title = "Chọn kiểu chữ";
  const fonts = availableFonts || [];
  if (fonts.length === 0) {
    fontSelect.innerHTML = '<option value="default">Mặc định (Comic)</option>';
  } else {
    fonts.forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f.id;
      opt.textContent = f.name;
      fontSelect.appendChild(opt);
    });
  }
  fontSelect.value = style.font || "default";
  fontSelect.addEventListener("change", () => {
    style.font = fontSelect.value;
    panel.dataset.font = fontSelect.value;
    schedule();
  });
  fontToolbar.appendChild(fontSelect);

  const boldBtn = document.createElement("button");
  boldBtn.type = "button";
  boldBtn.className = "ui-icon-btn ui-btn-ghost ui-btn-compact bold-toggle-btn";
  boldBtn.setAttribute("aria-label", "In đậm chữ");
  boldBtn.appendChild(window.createUiIcon("bold"));
  boldBtn.title = "In đậm chữ";
  if (style.bold === true) boldBtn.classList.add("active");
  boldBtn.addEventListener("click", () => {
    const next = !(style.bold === true);
    style.bold = next;
    panel.dataset.bold = String(next);
    boldBtn.classList.toggle("active", next);
    schedule();
  });
  fontToolbar.appendChild(boldBtn);

  const sizeGroup = document.createElement("div");
  sizeGroup.className = "ui-control-row font-size-group";
  const sizeLabel = document.createElement("span");
  sizeLabel.className = "ui-control-label";
  sizeLabel.textContent = "Kích thước:";
  const autoBtn = document.createElement("button");
  autoBtn.type = "button";
  autoBtn.className = "ui-btn ui-btn-ghost ui-btn-compact size-auto-btn";
  autoBtn.textContent = "Tự động";
  autoBtn.title = "Tự động vừa vùng";
  const sizeSlider = document.createElement("input");
  sizeSlider.type = "range";
  sizeSlider.className = "font-size-slider";
  sizeSlider.min = "10";
  sizeSlider.max = "60";
  sizeSlider.value = "20";
  const sizeValSpan = document.createElement("span");
  sizeValSpan.className = "ui-value";
  const sizeIsAuto = !style.fontSize || style.fontSize === "auto";
  if (sizeIsAuto) {
    autoBtn.classList.add("selected");
    sizeSlider.disabled = true;
    sizeValSpan.textContent = "Tự động";
  } else {
    sizeSlider.value = style.fontSize;
    sizeValSpan.textContent = style.fontSize + "px";
  }
  autoBtn.addEventListener("click", () => {
    const isAuto = !style.fontSize || style.fontSize === "auto";
    if (isAuto) {
      autoBtn.classList.remove("selected");
      sizeSlider.disabled = false;
      style.fontSize = String(sizeSlider.value);
      panel.dataset.fontSize = style.fontSize;
      sizeValSpan.textContent = style.fontSize + "px";
    } else {
      autoBtn.classList.add("selected");
      sizeSlider.disabled = true;
      style.fontSize = "auto";
      panel.dataset.fontSize = "auto";
      sizeValSpan.textContent = "Tự động";
    }
    schedule();
  });
  sizeSlider.addEventListener("input", () => {
    if (!style.fontSize || style.fontSize === "auto") return;
    style.fontSize = sizeSlider.value;
    panel.dataset.fontSize = style.fontSize;
    sizeValSpan.textContent = style.fontSize + "px";
    schedule();
  });
  sizeGroup.append(sizeLabel, autoBtn, sizeSlider, sizeValSpan);
  fontToolbar.appendChild(sizeGroup);

  body.appendChild(fontToolbar);
  buildAlignmentControls(body, panel, obj, pageIndex);
}

function buildAppearanceSection(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const colorToolbar = document.createElement("div");
  colorToolbar.className = "ui-control-row color-toolbar";
  const colorLabel = document.createElement("span");
  colorLabel.className = "ui-control-label";
  colorLabel.textContent = "Màu chữ";
  colorToolbar.appendChild(colorLabel);

  const colors = [
    { name: "Tự động tương phản", value: "auto", bg: "linear-gradient(135deg, #000 50%, #fff 50%)" },
    { name: "Trắng", value: "#ffffff", bg: "#ffffff" },
    { name: "Đen", value: "#000000", bg: "#000000" },
    { name: "Đỏ", value: "#e8432c", bg: "#e8432c" },
    { name: "Vàng", value: "#f1c40f", bg: "#f1c40f" },
  ];
  colors.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-swatch color-btn";
    btn.setAttribute("aria-pressed", style.color === c.value ? "true" : "false");
    btn.title = c.name;
    btn.style.background = c.bg;
    btn.addEventListener("click", () => {
      colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      style.color = c.value;
      panel.dataset.color = c.value;
      schedule();
    });
    colorToolbar.appendChild(btn);
  });

  const customPicker = document.createElement("input");
  customPicker.type = "color";
  customPicker.className = "ui-color-input custom-color-picker";
  customPicker.value = "#ffffff";
  customPicker.title = "Chọn màu tùy chỉnh";
  if (style.color && style.color !== "auto" && !colors.some((c) => c.value === style.color)) {
    customPicker.value = style.color;
  }
  customPicker.addEventListener("input", () => {
    colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
    style.color = customPicker.value;
    panel.dataset.color = customPicker.value;
    schedule();
  });
  colorToolbar.appendChild(customPicker);

  const strokeToolbar = document.createElement("div");
  strokeToolbar.className = "ui-control-row stroke-toolbar";
  const strokeLabel = document.createElement("span");
  strokeLabel.className = "ui-control-label";
  strokeLabel.textContent = "Viền chữ";
  const strokeSlider = document.createElement("input");
  strokeSlider.type = "range";
  strokeSlider.className = "stroke-width-slider";
  strokeSlider.min = "0";
  strokeSlider.max = "8";
  strokeSlider.value = "2";
  strokeSlider.title = "Độ dày viền chữ";
  const strokeValSpan = document.createElement("span");
  strokeValSpan.className = "ui-value";
  const strokeIsAuto = !style.strokeWidth || style.strokeWidth === "auto";
  if (strokeIsAuto) {
    strokeValSpan.textContent = "Tự động";
  } else {
    strokeSlider.value = style.strokeWidth;
    strokeValSpan.textContent = style.strokeWidth + "px";
  }
  const strokeColorPicker = document.createElement("input");
  strokeColorPicker.type = "color";
  strokeColorPicker.className = "ui-color-input stroke-color-picker";
  strokeColorPicker.value = (style.strokeColor && style.strokeColor !== "auto") ? style.strokeColor : "#000000";
  strokeColorPicker.title = "Màu viền chữ";
  strokeSlider.addEventListener("input", () => {
    style.strokeWidth = strokeSlider.value;
    panel.dataset.strokeWidth = strokeSlider.value;
    strokeValSpan.textContent = strokeSlider.value + "px";
    schedule();
  });
  strokeColorPicker.addEventListener("input", () => {
    style.strokeColor = strokeColorPicker.value;
    panel.dataset.strokeColor = strokeColorPicker.value;
    schedule();
  });
  strokeToolbar.append(strokeLabel, strokeSlider, strokeValSpan, strokeColorPicker);

  body.append(colorToolbar, strokeToolbar);
}

function buildBackgroundSection(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const bgToolbar = document.createElement("div");
  bgToolbar.className = "ui-control-row bg-toolbar";

  const toggleId = "bg-toggle-" + obj.id;
  const bgToggle = document.createElement("input");
  bgToggle.type = "checkbox";
  bgToggle.id = toggleId;
  bgToggle.className = "bg-toggle-checkbox";
  bgToggle.checked = !!(style.bgColor && style.bgColor !== "transparent");
  const toggleLabel = document.createElement("label");
  toggleLabel.htmlFor = toggleId;
  toggleLabel.className = "bg-toggle-label";
  toggleLabel.textContent = "Nền";

  const bgSelect = document.createElement("select");
  bgSelect.className = "ui-select";
  const bgColors = ["#ffffff", "#000000"];
  if (style.bgColor && style.bgColor !== "transparent" && !bgColors.includes(style.bgColor)) {
    bgColors.unshift(style.bgColor);
  }
  bgColors.forEach((val) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = val === "#ffffff" ? "Trắng" : val === "#000000" ? "Đen" : val;
    bgSelect.appendChild(opt);
  });
  bgSelect.value = (style.bgColor && style.bgColor !== "transparent") ? style.bgColor : "#ffffff";

  const radiusSlider = document.createElement("input");
  radiusSlider.type = "range";
  radiusSlider.className = "corner-radius-slider";
  radiusSlider.min = "0";
  radiusSlider.max = "20";
  radiusSlider.value = style.cornerRadius || "0";
  radiusSlider.title = "Độ bo góc nền";
  const radiusValSpan = document.createElement("span");
  radiusValSpan.className = "ui-value";
  radiusValSpan.textContent = (style.cornerRadius || "0") + "px";

  const updateEnabled = () => {
    bgSelect.disabled = !bgToggle.checked;
    radiusSlider.disabled = !bgToggle.checked;
  };

  bgToggle.addEventListener("change", () => {
    style.bgColor = bgToggle.checked ? bgSelect.value : "transparent";
    panel.dataset.bgColor = style.bgColor;
    updateEnabled();
    schedule();
  });
  bgSelect.addEventListener("change", () => {
    style.bgColor = bgSelect.value;
    panel.dataset.bgColor = bgSelect.value;
    schedule();
  });
  radiusSlider.addEventListener("input", () => {
    style.cornerRadius = radiusSlider.value;
    panel.dataset.cornerRadius = radiusSlider.value;
    radiusValSpan.textContent = radiusSlider.value + "px";
    schedule();
  });

  updateEnabled();
  bgToolbar.append(toggleLabel, bgToggle, bgSelect, radiusSlider, radiusValSpan);
  body.appendChild(bgToolbar);
}

const TEXT_OBJECT_MIN_SIZE = 10;

function getPageImageSize(pageIndex) {
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  if (page && page.width && page.height) return { w: page.width, h: page.height };
  const img = document.querySelector(".translation-canvas-host .page-image-wrap img");
  if (img && img.naturalWidth) return { w: img.naturalWidth, h: img.naturalHeight };
  return { w: Infinity, h: Infinity };
}

function buildGeometryControls(panel, obj, pageIndex) {
  const { w: W, h: H } = getPageImageSize(pageIndex);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const grid = document.createElement("div");
  grid.className = "geometry-grid";
  grid.dataset.geometryFor = obj.id;

  const mkField = (label, get, commit) => {
    const wrap = document.createElement("div");
    wrap.className = "ui-field geometry-field";
    const lbl = document.createElement("span");
    lbl.className = "geometry-field-label";
    lbl.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.className = "ui-input geometry-input";
    inp.dataset.geometryField = label;
    const refresh = () => { inp.value = String(get()); };
    refresh();
    inp.addEventListener("change", () => {
      const raw = parseInt(inp.value, 10);
      if (!Number.isFinite(raw)) { refresh(); return; }
      commit(raw);
      refresh();
      if (typeof window.syncOverlayForObject === "function") window.syncOverlayForObject(pageIndex, obj.id);
      if (typeof window.scheduleGeomPersist === "function") window.scheduleGeomPersist(pageIndex, obj.id);
    });
    wrap.append(lbl, inp);
    return { wrap, refresh };
  };

  const fx = mkField("X", () => obj.region.x1, (v) => {
    const w = obj.region.x2 - obj.region.x1;
    obj.region.x1 = clamp(Math.round(v), 0, W - w);
    obj.region.x2 = obj.region.x1 + w;
  });
  const fy = mkField("Y", () => obj.region.y1, (v) => {
    const h = obj.region.y2 - obj.region.y1;
    obj.region.y1 = clamp(Math.round(v), 0, H - h);
    obj.region.y2 = obj.region.y1 + h;
  });
  const fw = mkField("W", () => obj.region.x2 - obj.region.x1, (v) => {
    const newW = clamp(Math.round(v), TEXT_OBJECT_MIN_SIZE, W - obj.region.x1);
    obj.region.x2 = obj.region.x1 + newW;
  });
  const fh = mkField("H", () => obj.region.y2 - obj.region.y1, (v) => {
    const newH = clamp(Math.round(v), TEXT_OBJECT_MIN_SIZE, H - obj.region.y1);
    obj.region.y2 = obj.region.y1 + newH;
  });

  grid.append(fx.wrap, fy.wrap, fw.wrap, fh.wrap);
  panel.appendChild(grid);
}

function refreshGeometryControls(pageIndex, id) {
  const grid = document.querySelector(`.geometry-grid[data-geometry-for="${id}"]`);
  const obj = findTextObject(pageIndex, id);
  if (!grid || !obj || !obj.region) return;
  const r = obj.region;
  const set = (field, val) => {
    const inp = grid.querySelector(`input[data-geometry-field="${field}"]`);
    if (inp) inp.value = String(val);
  };
  set("X", r.x1);
  set("Y", r.y1);
  set("W", r.x2 - r.x1);
  set("H", r.y2 - r.y1);
}
window.refreshGeometryControls = refreshGeometryControls;

function buildAlignmentControls(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const mkGroup = (label, options, key) => {
    const group = document.createElement("div");
    group.className = "align-group";
    const lbl = document.createElement("span");
    lbl.className = "style-group-label";
    lbl.textContent = label;
    const row = document.createElement("div");
    row.className = "align-btn-row";
    options.forEach(([val, txt]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ui-btn ui-btn-ghost ui-btn-compact align-btn" + ((style[key] || DEFAULT_TEXT_OBJECT_STYLE[key]) === val ? " selected" : "");
      b.textContent = txt;
      b.addEventListener("click", () => {
        style[key] = val;
        panel.dataset[key] = val;
        row.querySelectorAll(".align-btn").forEach((x) => x.classList.remove("selected"));
        b.classList.add("selected");
        schedule();
      });
      row.appendChild(b);
    });
    group.append(lbl, row);
    return group;
  };

  body.append(
    mkGroup("Căn ngang:", [["left", "Trái"], ["center", "Giữa"], ["right", "Phải"]], "horizontalAlign"),
    mkGroup("Căn dọc:", [["top", "Trên"], ["middle", "Giữa"], ["bottom", "Dưới"]], "verticalAlign"),
  );
}

let _currentSaveStatus = "saved";
let _textSaving = 0;
let _textHasError = false;

function setSaveStatusContent(el, status) {
  const copy = {
    saved: ["check", "Đã lưu"],
    dirty: ["unsaved", "Chưa lưu"],
    saving: ["spinner", "Đang lưu…"],
    error: ["alert", "Lưu thất bại"],
  }[status];
  if (!copy) return;
  const [icon, text] = copy;
  el.replaceChildren(
    window.createUiIcon(icon, icon === "spinner" ? "ui-status-icon ui-icon-spinner" : "ui-status-icon"),
    document.createTextNode(text),
  );
}

function updateSaveStatus(status) {
  _currentSaveStatus = status;
  const statusEls = document.querySelectorAll(".editor-save-status");
  statusEls.forEach((el) => {
    el.className = `editor-save-status save-status-${status}`;
    if (status === "saved") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Tất cả thay đổi đã được lưu");
    } else if (status === "dirty") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Có thay đổi chưa lưu");
    } else if (status === "saving") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Đang lưu thay đổi");
    } else if (status === "error") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Lưu thay đổi thất bại");
    }
  });
}
window.updateSaveStatus = updateSaveStatus;

function refreshSaveStatus() {
  let status = "saved";
  const hasGeomDirty = typeof window.hasPendingGeom === "function" ? window.hasPendingGeom() : false;
  const isGeomSaving = typeof window.isGeomSaving === "function" ? window.isGeomSaving() : false;
  const hasGeomError = typeof window.hasGeomError === "function" ? window.hasGeomError() : false;

  if (_textHasError || hasGeomError) {
    status = "error";
  } else if (_textSaving > 0 || isGeomSaving) {
    status = "saving";
  } else if (_textDirty.size > 0 || hasGeomDirty) {
    status = "dirty";
  } else {
    status = "saved";
  }
  updateSaveStatus(status);
  return status;
}
window.refreshSaveStatus = refreshSaveStatus;

const _textDirty = new Map();
let _textTimer = null;

function _captureTextState(obj) {
  return {
    ocr_text: obj.ocr_text != null ? obj.ocr_text : "",
    translation: obj.translation != null ? obj.translation : "",
    style: obj.style
      ? JSON.parse(JSON.stringify(obj.style))
      : JSON.parse(JSON.stringify(DEFAULT_TEXT_OBJECT_STYLE)),
  };
}

function scheduleTextObjectPersist(pageIndex, id) {
  const obj = findTextObject(pageIndex, id);
  if (!obj) return;
  _textDirty.set(`${pageIndex}:${id}`, Object.assign({ pageIndex, id }, _captureTextState(obj)));
  _textHasError = false;
  refreshSaveStatus();
  clearTimeout(_textTimer);
  _textTimer = setTimeout(() => { flushTextObjectPersist().catch(() => {}); }, 800);
}
window.scheduleTextObjectPersist = scheduleTextObjectPersist;

async function _persistTextObjectsBulk(chapterId, items) {
  // One request for the whole batch; the server applies every patch under a
  // single manifest transaction instead of one full rewrite per object.
  const resp = await fetch("/api/text_object/update_bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chapter_id: chapterId,
      updates: items.map((p) => ({
        page_index: p.pageIndex,
        id: p.id,
        ocr_text: p.ocr_text,
        translation: p.translation,
        style: p.style,
      })),
    }),
  });
  if (!resp.ok) {
    throw new Error(getErrorMessage(resp.status, await parseApiResponse(resp)));
  }
  return "bulk";
}

async function _persistTextObjectsIndividually(chapterId, items) {
  const failures = [];
  await Promise.all(items.map(async (p) => {
    const obj = findTextObject(p.pageIndex, p.id);
    if (!obj) return;
    try {
      await apiTextObject("update", {
        chapter_id: chapterId,
        page_index: p.pageIndex,
        id: p.id,
        ocr_text: p.ocr_text,
        translation: p.translation,
        style: p.style,
      });
    } catch (err) {
      failures.push(err);
      if (chapterId === currentChapterId) {
        _textHasError = true;
        _textDirty.set(`${p.pageIndex}:${p.id}`, Object.assign({ pageIndex: p.pageIndex, id: p.id }, _captureTextState(obj)));
      }
    }
  }));
  return failures;
}

async function flushTextObjectPersist(pageIndex) {
  clearTimeout(_textTimer);
  _textTimer = null;
  if (_textDirty.size === 0) return;
  const items = [];
  _textDirty.forEach((v) => {
    if (pageIndex === undefined || v.pageIndex === pageIndex) items.push(v);
  });
  if (items.length === 0) return;
  const chapterId = currentChapterId;
  if (!chapterId) return;
  items.forEach((v) => _textDirty.delete(`${v.pageIndex}:${v.id}`));
  _textSaving += items.length;
  refreshSaveStatus();
  let failures = [];
  try {
    if (items.length === 1) {
      failures = await _persistTextObjectsIndividually(chapterId, items);
    } else {
      try {
        await _persistTextObjectsBulk(chapterId, items);
      } catch (bulkErr) {
        // Older servers (or a rejected payload) still get a working save.
        console.warn("Bulk text-object save failed; falling back per object:", bulkErr);
        failures = await _persistTextObjectsIndividually(chapterId, items);
      }
    }
  } finally {
    _textSaving -= items.length;
    refreshSaveStatus();
  }
  if (chapterId !== currentChapterId) return;
  if (failures.length) {
    showToast("Không lưu được nội dung: " + failures[0].message, "error");
    throw new Error("Không lưu được nội dung");
  }
}
window.flushTextObjectPersist = flushTextObjectPersist;

function cancelTextObjectPersist() {
  clearTimeout(_textTimer);
  _textTimer = null;
}

window.removePendingPersist = function removePendingPersist(pageIndex, id) {
  _textDirty.delete(`${pageIndex}:${id}`);
  if (typeof window.removePendingGeom === "function") window.removePendingGeom(pageIndex, id);
  refreshSaveStatus();
};

window.flushAllPendingPersists = async function flushAllPendingPersists(pageIndex) {
  const jobs = [flushTextObjectPersist(pageIndex)];
  if (typeof window.flushGeomPersist === "function") jobs.push(window.flushGeomPersist(pageIndex));
  await Promise.all(jobs);
};

window.cancelPendingPersist = function cancelPendingPersist() {
  cancelTextObjectPersist();
  _textDirty.clear();
  _textHasError = false;
  if (typeof window.cancelGeomPersist === "function") window.cancelGeomPersist();
  if (typeof window.clearPendingGeom === "function") window.clearPendingGeom();
  refreshSaveStatus();
};

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

  imgWrap.addEventListener("mousedown", onDown);
  imgWrap.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  imgWrap.addEventListener("click", onClick);

  window._editorDrawCleanup = function cleanupEditorDraw() {
    window.removeEventListener("mouseup", onUp);
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

let _editorSwitchingPage = false;
let _editorPendingSwitchIndex = null;

async function switchEditorPage(newIndex) {
  const chapterId = currentChapterId;
  const pages = currentManifest ? currentManifest.pages : null;
  if (!pages || newIndex < 0 || newIndex >= pages.length) return;

  if (_editorSwitchingPage) {
    _editorPendingSwitchIndex = newIndex;
    return;
  }
  _editorSwitchingPage = true;
  _editorPendingSwitchIndex = null;

  try {
    try {
      await Promise.race([
        flushAllPendingPersists(),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout lưu dữ liệu")), 3000)),
      ]);
    } catch (err) {
      console.warn("Lưu dữ liệu trước khi chuyển trang bị cảnh báo/lỗi:", err);
    }
    if (chapterId !== currentChapterId) return;
    editorState.activePageIndex = newIndex;
    editorState.selectedTextObjectId = null;
    renderEditor();
  } finally {
    _editorSwitchingPage = false;
    if (_editorPendingSwitchIndex !== null && _editorPendingSwitchIndex !== editorState.activePageIndex) {
      const nextTarget = _editorPendingSwitchIndex;
      _editorPendingSwitchIndex = null;
      void switchEditorPage(nextTarget);
    }
  }
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
  // Key the preview URL on the committed render revision when the server
  // reported one; only fall back to a clock bust when it is unavailable.
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

// Guards the one-way auto-sync from renderEditor against re-entry: while a sync is
// in flight for a page we must not start another from a nested renderEditor call,
// or a fast local server turns that into an unbounded rebuild cascade that blocks the
// main thread and makes the tab report "not responding".
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
  // Match the backend ownership rule exactly: an empty value can be a deliberate
  // user edit, so only the last machine-owned value may be advanced automatically.
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
  // Keep this eligibility predicate aligned with ensure_page_text_objects().
  // A box the server intentionally ignores must never request another ensure.
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

  const target = document.createElement("select");
  target.className = "chapter-translate-target";
  target.setAttribute("aria-label", "Ngôn ngữ bản dịch");
  [
    ["vi", "Tiếng Việt"],
    ["en", "English"],
  ].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    target.appendChild(option);
  });

  const budget = document.createElement("input");
  budget.type = "number";
  budget.min = "0.001";
  budget.max = "0.25";
  budget.step = "0.001";
  budget.value = "0.02";
  budget.className = "chapter-translate-budget";
  budget.title = "Ngân sách tối đa ước tính cho lần dịch chương (USD)";
  budget.setAttribute("aria-label", "Ngân sách dịch chương bằng USD");
  const provider = document.createElement("select");
  provider.className = "chapter-translate-provider";
  [["deepseek", "DeepSeek"], ["openai", "OpenAI"], ["openrouter", "OpenRouter"], ["experiential", "Experiential Labs"]].forEach(([value, label]) => provider.add(new Option(label, value)));
  provider.value = localStorage.getItem("manga_translation_provider") || "deepseek";
  const model = document.createElement("input");
  model.className = "ui-input chapter-translate-model";
  model.placeholder = "Model mặc định của provider";
  const syncModel = () => {
    model.value = localStorage.getItem(`manga_ai_model_${provider.value}`) || "";
    localStorage.setItem("manga_translation_provider", provider.value);
    budget.disabled = provider.value !== "deepseek";
    budgetLabel.hidden = provider.value !== "deepseek";
  };
  provider.addEventListener("change", syncModel);
  model.addEventListener("change", () => localStorage.setItem(`manga_ai_model_${provider.value}`, model.value.trim()));
  const providerLabel = document.createElement("label");
  providerLabel.className = "ui-field";
  providerLabel.textContent = "Dịch vụ AI";
  providerLabel.appendChild(provider);
  const modelLabel = document.createElement("label");
  modelLabel.className = "ui-field";
  modelLabel.textContent = "Model";
  modelLabel.appendChild(model);
  const targetLabel = document.createElement("label");
  targetLabel.className = "ui-field";
  targetLabel.textContent = "Dịch sang";
  targetLabel.appendChild(target);
  const budgetLabel = document.createElement("label");
  budgetLabel.className = "ui-field";
  budgetLabel.textContent = "Giới hạn chi phí (USD)";
  budgetLabel.appendChild(budget);

  const run = document.createElement("button");
  run.type = "button";
  run.className = "ui-btn ui-btn-primary chapter-translate-run";
  run.textContent = "Dịch tự động";

  run.addEventListener("click", async () => {
    const chapterId = window.currentChapterId;
    if (!chapterId) return;
    run.disabled = true;
    run.textContent = "Đang dịch…";
    summary.textContent = "Đang dịch…";
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      } else if (typeof window.flushTextObjectPersist === "function") {
        await window.flushTextObjectPersist();
      }
      if (chapterId !== window.currentChapterId) return;

      const response = await fetch("/api/translate/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          source_lang: document.getElementById("lang-select")?.value || "ja",
          target_lang: target.value,
          provider: provider.value,
          model: model.value.trim() || null,
          budget_usd: Number(budget.value || 0.02),
          force: false,
        }),
      });
      const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
      const data = await parse(response);
      if (!response.ok) {
        const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d?.detail || `HTTP ${s}`;
        throw new Error(getErr(response.status, data));
      }
      if (chapterId !== window.currentChapterId) return;
      window.currentManifest = data;
      const info = data.translation_run || {};
      const cost = Number(info.estimated_cost_usd || 0).toFixed(4);
      if (typeof window.showToast === "function") {
        window.showToast(`Đã dịch ${info.translated || 0} vùng · chi phí ~$${cost}`, "info");
      }
      renderEditor();
    } catch (err) {
      if (typeof window.showToast === "function") {
        window.showToast("Dịch tự động thất bại: " + err.message, "error");
      }
    } finally {
      run.disabled = false;
      run.textContent = "Dịch tự động";
      summary.textContent = "Dịch tự động";
    }
  });

  options.append(providerLabel, modelLabel, targetLabel, budgetLabel, run);
  syncModel();
  controls.append(summary, options);
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

function renderEditor() {
  const container = document.getElementById("page-view");
  if (!container) return;
  // Rendering must never discard debounced user edits.  Persistence is flushed
  // explicitly on navigation and actions instead of being reset per frame.
  if (!currentManifest || !currentManifest.pages || currentManifest.pages.length === 0) return;

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
          // User navigated away while the sync was in flight; let the next
          // renderEditor for the new page pick up its own auto-sync.
          _autoSyncingPageIndex = null;
          return;
        }
        // Do not rebuild solely because ensure answered successfully.  If the
        // backend made no change, rendering again would repeat the same request
        // forever for an ineligible or deliberately user-edited object.
        if (!manifest || !_autoSyncChangedPage(manifest, pageIndex)) {
          _autoSyncingPageIndex = null;
          return;
        }
        // Let the current render finish before applying the one real manifest
        // change returned by the server.
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
