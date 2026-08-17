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
}

async function createTextObject(pageIndex, shape, region) {
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  const manifest = await apiTextObject("create", {
    chapter_id: currentChapterId,
    page_index: pageIndex,
    shape,
    region,
  });
  const page = manifest.pages[pageIndex];
  const objs = page.text_objects || [];
  const obj = objs[objs.length - 1];
  if (!obj) throw new Error("Không tạo được text object");
  editorState.selectedTextObjectId = obj.id;
  applyManifestResponse(manifest, pageIndex, { id: obj.id });
  associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
    showToast("Không nhóm được OCR tự động, bạn vẫn có thể nhập tay: " + err.message, "info");
  });
}

async function deleteTextObject(pageIndex, id) {
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  const manifest = await apiTextObject("delete", {
    chapter_id: currentChapterId,
    page_index: pageIndex,
    id,
  });
  if (editorState.selectedTextObjectId === id) editorState.selectedTextObjectId = null;
  if (typeof window.removePendingPersist === "function") window.removePendingPersist(pageIndex, id);
  applyManifestResponse(manifest, pageIndex, { id });
}
window.deleteTextObject = deleteTextObject;

const DUPLICATE_OFFSET = 24;

async function duplicateTextObject(pageIndex, id) {
  const obj = findTextObject(pageIndex, id);
  if (!obj) throw new Error("Không tìm thấy text object");
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  const { w: W, h: H } = getPageImageSize(pageIndex);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const bw = obj.region.x2 - obj.region.x1;
  const bh = obj.region.y2 - obj.region.y1;
  const x1 = clamp(obj.region.x1 + DUPLICATE_OFFSET, 0, Math.max(0, W - bw));
  const y1 = clamp(obj.region.y1 + DUPLICATE_OFFSET, 0, Math.max(0, H - bh));

  const manifest = await apiTextObject("create", {
    chapter_id: currentChapterId,
    page_index: pageIndex,
    shape: obj.shape,
    region: { x1, y1, x2: x1 + bw, y2: y1 + bh },
  });
  const objs = manifest.pages[pageIndex].text_objects || [];
  const created = objs[objs.length - 1];
  if (!created) throw new Error("Không nhân đôi được text object");

  const updated = await apiTextObject("update", {
    chapter_id: currentChapterId,
    page_index: pageIndex,
    id: created.id,
    ocr_text: obj.ocr_text || "",
    translation: obj.translation || "",
    style: JSON.parse(JSON.stringify(obj.style || DEFAULT_TEXT_OBJECT_STYLE)),
  });
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
  const langEl = document.getElementById("lang-select");
  const lang = langEl ? langEl.value : "ja";
  const snapshot = collectPanelState(pageIndex, id);
  const manifest = await apiTextObject("ocr", {
    chapter_id: currentChapterId,
    page_index: pageIndex,
    id,
    lang,
  });
  if (!findTextObject(pageIndex, id)) return;
  applyManifestResponse(manifest, pageIndex, { skipOverlays: true, snapshot, id });
}
window.associateTextObjectOcr = associateTextObjectOcr;

function setEditorTool(tool) {
  editorState.tool = tool;
  document.querySelectorAll(".editor-tool-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tool === tool);
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
}
window.setSelectedTextObject = setSelectedTextObject;

function clearSelectedTextObject() {
  editorState.selectedTextObjectId = null;
  document.querySelectorAll(".text-object-overlay").forEach((el) => el.classList.remove("selected"));
  renderEditorPanel(editorState.activePageIndex);
}
window.clearSelectedTextObject = clearSelectedTextObject;

function renderTextObjectOverlays(pageIndex, page) {
  const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
  if (!wrapper) return;
  const imgWrap = wrapper.querySelector(".page-image-wrap");
  if (!imgWrap) return;
  const img = imgWrap.querySelector("img");
  imgWrap.querySelectorAll(".text-object-overlay:not(.drawing)").forEach((el) => el.remove());

  const render = () => {
    if (!img.naturalWidth || !img.clientWidth) return;
    const sx = img.clientWidth / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    (page.text_objects || []).forEach((obj) => {
      if (!obj || !obj.region) return;
      const overlay = document.createElement("div");
      overlay.className = "text-object-overlay" + (obj.shape === "ellipse" ? " ellipse" : "");
      overlay.dataset.pageIndex = String(pageIndex);
      overlay.dataset.objectId = obj.id;
      overlay.style.left = obj.region.x1 * sx + "px";
      overlay.style.top = obj.region.y1 * sy + "px";
      overlay.style.width = (obj.region.x2 - obj.region.x1) * sx + "px";
      overlay.style.height = (obj.region.y2 - obj.region.y1) * sy + "px";
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
    const empty = document.createElement("div");
    empty.className = "text-editor-empty";
    empty.textContent = "Select a text region on the image.";
    panel.appendChild(empty);
    panelHost.appendChild(panel);
    return;
  }
  panel.dataset.objectId = obj.id;

  const ocrLabel = document.createElement("label");
  ocrLabel.className = "text-editor-label";
  ocrLabel.textContent = "Chữ gốc / OCR";

  const ocrTa = document.createElement("textarea");
  ocrTa.className = "text-editor-textarea ocr-textarea";
  ocrTa.dataset.textObjectId = obj.id;
  ocrTa.rows = 4;
  ocrTa.placeholder = "Chữ gốc (sửa tay được)...";
  ocrTa.value = obj.ocr_text || "";
  ocrTa.addEventListener("input", () => {
    obj.ocr_text = ocrTa.value;
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  const trLabel = document.createElement("label");
  trLabel.className = "text-editor-label";
  trLabel.textContent = "Bản dịch";

  const trTa = document.createElement("textarea");
  trTa.className = "text-editor-textarea translation-textarea";
  trTa.dataset.textObjectId = obj.id;
  trTa.rows = 4;
  trTa.placeholder = "Nhập bản dịch...";
  trTa.value = obj.translation || "";
  trTa.addEventListener("input", () => {
    obj.translation = trTa.value;
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  panel.append(ocrLabel, ocrTa, trLabel, trTa);

  buildGeometryControls(panel, obj, pageIndex);

  const textBody = buildPanelSection(panel, "Văn bản", true);
  buildTextSection(textBody, panel, obj, pageIndex);

  const appearanceBody = buildPanelSection(panel, "Kiểu dáng", false);
  buildAppearanceSection(appearanceBody, panel, obj, pageIndex);

  const backgroundBody = buildPanelSection(panel, "Nền", false);
  buildBackgroundSection(backgroundBody, panel, obj, pageIndex);

  const actions = document.createElement("div");
  actions.className = "text-object-actions";

  const ocrBtn = document.createElement("button");
  ocrBtn.type = "button";
  ocrBtn.className = "text-object-action-btn";
  ocrBtn.textContent = "Nhóm OCR lại";
  ocrBtn.addEventListener("click", () => {
    associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
      showToast("Không nhóm được OCR: " + err.message, "info");
    });
  });

  const dupBtn = document.createElement("button");
  dupBtn.type = "button";
  dupBtn.className = "text-object-action-btn";
  dupBtn.textContent = "Nhân đôi";
  dupBtn.title = "Tạo bản sao vùng chữ này";
  dupBtn.addEventListener("click", () => {
    duplicateTextObject(pageIndex, obj.id).catch((err) => {
      showToast("Nhân đôi text object thất bại: " + err.message, "error");
    });
  });

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "text-object-action-btn danger";
  delBtn.textContent = "Xóa";
  delBtn.title = "Xóa text object";
  delBtn.addEventListener("click", () => {
    if (delBtn.dataset.armed !== "1") {
      delBtn.dataset.armed = "1";
      delBtn.classList.add("confirming");
      delBtn.textContent = "Chắc chắn xóa?";
      return;
    }
    deleteTextObject(pageIndex, obj.id).catch((err) => {
      showToast("Xóa text object thất bại: " + err.message, "error");
    });
  });

  actions.append(ocrBtn, dupBtn, delBtn);
  panel.appendChild(actions);

  syncStyleDataset(panel, obj.style);
  panelHost.appendChild(panel);
}

function buildPanelSection(panel, title, open) {
  const section = document.createElement("div");
  section.className = "text-editor-section" + (open ? " open" : "");

  const header = document.createElement("button");
  header.type = "button";
  header.className = "text-editor-section-header";
  header.setAttribute("aria-expanded", open ? "true" : "false");
  header.innerHTML = `<span>${title}</span><span class="section-caret" aria-hidden="true">▸</span>`;
  header.addEventListener("click", () => {
    const isOpen = section.classList.toggle("open");
    header.setAttribute("aria-expanded", isOpen ? "true" : "false");
  });

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
  fontToolbar.className = "font-style-toolbar";

  const fontSelect = document.createElement("select");
  fontSelect.className = "font-family-select";
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
  boldBtn.className = "bold-toggle-btn";
  boldBtn.textContent = "B";
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
  sizeGroup.className = "font-size-group";
  const sizeLabel = document.createElement("span");
  sizeLabel.className = "size-label";
  sizeLabel.textContent = "Cỡ:";
  const autoBtn = document.createElement("button");
  autoBtn.type = "button";
  autoBtn.className = "size-auto-btn";
  autoBtn.textContent = "Auto";
  autoBtn.title = "Tự động vừa ô";
  const sizeSlider = document.createElement("input");
  sizeSlider.type = "range";
  sizeSlider.className = "font-size-slider";
  sizeSlider.min = "10";
  sizeSlider.max = "60";
  sizeSlider.value = "20";
  const sizeValSpan = document.createElement("span");
  sizeValSpan.className = "font-size-val";
  const sizeIsAuto = !style.fontSize || style.fontSize === "auto";
  if (sizeIsAuto) {
    autoBtn.classList.add("selected");
    sizeSlider.disabled = true;
    sizeValSpan.textContent = "Auto";
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
      sizeValSpan.textContent = "Auto";
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
  colorToolbar.className = "color-toolbar";
  const colorLabel = document.createElement("span");
  colorLabel.className = "color-label";
  colorLabel.textContent = "Màu chữ:";
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
    btn.className = "color-btn" + (style.color === c.value ? " selected" : "");
    btn.title = c.name;
    btn.style.background = c.bg;
    btn.addEventListener("click", () => {
      colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      style.color = c.value;
      panel.dataset.color = c.value;
      schedule();
    });
    colorToolbar.appendChild(btn);
  });

  const customPicker = document.createElement("input");
  customPicker.type = "color";
  customPicker.className = "box-color-picker custom-color-picker";
  customPicker.value = "#ffffff";
  customPicker.title = "Chọn màu tùy chỉnh";
  if (style.color && style.color !== "auto" && !colors.some((c) => c.value === style.color)) {
    customPicker.value = style.color;
  }
  customPicker.addEventListener("input", () => {
    colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
    style.color = customPicker.value;
    panel.dataset.color = customPicker.value;
    schedule();
  });
  colorToolbar.appendChild(customPicker);

  const strokeToolbar = document.createElement("div");
  strokeToolbar.className = "stroke-toolbar";
  const strokeLabel = document.createElement("span");
  strokeLabel.className = "style-group-label";
  strokeLabel.textContent = "Viền chữ:";
  const strokeSlider = document.createElement("input");
  strokeSlider.type = "range";
  strokeSlider.className = "stroke-width-slider";
  strokeSlider.min = "0";
  strokeSlider.max = "8";
  strokeSlider.value = "2";
  strokeSlider.title = "Độ dày viền chữ";
  const strokeValSpan = document.createElement("span");
  strokeValSpan.className = "style-val-span";
  const strokeIsAuto = !style.strokeWidth || style.strokeWidth === "auto";
  if (strokeIsAuto) {
    strokeValSpan.textContent = "Auto";
  } else {
    strokeSlider.value = style.strokeWidth;
    strokeValSpan.textContent = style.strokeWidth + "px";
  }
  const strokeColorPicker = document.createElement("input");
  strokeColorPicker.type = "color";
  strokeColorPicker.className = "box-color-picker stroke-color-picker";
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
  bgToolbar.className = "bg-toolbar";

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
  bgSelect.className = "bg-color-select";
  const bgColors = ["#ffffff", "#000000"];
  if (style.bgColor && style.bgColor !== "transparent" && !bgColors.includes(style.bgColor)) {
    bgColors.unshift(style.bgColor);
  }
  bgColors.forEach((val) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = val === "#ffffff" ? "Nền Trắng" : val === "#000000" ? "Nền Đen" : val;
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
  radiusValSpan.className = "style-val-span";
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
    wrap.className = "geometry-field";
    const lbl = document.createElement("span");
    lbl.className = "geometry-field-label";
    lbl.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.className = "geometry-input";
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
      b.className = "align-btn" + ((style[key] || DEFAULT_TEXT_OBJECT_STYLE[key]) === val ? " selected" : "");
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

function updateSaveStatus(status) {
  _currentSaveStatus = status;
  const statusEls = document.querySelectorAll(".editor-save-status");
  statusEls.forEach((el) => {
    el.className = `editor-save-status save-status-${status}`;
    if (status === "saved") {
      el.textContent = "✓ Đã lưu";
      el.setAttribute("aria-label", "Tất cả thay đổi đã được lưu");
    } else if (status === "dirty") {
      el.textContent = "● Chưa lưu";
      el.setAttribute("aria-label", "Có thay đổi chưa lưu");
    } else if (status === "saving") {
      el.textContent = "⏳ Đang lưu...";
      el.setAttribute("aria-label", "Đang lưu thay đổi");
    } else if (status === "error") {
      el.textContent = "⚠️ Lưu thất bại";
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

async function flushTextObjectPersist(pageIndex) {
  clearTimeout(_textTimer);
  _textTimer = null;
  if (_textDirty.size === 0) return;
  const items = [];
  _textDirty.forEach((v) => {
    if (pageIndex === undefined || v.pageIndex === pageIndex) items.push(v);
  });
  if (items.length === 0) return;
  items.forEach((v) => _textDirty.delete(`${v.pageIndex}:${v.id}`));
  _textSaving += items.length;
  refreshSaveStatus();
  const failures = [];
  try {
    await Promise.all(items.map(async (p) => {
      const obj = findTextObject(p.pageIndex, p.id);
      if (!obj) return;
      try {
        await apiTextObject("update", {
          chapter_id: currentChapterId,
          page_index: p.pageIndex,
          id: p.id,
          ocr_text: p.ocr_text,
          translation: p.translation,
          style: p.style,
        });
      } catch (err) {
        failures.push(err);
        _textHasError = true;
        _textDirty.set(`${p.pageIndex}:${p.id}`, Object.assign({ pageIndex: p.pageIndex, id: p.id }, _captureTextState(obj)));
      }
    }));
  } finally {
    _textSaving -= items.length;
    refreshSaveStatus();
  }
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

  const updateTemp = (x, y) => {
    if (!temp || !start) return;
    temp.style.left = Math.min(start.x, x) + "px";
    temp.style.top = Math.min(start.y, y) + "px";
    temp.style.width = Math.abs(x - start.x) + "px";
    temp.style.height = Math.abs(y - start.y) + "px";
  };

  const onDown = (e) => {
    if (editorState.tool === "select") return;
    if (e.button !== 0) return;
    if (e.target.closest(".text-object-overlay")) return;
    e.preventDefault();
    const rect = imgWrap.getBoundingClientRect();
    drawing = true;
    start = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    last = start;
    temp = document.createElement("div");
    temp.className = "text-object-overlay drawing" + (editorState.tool === "ellipse" ? " ellipse" : "");
    imgWrap.appendChild(temp);
    updateTemp(start.x, start.y);
  };

  const onMove = (e) => {
    if (!drawing) return;
    const rect = imgWrap.getBoundingClientRect();
    last = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    updateTemp(last.x, last.y);
  };

  const onUp = () => {
    if (!drawing) return;
    drawing = false;
    if (temp) { temp.remove(); temp = null; }
    if (!start || !last) { start = null; last = null; return; }
    if (!img.naturalWidth || !img.clientWidth) { start = null; last = null; return; }
    const sx = img.naturalWidth / img.clientWidth;
    const sy = img.naturalHeight / img.clientHeight;
    const x1 = Math.round(Math.min(start.x, last.x) * sx);
    const y1 = Math.round(Math.min(start.y, last.y) * sy);
    const x2 = Math.round(Math.max(start.x, last.x) * sx);
    const y2 = Math.round(Math.max(start.y, last.y) * sy);
    const shape = editorState.tool;
    start = null;
    last = null;
    if (x2 - x1 < 10 || y2 - y1 < 10) return;
    createTextObject(pageIndex, shape, { x1, y1, x2, y2 }).catch((err) => {
      showToast("Tạo text object thất bại: " + err.message, "error");
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
  img.src = (page.clean || page.original) + "?t=" + Date.now();
  img.draggable = false;
  imgWrap.appendChild(img);
  block.appendChild(imgWrap);
  wrapper.appendChild(block);
  return wrapper;
}

async function switchEditorPage(newIndex) {
  const pages = currentManifest ? currentManifest.pages : null;
  if (!pages || newIndex < 0 || newIndex >= pages.length) return;
  try {
    await flushAllPendingPersists();
  } catch (err) {
    showToast("Không thể chuyển trang vì lưu dữ liệu thất bại.", "error");
    return;
  }
  editorState.activePageIndex = newIndex;
  editorState.selectedTextObjectId = null;
  renderEditor();
}

function showRenderResult(pageIndex, outputPath) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return;
  let resultBox = panelHost.querySelector(".render-result");
  if (!resultBox) {
    resultBox = document.createElement("div");
    resultBox.className = "render-result";
    panelHost.appendChild(resultBox);
  }
  const cacheBust = "?t=" + Date.now();
  resultBox.innerHTML = "";

  const label = document.createElement("div");
  label.className = "render-result-label";
  label.textContent = "Kết quả:";
  resultBox.appendChild(label);

  const img = document.createElement("img");
  img.src = outputPath + cacheBust;
  resultBox.appendChild(img);

  const link = document.createElement("a");
  link.href = outputPath + cacheBust;
  link.download = `page_${pageIndex + 1}_rendered.png`;
  link.className = "download-link";
  link.textContent = "Tải ảnh này về";
  resultBox.appendChild(link);
}

function renderEditor() {
  const container = document.getElementById("page-view");
  if (!container) return;
  cancelTextObjectPersist();
  if (typeof window.cancelGeomPersist === "function") window.cancelGeomPersist();
  if (!currentManifest || !currentManifest.pages || currentManifest.pages.length === 0) return;

  if (currentChapterId && editorState.lastChapterId !== currentChapterId) {
    editorState.lastChapterId = currentChapterId;
    editorState.activePageIndex = 0;
    editorState.selectedTextObjectId = null;
  }

  const pages = currentManifest.pages;
  editorState.activePageIndex = Math.max(0, Math.min(editorState.activePageIndex, pages.length - 1));
  const pageIndex = editorState.activePageIndex;
  const page = pages[pageIndex];

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

  const title = document.createElement("div");
  title.className = "translation-toolbar-title";
  title.innerHTML = '<strong>Biên tập bản dịch</strong><span>Chọn công cụ, kéo vùng quanh bong bóng chữ để tạo text object.</span>';

  const tools = document.createElement("div");
  tools.className = "editor-tools";
  [
    { key: "select", label: "Chọn" },
    { key: "rectangle", label: "Hình chữ nhật" },
    { key: "ellipse", label: "Hình ellipse" },
  ].forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "editor-tool-btn" + (editorState.tool === t.key ? " active" : "");
    btn.dataset.tool = t.key;
    btn.textContent = t.label;
    btn.addEventListener("click", () => setEditorTool(t.key));
    tools.appendChild(btn);
  });

  const addBoxBtn = document.createElement("button");
  addBoxBtn.type = "button";
  addBoxBtn.className = "editor-tool-btn";
  addBoxBtn.textContent = "Thêm ô chữ";
  addBoxBtn.title = "Tạo ô chữ mới ở giữa trang";
  addBoxBtn.addEventListener("click", () => {
    addTextObjectBox().catch((err) => {
      showToast("Không thêm được ô chữ: " + err.message, "error");
    });
  });
  tools.appendChild(addBoxBtn);

  const renderBtn = document.createElement("button");
  renderBtn.type = "button";
  renderBtn.className = "render-btn editor-render-btn";
  renderBtn.textContent = "Chèn chữ vào ảnh";
  renderBtn.addEventListener("click", () => renderTranslations(pageIndex));

  const saveStatus = document.createElement("div");
  const initialStatus = refreshSaveStatus();
  saveStatus.className = `editor-save-status save-status-${initialStatus}`;
  if (initialStatus === "saved") {
    saveStatus.textContent = "✓ Đã lưu";
  } else if (initialStatus === "dirty") {
    saveStatus.textContent = "● Chưa lưu";
  } else if (initialStatus === "saving") {
    saveStatus.textContent = "⏳ Đang lưu...";
  } else if (initialStatus === "error") {
    saveStatus.textContent = "⚠️ Lưu thất bại";
  }

  toolbar.append(title, tools, renderBtn, saveStatus);

  const nav = document.createElement("nav");
  nav.className = "translation-page-nav workspace-nav-bar";
  nav.setAttribute("aria-label", "Điều hướng trang biên tập");

  const prev = document.createElement("button");
  prev.type = "button";
  prev.className = "translation-nav-btn workspace-nav-btn";
  prev.textContent = "← Trước";
  prev.setAttribute("aria-label", "Trang trước");
  prev.disabled = pageIndex === 0;
  prev.addEventListener("click", () => switchEditorPage(pageIndex - 1));

  const position = document.createElement("div");
  position.className = "translation-position workspace-nav-position";
  position.setAttribute("aria-live", "polite");

  const jumpWrap = document.createElement("label");
  jumpWrap.className = "workspace-nav-jump-wrap";
  jumpWrap.textContent = "Trang ";

  const jumpInput = document.createElement("input");
  jumpInput.type = "number";
  jumpInput.min = "1";
  jumpInput.max = String(pages.length);
  jumpInput.value = String(pageIndex + 1);
  jumpInput.className = "workspace-nav-jump-input";
  jumpInput.setAttribute("aria-label", "Nhảy tới số trang");

  const doJump = () => {
    const targetIndex = parsePageNumber(jumpInput.value, pages.length);
    if (targetIndex === null) {
      jumpInput.value = String(pageIndex + 1);
      return;
    }
    if (targetIndex !== pageIndex) {
      switchEditorPage(targetIndex);
    }
  };

  jumpInput.addEventListener("change", doJump);
  jumpInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      doJump();
    }
  });

  jumpWrap.appendChild(jumpInput);

  const totalText = document.createElement("span");
  totalText.textContent = ` / ${pages.length}`;

  const labelSpan = document.createElement("span");
  const labelText = typeof pageLabel === "function" ? pageLabel(pages, pageIndex) : `Trang ${pageIndex + 1}`;
  labelSpan.textContent = ` · ${labelText}`;

  position.append(jumpWrap, totalText, labelSpan);

  if (page.rendered) {
    const badge = document.createElement("span");
    badge.className = "page-status-badge rendered";
    badge.textContent = "Đã xuất ảnh";
    position.appendChild(badge);
  } else if (page.skipped) {
    const badge = document.createElement("span");
    badge.className = "page-status-badge skipped";
    badge.textContent = "Bỏ qua";
    position.appendChild(badge);
  }

  const next = document.createElement("button");
  next.type = "button";
  next.className = "translation-nav-btn workspace-nav-btn";
  next.textContent = "Sau →";
  next.setAttribute("aria-label", "Trang sau");
  next.disabled = pageIndex === pages.length - 1;
  next.addEventListener("click", () => switchEditorPage(pageIndex + 1));

  nav.append(prev, position, next);

  const body = document.createElement("div");
  body.className = "translation-workspace-body";

  const canvasHost = document.createElement("main");
  canvasHost.className = "translation-canvas-host";

  const panelHost = document.createElement("aside");
  panelHost.className = "translation-panel-host";
  panelHost.setAttribute("aria-label", "Bảng biên tập vùng chữ");

  body.append(canvasHost, panelHost);

  shell.append(toolbar, nav, body);
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
}
