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
    const autoOpt = document.createElement("option");
    autoOpt.value = "auto";
    autoOpt.textContent = "Tự động (AI chọn)";
    fontSelect.appendChild(autoOpt);
    const groups = new Map();
    fonts.forEach((f) => {
      if (!f || f.id === "auto") return;
      const category = f.category || "other";
      let group = groups.get(category);
      if (!group) {
        group = document.createElement("optgroup");
        group.label = category === "default" ? "Mặc định" : category;
        groups.set(category, group);
        fontSelect.appendChild(group);
      }
      const opt = document.createElement("option");
      opt.value = f.id;
      opt.textContent = f.name;
      group.appendChild(opt);
    });
  }
  fontSelect.value = obj.font_selection_mode === "auto" ? "auto" : (style.font || "default");
  fontSelect.addEventListener("change", () => {
    style.font = fontSelect.value;
    obj.font_selection_mode = fontSelect.value === "auto" ? "auto" : "user";
    obj.font_ai_id = null;
    obj.font_match = null;
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
