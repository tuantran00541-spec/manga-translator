let _activeFontPopCleanup = null;

function renderEditorPanel(pageIndex) {
  if (typeof _activeFontPopCleanup === "function") {
    _activeFontPopCleanup();
    _activeFontPopCleanup = null;
  }
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
  window.editorHistory?.watch(pageIndex, obj);

  const fitTextareaHeight = (ta) => {
    ta.style.height = "auto";
    if (ta.scrollHeight > 0) ta.style.height = `${Math.min(100, ta.scrollHeight + 2)}px`;
  };

  const ocrLabel = document.createElement("label");
  ocrLabel.className = "ui-field text-editor-field";
  ocrLabel.textContent = "Nội dung gốc / OCR";

  const ocrTa = document.createElement("textarea");
  ocrTa.className = "ui-textarea text-editor-textarea ocr-textarea";
  ocrTa.dataset.textObjectId = obj.id;
  ocrTa.rows = 1;
  ocrTa.placeholder = "Nhập hoặc hiệu chỉnh nội dung gốc…";
  ocrTa.value = obj.ocr_text || "";
  ocrTa.addEventListener("input", () => {
    obj.ocr_text = ocrTa.value;
    fitTextareaHeight(ocrTa);
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  const trLabel = document.createElement("label");
  trLabel.className = "ui-field text-editor-field";
  trLabel.textContent = "Bản dịch";

  const trTa = document.createElement("textarea");
  trTa.className = "ui-textarea text-editor-textarea translation-textarea";
  trTa.dataset.textObjectId = obj.id;
  trTa.rows = 1;
  trTa.placeholder = "Nhập nội dung bản dịch…";
  trTa.value = obj.translation || "";
  trTa.addEventListener("input", () => {
    obj.translation = trTa.value;
    fitTextareaHeight(trTa);
    scheduleTextObjectPersist(pageIndex, obj.id);
  });

  ocrLabel.appendChild(ocrTa);
  trLabel.appendChild(trTa);

  const sections = document.createElement("div");
  sections.className = "text-editor-sections";

  const textBody = buildPanelSection(sections, "Nội dung chữ", true);
  textBody.append(ocrLabel, trLabel);
  const layoutIssueMessages = {
    balloon_overset: "Bản dịch dài hơn phần lòng ô có thể dàn ở cỡ đọc được.",
    placement_unverified: "Chưa xác nhận được vị trí chữ trên ảnh gốc. Chỉnh vùng chữ trên ảnh để đặt lại vị trí.",
    placement_ambiguous: "Nhiều bản dịch cùng trỏ vào vùng chữ này. Kiểm tra lại các vùng bị trùng.",
    placement_outside_bounds: "Chữ vượt vùng đã xác nhận. Chỉnh cỡ chữ hoặc vùng chữ.",
    placement_overset: "Bản dịch quá dài so với vùng chữ gốc. Rút gọn lời hoặc chỉnh vùng chữ.",
    skipped_collision: "Vùng chữ chồng với vùng khác. Chỉnh vị trí hoặc cỡ chữ.",
    unanchored_glyph: "Ký tự đứng riêng trong vùng ảnh gốc trống. Kiểm tra bản dịch hoặc vùng chữ.",
  };
  const issueType = page?.render_layout_issue_types?.[obj.id];
  const layoutIssue = page?.rendered && issueType
    ? (layoutIssueMessages[issueType] || "Bố cục vùng chữ cần kiểm tra lại trước khi xuất.")
    : "";
  if (layoutIssue) {
    const hint = document.createElement("small");
    hint.className = "editor-page-status";
    hint.textContent = layoutIssue;
    textBody.appendChild(hint);
  }

  const typographyBody = buildPanelSection(sections, "Kiểu chữ", true);
  window.stylePresets?.buildSection(typographyBody, obj, pageIndex);
  buildTextSection(typographyBody, panel, obj, pageIndex);

  const appearanceBody = buildPanelSection(sections, "Màu chữ", false);
  buildAppearanceSection(appearanceBody, panel, obj, pageIndex);

  panel.appendChild(sections);

  const actions = document.createElement("div");
  actions.className = "text-object-actions";

  const ocrBtn = document.createElement("button");
  ocrBtn.type = "button";
  ocrBtn.className = "ui-btn ui-btn-ghost text-object-action-btn";
  ocrBtn.textContent = "OCR lại";
  ocrBtn.title = "Nhận dạng lại bằng OCR";
  ocrBtn.addEventListener("click", () => {
    associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
      showToast("Không thể nhận dạng lại bằng OCR: " + err.message, "info");
    });
  });

  const dupBtn = document.createElement("button");
  dupBtn.type = "button";
  dupBtn.className = "ui-btn ui-btn-ghost text-object-action-btn";
  dupBtn.textContent = "Nhân đôi";
  dupBtn.title = "Nhân đôi vùng chữ (Ctrl+D)";
  dupBtn.addEventListener("click", () => {
    window.duplicateTextObject?.(pageIndex, obj.id)?.then((newId) => {
      if (!newId) return;
      window.refreshReviewOverlays?.();
      showToast("Đã nhân đôi vùng chữ.", "success");
    }).catch((err) => {
      showToast("Không thể nhân đôi vùng chữ: " + err.message, "error");
    });
  });

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "ui-btn ui-btn-danger text-object-action-btn text-object-delete-btn";
  delBtn.textContent = "Xóa";
  delBtn.title = "Xóa vùng chữ";
  delBtn.addEventListener("click", () => {
    if (delBtn.dataset.armed !== "1") {
      delBtn.dataset.armed = "1";
      delBtn.textContent = "Xác nhận";
      return;
    }
    deleteTextObject(pageIndex, obj.id).then(() => {
      window.refreshReviewOverlays?.();
    }).catch((err) => {
      showToast("Không thể xóa vùng chữ: " + err.message, "error");
    });
  });

  actions.append(ocrBtn, dupBtn, delBtn);
  panel.appendChild(actions);

  syncStyleDataset(panel, obj.style);
  panelHost.appendChild(panel);
  fitTextareaHeight(ocrTa);
  fitTextareaHeight(trTa);
  requestAnimationFrame(() => {
    if (!panel.isConnected) return;
    fitTextareaHeight(ocrTa);
    fitTextareaHeight(trTa);
  });
}

function buildPanelSection(container, title, open) {
  const section = document.createElement("details");
  section.className = "ui-disclosure inspector-section text-editor-section";
  section.open = open;

  const header = document.createElement("summary");
  const label = document.createElement("span");
  label.textContent = title;
  header.append(label, window.createUiIcon("chevron-down", "inspector-chevron"));

  const body = document.createElement("div");
  body.className = "text-editor-section-body";

  section.append(header, body);
  container.appendChild(section);
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


function shownFontId(obj) {
  const font = obj.style?.font;
  if (obj.font_selection_mode === "ai" && obj.font_ai_id) return obj.font_ai_id;
  return font && font !== "auto" ? font : "default";
}
window.shownFontId = shownFontId;

const FONT_CATEGORY_LABELS = {
  default: "Mặc định",
  dialogue: "Lời thoại",
  narration: "Dẫn truyện",
  emphasis: "Nhấn mạnh",
  thought: "Suy nghĩ",
  sfx: "Hiệu ứng",
  horror: "Kinh dị",
  romance: "Lãng mạn",
  skill: "Chiêu thức",
  system: "Hệ thống",
  other: "Khác",
};

function cleanFontLabel(name) {
  const raw = String(name || "").trim();
  const wrapped = raw.match(/^Mac-dinh(?:-\d+)?\s*\((.+)\)$/i);
  if (wrapped) return wrapped[1].trim();
  return raw || "Mặc định";
}

function buildTextSection(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const fontRow = document.createElement("div");
  fontRow.className = "ui-control-row font-style-toolbar font-style-row";

  const fontSelect = document.createElement("select");
  fontSelect.className = "ui-select";
  fontSelect.hidden = true;
  fontSelect.setAttribute("aria-hidden", "true");
  fontSelect.tabIndex = -1;
  const rawFonts = availableFonts || [];
  const entries = (rawFonts.length ? rawFonts : [{ id: "default", name: "Mặc định (Comic)", category: "default" }])
    .filter((f) => f && f.id !== "auto")
    .map((f) => {
      const cat = f.category || "other";
      return {
        id: f.id,
        label: cleanFontLabel(f.name),
        category: cat,
        categoryLabel: FONT_CATEGORY_LABELS[cat] || cat,
      };
    });

  const groups = new Map();
  entries.forEach((entry) => {
    let group = groups.get(entry.category);
    if (!group) {
      group = document.createElement("optgroup");
      group.label = entry.categoryLabel;
      groups.set(entry.category, group);
      fontSelect.appendChild(group);
    }
    const opt = document.createElement("option");
    opt.value = entry.id;
    opt.textContent = entry.label;
    group.appendChild(opt);
  });

  fontSelect.value = shownFontId(obj);
  if (!fontSelect.value) fontSelect.value = entries[0]?.id || "default";

  const fontBtn = document.createElement("button");
  fontBtn.type = "button";
  fontBtn.className = "ui-select font-picker-trigger";
  fontBtn.title = "Chọn phông chữ";
  fontBtn.setAttribute("aria-haspopup", "listbox");
  fontBtn.setAttribute("aria-expanded", "false");
  const fontBtnText = document.createElement("span");
  fontBtnText.className = "font-picker-value";
  const syncFontLabel = () => {
    const active = entries.find((e) => e.id === fontSelect.value) || entries[0];
    fontBtnText.textContent = active ? active.label : "Mặc định";
  };
  syncFontLabel();
  fontBtn.append(fontBtnText, window.createUiIcon("chevron-down", "font-picker-caret"));

  fontSelect.addEventListener("change", () => {
    style.font = fontSelect.value;
    obj.font_selection_mode = "user";
    obj.font_ai_id = null;
    obj.font_match = null;
    panel.dataset.font = fontSelect.value;
    syncFontLabel();
    schedule();
  });

  let activePop = null;
  const closeFontPop = (restoreFocus = false) => {
    if (!activePop) return;
    const cleanup = activePop.cleanup;
    activePop = null;
    if (_activeFontPopCleanup === cleanup) _activeFontPopCleanup = null;
    cleanup();
    fontBtn.setAttribute("aria-expanded", "false");
    fontBtn.classList.remove("is-open");
    if (restoreFocus && fontBtn.isConnected) fontBtn.focus();
  };

  fontBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    if (activePop) {
      closeFontPop();
      return;
    }
    if (typeof _activeFontPopCleanup === "function") {
      _activeFontPopCleanup();
      _activeFontPopCleanup = null;
    }
    document.querySelectorAll(".font-picker-pop").forEach((el) => el.remove());

    const pop = document.createElement("div");
    pop.className = "font-picker-pop";
    pop.setAttribute("role", "listbox");
    pop.setAttribute("aria-label", "Danh sách phông chữ");

    const search = document.createElement("input");
    search.type = "search";
    search.className = "ui-input font-picker-search";
    search.placeholder = "Tìm phông chữ…";
    search.setAttribute("aria-label", "Tìm phông chữ");

    const catsRow = document.createElement("div");
    catsRow.className = "font-picker-cats";
    let activeCat = "";
    const catKeys = ["", ...groups.keys()];
    catKeys.forEach((key) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "font-picker-cat" + (key === "" ? " is-active" : "");
      chip.textContent = key ? (FONT_CATEGORY_LABELS[key] || key) : "Tất cả";
      chip.addEventListener("click", () => {
        activeCat = key;
        catsRow.querySelectorAll(".font-picker-cat").forEach((c) => c.classList.toggle("is-active", c === chip));
        renderFontList();
      });
      catsRow.appendChild(chip);
    });

    const list = document.createElement("div");
    list.className = "font-picker-list";

    const renderFontList = () => {
      const q = search.value.trim().toLowerCase();
      const filtered = entries.filter((e) => {
        if (activeCat && e.category !== activeCat) return false;
        if (!q) return true;
        return e.label.toLowerCase().includes(q) || e.categoryLabel.toLowerCase().includes(q);
      });
      list.replaceChildren();
      if (!filtered.length) {
        const empty = document.createElement("div");
        empty.className = "font-picker-empty";
        empty.textContent = "Không tìm thấy phông phù hợp.";
        list.appendChild(empty);
        return;
      }
      let lastCat = null;
      filtered.forEach((entry) => {
        if (!activeCat && entry.category !== lastCat) {
          lastCat = entry.category;
          const head = document.createElement("div");
          head.className = "font-picker-group";
          head.textContent = entry.categoryLabel;
          list.appendChild(head);
        }
        const opt = document.createElement("button");
        opt.type = "button";
        const selected = entry.id === fontSelect.value;
        opt.className = "font-picker-item" + (selected ? " is-selected" : "");
        opt.setAttribute("role", "option");
        opt.setAttribute("aria-selected", String(selected));
        const nameSpan = document.createElement("span");
        nameSpan.textContent = entry.label;
        opt.appendChild(nameSpan);
        if (selected) opt.appendChild(window.createUiIcon("check", "font-picker-check"));
        opt.addEventListener("click", () => {
          fontSelect.value = entry.id;
          fontSelect.dispatchEvent(new Event("change", { bubbles: true }));
          closeFontPop(true);
        });
        list.appendChild(opt);
      });
    };

    search.addEventListener("input", renderFontList);
    renderFontList();
    pop.append(search, catsRow, list);
    document.body.appendChild(pop);

    const rect = fontBtn.getBoundingClientRect();
    const width = Math.max(256, Math.round(rect.width + 34));
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
    pop.style.width = `${width}px`;
    pop.style.left = `${left}px`;
    const spaceBelow = window.innerHeight - rect.bottom - 10;
    const spaceAbove = rect.top - 10;
    if (spaceBelow >= 240 || spaceBelow >= spaceAbove) {
      pop.style.top = `${Math.round(rect.bottom + 4)}px`;
      pop.style.maxHeight = `${Math.min(340, Math.max(180, Math.floor(spaceBelow)))}px`;
    } else {
      pop.style.bottom = `${Math.round(window.innerHeight - rect.top + 4)}px`;
      pop.style.maxHeight = `${Math.min(340, Math.max(180, Math.floor(spaceAbove)))}px`;
    }

    const onOutside = (e) => {
      if (!fontBtn.isConnected || (!pop.contains(e.target) && !fontBtn.contains(e.target))) {
        closeFontPop();
      }
    };
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        closeFontPop(true);
      }
    };
    const onScrollOrResize = (e) => {
      if (e?.type === "scroll" && pop.contains(e.target)) return;
      closeFontPop();
    };
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onScrollOrResize, true);
    window.addEventListener("scroll", onScrollOrResize, true);

    const cleanup = () => {
      document.removeEventListener("pointerdown", onOutside, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", onScrollOrResize, true);
      window.removeEventListener("scroll", onScrollOrResize, true);
      pop.remove();
    };
    activePop = { cleanup };
    _activeFontPopCleanup = cleanup;
    fontBtn.setAttribute("aria-expanded", "true");
    fontBtn.classList.add("is-open");
    search.focus();
  });

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
  fontRow.append(fontSelect, fontBtn, boldBtn);

  const sizeGroup = document.createElement("div");
  sizeGroup.className = "ui-control-row font-size-group";
  const sizeLabel = document.createElement("span");
  sizeLabel.className = "ui-control-label";
  sizeLabel.textContent = "Cỡ chữ";
  const autoBtn = document.createElement("button");
  autoBtn.type = "button";
  autoBtn.className = "ui-btn ui-btn-ghost ui-btn-compact size-auto-btn";
  autoBtn.textContent = "Tự động";
  autoBtn.title = "Tự động vừa vùng";
  const sizeSlider = document.createElement("input");
  sizeSlider.type = "range";
  sizeSlider.className = "font-size-slider";
  sizeSlider.min = "10";
  sizeSlider.max = "120";
  sizeSlider.value = "20";
  const sizeValSpan = document.createElement("span");
  sizeValSpan.className = "ui-value font-size-value";
  const syncSizeUI = () => {
    const isAuto = !style.fontSize || style.fontSize === "auto";
    autoBtn.classList.toggle("selected", isAuto);
    sizeSlider.disabled = isAuto;
    sizeValSpan.hidden = isAuto;
    if (!isAuto) {
      sizeSlider.value = style.fontSize;
      sizeValSpan.textContent = style.fontSize + "px";
    }
  };
  syncSizeUI();
  autoBtn.addEventListener("click", () => {
    const isAuto = !style.fontSize || style.fontSize === "auto";
    style.fontSize = isAuto ? String(sizeSlider.value) : "auto";
    panel.dataset.fontSize = style.fontSize;
    syncSizeUI();
    schedule();
  });
  sizeSlider.addEventListener("input", () => {
    if (!style.fontSize || style.fontSize === "auto") return;
    style.fontSize = sizeSlider.value;
    panel.dataset.fontSize = style.fontSize;
    sizeValSpan.textContent = style.fontSize + "px";
    schedule();
  });
  sizeGroup.append(sizeLabel, sizeSlider, sizeValSpan, autoBtn);

  body.append(fontRow, sizeGroup);
  buildAlignmentControls(body, panel, obj, pageIndex);

  const shapeGroup = document.createElement("div");
  shapeGroup.className = "align-group shape-group";
  const shapeLbl = document.createElement("span");
  shapeLbl.className = "style-group-label";
  shapeLbl.textContent = "Hình dạng vùng";
  const shapeRow = document.createElement("div");
  shapeRow.className = "align-btn-row";
  [
    ["rectangle", "Chữ nhật"],
    ["ellipse", "Elip"],
  ].forEach(([val, txt]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ui-btn ui-btn-ghost ui-btn-compact align-btn" + ((obj.shape || "rectangle") === val ? " selected" : "");
    b.textContent = txt;
    b.addEventListener("click", () => {
      obj.shape = val;
      shapeRow.querySelectorAll(".align-btn").forEach((x) => x.classList.remove("selected"));
      b.classList.add("selected");
      schedule();
    });
    shapeRow.appendChild(b);
  });
  shapeGroup.append(shapeLbl, shapeRow);
  body.append(shapeGroup);
}

function _ensureColorProfile(obj) {
  if (!obj.ocr_color_profile || typeof obj.ocr_color_profile !== "object") {
    const baseFill = (obj.style?.color && obj.style.color !== "auto")
      ? obj.style.color
      : (obj.ocr_text_color || "#111111");
    obj.ocr_color_profile = { fill: /^#[0-9a-f]{6}$/i.test(baseFill) ? baseFill.toLowerCase() : "#111111" };
  }
  return obj.ocr_color_profile;
}

function buildAppearanceSection(body, panel, obj, pageIndex) {
  const style = obj.style || (obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE));
  const schedule = () => scheduleTextObjectPersist(pageIndex, obj.id);

  const colorToolbar = document.createElement("div");
  colorToolbar.className = "ui-control-row color-toolbar";
  const colorLabel = document.createElement("span");
  colorLabel.className = "ui-control-label";
  colorLabel.textContent = "Màu chữ";
  const swatchRow = document.createElement("div");
  swatchRow.className = "color-swatch-row";

  const colors = [
    { name: "Tự động tương phản / Màu gốc OCR", value: "auto", bg: "linear-gradient(135deg, #000 50%, #fff 50%)" },
    { name: "Trắng", value: "#ffffff", bg: "#ffffff" },
    { name: "Đen", value: "#000000", bg: "#000000" },
    { name: "Đỏ", value: "#e8432c", bg: "#e8432c" },
    { name: "Vàng", value: "#f1c40f", bg: "#f1c40f" },
  ];
  const syncColorSwatches = () => {
    swatchRow.querySelectorAll(".color-btn").forEach((b) => {
      b.setAttribute("aria-pressed", style.color === b.dataset.colorValue ? "true" : "false");
    });
  };
  colors.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-swatch color-btn";
    btn.dataset.colorValue = c.value;
    btn.setAttribute("aria-pressed", style.color === c.value ? "true" : "false");
    btn.title = c.name;
    btn.style.background = c.bg;
    btn.addEventListener("click", () => {
      style.color = c.value;
      panel.dataset.color = c.value;
      syncColorSwatches();
      schedule();
    });
    swatchRow.appendChild(btn);
  });

  const customPicker = document.createElement("input");
  customPicker.type = "color";
  customPicker.className = "ui-color-input ui-color-swatch custom-color-picker";
  customPicker.value = "#ffffff";
  customPicker.title = "Chọn màu tùy chỉnh";
  if (style.color && style.color !== "auto" && !colors.some((c) => c.value === style.color)) {
    customPicker.value = style.color;
  }
  customPicker.addEventListener("input", () => {
    style.color = customPicker.value;
    panel.dataset.color = customPicker.value;
    syncColorSwatches();
    schedule();
  });

  const eyedropBtn = document.createElement("button");
  eyedropBtn.type = "button";
  eyedropBtn.className = "ui-icon-btn ui-btn-ghost ui-btn-compact inspector-eyedrop-btn";
  eyedropBtn.title = "Hút màu trực tiếp từ ảnh (I)";
  eyedropBtn.setAttribute("aria-label", "Hút màu từ ảnh");
  eyedropBtn.appendChild(window.createUiIcon("eyedropper"));
  eyedropBtn.addEventListener("click", () => {
    window.selectReviewTool?.("eyedropper");
    window.showToast?.("Nhấp lên ảnh để hút màu chữ (Shift: viền, Alt: nền).", "info");
  });

  swatchRow.append(customPicker, eyedropBtn);
  colorToolbar.append(colorLabel, swatchRow);

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
  strokeValSpan.className = "ui-value stroke-width-value";
  const strokeColorPicker = document.createElement("input");
  strokeColorPicker.type = "color";
  strokeColorPicker.className = "ui-color-input ui-color-swatch stroke-color-picker";
  strokeColorPicker.title = "Màu viền chữ";
  const strokeAutoBtn = document.createElement("button");
  strokeAutoBtn.type = "button";
  strokeAutoBtn.className = "ui-btn ui-btn-ghost ui-btn-compact stroke-auto-btn";
  strokeAutoBtn.textContent = "Tự động";
  strokeAutoBtn.title = "Dùng độ dày và màu viền tự động";

  const syncStrokeUI = () => {
    const wAuto = !style.strokeWidth || style.strokeWidth === "auto";
    const cAuto = !style.strokeColor || style.strokeColor === "auto";
    strokeAutoBtn.classList.toggle("selected", wAuto && cAuto);
    if (wAuto) {
      strokeValSpan.textContent = "Tự động";
    } else {
      strokeSlider.value = style.strokeWidth;
      strokeValSpan.textContent = style.strokeWidth + "px";
    }
    const defaultStroke = obj.ocr_color_profile?.stroke_color || "#000000";
    strokeColorPicker.value = cAuto ? defaultStroke : style.strokeColor;
  };
  syncStrokeUI();

  strokeSlider.addEventListener("input", () => {
    style.strokeWidth = strokeSlider.value;
    panel.dataset.strokeWidth = strokeSlider.value;
    syncStrokeUI();
    schedule();
  });
  strokeColorPicker.addEventListener("input", () => {
    style.strokeColor = strokeColorPicker.value;
    panel.dataset.strokeColor = strokeColorPicker.value;
    syncStrokeUI();
    schedule();
  });
  strokeAutoBtn.addEventListener("click", () => {
    style.strokeWidth = "auto";
    style.strokeColor = "auto";
    panel.dataset.strokeWidth = "auto";
    panel.dataset.strokeColor = "auto";
    syncStrokeUI();
    schedule();
  });
  strokeToolbar.append(strokeLabel, strokeColorPicker, strokeSlider, strokeValSpan, strokeAutoBtn);

  // Box background fill & corner radius (already supported in backend renderer)
  const bgToolbar = document.createElement("div");
  bgToolbar.className = "ui-control-row bg-toolbar";
  const bgLabel = document.createElement("span");
  bgLabel.className = "ui-control-label";
  bgLabel.textContent = "Nền khung";
  const bgSwatchRow = document.createElement("div");
  bgSwatchRow.className = "color-swatch-row";
  const bgChoices = [
    { name: "Trong suốt", value: "transparent", bg: "repeating-conic-gradient(#bbb 0% 25%, #fff 0% 50%) 50% / 10px 10px" },
    { name: "Trắng", value: "#ffffff", bg: "#ffffff" },
    { name: "Đen", value: "#000000", bg: "#000000" },
    { name: "Kem", value: "#f7f3e8", bg: "#f7f3e8" },
  ];
  const bgPicker = document.createElement("input");
  bgPicker.type = "color";
  bgPicker.className = "ui-color-input ui-color-swatch bg-color-picker";
  bgPicker.title = "Màu nền khung tùy chỉnh";

  const radiusRow = document.createElement("div");
  radiusRow.className = "ui-control-row corner-radius-toolbar";
  const radiusLabel = document.createElement("span");
  radiusLabel.className = "ui-control-label";
  radiusLabel.textContent = "Bo góc";
  const radiusSlider = document.createElement("input");
  radiusSlider.type = "range";
  radiusSlider.className = "stroke-width-slider corner-radius-slider";
  radiusSlider.min = "0";
  radiusSlider.max = "32";
  radiusSlider.value = String(style.cornerRadius || "0");
  const radiusVal = document.createElement("span");
  radiusVal.className = "ui-value corner-radius-value";

  const syncBgUI = () => {
    const curBg = style.bgColor || "transparent";
    const isTrans = curBg === "transparent" || curBg === "none";
    bgSwatchRow.querySelectorAll(".bg-swatch-btn").forEach((b) => {
      b.setAttribute("aria-pressed", curBg === b.dataset.bgValue ? "true" : "false");
    });
    if (!isTrans && /^#[0-9a-f]{6}$/i.test(curBg)) bgPicker.value = curBg;
    else bgPicker.value = "#ffffff";
    radiusSlider.disabled = isTrans;
    radiusRow.classList.toggle("is-disabled", isTrans);
    radiusVal.textContent = `${style.cornerRadius || 0}px`;
  };

  bgChoices.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-swatch bg-swatch-btn";
    btn.dataset.bgValue = c.value;
    btn.title = c.name;
    btn.style.background = c.bg;
    btn.addEventListener("click", () => {
      style.bgColor = c.value;
      panel.dataset.bgColor = c.value;
      syncBgUI();
      schedule();
    });
    bgSwatchRow.appendChild(btn);
  });
  bgPicker.addEventListener("input", () => {
    style.bgColor = bgPicker.value;
    panel.dataset.bgColor = bgPicker.value;
    syncBgUI();
    schedule();
  });
  bgSwatchRow.appendChild(bgPicker);
  bgToolbar.append(bgLabel, bgSwatchRow);

  radiusSlider.addEventListener("input", () => {
    style.cornerRadius = radiusSlider.value;
    panel.dataset.cornerRadius = radiusSlider.value;
    syncBgUI();
    schedule();
  });
  radiusRow.append(radiusLabel, radiusSlider, radiusVal);
  syncBgUI();

  // Gradient & Glow controls (ocr_color_profile)
  const fxBlock = document.createElement("div");
  fxBlock.className = "color-effects-block";

  const gradRow = document.createElement("div");
  gradRow.className = "ui-control-row gradient-toolbar";
  const gradLabel = document.createElement("span");
  gradLabel.className = "ui-control-label";
  gradLabel.textContent = "Chuyển sắc";
  const gradModes = document.createElement("div");
  gradModes.className = "align-btn-row";
  const gradStopsRow = document.createElement("div");
  gradStopsRow.className = "gradient-stops-row";

  const syncGradUI = () => {
    const prof = obj.ocr_color_profile;
    const activeDir = prof?.gradient?.direction || "none";
    gradModes.querySelectorAll(".align-btn").forEach((b) => {
      b.classList.toggle("selected", b.dataset.gradDir === activeDir);
    });
    gradStopsRow.replaceChildren();
    if (activeDir === "none" || !Array.isArray(prof?.gradient?.stops)) {
      gradStopsRow.hidden = true;
      return;
    }
    gradStopsRow.hidden = false;
    prof.gradient.stops.forEach((hex, idx) => {
      const inp = document.createElement("input");
      inp.type = "color";
      inp.className = "ui-color-input ui-color-swatch gradient-stop-picker";
      inp.value = /^#[0-9a-f]{6}$/i.test(hex) ? hex : "#ffaa00";
      inp.title = `Màu điểm dừng ${idx + 1}`;
      inp.addEventListener("input", () => {
        const p = _ensureColorProfile(obj);
        if (!p.gradient?.stops) return;
        p.gradient.stops[idx] = inp.value;
        p.fill = p.gradient.stops[Math.floor(p.gradient.stops.length / 2)] || inp.value;
        style.color = "auto";
        panel.dataset.color = "auto";
        syncColorSwatches();
        schedule();
      });
      gradStopsRow.appendChild(inp);
    });
  };

  [
    ["none", "Tắt"],
    ["vertical", "Dọc"],
    ["horizontal", "Ngang"],
  ].forEach(([dir, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ui-btn ui-btn-ghost ui-btn-compact align-btn";
    b.dataset.gradDir = dir;
    b.textContent = label;
    b.addEventListener("click", () => {
      if (dir === "none") {
        if (obj.ocr_color_profile && typeof obj.ocr_color_profile === "object") {
          delete obj.ocr_color_profile.gradient;
        }
      } else {
        const p = _ensureColorProfile(obj);
        const existingStops = p.gradient?.stops;
        const base = /^#[0-9a-f]{6}$/i.test(p.fill || "") ? p.fill : "#f59e0b";
        const stops = Array.isArray(existingStops) && existingStops.length >= 2
          ? existingStops.slice(0, dir === "vertical" ? 3 : 2)
          : (dir === "vertical" ? [base, "#f97316", "#b91c1c"] : [base, "#ec4899"]);
        if (dir === "vertical" && stops.length === 2) stops.splice(1, 0, base);
        p.gradient = { direction: dir, stops };
        style.color = "auto";
        panel.dataset.color = "auto";
        syncColorSwatches();
      }
      syncGradUI();
      schedule();
    });
    gradModes.appendChild(b);
  });
  gradRow.append(gradLabel, gradModes);
  syncGradUI();

  const glowRow = document.createElement("div");
  glowRow.className = "ui-control-row glow-toolbar";
  const glowLabel = document.createElement("span");
  glowLabel.className = "ui-control-label";
  glowLabel.textContent = "Hào quang";
  const glowToggle = document.createElement("button");
  glowToggle.type = "button";
  glowToggle.className = "ui-btn ui-btn-ghost ui-btn-compact align-btn";
  const glowColorPicker = document.createElement("input");
  glowColorPicker.type = "color";
  glowColorPicker.className = "ui-color-input ui-color-swatch glow-color-picker";
  glowColorPicker.title = "Màu hào quang (Glow)";
  const glowSlider = document.createElement("input");
  glowSlider.type = "range";
  glowSlider.className = "stroke-width-slider glow-radius-slider";
  glowSlider.min = "2";
  glowSlider.max = "12";
  const glowVal = document.createElement("span");
  glowVal.className = "ui-value glow-radius-value";

  const syncGlowUI = () => {
    const gc = obj.ocr_color_profile?.glow_color;
    const on = Boolean(gc && /^#[0-9a-f]{6}$/i.test(gc));
    glowToggle.textContent = on ? "Bật" : "Tắt";
    glowToggle.classList.toggle("selected", on);
    glowColorPicker.disabled = !on;
    glowSlider.disabled = !on;
    glowColorPicker.value = on ? gc : "#ff5522";
    const rad = on ? Math.max(2, Math.min(12, Number(obj.ocr_color_profile?.glow_radius || 5))) : 5;
    glowSlider.value = String(rad);
    glowVal.textContent = on ? `${rad}px` : "—";
  };

  glowToggle.addEventListener("click", () => {
    const on = Boolean(obj.ocr_color_profile?.glow_color);
    if (on) {
      delete obj.ocr_color_profile.glow_color;
      delete obj.ocr_color_profile.glow_radius;
    } else {
      const p = _ensureColorProfile(obj);
      p.glow_color = glowColorPicker.value || "#ff5522";
      p.glow_radius = Math.max(2, Math.min(12, Number(glowSlider.value || 5)));
      style.color = "auto";
      panel.dataset.color = "auto";
      syncColorSwatches();
    }
    syncGlowUI();
    schedule();
  });
  glowColorPicker.addEventListener("input", () => {
    const p = _ensureColorProfile(obj);
    p.glow_color = glowColorPicker.value;
    p.glow_radius = Math.max(2, Math.min(12, Number(glowSlider.value || 5)));
    style.color = "auto";
    panel.dataset.color = "auto";
    syncColorSwatches();
    syncGlowUI();
    schedule();
  });
  glowSlider.addEventListener("input", () => {
    const p = _ensureColorProfile(obj);
    if (!p.glow_color) p.glow_color = glowColorPicker.value || "#ff5522";
    p.glow_radius = Number(glowSlider.value);
    syncGlowUI();
    schedule();
  });
  glowRow.append(glowLabel, glowToggle, glowColorPicker, glowSlider, glowVal);
  syncGlowUI();

  fxBlock.append(gradRow, gradStopsRow, glowRow);
  body.append(colorToolbar, strokeToolbar, bgToolbar, radiusRow, fxBlock);
}

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
    mkGroup("Căn ngang", [["left", "Trái"], ["center", "Giữa"], ["right", "Phải"]], "horizontalAlign"),
    mkGroup("Căn dọc", [["top", "Trên"], ["middle", "Giữa"], ["bottom", "Dưới"]], "verticalAlign"),
  );
}
