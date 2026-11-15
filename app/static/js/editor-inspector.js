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

  const typographyBody = buildPanelSection(sections, "Kiểu chữ", false);
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
    deleteTextObject(pageIndex, obj.id).catch((err) => {
      showToast("Không thể xóa vùng chữ: " + err.message, "error");
    });
  });

  actions.append(ocrBtn, delBtn);
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
  if (obj.font_selection_mode !== "user" && obj.font_ai_id) return obj.font_ai_id;
  const font = obj.style?.font;
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
    activePop.cleanup();
    activePop = null;
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

    activePop = {
      cleanup: () => {
        document.removeEventListener("pointerdown", onOutside, true);
        document.removeEventListener("keydown", onKey, true);
        window.removeEventListener("resize", onScrollOrResize, true);
        window.removeEventListener("scroll", onScrollOrResize, true);
        pop.remove();
      },
    };
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
  sizeSlider.max = "60";
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
      swatchRow.querySelectorAll(".color-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      style.color = c.value;
      panel.dataset.color = c.value;
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
    swatchRow.querySelectorAll(".color-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
    style.color = customPicker.value;
    panel.dataset.color = customPicker.value;
    schedule();
  });
  swatchRow.appendChild(customPicker);
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
  const strokeIsAuto = !style.strokeWidth || style.strokeWidth === "auto";
  if (strokeIsAuto) {
    strokeValSpan.textContent = "Tự động";
  } else {
    strokeSlider.value = style.strokeWidth;
    strokeValSpan.textContent = style.strokeWidth + "px";
  }
  const strokeColorPicker = document.createElement("input");
  strokeColorPicker.type = "color";
  strokeColorPicker.className = "ui-color-input ui-color-swatch stroke-color-picker";
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
  strokeToolbar.append(strokeLabel, strokeColorPicker, strokeSlider, strokeValSpan);

  body.append(colorToolbar, strokeToolbar);
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
