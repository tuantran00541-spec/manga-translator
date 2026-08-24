function createBoxItem(pageIndex, boxIndex) {
  const item = document.createElement("div");
  item.className = "box-item";

  const header = document.createElement("div");
  header.className = "box-header";

  const original = document.createElement("div");
  original.className = "original";
  original.textContent = "Đang OCR...";
  header.appendChild(original);

  const removeBtn = document.createElement("button");
  removeBtn.className = "remove-box-btn";
  removeBtn.textContent = "Xóa";
  removeBtn.addEventListener("click", () => removeBoxAndRepaint(pageIndex, boxIndex, item));
  header.appendChild(removeBtn);

  item.appendChild(header);

  const styleToolbar = document.createElement("div");
  styleToolbar.className = "font-style-toolbar";

  const fontSelect = document.createElement("select");
  fontSelect.className = "font-family-select";
  fontSelect.title = "Chọn kiểu chữ";

  if (availableFonts.length === 0) {
    fontSelect.innerHTML = '<option value="default">Mặc định (Comic)</option>';
  } else {
    availableFonts.forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f.id;
      opt.textContent = f.name;
      fontSelect.appendChild(opt);
    });
  }

  fontSelect.addEventListener("change", (e) => {
    textarea.dataset.font = e.target.value;
  });
  styleToolbar.appendChild(fontSelect);

  const boldBtn = document.createElement("button");
  boldBtn.type = "button";
  boldBtn.className = "bold-toggle-btn";
  boldBtn.textContent = "B";
  boldBtn.title = "In đậm chữ";
  boldBtn.addEventListener("click", () => {
    const isBold = textarea.dataset.bold === "true";
    textarea.dataset.bold = isBold ? "false" : "true";
    boldBtn.classList.toggle("active", !isBold);
  });
  styleToolbar.appendChild(boldBtn);

  const sizeGroup = document.createElement("div");
  sizeGroup.className = "font-size-group";

  const sizeLabel = document.createElement("span");
  sizeLabel.className = "size-label";
  sizeLabel.textContent = "Cỡ:";

  const sizeValSpan = document.createElement("span");
  sizeValSpan.className = "font-size-val";
  sizeValSpan.textContent = "Auto";

  const autoBtn = document.createElement("button");
  autoBtn.type = "button";
  autoBtn.className = "size-auto-btn selected";
  autoBtn.textContent = "Auto";
  autoBtn.title = "Tự động vừa ô";

  const sizeSlider = document.createElement("input");
  sizeSlider.type = "range";
  sizeSlider.className = "font-size-slider";
  sizeSlider.min = "10";
  sizeSlider.max = "60";
  sizeSlider.value = "20";
  sizeSlider.disabled = true;

  autoBtn.addEventListener("click", () => {
    const isAuto = textarea.dataset.fontSize === "auto" || !textarea.dataset.fontSize;
    if (isAuto) {
      autoBtn.classList.remove("selected");
      sizeSlider.disabled = false;
      textarea.dataset.fontSize = sizeSlider.value;
      sizeValSpan.textContent = sizeSlider.value + "px";
    } else {
      autoBtn.classList.add("selected");
      sizeSlider.disabled = true;
      textarea.dataset.fontSize = "auto";
      sizeValSpan.textContent = "Auto";
    }
  });

  sizeSlider.addEventListener("input", (e) => {
    if (textarea.dataset.fontSize !== "auto") {
      textarea.dataset.fontSize = e.target.value;
      sizeValSpan.textContent = e.target.value + "px";
    }
  });

  sizeGroup.appendChild(sizeLabel);
  sizeGroup.appendChild(autoBtn);
  sizeGroup.appendChild(sizeSlider);
  sizeGroup.appendChild(sizeValSpan);
  styleToolbar.appendChild(sizeGroup);

  item.appendChild(styleToolbar);

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
  strokeValSpan.textContent = "Auto";

  const strokeColorPicker = document.createElement("input");
  strokeColorPicker.type = "color";
  strokeColorPicker.className = "box-color-picker";
  strokeColorPicker.value = "#000000";
  strokeColorPicker.title = "Màu viền chữ";

  strokeSlider.addEventListener("input", (e) => {
    textarea.dataset.strokeWidth = e.target.value;
    strokeValSpan.textContent = e.target.value + "px";
  });

  strokeColorPicker.addEventListener("input", (e) => {
    textarea.dataset.strokeColor = e.target.value;
  });

  strokeToolbar.appendChild(strokeLabel);
  strokeToolbar.appendChild(strokeSlider);
  strokeToolbar.appendChild(strokeValSpan);
  strokeToolbar.appendChild(strokeColorPicker);
  item.appendChild(strokeToolbar);

  const bgToolbar = document.createElement("div");
  bgToolbar.className = "bg-toolbar";

  const bgLabel = document.createElement("span");
  bgLabel.className = "style-group-label";
  bgLabel.textContent = "Nền & Bo góc:";

  const bgSelect = document.createElement("select");
  bgSelect.className = "bg-color-select";
  bgSelect.innerHTML = `
    <option value="transparent">Trong suốt</option>
    <option value="#ffffff">Nền Trắng</option>
    <option value="#000000">Nền Đen</option>
  `;

  const radiusSlider = document.createElement("input");
  radiusSlider.type = "range";
  radiusSlider.className = "corner-radius-slider";
  radiusSlider.min = "0";
  radiusSlider.max = "20";
  radiusSlider.value = "0";
  radiusSlider.title = "Độ bo góc nền";

  const radiusValSpan = document.createElement("span");
  radiusValSpan.className = "style-val-span";
  radiusValSpan.textContent = "0px";

  bgSelect.addEventListener("change", (e) => {
    textarea.dataset.bgColor = e.target.value;
  });

  radiusSlider.addEventListener("input", (e) => {
    textarea.dataset.cornerRadius = e.target.value;
    radiusValSpan.textContent = e.target.value + "px";
  });

  bgToolbar.appendChild(bgLabel);
  bgToolbar.appendChild(bgSelect);
  bgToolbar.appendChild(radiusSlider);
  bgToolbar.appendChild(radiusValSpan);
  item.appendChild(bgToolbar);

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

  const textarea = document.createElement("textarea");
  textarea.rows = 2;
  textarea.dataset.pageIndex = pageIndex;
  textarea.dataset.boxIndex = boxIndex;
  textarea.dataset.color = "auto";
  textarea.dataset.font = "default";
  textarea.dataset.fontSize = "auto";
  textarea.dataset.bold = "false";
  textarea.dataset.strokeWidth = "auto";
  textarea.dataset.strokeColor = "auto";
  textarea.dataset.bgColor = "transparent";
  textarea.dataset.cornerRadius = "0";

  const customPicker = document.createElement("input");
  customPicker.type = "color";
  customPicker.className = "box-color-picker";
  customPicker.value = "#ffffff";
  customPicker.title = "Chọn màu tùy chỉnh";

  colors.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "color-btn" + (c.value === "auto" ? " selected" : "");
    btn.title = c.name;
    btn.style.background = c.bg;
    btn.addEventListener("click", () => {
      colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      textarea.dataset.color = c.value;
    });
    colorToolbar.appendChild(btn);
  });

  customPicker.addEventListener("input", (e) => {
    colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
    textarea.dataset.color = e.target.value;
  });
  colorToolbar.appendChild(customPicker);

  item.appendChild(colorToolbar);
  item.appendChild(textarea);

  const draftKey = `${pageIndex}_${boxIndex}`;
  const drafts = currentManifest.drafts || {};
  if (drafts[draftKey]) {
    const d = drafts[draftKey];
    if (d.text) textarea.value = d.text;
    if (d.color && d.color !== "auto") {
      textarea.dataset.color = d.color;
      colorToolbar.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
    }
    if (d.font) { textarea.dataset.font = d.font; fontSelect.value = d.font; }
    if (d.fontSize && d.fontSize !== "auto") {
      textarea.dataset.fontSize = d.fontSize;
      autoBtn.classList.remove("selected");
      sizeSlider.disabled = false;
      sizeSlider.value = d.fontSize;
      sizeValSpan.textContent = d.fontSize + "px";
    }
    if (d.bold === true || d.bold === "true") {
      textarea.dataset.bold = "true";
      boldBtn.classList.add("active");
    }
    if (d.strokeWidth && d.strokeWidth !== "auto") {
      textarea.dataset.strokeWidth = d.strokeWidth;
      strokeSlider.value = d.strokeWidth;
      strokeValSpan.textContent = d.strokeWidth + "px";
    }
    if (d.strokeColor && d.strokeColor !== "auto") {
      textarea.dataset.strokeColor = d.strokeColor;
      strokeColorPicker.value = d.strokeColor;
    }
    if (d.bgColor && d.bgColor !== "transparent") {
      textarea.dataset.bgColor = d.bgColor;
      bgSelect.value = d.bgColor;
    }
    if (d.cornerRadius && parseInt(d.cornerRadius) > 0) {
      textarea.dataset.cornerRadius = d.cornerRadius;
      radiusSlider.value = d.cornerRadius;
      radiusValSpan.textContent = d.cornerRadius + "px";
    }
  }

  const triggerSave = () => scheduleSaveDraft();
  textarea.addEventListener("input", triggerSave);
  fontSelect.addEventListener("change", triggerSave);
  boldBtn.addEventListener("click", triggerSave);
  sizeSlider.addEventListener("change", triggerSave);
  autoBtn.addEventListener("click", triggerSave);
  strokeSlider.addEventListener("change", triggerSave);
  strokeColorPicker.addEventListener("change", triggerSave);
  bgSelect.addEventListener("change", triggerSave);
  radiusSlider.addEventListener("change", triggerSave);
  customPicker.addEventListener("change", triggerSave);

  fetchOcr(pageIndex, boxIndex, original);

  return item;
}
