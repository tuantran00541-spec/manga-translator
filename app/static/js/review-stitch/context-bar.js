import { state } from "./state.js";

const GAP = 10;

function cleanFontLabel(name) {
  const raw = String(name || "").trim();
  const wrapped = raw.match(/^Mac-dinh(?:-\d+)?\s*\((.+)\)$/i);
  if (wrapped) return wrapped[1].trim();
  return raw || "Mặc định";
}

function resolveColorHex(raw, fallback) {
  const s = String(raw || "").trim();
  return /^#[0-9a-f]{6}$/i.test(s) ? s : fallback;
}

/**
 * Floating Contextual TypeBar inspired by Koharu's ContextBar + TypeBar.
 * Anchors cleanly below (or above) the active text selection on the canvas and
 * applies typography / color / alignment changes across single or multi-selection
 * in one atomic undoable batch.
 */
export function mountContextTypeBar(shell, signal, hooks) {
  shell.querySelector(".review-context-typebar")?.remove();

  const bar = document.createElement("div");
  bar.className = "review-context-typebar";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "Thanh chỉnh chữ nhanh");
  bar.hidden = true;

  const badge = document.createElement("span");
  badge.className = "review-ctx-count";
  badge.hidden = true;

  const fontSelect = document.createElement("select");
  fontSelect.className = "ui-select review-ctx-font";
  fontSelect.setAttribute("aria-label", "Phông chữ");
  fontSelect.title = "Phông chữ";

  const sizeGroup = document.createElement("div");
  sizeGroup.className = "review-ctx-size-group";
  const sizeDown = document.createElement("button");
  sizeDown.type = "button";
  sizeDown.className = "ui-btn ui-btn-ghost review-ctx-icon-btn";
  sizeDown.setAttribute("aria-label", "Giảm cỡ chữ");
  sizeDown.title = "Giảm cỡ chữ";
  sizeDown.append(window.createUiIcon("minus"));

  const sizeInput = document.createElement("input");
  sizeInput.type = "number";
  sizeInput.min = "6";
  sizeInput.max = "96";
  sizeInput.step = "1";
  sizeInput.placeholder = "Auto";
  sizeInput.className = "ui-input review-ctx-size-input";
  sizeInput.setAttribute("aria-label", "Cỡ chữ");
  sizeInput.title = "Cỡ chữ (để trống = Tự động)";

  const sizeUp = document.createElement("button");
  sizeUp.type = "button";
  sizeUp.className = "ui-btn ui-btn-ghost review-ctx-icon-btn";
  sizeUp.setAttribute("aria-label", "Tăng cỡ chữ");
  sizeUp.title = "Tăng cỡ chữ";
  sizeUp.append(window.createUiIcon("plus"));

  const sizeAuto = document.createElement("button");
  sizeAuto.type = "button";
  sizeAuto.className = "ui-btn ui-btn-ghost review-ctx-pill";
  sizeAuto.textContent = "Auto";
  sizeAuto.title = "Cỡ chữ tự động";
  sizeGroup.append(sizeDown, sizeInput, sizeUp, sizeAuto);

  const boldBtn = document.createElement("button");
  boldBtn.type = "button";
  boldBtn.className = "ui-btn ui-btn-ghost review-ctx-icon-btn";
  boldBtn.setAttribute("aria-label", "In đậm");
  boldBtn.title = "In đậm";
  boldBtn.append(window.createUiIcon("bold"));

  const alignGroup = document.createElement("div");
  alignGroup.className = "review-ctx-align-group";
  alignGroup.setAttribute("role", "group");
  alignGroup.setAttribute("aria-label", "Căn lề ngang");
  const alignBtns = [
    ["left", "align-left", "Căn trái"],
    ["center", "align-center", "Căn giữa"],
    ["right", "align-right", "Căn phải"],
  ].map(([val, icon, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ui-btn ui-btn-ghost review-ctx-icon-btn";
    b.dataset.align = val;
    b.setAttribute("aria-label", label);
    b.title = label;
    b.append(window.createUiIcon(icon));
    alignGroup.appendChild(b);
    return b;
  });

  const colorGroup = document.createElement("div");
  colorGroup.className = "review-ctx-colors";
  const textColor = document.createElement("input");
  textColor.type = "color";
  textColor.className = "ui-color-swatch review-ctx-color";
  textColor.setAttribute("aria-label", "Màu chữ");
  textColor.title = "Màu chữ";
  const strokeColor = document.createElement("input");
  strokeColor.type = "color";
  strokeColor.className = "ui-color-swatch review-ctx-stroke";
  strokeColor.setAttribute("aria-label", "Màu viền chữ");
  strokeColor.title = "Màu viền chữ";
  colorGroup.append(textColor, strokeColor);

  const sep = document.createElement("span");
  sep.className = "review-ctx-sep";

  const dupBtn = document.createElement("button");
  dupBtn.type = "button";
  dupBtn.className = "ui-btn ui-btn-ghost review-ctx-icon-btn";
  dupBtn.setAttribute("aria-label", "Nhân đôi vùng chọn (Ctrl+D)");
  dupBtn.title = "Nhân đôi vùng chọn (Ctrl+D)";
  dupBtn.append(window.createUiIcon("copy"));

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "ui-btn ui-btn-ghost review-ctx-icon-btn review-ctx-danger";
  delBtn.setAttribute("aria-label", "Xóa vùng chọn (Delete)");
  delBtn.title = "Xóa vùng chọn (Delete)";
  delBtn.append(window.createUiIcon("trash"));

  bar.append(badge, fontSelect, sizeGroup, boldBtn, alignGroup, colorGroup, sep, dupBtn, delBtn);
  const hostPane = shell._canvasPane || shell;
  hostPane.appendChild(bar);

  let fontSig = "";
  const syncFonts = () => {
    const list = (window.availableFonts || []).filter((f) => f && f.id !== "auto");
    const entries = list.length ? list : [{ id: "default", name: "Mặc định" }];
    const sig = entries.map((e) => e.id).join("|");
    if (sig === fontSig) return;
    fontSig = sig;
    fontSelect.replaceChildren();
    entries.forEach((f) => fontSelect.add(new Option(cleanFontLabel(f.name), f.id)));
  };

  const applyToSelection = (mutator) => {
    const targets = hooks.getSelectedTargets(shell);
    if (!targets.length) return;
    const batchTargets = targets.map((t) => ({ pageIndex: t.pageIndex, id: t.id }));
    const run = () => {
      targets.forEach(({ pageIndex, id, obj }) => {
        obj.style = Object.assign({}, window.DEFAULT_TEXT_OBJECT_STYLE, obj.style || {});
        mutator(obj);
        window.scheduleTextObjectPersist?.(pageIndex, id);
        hooks.syncOverlay(shell, pageIndex, id);
      });
    };
    if (window.editorHistory) window.editorHistory.batch(batchTargets, run);
    else run();
    const activePage = Number(window.editorState?.activePageIndex ?? targets[0].pageIndex);
    window.renderEditorPanel?.(activePage);
    sync();
  };

  const stepSize = (delta) => {
    const targets = hooks.getSelectedTargets(shell);
    if (!targets.length) return;
    const first = targets[0].obj.style?.fontSize;
    const base = Number.isFinite(Number(first)) && Number(first) > 0 ? Number(first) : 20;
    const next = Math.max(6, Math.min(96, Math.round(base + delta)));
    applyToSelection((obj) => { obj.style.fontSize = String(next); });
  };

  fontSelect.addEventListener("change", () => {
    const val = fontSelect.value;
    applyToSelection((obj) => {
      obj.style.font = val;
      obj.font_selection_mode = "user";
      obj.font_ai_id = null;
      obj.font_match = null;
    });
  }, { signal });

  sizeDown.addEventListener("click", () => stepSize(-2), { signal });
  sizeUp.addEventListener("click", () => stepSize(2), { signal });
  sizeAuto.addEventListener("click", () => applyToSelection((obj) => { obj.style.fontSize = "auto"; }), { signal });
  sizeInput.addEventListener("change", () => {
    const raw = sizeInput.value.trim();
    if (!raw) return applyToSelection((obj) => { obj.style.fontSize = "auto"; });
    const n = Math.max(6, Math.min(96, Math.round(Number(raw))));
    if (Number.isFinite(n)) applyToSelection((obj) => { obj.style.fontSize = String(n); });
  }, { signal });

  boldBtn.addEventListener("click", () => {
    const next = boldBtn.getAttribute("aria-pressed") !== "true";
    applyToSelection((obj) => { obj.style.bold = next; });
  }, { signal });

  alignBtns.forEach((b) => {
    b.addEventListener("click", () => {
      applyToSelection((obj) => { obj.style.horizontalAlign = b.dataset.align; });
    }, { signal });
  });

  textColor.addEventListener("input", () => {
    applyToSelection((obj) => { obj.style.color = textColor.value; });
  }, { signal });

  strokeColor.addEventListener("input", () => {
    applyToSelection((obj) => {
      obj.style.strokeColor = strokeColor.value;
      if (obj.style.strokeWidth === "0" || obj.style.strokeWidth === 0) obj.style.strokeWidth = "2";
    });
  }, { signal });

  dupBtn.addEventListener("click", () => hooks.duplicateSelection(shell), { signal });
  delBtn.addEventListener("click", () => hooks.deleteSelection(shell), { signal });

  // Prevent pointer clicks on the bar from clearing selection on the canvas below.
  bar.addEventListener("pointerdown", (e) => e.stopPropagation(), { signal });

  function position(targets) {
    const viewport = shell.querySelector(".review-document-viewport");
    const image = shell.querySelector(".review-stitched-image");
    if (!viewport || !image) { bar.hidden = true; return; }
    const hostRect = hostPane.getBoundingClientRect();
    const vpRect = viewport.getBoundingClientRect();
    const imgRect = image.getBoundingClientRect();
    const scale = Number(image.dataset.zoomScale || 1);
    if (!imgRect.width || !scale) { bar.hidden = true; return; }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const { desc, obj } of targets) {
      const r = hooks.sourceRegion(desc, obj.region);
      minX = Math.min(minX, r.x1);
      minY = Math.min(minY, r.y1);
      maxX = Math.max(maxX, r.x2);
      maxY = Math.max(maxY, r.y2);
    }
    if (!Number.isFinite(minX)) { bar.hidden = true; return; }

    const selLeft = imgRect.left + minX * scale;
    const selRight = imgRect.left + maxX * scale;
    const selTop = imgRect.top + minY * scale;
    const selBottom = imgRect.top + maxY * scale;

    // Hide if selection is scrolled completely out of the viewport.
    if (selBottom < vpRect.top || selTop > vpRect.bottom) {
      bar.hidden = true;
      return;
    }

    bar.hidden = false;
    const barRect = bar.getBoundingClientRect();
    const bw = barRect.width || 420;
    const bh = barRect.height || 38;

    const minLeft = Math.max(66, vpRect.left - hostRect.left + 8);
    const maxLeft = Math.max(minLeft, hostRect.width - bw - 12);
    const center = ((selLeft + selRight) / 2) - hostRect.left - bw / 2;
    const left = Math.max(minLeft, Math.min(maxLeft, center));

    const minTop = vpRect.top - hostRect.top + 8;
    const maxTop = vpRect.bottom - hostRect.top - bh - 10;
    let top = selBottom - hostRect.top + GAP;
    if (top > maxTop) {
      const above = selTop - hostRect.top - bh - GAP;
      top = above >= minTop ? above : maxTop;
    }
    top = Math.max(minTop, Math.min(maxTop, top));

    bar.style.left = `${Math.round(left)}px`;
    bar.style.top = `${Math.round(top)}px`;
  }

  function sync() {
    if (state.variant !== "clean" || state.tool !== "select" || !state.showOverlays || state.peekOriginal) {
      bar.hidden = true;
      return;
    }
    const targets = hooks.getSelectedTargets(shell);
    if (!targets.length) {
      bar.hidden = true;
      return;
    }

    syncFonts();
    if (targets.length > 1) {
      badge.hidden = false;
      badge.textContent = `${targets.length} vùng`;
    } else {
      badge.hidden = true;
    }

    const primary = targets[0].obj;
    const style = Object.assign({}, window.DEFAULT_TEXT_OBJECT_STYLE, primary.style || {});
    const shownFont = window.shownFontId ? window.shownFontId(primary) : (style.font || "default");
    if ([...fontSelect.options].some((o) => o.value === shownFont)) fontSelect.value = shownFont;

    const isAutoSize = !style.fontSize || style.fontSize === "auto";
    sizeInput.value = isAutoSize ? "" : String(style.fontSize);
    sizeAuto.classList.toggle("active", isAutoSize);
    sizeAuto.setAttribute("aria-pressed", String(isAutoSize));

    const isBold = style.bold === true;
    boldBtn.classList.toggle("active", isBold);
    boldBtn.setAttribute("aria-pressed", String(isBold));

    const hAlign = style.horizontalAlign || "center";
    alignBtns.forEach((b) => {
      const active = b.dataset.align === hAlign;
      b.classList.toggle("active", active);
      b.setAttribute("aria-pressed", String(active));
    });

    textColor.value = resolveColorHex(style.color, resolveColorHex(primary.ocr_text_color, "#111111"));
    strokeColor.value = resolveColorHex(style.strokeColor, "#ffffff");

    position(targets);
  }

  const viewport = shell.querySelector(".review-document-viewport");
  let raf = 0;
  const scheduleSync = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; sync(); });
  };
  viewport?.addEventListener("scroll", scheduleSync, { passive: true, signal });
  window.addEventListener("resize", scheduleSync, { signal });
  document.addEventListener("text-object-edited", scheduleSync, { signal });
  signal.addEventListener("abort", () => { cancelAnimationFrame(raf); bar.remove(); }, { once: true });

  shell._syncContextBar = sync;
  return { sync, bar };
}
