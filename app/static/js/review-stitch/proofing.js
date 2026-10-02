import { clearSelection, ownedObjects, renderOverlays, selectObject, sourceRegion } from "./overlays.js";
import { state } from "./state.js";

const SHORTCUT_HELP = [
  ["A / D", "Lát trước / lát sau"],
  ["Alt + mũi tên lên / xuống", "Vùng chữ trước / sau"],
  ["Enter", "Sửa bản dịch của vùng đang chọn"],
  ["Ctrl + Enter", "Lưu và sang vùng sau"],
  ["Esc", "Thoát ô nhập, quay về trang"],
  ["Ctrl + Z / Ctrl + Y", "Hoàn tác / làm lại"],
  ["Ctrl + F / Ctrl + H", "Soát chữ: tìm / thay thế"],
  ["1 – 9", "Áp kiểu chữ mẫu cho vùng đang chọn"],
  ["Delete", "Xóa vùng đang chọn"],
  ["V R O B E H Z", "Chọn, chữ nhật, elip, cọ, tẩy, bàn tay, thu phóng"],
  ["?", "Mở / đóng bảng phím tắt"],
];

const isEditable = (el) => el?.isContentEditable || ["input", "textarea", "select"].includes(el?.tagName?.toLowerCase());

const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function iconButton(icon, label, className) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `ui-btn ui-btn-ghost ui-btn-compact ${className}`;
  b.setAttribute("aria-label", label);
  b.title = label;
  b.append(window.createUiIcon(icon));
  return b;
}

// Text objects of the chapter in reading order: slice by slice, then top to bottom, left to right.
export function orderedObjects(shell) {
  return ownedObjects(shell)
    .map(({ desc, pageIndex, obj }) => ({ pageIndex, obj, r: sourceRegion(desc, obj.region) }))
    .sort((a, b) => (a.r.y1 - b.r.y1) || (a.r.x1 - b.r.x1))
    .map(({ pageIndex, obj }) => ({ pageIndex, obj }));
}

function overlayFor(shell, pageIndex, id) {
  return shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`);
}

async function ensureCleanView(shell, pageIndex, id) {
  if (state.variant !== "clean") {
    state.variant = "clean";
    shell._syncVariantUI?.();
    shell._rerender?.();
  }
  for (let i = 0; i < 40; i++) {
    if (overlayFor(shell, pageIndex, id)) return true;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return false;
}

// Selects a text object on the page and scrolls it into view.
export async function goTo(shell, pageIndex, id, { focusTranslation = false } = {}) {
  if (!(await ensureCleanView(shell, pageIndex, id))) return false;
  selectObject(shell, pageIndex, id);
  overlayFor(shell, pageIndex, id)?.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
  if (focusTranslation) {
    const field = shell.querySelector(".review-floating-inspector .translation-textarea");
    if (field) {
      field.focus();
      field.setSelectionRange(field.value.length, field.value.length);
    }
  }
  return true;
}

function stepObject(shell, delta, options) {
  const list = orderedObjects(shell);
  if (!list.length) return;
  const selected = window.editorState?.selectedTextObjectId;
  const pageIndex = Number(window.editorState?.activePageIndex);
  let index = list.findIndex((item) => item.obj.id === selected && item.pageIndex === pageIndex);
  if (index < 0) {
    const visible = Number(shell.closest(".review-workspace-shell")?.dataset.reviewCanonicalIndex);
    index = list.findIndex((item) => item.pageIndex >= visible);
    if (index < 0) index = 0;
    else if (delta > 0) index -= 1;
  }
  const next = list[Math.max(0, Math.min(list.length - 1, index + delta))];
  if (next) void goTo(shell, next.pageIndex, next.obj.id, options);
}

function stepSlice(shell, delta) {
  const descriptors = shell._descriptors || [];
  const viewport = shell.querySelector(".review-document-viewport");
  const image = shell.querySelector(".review-stitched-image");
  if (!descriptors.length || !viewport || !image) return;
  const current = Number(shell.closest(".review-workspace-shell")?.dataset.reviewCanonicalIndex);
  let index = descriptors.findIndex((d) => Number(d.item.canonicalIndex) === current);
  if (index < 0) index = 0;
  const target = descriptors[Math.max(0, Math.min(descriptors.length - 1, index + delta))];
  const scale = Number(image.dataset.zoomScale || 1);
  const offset = image.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
  viewport.scrollTo({ top: viewport.scrollTop + offset + target.sourceY1 * scale - 8, behavior: "smooth" });
}

function applyPresetKey(shell, digit) {
  const preset = window.stylePresets?.list()?.[digit - 1];
  const id = window.editorState?.selectedTextObjectId;
  const pageIndex = Number(window.editorState?.activePageIndex);
  if (!id) return;
  if (!preset) {
    window.showToast?.(`Chưa có kiểu chữ mẫu số ${digit}. Lưu kiểu mẫu trong mục Kiểu chữ.`, "info");
    return;
  }
  if (window.stylePresets.applyPreset(preset, [{ pageIndex, id }])) {
    window.renderEditorPanel?.(pageIndex);
    window.showToast?.(`Đã áp kiểu "${preset.name}".`, "success");
  }
}

function mountHelp(shell, signal) {
  const dialog = document.createElement("div");
  dialog.className = "review-shortcut-help";
  dialog.hidden = true;
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-label", "Phím tắt");
  const title = document.createElement("strong");
  title.textContent = "Phím tắt";
  const list = document.createElement("dl");
  SHORTCUT_HELP.forEach(([keys, text]) => {
    const dt = document.createElement("dt");
    dt.textContent = keys;
    const dd = document.createElement("dd");
    dd.textContent = text;
    list.append(dt, dd);
  });
  dialog.append(title, list);
  shell.appendChild(dialog);
  dialog.addEventListener("click", () => { dialog.hidden = true; }, { signal });
  return () => { dialog.hidden = !dialog.hidden; };
}

function mountProofPanel(shell, signal) {
  const panel = document.createElement("aside");
  panel.className = "review-proof-panel";
  panel.hidden = true;
  panel.setAttribute("aria-label", "Soát chữ toàn chương");

  const header = document.createElement("div");
  header.className = "review-proof-header";
  const title = document.createElement("strong");
  title.textContent = "Soát chữ";
  const count = document.createElement("span");
  count.className = "review-proof-count";
  const close = iconButton("close", "Đóng bảng soát chữ", "review-proof-close");
  header.append(title, count, close);

  const search = document.createElement("div");
  search.className = "review-proof-search";
  const find = document.createElement("input");
  find.type = "search";
  find.className = "ui-input review-proof-find";
  find.placeholder = "Tìm (Ctrl+F)…";
  find.setAttribute("aria-label", "Tìm chữ trong chương");
  const field = document.createElement("select");
  field.className = "ui-select review-proof-field";
  field.setAttribute("aria-label", "Tìm trong");
  field.add(new Option("Bản dịch", "translation"));
  field.add(new Option("Chữ gốc", "ocr_text"));
  const matchCase = document.createElement("button");
  matchCase.type = "button";
  matchCase.className = "ui-btn ui-btn-ghost ui-btn-compact review-proof-case";
  matchCase.textContent = "Aa";
  matchCase.title = "Phân biệt hoa thường";
  matchCase.setAttribute("aria-pressed", "false");
  search.append(find, field, matchCase);

  const replaceRow = document.createElement("div");
  replaceRow.className = "review-proof-replace";
  const replaceInput = document.createElement("input");
  replaceInput.className = "ui-input review-proof-replace-input";
  replaceInput.placeholder = "Thay bằng (Ctrl+H)…";
  replaceInput.setAttribute("aria-label", "Thay bằng");
  const replaceAll = document.createElement("button");
  replaceAll.type = "button";
  replaceAll.className = "ui-btn ui-btn-ghost ui-btn-compact review-proof-replace-all";
  replaceAll.textContent = "Thay tất cả";
  replaceRow.append(replaceInput, replaceAll);

  const tools = document.createElement("div");
  tools.className = "review-proof-tools";
  const checkAllLabel = document.createElement("label");
  checkAllLabel.className = "review-proof-check-all";
  const checkAll = document.createElement("input");
  checkAll.type = "checkbox";
  checkAll.setAttribute("aria-label", "Chọn mọi vùng đang hiện");
  checkAllLabel.append(checkAll, document.createTextNode(" Tất cả"));
  const untranslatedLabel = document.createElement("label");
  untranslatedLabel.className = "review-proof-untranslated-label";
  const untranslated = document.createElement("input");
  untranslated.type = "checkbox";
  untranslated.className = "review-proof-untranslated";
  untranslatedLabel.append(untranslated, document.createTextNode(" Chưa dịch"));
  const presetSelect = document.createElement("select");
  presetSelect.className = "ui-select review-proof-preset";
  presetSelect.setAttribute("aria-label", "Kiểu chữ mẫu cho các vùng đã chọn");
  const presetApply = document.createElement("button");
  presetApply.type = "button";
  presetApply.className = "ui-btn ui-btn-ghost ui-btn-compact review-proof-preset-apply";
  tools.append(checkAllLabel, untranslatedLabel, presetSelect, presetApply);

  const list = document.createElement("ul");
  list.className = "review-proof-list";
  panel.append(header, search, replaceRow, tools, list);
  shell.appendChild(panel);

  const checked = new Set();
  const rowKey = (pageIndex, id) => `${pageIndex}:${id}`;

  const matcher = () => {
    const query = find.value;
    if (!query) return null;
    return new RegExp(escapeRegExp(query), matchCase.getAttribute("aria-pressed") === "true" ? "g" : "gi");
  };

  const visibleItems = () => {
    const re = matcher();
    return orderedObjects(shell).filter(({ obj }) => {
      if (untranslated.checked && String(obj.translation || "").trim()) return false;
      if (!re) return true;
      re.lastIndex = 0;
      return re.test(String(obj[field.value] || ""));
    });
  };

  const syncPresetControls = () => {
    const presets = window.stylePresets?.list() || [];
    const selected = presetSelect.value;
    presetSelect.replaceChildren(new Option(presets.length ? "Kiểu mẫu…" : "Chưa có kiểu mẫu", ""));
    presets.forEach((p) => presetSelect.add(new Option(p.name, p.name)));
    if (presets.some((p) => p.name === selected)) presetSelect.value = selected;
    presetApply.textContent = `Áp cho ${checked.size} vùng`;
    presetApply.disabled = !presetSelect.value || !checked.size;
  };

  const syncCount = () => {
    const all = orderedObjects(shell);
    const done = all.filter(({ obj }) => String(obj.translation || "").trim()).length;
    const shown = list.children.length;
    count.textContent = find.value || untranslated.checked
      ? `${shown} / ${all.length} vùng`
      : `${all.length} vùng · ${done} đã dịch`;
  };

  const autoSize = (ta) => {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(160, ta.scrollHeight + 2)}px`;
  };

  const focusRow = (row) => {
    const ta = row?.querySelector(".review-proof-translation");
    if (!ta) return;
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  };

  function buildRow({ pageIndex, obj }) {
    const row = document.createElement("li");
    row.className = "review-proof-row";
    row.dataset.pageIndex = String(pageIndex);
    row.dataset.objectId = String(obj.id);
    if (pageIndex === Number(window.editorState?.activePageIndex) && obj.id === window.editorState?.selectedTextObjectId) row.classList.add("active");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.className = "review-proof-check";
    box.checked = checked.has(rowKey(pageIndex, obj.id));
    box.setAttribute("aria-label", `Chọn vùng ở lát ${pageIndex + 1}`);
    const slice = document.createElement("span");
    slice.className = "review-proof-slice";
    slice.textContent = `Lát ${pageIndex + 1}`;
    const source = document.createElement("div");
    source.className = "review-proof-source";
    source.textContent = obj.ocr_text || "—";
    const ta = document.createElement("textarea");
    ta.className = "ui-textarea review-proof-translation";
    ta.rows = 1;
    ta.value = obj.translation || "";
    ta.placeholder = "Chưa dịch";
    ta.dataset.textObjectId = String(obj.id);
    ta.setAttribute("aria-label", `Bản dịch, lát ${pageIndex + 1}`);
    box.addEventListener("change", () => {
      if (box.checked) checked.add(rowKey(pageIndex, obj.id));
      else checked.delete(rowKey(pageIndex, obj.id));
      syncPresetControls();
    });
    ta.addEventListener("focus", () => {
      list.querySelectorAll(".review-proof-row.active").forEach((r) => r.classList.remove("active"));
      row.classList.add("active");
      const live = window.findTextObject?.(pageIndex, obj.id);
      if (live) window.editorHistory?.watch(pageIndex, live);
      void goTo(shell, pageIndex, obj.id);
    });
    ta.addEventListener("input", () => {
      const live = window.findTextObject?.(pageIndex, obj.id);
      if (!live) return;
      live.translation = ta.value;
      autoSize(ta);
      window.scheduleTextObjectPersist?.(pageIndex, obj.id);
      syncCount();
    });
    ta.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        focusRow(event.shiftKey ? row.previousElementSibling : row.nextElementSibling);
      }
    });
    row.addEventListener("click", (event) => {
      if (event.target.closest("textarea, input")) return;
      focusRow(row);
    });
    row.append(box, slice, source, ta);
    return row;
  }

  const rebuild = () => {
    if (panel.hidden) return;
    const items = visibleItems();
    list.replaceChildren(...items.map(buildRow));
    list.querySelectorAll(".review-proof-translation").forEach(autoSize);
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "review-proof-empty";
      empty.textContent = find.value ? "Không có vùng nào khớp." : "Chưa có vùng chữ nào.";
      list.append(empty);
    }
    checkAll.checked = items.length > 0 && items.every(({ pageIndex, obj }) => checked.has(rowKey(pageIndex, obj.id)));
    syncPresetControls();
    syncCount();
  };

  const setOpen = (open, focus = "find") => {
    panel.hidden = !open;
    shell.classList.toggle("review-proof-open", open);
    toggle.setAttribute("aria-pressed", String(open));
    toggle.classList.toggle("ui-btn-primary", open);
    if (!open) return;
    window.stylePresets?.load();
    rebuild();
    (focus === "replace" ? replaceInput : find).focus();
  };

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "ui-btn ui-btn-ghost ui-btn-compact review-proof-toggle";
  toggle.setAttribute("aria-pressed", "false");
  toggle.title = "Soát chữ toàn chương (Ctrl+F)";
  const toggleLabel = document.createElement("span");
  toggleLabel.className = "review-proof-toggle-label";
  toggleLabel.textContent = "Soát chữ";
  toggle.setAttribute("aria-label", "Soát chữ toàn chương");
  toggle.append(window.createUiIcon("list"), toggleLabel);
  toggle.addEventListener("click", () => setOpen(panel.hidden), { signal });
  close.addEventListener("click", () => setOpen(false), { signal });

  let searchTimer = 0;
  find.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(rebuild, 150); }, { signal });
  field.addEventListener("change", rebuild, { signal });
  untranslated.addEventListener("change", rebuild, { signal });
  matchCase.addEventListener("click", () => {
    matchCase.setAttribute("aria-pressed", String(matchCase.getAttribute("aria-pressed") !== "true"));
    matchCase.classList.toggle("ui-btn-primary", matchCase.getAttribute("aria-pressed") === "true");
    rebuild();
  }, { signal });
  find.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      focusRow(list.querySelector(".review-proof-row"));
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (find.value) { find.value = ""; rebuild(); } else setOpen(false);
    }
  }, { signal });
  checkAll.addEventListener("change", () => {
    visibleItems().forEach(({ pageIndex, obj }) => {
      if (checkAll.checked) checked.add(rowKey(pageIndex, obj.id));
      else checked.delete(rowKey(pageIndex, obj.id));
    });
    list.querySelectorAll(".review-proof-check").forEach((box) => { box.checked = checkAll.checked; });
    syncPresetControls();
  }, { signal });
  presetSelect.addEventListener("change", syncPresetControls, { signal });
  presetApply.addEventListener("click", () => {
    const preset = (window.stylePresets?.list() || []).find((p) => p.name === presetSelect.value);
    const targets = [...checked].map((key) => {
      const [pageIndex, ...rest] = key.split(":");
      return { pageIndex: Number(pageIndex), id: rest.join(":") };
    });
    const applied = window.stylePresets?.applyPreset(preset, targets) || 0;
    if (!applied) return;
    const selected = window.editorState?.selectedTextObjectId;
    if (selected) window.renderEditorPanel?.(Number(window.editorState.activePageIndex));
    window.showToast?.(`Đã áp kiểu "${preset.name}" cho ${applied} vùng. Ctrl+Z để hoàn tác.`, "success");
  }, { signal });

  replaceAll.addEventListener("click", () => {
    const re = matcher();
    if (!re) {
      find.focus();
      return window.showToast?.("Nhập chữ cần tìm trước.", "info");
    }
    const key = field.value;
    const targets = visibleItems();
    let occurrences = 0;
    const changed = window.editorHistory.batch(targets.map(({ pageIndex, obj }) => ({ pageIndex, id: obj.id })), () => {
      targets.forEach(({ pageIndex, obj }) => {
        const text = String(obj[key] || "");
        re.lastIndex = 0;
        const hits = text.match(re)?.length || 0;
        if (!hits) return;
        occurrences += hits;
        obj[key] = text.replace(re, () => replaceInput.value);
        window.scheduleTextObjectPersist?.(pageIndex, obj.id);
      });
    });
    rebuild();
    if (window.editorState?.selectedTextObjectId) window.renderEditorPanel?.(Number(window.editorState.activePageIndex));
    window.showToast?.(
      changed ? `Đã thay ${occurrences} chỗ trong ${changed} vùng. Ctrl+Z để hoàn tác.` : "Không có chỗ nào để thay.",
      changed ? "success" : "info",
    );
  }, { signal });
  replaceInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); replaceAll.click(); }
  }, { signal });

  document.addEventListener("text-object-edited", (event) => {
    if (panel.hidden) return;
    const { pageIndex, id } = event.detail || {};
    const row = list.querySelector(`.review-proof-row[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`);
    const obj = window.findTextObject?.(pageIndex, id);
    if (!row || !obj) return;
    const ta = row.querySelector(".review-proof-translation");
    if (ta && document.activeElement !== ta && ta.value !== (obj.translation || "")) { ta.value = obj.translation || ""; autoSize(ta); }
    const source = row.querySelector(".review-proof-source");
    if (source) source.textContent = obj.ocr_text || "—";
    syncCount();
  }, { signal });
  document.addEventListener("editor-history-applied", rebuild, { signal });
  document.addEventListener("style-presets-changed", syncPresetControls, { signal });

  return { toggle, setOpen, isOpen: () => !panel.hidden, rebuild };
}

function mountHistoryButtons(signal) {
  const undo = iconButton("undo", "Hoàn tác (Ctrl+Z)", "review-undo-btn");
  const redo = iconButton("redo", "Làm lại (Ctrl+Y)", "review-redo-btn");
  const sync = () => {
    undo.disabled = !window.editorHistory?.canUndo();
    redo.disabled = !window.editorHistory?.canRedo();
  };
  undo.addEventListener("click", () => window.editorHistory?.undo(), { signal });
  redo.addEventListener("click", () => window.editorHistory?.redo(), { signal });
  document.addEventListener("editor-history-changed", sync, { signal });
  sync();
  return [undo, redo];
}

export function installProofing(shell, signal) {
  const left = shell.querySelector(".review-docbar-left");
  const proof = mountProofPanel(shell, signal);
  const toggleHelp = mountHelp(shell, signal);
  const help = document.createElement("button");
  help.type = "button";
  help.className = "ui-btn ui-btn-ghost ui-btn-compact review-help-btn";
  help.textContent = "?";
  help.title = "Phím tắt (?)";
  help.setAttribute("aria-label", "Phím tắt");
  help.addEventListener("click", toggleHelp, { signal });
  left?.append(...mountHistoryButtons(signal), proof.toggle, help);
  const viewport = shell.querySelector(".review-document-viewport");
  if (viewport) viewport.tabIndex = -1;
  const onCanvas = (target) => target === document.body || target === document.documentElement || Boolean(target?.closest?.(".review-document-viewport"));

  document.addEventListener("editor-history-applied", (event) => {
    renderOverlays(shell, signal);
    const focus = event.detail?.focus;
    if (focus?.id) void goTo(shell, focus.pageIndex, focus.id);
    else clearSelection(shell);
  }, { signal });

  window.addEventListener("keydown", (event) => {
    if (shell.closest(".review-workspace-shell")?.classList.contains("review-busy")) return;
    const key = event.key;
    const ctrl = event.ctrlKey || event.metaKey;
    if (ctrl && !event.altKey && (key === "f" || key === "F" || key === "h" || key === "H")) {
      event.preventDefault();
      proof.setOpen(true, key.toLowerCase() === "h" ? "replace" : "find");
      return;
    }
    const target = event.target;
    if (isEditable(target)) {
      const inInspector = target.closest?.(".review-floating-inspector");
      if (key === "Escape" && (inInspector || target.closest?.(".review-proof-list"))) {
        event.preventDefault();
        target.blur();
        shell.querySelector(".review-document-viewport")?.focus({ preventScroll: true });
      } else if (key === "Enter" && ctrl && inInspector && target.classList.contains("translation-textarea")) {
        event.preventDefault();
        stepObject(shell, event.shiftKey ? -1 : 1, { focusTranslation: true });
      }
      return;
    }
    if (ctrl && !event.altKey && (key === "z" || key === "Z")) {
      event.preventDefault();
      if (event.shiftKey) window.editorHistory?.redo();
      else window.editorHistory?.undo();
      return;
    }
    if (ctrl && !event.altKey && (key === "y" || key === "Y")) {
      event.preventDefault();
      window.editorHistory?.redo();
      return;
    }
    if (ctrl) return;
    if (event.altKey && (key === "ArrowDown" || key === "ArrowUp")) {
      event.preventDefault();
      stepObject(shell, key === "ArrowDown" ? 1 : -1);
      return;
    }
    if (event.altKey) return;
    if (key === "?") {
      event.preventDefault();
      toggleHelp();
    } else if (key === "a" || key === "A" || key === "d" || key === "D") {
      event.preventDefault();
      stepSlice(shell, key.toLowerCase() === "d" ? 1 : -1);
    } else if (key === "Enter" && onCanvas(target) && window.editorState?.selectedTextObjectId) {
      event.preventDefault();
      void goTo(shell, Number(window.editorState.activePageIndex), window.editorState.selectedTextObjectId, { focusTranslation: true });
    } else if (/^[1-9]$/.test(key) && onCanvas(target) && window.editorState?.selectedTextObjectId) {
      event.preventDefault();
      applyPresetKey(shell, Number(key));
    }
  }, { signal });
}
