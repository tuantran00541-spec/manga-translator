import { MIN_BOX, chapterKey, isAutoSynced, markAutoSynced, parseSelectionKey, selectionKey, state } from "./state.js";
import { setTool, syncTool } from "./tools.js";

export function descriptorFor(shell, pageIndex) {
  return (shell._descriptors || []).find((d) => Number(d.item.canonicalIndex) === Number(pageIndex)) || null;
}

export function ownedObjects(shell) {
  const out = [];
  for (const desc of shell._descriptors || []) {
    const pageIndex = Number(desc.item.canonicalIndex), page = window.currentManifest?.pages?.[pageIndex];
    if (!page || page.skipped) continue;
    for (const obj of page.text_objects || []) {
      if (!obj?.id || !obj.region) continue;
      if (obj.source_missing || obj.joined_into || obj.overlap_dropped) continue;
      const center = (Number(obj.region.y1) + Number(obj.region.y2)) / 2;
      if (center >= desc.localY1 && center < desc.localY2) out.push({ desc, pageIndex, obj });
    }
  }
  return out;
}

export const sourceRegion = (desc, r) => ({
  x1: Number(r.x1), y1: desc.sourceY1 + (Number(r.y1) - desc.localY1),
  x2: Number(r.x2), y2: desc.sourceY1 + (Number(r.y2) - desc.localY1),
});

function _hexLuminance(color) {
  const m = /^#([0-9a-f]{6})$/i.exec(String(color || "").trim());
  if (!m) return 0;
  const r = parseInt(m[1].slice(0, 2), 16);
  const g = parseInt(m[1].slice(2, 4), 16);
  const b = parseInt(m[1].slice(4, 6), 16);
  return (r * 299 + g * 587 + b * 114) / 1000;
}

const _loadedWebFonts = new Map();

function _webFontFamily(fontId) {
  const cleanId = String(fontId || "default").trim() || "default";
  const safe = cleanId.replace(/[^a-zA-Z0-9_-]/g, "_");
  return `mt_font_${safe}`;
}

function _ensureWebFont(fontId) {
  const cleanId = String(fontId || "default").trim() || "default";
  const family = _webFontFamily(cleanId);
  if (typeof FontFace !== "function" || !document.fonts) return family;
  if (_loadedWebFonts.has(family)) return family;
  const url = `/api/fonts/${encodeURIComponent(cleanId)}/file`;
  const face = new FontFace(family, `url("${url}")`);
  const promise = face.load()
    .then((loaded) => {
      document.fonts.add(loaded);
      return family;
    })
    .catch(() => null);
  _loadedWebFonts.set(family, promise);
  return family;
}

function _renderOverlayPreview(overlay, obj, width, height) {
  let preview = overlay.querySelector(".review-overlay-preview");
  const text = String(obj.translation || "").trim();
  overlay.classList.toggle("is-translated", Boolean(text));
  overlay.classList.toggle("is-untranslated", !text);
  overlay.classList.toggle("ellipse", obj.shape === "ellipse");

  if (!text) {
    if (preview) preview.remove();
    return;
  }
  if (!preview) {
    preview = document.createElement("div");
    preview.className = "review-overlay-preview";
    const span = document.createElement("span");
    span.className = "review-overlay-preview-text";
    preview.appendChild(span);
    overlay.appendChild(preview);
  }
  const span = preview.querySelector(".review-overlay-preview-text");
  span.textContent = text;

  const style = obj.style || {};
  const hAlign = style.horizontalAlign || "center";
  const vAlign = style.verticalAlign || "middle";
  preview.style.justifyContent = hAlign === "left" ? "flex-start" : hAlign === "right" ? "flex-end" : "center";
  preview.style.alignItems = vAlign === "top" ? "flex-start" : vAlign === "bottom" ? "flex-end" : "center";
  preview.style.textAlign = hAlign;

  const bg = style.bgColor && style.bgColor !== "transparent" && style.bgColor !== "none" ? style.bgColor : "transparent";
  preview.style.background = bg;
  if (obj.shape === "ellipse") {
    preview.style.borderRadius = "999px";
  } else {
    const cr = Math.max(0, Number(style.cornerRadius || 0));
    preview.style.borderRadius = cr > 0 ? `${cr}px` : "0px";
  }

  const resolvedFontId = typeof window.shownFontId === "function"
    ? window.shownFontId(obj)
    : (obj.style?.font && obj.style.font !== "auto" ? obj.style.font : "default");
  const webFamily = _ensureWebFont(resolvedFontId);
  span.style.fontFamily = `"${webFamily}", "Comic Neue", var(--font-ui)`;

  const fixedSize = Number(style.fontSize);
  const autoSize = Math.max(11, Math.min(34, Math.round(Math.min(height * 0.34, Math.sqrt((width * height) / Math.max(8, text.length * 1.4))))));
  span.style.fontSize = `${Number.isFinite(fixedSize) && fixedSize > 0 ? fixedSize : autoSize}px`;
  span.style.fontWeight = style.bold === true ? "700" : "500";

  const colorAuto = !style.color || style.color === "auto";
  const profile = colorAuto && obj.ocr_color_profile && typeof obj.ocr_color_profile === "object" ? obj.ocr_color_profile : null;
  const grad = profile?.gradient;
  const hasGradient = Boolean(grad && Array.isArray(grad.stops) && grad.stops.length >= 2);
  const resolvedFill = !colorAuto ? style.color : (profile?.fill || obj.ocr_text_color || "#111111");

  if (hasGradient) {
    const angle = grad.direction === "horizontal" ? "90deg" : "180deg";
    span.style.backgroundImage = `linear-gradient(${angle}, ${grad.stops.join(", ")})`;
    span.style.webkitBackgroundClip = "text";
    span.style.backgroundClip = "text";
    span.style.color = "transparent";
  } else {
    span.style.backgroundImage = "";
    span.style.webkitBackgroundClip = "";
    span.style.backgroundClip = "";
    span.style.color = resolvedFill;
  }

  const autoContrastStroke = _hexLuminance(resolvedFill) > 128 ? "#000000" : "#ffffff";
  const strokeC = (style.strokeColor && style.strokeColor !== "auto")
    ? style.strokeColor
    : (profile?.stroke_color || autoContrastStroke);
  const strokeW = style.strokeWidth === "0" || style.strokeWidth === 0 ? 0 : Math.min(3, Math.max(1, Number(style.strokeWidth) || 1));

  if (hasGradient) {
    // text-shadow paints over background-clip:text in Chromium; use filter drop-shadow so gradient stays crisp.
    span.style.textShadow = "";
    const drops = [];
    if (strokeW > 0) {
      drops.push(
        `drop-shadow(-${strokeW}px 0 0 ${strokeC})`,
        `drop-shadow(${strokeW}px 0 0 ${strokeC})`,
        `drop-shadow(0 -${strokeW}px 0 ${strokeC})`,
        `drop-shadow(0 ${strokeW}px 0 ${strokeC})`,
      );
    }
    if (profile?.glow_color) {
      const gr = Math.max(2, Math.min(12, Number(profile.glow_radius || 5)));
      drops.push(`drop-shadow(0 0 ${gr}px ${profile.glow_color})`);
    }
    span.style.filter = drops.join(" ");
  } else {
    span.style.filter = "";
    const shadows = [];
    if (strokeW > 0) {
      shadows.push(
        `-${strokeW}px 0 0 ${strokeC}`,
        `${strokeW}px 0 0 ${strokeC}`,
        `0 -${strokeW}px 0 ${strokeC}`,
        `0 ${strokeW}px 0 ${strokeC}`,
      );
    }
    if (profile?.glow_color) {
      const gr = Math.max(2, Math.min(12, Number(profile.glow_radius || 5)));
      shadows.push(`0 0 ${gr * 2}px ${profile.glow_color}`);
    }
    span.style.textShadow = shadows.join(", ");
  }
}

export function syncOverlay(shell, pageIndex, id) {
  const desc = descriptorFor(shell, pageIndex);
  const obj = window.findTextObject?.(pageIndex, id);
  const overlay = shell.querySelector(
    `.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`,
  );
  if (!desc || !obj?.region || !overlay) return false;
  const r = sourceRegion(desc, obj.region);
  const w = Math.max(MIN_BOX, r.x2 - r.x1);
  const h = Math.max(MIN_BOX, r.y2 - r.y1);
  Object.assign(overlay.style, {
    left: `${r.x1}px`,
    top: `${r.y1}px`,
    width: `${w}px`,
    height: `${h}px`,
  });
  _renderOverlayPreview(overlay, obj, w, h);
  return true;
}

export function installOverlaySync() {
  window.syncOverlayForObject = (pageIndex, id) => {
    const shell = document.querySelector("#page-view.review-mode .review-document-shell");
    if (shell) syncOverlay(shell, pageIndex, id);
  };
}

export function getSelectedTargets(shell) {
  const out = [];
  const seen = new Set();
  for (const key of state.selectedKeys || []) {
    const parsed = parseSelectionKey(key);
    if (!parsed) continue;
    const obj = window.findTextObject?.(parsed.pageIndex, parsed.id);
    const desc = descriptorFor(shell, parsed.pageIndex);
    if (!obj?.region || !desc) continue;
    const k = selectionKey(parsed.pageIndex, parsed.id);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ pageIndex: parsed.pageIndex, id: parsed.id, obj, desc });
  }
  if (!out.length && window.editorState?.selectedTextObjectId) {
    const pageIndex = Number(window.editorState.activePageIndex || 0);
    const id = window.editorState.selectedTextObjectId;
    const obj = window.findTextObject?.(pageIndex, id);
    const desc = descriptorFor(shell, pageIndex);
    if (obj?.region && desc) out.push({ pageIndex, id, obj, desc });
  }
  return out;
}

function _syncSelectionHighlight(shell) {
  const keys = state.selectedKeys;
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => {
    const k = selectionKey(el.dataset.pageIndex, el.dataset.objectId);
    el.classList.toggle("selected", keys.has(k));
  });
}

export function selectMany(shell, items) {
  state.selectedKeys.clear();
  const valid = [];
  for (const item of items || []) {
    if (!item || item.id == null) continue;
    const pageIndex = Number(item.pageIndex);
    const obj = window.findTextObject?.(pageIndex, item.id);
    if (!obj) continue;
    state.selectedKeys.add(selectionKey(pageIndex, item.id));
    valid.push({ pageIndex, id: item.id });
  }
  if (!valid.length) return clearSelection(shell);
  const primary = valid[0];
  if (window.editorState) {
    window.editorState.activePageIndex = primary.pageIndex;
    window.editorState.selectedTextObjectId = primary.id;
  }
  _syncSelectionHighlight(shell);
  const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
  if (floating) {
    floating.hidden = false;
    floating.classList.add("is-open");
  }
  window.renderEditorPanel?.(primary.pageIndex);
  shell._syncContextBar?.();
  shell._refreshLayers?.();
}

export function selectObject(shell, pageIndex, id, { additive = false, toggle = false } = {}) {
  if (!window.editorState) return;
  const idx = Number(pageIndex);
  const key = selectionKey(idx, id);
  if (toggle && state.selectedKeys.has(key)) {
    state.selectedKeys.delete(key);
    if (!state.selectedKeys.size) return clearSelection(shell);
    const remaining = parseSelectionKey([...state.selectedKeys][0]);
    if (remaining) {
      window.editorState.activePageIndex = remaining.pageIndex;
      window.editorState.selectedTextObjectId = remaining.id;
      _syncSelectionHighlight(shell);
      window.renderEditorPanel?.(remaining.pageIndex);
      shell._syncContextBar?.();
      shell._refreshLayers?.();
    }
    return;
  }
  if (!additive) state.selectedKeys.clear();
  state.selectedKeys.add(key);
  window.editorState.activePageIndex = idx;
  window.editorState.selectedTextObjectId = id;
  _syncSelectionHighlight(shell);
  const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
  if (floating) {
    floating.hidden = false;
    floating.classList.add("is-open");
  }
  window.renderEditorPanel?.(idx);
  shell._syncContextBar?.();
  shell._refreshLayers?.();
}

export function clearSelection(shell) {
  state.selectedKeys.clear();
  if (window.editorState) window.editorState.selectedTextObjectId = null;
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.remove("selected"));
  const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
  if (floating && !floating.classList.contains("koharu-docked-inspector")) {
    floating.classList.remove("is-open");
    floating.hidden = true;
  }
  const panel = shell.querySelector(".translation-panel-host");
  if (panel) panel.innerHTML = '<div class="ui-empty-state koharu-empty-props">Chọn một vùng chữ trên trang hoặc trong danh sách Lớp chữ để chỉnh sửa.</div>';
  shell._syncContextBar?.();
  shell._refreshLayers?.();
}

export function startInlineTextEdit(shell, pageIndex, id) {
  if (state.variant !== "clean") return false;
  const idx = Number(pageIndex);
  const obj = window.findTextObject?.(idx, id);
  const overlay = shell.querySelector(
    `.review-text-object-overlay[data-page-index="${idx}"][data-object-id="${CSS.escape(String(id))}"]`,
  );
  if (!obj || !overlay) return false;
  if (overlay.querySelector(".review-inline-editor")) return true;

  selectObject(shell, idx, id);
  window.editorHistory?.watch(idx, obj);

  const preview = overlay.querySelector(".review-overlay-preview");
  if (preview) preview.hidden = true;
  overlay.classList.add("is-inline-editing");

  const ta = document.createElement("textarea");
  ta.className = "review-inline-editor";
  ta.value = obj.translation || "";
  ta.placeholder = "Nhập bản dịch…";
  ta.setAttribute("aria-label", "Sửa bản dịch trực tiếp trên trang");
  const style = obj.style || {};
  ta.style.textAlign = style.horizontalAlign || "center";
  ta.style.fontWeight = style.bold === true ? "700" : "500";
  const fixedSize = Number(style.fontSize);
  if (Number.isFinite(fixedSize) && fixedSize > 0) ta.style.fontSize = `${fixedSize}px`;

  let closed = false;
  const finish = ({ restoreFocus = false } = {}) => {
    if (closed) return;
    closed = true;
    ta.remove();
    overlay.classList.remove("is-inline-editing");
    if (preview) preview.hidden = false;
    syncOverlay(shell, idx, id);
    window.renderEditorPanel?.(idx);
    shell._syncContextBar?.();
    if (restoreFocus) shell.querySelector(".review-document-viewport")?.focus({ preventScroll: true });
  };

  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  ta.addEventListener("click", (e) => e.stopPropagation());
  ta.addEventListener("dblclick", (e) => e.stopPropagation());
  ta.addEventListener("input", () => {
    obj.translation = ta.value;
    window.scheduleTextObjectPersist?.(idx, id);
    const panelTa = shell.querySelector(`.review-floating-inspector .translation-textarea[data-text-object-id="${CSS.escape(String(id))}"]`);
    if (panelTa && panelTa.value !== ta.value) panelTa.value = ta.value;
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      finish({ restoreFocus: true });
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      e.stopPropagation();
      finish({ restoreFocus: false });
      shell._stepObjectFromCanvas?.(e.shiftKey ? -1 : 1, { inlineEdit: true });
    }
  });
  ta.addEventListener("blur", () => finish({ restoreFocus: false }));

  overlay.appendChild(ta);
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  return true;
}

export function overlayMode(event, overlay) {
  const rect = overlay.getBoundingClientRect(), edge = Math.max(4, Math.min(10, Math.min(rect.width, rect.height) / 3));
  const l = event.clientX - rect.left < edge, r = rect.right - event.clientX < edge, t = event.clientY - rect.top < edge, b = rect.bottom - event.clientY < edge;
  if (t && l) { return "nw"; } if (t && r) { return "ne"; } if (b && l) { return "sw"; } if (b && r) return "se";
  if (l) { return "w"; } if (r) { return "e"; } if (t) { return "n"; } if (b) { return "s"; } return "move";
}

export function cursor(mode) {
  if (mode === "nw" || mode === "se") return "nwse-resize";
  if (mode === "ne" || mode === "sw") return "nesw-resize";
  if (mode === "n" || mode === "s") return "ns-resize";
  if (mode === "e" || mode === "w") return "ew-resize";
  return "move";
}

export function installTransform(shell, overlay, desc, pageIndex, obj, signal) {
  let drag = null;
  const image = shell.querySelector(".review-stitched-image");
  overlay.addEventListener("pointerdown", (event) => {
    if (state.variant !== "clean" || state.tool !== "select" || event.button !== 0) return;
    if (event.target.closest(".review-inline-editor")) return;
    event.preventDefault(); event.stopPropagation();
    const key = selectionKey(pageIndex, obj.id);
    if (event.shiftKey) {
      selectObject(shell, pageIndex, obj.id, { additive: true, toggle: true });
      return;
    }
    if (!state.selectedKeys.has(key)) {
      selectObject(shell, pageIndex, obj.id);
    }
    const rect = image.getBoundingClientRect(), W = Number(image.dataset.sourceWidth || 0), H = Number(image.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W || !H) return;
    const targets = getSelectedTargets(shell);
    const mode = targets.length > 1 ? "move" : overlayMode(event, overlay);
    drag = {
      mode,
      x: (event.clientX - rect.left) * W / rect.width,
      y: (event.clientY - rect.top) * H / rect.height,
      items: targets.map((t) => ({ ...t, original: { ...t.obj.region } })),
    };
    overlay.classList.add("transforming"); overlay.setPointerCapture?.(event.pointerId);
  }, { signal });
  overlay.addEventListener("pointermove", (event) => {
    if (!drag) {
      if (state.tool === "select") {
        overlay.style.cursor = state.selectedKeys.size > 1 ? "move" : cursor(overlayMode(event, overlay));
      }
      return;
    }
    const rect = image.getBoundingClientRect(), W0 = Number(image.dataset.sourceWidth || 0), H0 = Number(image.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W0 || !H0) return;
    const x = (event.clientX - rect.left) * W0 / rect.width, y = (event.clientY - rect.top) * H0 / rect.height, dx = x - drag.x, dy = y - drag.y;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    for (const item of drag.items) {
      const o = item.original, W = item.desc.img.naturalWidth, minY = item.desc.localY1, maxY = item.desc.localY2;
      let { x1, y1, x2, y2 } = o;
      if (drag.mode === "move") {
        const w = o.x2 - o.x1, h = o.y2 - o.y1;
        x1 = clamp(o.x1 + dx, 0, Math.max(0, W - w)); y1 = clamp(o.y1 + dy, minY, Math.max(minY, maxY - h)); x2 = x1 + w; y2 = y1 + h;
      } else {
        if (drag.mode.includes("w")) x1 = clamp(o.x1 + dx, 0, o.x2 - MIN_BOX);
        if (drag.mode.includes("e")) x2 = clamp(o.x2 + dx, o.x1 + MIN_BOX, W);
        if (drag.mode.includes("n")) y1 = clamp(o.y1 + dy, minY, o.y2 - MIN_BOX);
        if (drag.mode.includes("s")) y2 = clamp(o.y2 + dy, o.y1 + MIN_BOX, maxY);
      }
      item.obj.region = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
      syncOverlay(shell, item.pageIndex, item.id);
    }
    shell._syncContextBar?.();
  }, { signal });
  const end = () => {
    if (!drag) return;
    const moved = drag.items; drag = null; overlay.classList.remove("transforming");
    const changed = moved.filter((item) => {
      const r = item.obj.region, o = item.original;
      return r.x1 !== o.x1 || r.y1 !== o.y1 || r.x2 !== o.x2 || r.y2 !== o.y2;
    });
    if (!changed.length) return;
    if (changed.length > 1 && window.editorHistory) {
      const finals = changed.map((item) => ({ ...item.obj.region }));
      changed.forEach((item) => { item.obj.region = { ...item.original }; });
      window.editorHistory.batch(changed.map((i) => ({ pageIndex: i.pageIndex, id: i.id })), () => {
        changed.forEach((item, idx) => {
          item.obj.region = finals[idx];
          window.scheduleGeomPersist?.(item.pageIndex, item.id);
        });
      });
    } else {
      changed.forEach((item) => window.scheduleGeomPersist?.(item.pageIndex, item.id));
    }
    shell._syncContextBar?.();
  };
  overlay.addEventListener("pointerup", end, { signal }); overlay.addEventListener("pointercancel", end, { signal });
}

export function createInlineEditor(shell, desc, pageIndex, obj, signal) {
  const overlay = document.createElement("div");
  overlay.className = `text-object-overlay review-text-object-overlay${obj.shape === "ellipse" ? " ellipse" : ""}`;
  overlay.dataset.pageIndex = String(pageIndex);
  overlay.dataset.objectId = String(obj.id);
  overlay.addEventListener("click", (event) => {
    if (event.target.closest(".review-inline-editor")) return;
    event.preventDefault();
    event.stopPropagation();
    if (!event.shiftKey && !state.selectedKeys.has(selectionKey(pageIndex, obj.id))) {
      selectObject(shell, pageIndex, obj.id);
    }
  }, { signal });
  overlay.addEventListener("dblclick", (event) => {
    if (state.variant !== "clean" || state.tool !== "select") return;
    event.preventDefault();
    event.stopPropagation();
    startInlineTextEdit(shell, pageIndex, obj.id);
  }, { signal });
  shell.querySelector(".review-stitched-image")?.appendChild(overlay);
  syncOverlay(shell, pageIndex, obj.id);
  installTransform(shell, overlay, desc, pageIndex, obj, signal);
}

export function renderOverlays(shell, signal) {
  const image = shell.querySelector(".review-stitched-image"); if (!image) return;
  image.querySelectorAll(".review-text-object-overlay,.review-text-drawing,.review-marquee-box").forEach((n) => n.remove());
  if (state.variant !== "clean") {
    shell._refreshLayers?.();
    shell._syncStatusBar?.();
    return;
  }
  for (const { desc, pageIndex, obj } of ownedObjects(shell)) createInlineEditor(shell, desc, pageIndex, obj, signal);
  _syncSelectionHighlight(shell);
  syncTool(shell);
  shell._syncContextBar?.();
  shell._refreshLayers?.();
  shell._syncStatusBar?.();
}

export function restoreWorkspaceState(shell) {
  if (shell._restoreApplied) return;
  const workspace = shell.closest(".review-workspace-shell");
  const state = workspace?._reviewRestoreState;
  if (!state || state.chapterId !== chapterKey()) return;
  shell._restoreApplied = true;
  workspace._reviewRestoreState = null;

  const viewport = shell.querySelector(".review-document-viewport");
  if (viewport) {
    viewport.scrollLeft = Math.max(0, Number(state.scrollLeft || 0));
    viewport.scrollTop = Math.max(0, Number(state.scrollTop || 0));
  }

  const pageIndex = Number(state.activePageIndex);
  const id = state.selectedTextObjectId;
  if (state.variant === "clean" && Number.isInteger(pageIndex) && id) {
    const overlay = shell.querySelector(
      `.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`,
    );
    if (overlay) selectObject(shell, pageIndex, id);
  }
}

export async function waitForOcr(shell, pageIndex, id, signal) {
  const start = Date.now();
  while (!signal.aborted && Date.now() - start < 8000) {
    if (window.findTextObject?.(pageIndex, id)?.ocr_text?.trim()) break;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  syncOverlay(shell, pageIndex, id);
  if (window.editorState?.selectedTextObjectId === id) {
    window.renderEditorPanel?.(Number(pageIndex));
  }
}

export async function ensureObjects(shell) {
  if (typeof window.ensureAutoTextObjects !== "function") return;
  for (const desc of shell._descriptors || []) {
    const pageIndex = Number(desc.item.canonicalIndex), key = `${chapterKey()}:${pageIndex}`;
    if (isAutoSynced(key)) continue;
    try {
      await window.ensureAutoTextObjects(pageIndex);
      markAutoSynced(key);
    } catch (err) {
      console.warn("Could not ensure text objects", pageIndex, err);
    }
  }
}

export function sourceCoverage(desc) {
  return { y1: desc.sourceY1 - desc.localY1, y2: desc.sourceY1 + (desc.img.naturalHeight - desc.localY1) };
}

export function reviewPageIndexAtSourceY(descriptors, sourceY) {
  const pages = descriptors || [];
  if (!pages.length || !Number.isFinite(Number(sourceY))) return null;
  const y = Number(sourceY);
  let low = 0, high = pages.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const desc = pages[middle];
    const start = Number(desc?.sourceY1), end = Number(desc?.sourceY2);
    const pageIndex = Number(desc?.item?.canonicalIndex);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || !Number.isInteger(pageIndex)) return null;
    if (y < start) high = middle - 1;
    else if (y >= end) low = middle + 1;
    else return pageIndex;
  }
  const nearest = pages[Math.max(0, Math.min(pages.length - 1, low))];
  const pageIndex = Number(nearest?.item?.canonicalIndex);
  return Number.isInteger(pageIndex) ? pageIndex : null;
}

export function ownerForRegion(shell, y1, y2) {
  const center = (y1 + y2) / 2;
  const complete = (shell._descriptors || []).filter((d) => { const c = sourceCoverage(d); return y1 >= c.y1 && y2 <= c.y2; });
  if (!complete.length) return null;
  return complete.find((d) => center >= d.sourceY1 && center < d.sourceY2) || complete.sort((a, b) => {
    const ca = sourceCoverage(a), cb = sourceCoverage(b);
    return Math.min(y1 - cb.y1, cb.y2 - y2) - Math.min(y1 - ca.y1, ca.y2 - y2);
  })[0];
}

export function sourcePoint(image, event) {
  const rect = image.getBoundingClientRect(), W = Number(image.dataset.sourceWidth || 0), H = Number(image.dataset.sourceHeight || 0);
  if (!rect.width || !rect.height || !W || !H) return null;
  return { x: Math.max(0, Math.min(W, (event.clientX - rect.left) * W / rect.width)), y: Math.max(0, Math.min(H, (event.clientY - rect.top) * H / rect.height)) };
}

export function installTextDrawing(shell, signal) {
  const image = shell.querySelector(".review-stitched-image");
  let drawing = null;
  let marquee = null;
  let suppressNextEmptyClick = false;
  const update = (p) => {
    if (!drawing || !p) return;
    const x1 = Math.min(drawing.start.x, p.x), y1 = Math.min(drawing.start.y, p.y), x2 = Math.max(drawing.start.x, p.x), y2 = Math.max(drawing.start.y, p.y);
    Object.assign(drawing.preview.style, { left: `${x1}px`, top: `${y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px` }); drawing.last = p;
  };
  const updateMarquee = (p) => {
    if (!marquee || !p) return;
    const x1 = Math.min(marquee.start.x, p.x), y1 = Math.min(marquee.start.y, p.y), x2 = Math.max(marquee.start.x, p.x), y2 = Math.max(marquee.start.y, p.y);
    Object.assign(marquee.box.style, { left: `${x1}px`, top: `${y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px` });
    marquee.last = p;
    if (x2 - x1 >= 6 || y2 - y1 >= 6) marquee.moved = true;
  };
  image.addEventListener("pointerdown", (e) => {
    if (state.variant !== "clean" || e.button !== 0 || e.target.closest(".review-text-object-overlay")) return;
    if (state.tool === "select") {
      const p = sourcePoint(image, e); if (!p) return;
      const box = document.createElement("div");
      box.className = "review-marquee-box";
      image.appendChild(box);
      marquee = { start: p, last: p, box, additive: e.shiftKey, moved: false };
      image.setPointerCapture?.(e.pointerId);
      updateMarquee(p);
      return;
    }
    if (!["rectangle", "ellipse"].includes(state.tool)) return;
    const p = sourcePoint(image, e); if (!p) { return; } e.preventDefault();
    const preview = document.createElement("div"); preview.className = `text-object-overlay drawing review-text-drawing${state.tool === "ellipse" ? " ellipse" : ""}`; image.appendChild(preview);
    drawing = { tool: state.tool, start: p, last: p, preview }; image.setPointerCapture?.(e.pointerId); update(p);
  }, { signal });
  image.addEventListener("pointermove", (e) => {
    if (drawing) update(sourcePoint(image, e));
    else if (marquee) updateMarquee(sourcePoint(image, e));
  }, { signal });
  const finish = async () => {
    if (marquee) {
      const m = marquee; marquee = null; m.box.remove();
      if (!m.moved) return;
      suppressNextEmptyClick = true;
      const x1 = Math.min(m.start.x, m.last.x), y1 = Math.min(m.start.y, m.last.y), x2 = Math.max(m.start.x, m.last.x), y2 = Math.max(m.start.y, m.last.y);
      const hits = [];
      if (m.additive) {
        for (const t of getSelectedTargets(shell)) hits.push({ pageIndex: t.pageIndex, id: t.id });
      }
      for (const { desc, pageIndex, obj } of ownedObjects(shell)) {
        const r = sourceRegion(desc, obj.region);
        if (r.x2 >= x1 && r.x1 <= x2 && r.y2 >= y1 && r.y1 <= y2) hits.push({ pageIndex, id: obj.id });
      }
      selectMany(shell, hits);
      return;
    }
    if (!drawing) return;
    const d = drawing; drawing = null; d.preview.remove();
    const x1 = Math.round(Math.min(d.start.x, d.last.x)), y1 = Math.round(Math.min(d.start.y, d.last.y)), x2 = Math.round(Math.max(d.start.x, d.last.x)), y2 = Math.round(Math.max(d.start.y, d.last.y));
    if (x2 - x1 < MIN_BOX || y2 - y1 < MIN_BOX) return;
    const owner = ownerForRegion(shell, y1, y2);
    if (!owner) return window.showToast?.("Vùng chọn vượt qua ranh giới nối an toàn giữa hai ảnh. Hãy thu vùng lại một chút.", "warning");
    const pageIndex = Number(owner.item.canonicalIndex), localY1 = Math.round(owner.localY1 + y1 - owner.sourceY1), localY2 = Math.round(owner.localY1 + y2 - owner.sourceY1);
    const region = { x1: Math.max(0, x1), y1: Math.max(0, localY1), x2: Math.min(owner.img.naturalWidth, x2), y2: Math.min(owner.img.naturalHeight, localY2) };
    if (region.x2 - region.x1 < MIN_BOX || region.y2 - region.y1 < MIN_BOX) return;
    try {
      if (typeof window.createTextObject !== "function") throw new Error("Công cụ tạo vùng chữ chưa sẵn sàng");
      await window.createTextObject(pageIndex, d.tool, region);
      const id = window.editorState?.selectedTextObjectId; window.editorState.activePageIndex = pageIndex; renderOverlays(shell, signal);
      if (id) { selectObject(shell, pageIndex, id); void waitForOcr(shell, pageIndex, id, signal); }
      setTool(shell, "select");
    } catch (err) { window.showToast?.("Không thể tạo vùng chữ: " + err.message, "error"); }
  };
  window.addEventListener("pointerup", finish, { signal }); window.addEventListener("pointercancel", finish, { signal });
  image.addEventListener("click", (e) => {
    if (suppressNextEmptyClick) { suppressNextEmptyClick = false; return; }
    if (state.tool === "select" && !e.target.closest(".review-text-object-overlay")) clearSelection(shell);
  }, { signal });
}

export function mountTextInspector(workspace, shell, signal) {
  const dockedHost = shell._propsHost || null;
  const floating = document.createElement("aside");
  floating.className = dockedHost
    ? "review-floating-inspector koharu-docked-inspector is-open"
    : "review-floating-inspector";
  floating.hidden = !dockedHost;
  floating.setAttribute("aria-label", "Thuộc tính vùng chữ");

  if (!dockedHost) {
    const header = document.createElement("div");
    header.className = "review-floating-inspector-header";
    const title = document.createElement("strong");
    title.textContent = "Văn bản";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "ui-icon-btn ui-btn-ghost ui-btn-compact";
    close.setAttribute("aria-label", "Đóng thuộc tính");
    close.append(window.createUiIcon("close"));
    header.append(title, close);
    floating.append(header);
    close.addEventListener("click", () => clearSelection(shell), { signal });
  }

  const host = document.createElement("section");
  host.className = "translation-panel-host review-text-inspector-host";
  host.setAttribute("aria-label", "Chỉnh sửa chữ gốc, bản dịch và kiểu chữ");
  host.innerHTML = '<div class="ui-empty-state koharu-empty-props">Chọn một vùng chữ trên trang hoặc trong danh sách Lớp chữ để chỉnh sửa.</div>';

  floating.append(host);
  (dockedHost || shell).appendChild(floating);
  shell._floatingInspector = floating;
}
