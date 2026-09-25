import { MIN_BOX, chapterKey, isAutoSynced, markAutoSynced, state } from "./state.js";
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

export function syncOverlay(shell, pageIndex, id) {
  const desc = descriptorFor(shell, pageIndex);
  const obj = window.findTextObject?.(pageIndex, id);
  const overlay = shell.querySelector(
    `.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`,
  );
  if (!desc || !obj?.region || !overlay) return false;
  const r = sourceRegion(desc, obj.region);
  Object.assign(overlay.style, {
    left: `${r.x1}px`,
    top: `${r.y1}px`,
    width: `${Math.max(MIN_BOX, r.x2 - r.x1)}px`,
    height: `${Math.max(MIN_BOX, r.y2 - r.y1)}px`,
  });
  return true;
}

export function installOverlaySync() {
  if (!state.baseSyncOverlayForObject) state.baseSyncOverlayForObject = window.syncOverlayForObject || null;
  window.syncOverlayForObject = (pageIndex, id) => {
    const shell = document.querySelector("#page-view.review-mode .review-document-shell");
    if (shell && syncOverlay(shell, pageIndex, id)) return;
    state.baseSyncOverlayForObject?.(pageIndex, id);
  };
}

export function selectObject(shell, pageIndex, id) {
  if (!window.editorState) return;
  window.editorState.activePageIndex = Number(pageIndex);
  window.editorState.selectedTextObjectId = id;
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.toggle("selected", Number(el.dataset.pageIndex) === Number(pageIndex) && el.dataset.objectId === String(id)));
  const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
  if (floating) {
    floating.hidden = false;
    floating.classList.add("is-open");
  }
  window.renderEditorPanel?.(Number(pageIndex));
}

export function clearSelection(shell) {
  if (window.editorState) window.editorState.selectedTextObjectId = null;
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.remove("selected"));
  const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
  if (floating) {
    floating.classList.remove("is-open");
    floating.hidden = true;
  }
  const panel = shell.querySelector(".translation-panel-host");
  if (panel) panel.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên trang để chỉnh sửa.</div>';
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
    event.preventDefault(); event.stopPropagation(); selectObject(shell, pageIndex, obj.id);
    const rect = image.getBoundingClientRect(), W = Number(image.dataset.sourceWidth || 0), H = Number(image.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W || !H) return;
    drag = { mode: overlayMode(event, overlay), x: (event.clientX - rect.left) * W / rect.width, y: (event.clientY - rect.top) * H / rect.height, original: { ...obj.region } };
    overlay.classList.add("transforming"); overlay.setPointerCapture?.(event.pointerId);
  }, { signal });
  overlay.addEventListener("pointermove", (event) => {
    if (!drag) { if (state.tool === "select") { overlay.style.cursor = cursor(overlayMode(event, overlay)); } return; }
    const rect = image.getBoundingClientRect(), W0 = Number(image.dataset.sourceWidth || 0), H0 = Number(image.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W0 || !H0) return;
    const x = (event.clientX - rect.left) * W0 / rect.width, y = (event.clientY - rect.top) * H0 / rect.height, dx = x - drag.x, dy = y - drag.y;
    const o = drag.original, W = desc.img.naturalWidth, minY = 0, maxY = desc.img.naturalHeight, clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
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
    obj.region = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
    syncOverlay(shell, pageIndex, obj.id);
  }, { signal });
  const end = () => {
    if (!drag) return;
    const o = drag.original; drag = null; overlay.classList.remove("transforming");
    const r = obj.region;
    if (r.x1 !== o.x1 || r.y1 !== o.y1 || r.x2 !== o.x2 || r.y2 !== o.y2) {
      window.scheduleGeomPersist?.(pageIndex, obj.id); window.refreshGeometryControls?.(pageIndex, obj.id);
    }
  };
  overlay.addEventListener("pointerup", end, { signal }); overlay.addEventListener("pointercancel", end, { signal });
}

export function createInlineEditor(shell, desc, pageIndex, obj, signal) {
  const overlay = document.createElement("div");
  overlay.className = `text-object-overlay review-text-object-overlay${obj.shape === "ellipse" ? " ellipse" : ""}`;
  overlay.dataset.pageIndex = String(pageIndex);
  overlay.dataset.objectId = String(obj.id);
  overlay.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    selectObject(shell, pageIndex, obj.id);
  }, { signal });
  shell.querySelector(".review-stitched-image")?.appendChild(overlay);
  syncOverlay(shell, pageIndex, obj.id);
  installTransform(shell, overlay, desc, pageIndex, obj, signal);
}

export function renderOverlays(shell, signal) {
  const image = shell.querySelector(".review-stitched-image"); if (!image) return;
  image.querySelectorAll(".review-text-object-overlay,.review-text-drawing").forEach((n) => n.remove());
  if (state.variant !== "clean") return;
  for (const { desc, pageIndex, obj } of ownedObjects(shell)) createInlineEditor(shell, desc, pageIndex, obj, signal);
  syncTool(shell);
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
  const image = shell.querySelector(".review-stitched-image"); let drawing = null;
  const update = (p) => {
    if (!drawing || !p) return;
    const x1 = Math.min(drawing.start.x, p.x), y1 = Math.min(drawing.start.y, p.y), x2 = Math.max(drawing.start.x, p.x), y2 = Math.max(drawing.start.y, p.y);
    Object.assign(drawing.preview.style, { left: `${x1}px`, top: `${y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px` }); drawing.last = p;
  };
  image.addEventListener("pointerdown", (e) => {
    if (state.variant !== "clean" || !["rectangle", "ellipse"].includes(state.tool) || e.button !== 0 || e.target.closest(".review-text-object-overlay")) return;
    const p = sourcePoint(image, e); if (!p) { return; } e.preventDefault();
    const preview = document.createElement("div"); preview.className = `text-object-overlay drawing review-text-drawing${state.tool === "ellipse" ? " ellipse" : ""}`; image.appendChild(preview);
    drawing = { tool: state.tool, start: p, last: p, preview }; image.setPointerCapture?.(e.pointerId); update(p);
  }, { signal });
  image.addEventListener("pointermove", (e) => { if (drawing) update(sourcePoint(image, e)); }, { signal });
  const finish = async () => {
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
  image.addEventListener("click", (e) => { if (state.tool === "select" && !e.target.closest(".review-text-object-overlay")) clearSelection(shell); }, { signal });
}

export function mountTextInspector(workspace, shell, signal) {
  const floating = document.createElement("aside");
  floating.className = "review-floating-inspector";
  floating.hidden = true;
  floating.setAttribute("aria-label", "Thuộc tính vùng chữ");

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

  const host = document.createElement("section");
  host.className = "translation-panel-host review-text-inspector-host";
  host.setAttribute("aria-label", "Chỉnh OCR, bản dịch và typography");
  host.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên ảnh để chỉnh sửa.</div>';

  floating.append(header, host);
  shell.appendChild(floating);
  shell._floatingInspector = floating;

  close.addEventListener("click", () => clearSelection(shell), { signal });

}
