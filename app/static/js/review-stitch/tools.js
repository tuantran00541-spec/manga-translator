import { state } from "./state.js";

export function syncToolButtons() {
  document.querySelectorAll(".review-rail-tool[data-tool]").forEach((b) => { const active = b.dataset.tool === state.tool; b.classList.toggle("active", active); b.setAttribute("aria-pressed", String(active)); });
}

export function syncBrushTip(shell) {
  const tipBtn = (shell || document).querySelector(".review-rail-brush-tip");
  if (!tipBtn) return;
  const diameter = Math.round(Math.max(8, Math.min(200, (state.brushRadius || 24) * 2)));
  const dotSize = Math.max(3, Math.min(18, Math.round(2 + (16 * diameter) / 200)));
  const dot = tipBtn.querySelector(".review-rail-brush-dot");
  const num = tipBtn.querySelector(".review-rail-brush-num");
  if (dot) { dot.style.width = `${dotSize}px`; dot.style.height = `${dotSize}px`; }
  if (num) num.textContent = String(diameter);
  tipBtn.title = `Cỡ cọ: ${diameter}px (phím [ / ] hoặc Alt + cuộn chuột)`;
  tipBtn.setAttribute("aria-label", `Cỡ cọ ${diameter} pixel`);
}

export function syncTool(shell) {
  const image = shell?.querySelector(".review-stitched-image"), viewport = shell?.querySelector(".review-document-viewport"); if (!image || !viewport) return;
  image.dataset.activeTool = state.tool; viewport.dataset.activeTool = state.tool;
  const paint = ["brush", "remove", "eraser"].includes(state.tool), text = ["rectangle", "ellipse"].includes(state.tool);
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => { el.classList.toggle("is-inert", !(state.tool === "select" && state.variant === "clean")); el.classList.toggle("tool-muted", state.variant === "clean" && state.tool !== "select"); });
  image.classList.toggle("text-draw-mode", text && state.variant === "clean"); image.classList.toggle("brush-mode", paint && state.variant === "clean"); syncToolButtons(); syncBrushTip(shell); shell._syncBrushBar?.(); shell._syncContextBar?.();
}

export function setTool(shell, next) {
  if (!["select", "rectangle", "ellipse", "brush", "remove", "eraser", "eyedropper", "hand", "zoom"].includes(next)) next = "select";
  state.tool = next;
  if (state.variant !== "clean" && ["rectangle", "ellipse", "brush", "remove", "eraser"].includes(state.tool)) { state.variant = "clean"; shell._syncVariantUI?.(); shell._rerender?.(); }
  if (window.editorState) window.editorState.tool = ["rectangle", "ellipse"].includes(state.tool) ? state.tool : "select";
  window.setEditorTool?.(window.editorState?.tool || "select"); syncTool(shell);
}

export function toolButton(shell, signal, name, icon, title, description, action = null) {
  const b = document.createElement("button"); b.type = "button"; b.className = "review-rail-tool"; b.dataset.tool = name; b.setAttribute("aria-label", `${title}: ${description}`); b.title = `${title}: ${description}`; b.setAttribute("aria-pressed", "false"); b.append(window.createUiIcon(icon));
  const tip = document.createElement("span"); tip.className = "review-tool-tooltip"; const strong = document.createElement("strong"); strong.textContent = title; const text = document.createElement("span"); text.textContent = description; tip.append(strong, text); b.appendChild(tip);
  b.addEventListener("click", () => action ? action() : setTool(shell, name), { signal }); return b;
}

export function mountToolRail(shell, signal) {
  document.body.classList.remove("review-tool-rail-active");
  shell.querySelector(".review-tool-rail")?.remove();
  const rail = document.createElement("div");
  rail.className = "review-tool-rail";
  rail.setAttribute("role", "toolbar");
  rail.setAttribute("aria-label", "Công cụ chỉnh sửa ảnh và trình bày chữ");

  const main = document.createElement("div");
  main.className = "review-tool-group";
  [
    ["select","cursor","Chọn (V)","Chọn, kéo chọn nhiều vùng, nhấn đúp để sửa chữ trực tiếp."],
    ["rectangle","rect-select","Vùng chữ nhật (R)","Kéo quanh bong bóng thoại để tạo vùng nhận dạng chữ."],
    ["ellipse","ellipse-select","Vùng elip (O)","Kéo quanh bong bóng thoại tròn để tạo vùng nhận dạng chữ."],
    ["remove","remove-brush","Cọ xóa nhanh (J)","Tô trực tiếp lên chữ sót — thả chuột tự làm sạch ngay (giữ Shift: ép LaMa)."],
    ["brush","brush","Cọ tẩy nền (B)","Tô nhiều vùng rồi chọn phương thức xử lý hoặc khôi phục ảnh gốc."],
    ["eraser","eraser","Tẩy vùng tô (E)","Xóa phần vùng tô đã đánh dấu nhầm."],
    ["eyedropper","eyedropper","Hút màu (I)","Nhấp lên ảnh để lấy màu chữ. Giữ Shift: màu viền, Alt: nền khung."],
    ["hand","hand","Bàn tay (H)","Kéo trang tự do (hoặc giữ Space / chuột giữa)."],
    ["zoom","zoom","Thu phóng (Z)","Nhấp để phóng to, Alt + nhấp để thu nhỏ."],
  ].forEach((x) => main.appendChild(toolButton(shell, signal, ...x)));

  const tipWrap = document.createElement("div");
  tipWrap.className = "review-rail-footer";
  const tipBtn = document.createElement("button");
  tipBtn.type = "button";
  tipBtn.className = "review-rail-brush-tip";
  const dotBox = document.createElement("span");
  dotBox.className = "review-rail-brush-dot-box";
  const dot = document.createElement("span");
  dot.className = "review-rail-brush-dot";
  dotBox.appendChild(dot);
  const num = document.createElement("span");
  num.className = "review-rail-brush-num";
  tipBtn.append(dotBox, num);
  tipBtn.addEventListener("click", () => {
    if (!["brush", "remove", "eraser"].includes(state.tool)) setTool(shell, "remove");
    shell._focusBrushSlider?.();
  }, { signal });
  tipWrap.appendChild(tipBtn);

  rail.append(main, tipWrap);
  (shell._canvasPane || shell).appendChild(rail);
  signal.addEventListener("abort", () => rail.remove(), { once: true });
  syncToolButtons();
  syncBrushTip(shell);
}

export function mountActions(shell) {
  const actions = shell.querySelector(".review-docbar-actions");
  if (!actions) return;
  actions.replaceChildren();

  const status = document.createElement("span");
  status.className = "editor-save-status";
  status.setAttribute("role", "status");
  actions.appendChild(status);
  window.refreshSaveStatus?.();

  if (typeof window.buildChapterExportButton === "function") {
    actions.appendChild(window.buildChapterExportButton());
  }
}
