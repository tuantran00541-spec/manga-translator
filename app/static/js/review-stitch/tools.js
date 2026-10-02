import { state } from "./state.js";

export function syncToolButtons() {
  document.querySelectorAll(".review-rail-tool[data-tool]").forEach((b) => { const active = b.dataset.tool === state.tool; b.classList.toggle("active", active); b.setAttribute("aria-pressed", String(active)); });
}

export function syncTool(shell) {
  const image = shell?.querySelector(".review-stitched-image"), viewport = shell?.querySelector(".review-document-viewport"); if (!image || !viewport) return;
  image.dataset.activeTool = state.tool; viewport.dataset.activeTool = state.tool;
  const paint = ["brush", "eraser"].includes(state.tool), text = ["rectangle", "ellipse"].includes(state.tool);
  shell.querySelectorAll(".review-text-object-overlay").forEach((el) => { el.classList.toggle("is-inert", !(state.tool === "select" && state.variant === "clean")); el.classList.toggle("tool-muted", state.variant === "clean" && state.tool !== "select"); });
  image.classList.toggle("text-draw-mode", text && state.variant === "clean"); image.classList.toggle("brush-mode", paint && state.variant === "clean"); syncToolButtons(); shell._syncBrushBar?.();
}

export function setTool(shell, next) {
  if (!["select", "rectangle", "ellipse", "brush", "eraser", "hand", "zoom"].includes(next)) next = "select";
  state.tool = next;
  if (state.variant !== "clean" && ["rectangle", "ellipse", "brush", "eraser"].includes(state.tool)) { state.variant = "clean"; shell._syncVariantUI?.(); shell._rerender?.(); }
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
  rail.setAttribute("aria-label", "Công cụ chỉnh sửa ảnh và lettering");

  const main = document.createElement("div");
  main.className = "review-tool-group";
  [
    ["select","cursor","Chọn","Chọn, kéo và thay đổi kích thước vùng chữ."],
    ["rectangle","rect-select","Vùng chữ nhật","Kéo quanh bong bóng để tạo vùng OCR."],
    ["ellipse","ellipse-select","Vùng elip","Kéo quanh bong bóng tròn để tạo vùng OCR."],
    ["brush","brush","Cọ Inpaint","Tô vùng cần xóa, hoặc vùng bị xóa ẩu cần khôi phục ảnh gốc, rồi chọn cách xử lý."],
    ["eraser","eraser","Tẩy mask","Xóa phần mask inpaint đã tô nhầm."],
    ["hand","hand","Bàn tay","Kéo trang tự do theo mọi hướng."],
    ["zoom","zoom","Thu phóng","Nhấp để phóng to, Alt + nhấp để thu nhỏ."],
  ].forEach((x) => main.appendChild(toolButton(shell, signal, ...x)));

  rail.appendChild(main);
  shell.appendChild(rail);
  signal.addEventListener("abort", () => rail.remove(), { once: true });
  syncToolButtons();
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
