import { descriptorFor } from "./overlays.js";
import { ZOOM_STEPS, state } from "./state.js";

function viewportHorizontalPadding(viewport) {
  if (!viewport || typeof window === "undefined" || !window.getComputedStyle) {
    return 48;
  }
  const style = window.getComputedStyle(viewport);
  const left = parseFloat(style.paddingLeft) || 0;
  const right = parseFloat(style.paddingRight) || 0;
  return Math.max(0, left + right);
}

export function zoomScale(viewport, width) {
  const numericWidth = Number(width);
  if (!Number.isFinite(numericWidth) || numericWidth <= 0) return 1;
  if (state.fitWidth) {
    if (!viewport || viewport.clientWidth <= 0) return 1;
    const padX = viewportHorizontalPadding(viewport);
    const available = Math.max(1, viewport.clientWidth - padX);
    return Math.min(1, available / numericWidth);
  }
  const numericZoom = Number(state.zoom);
  const safeZoom =
    Number.isFinite(numericZoom) && numericZoom > 0 ? numericZoom : 100;
  return Math.max(0.05, safeZoom / 100);
}

export function applyZoom(shell, focus = null) {
  if (!shell) return;
  const viewport = shell.querySelector(".review-document-viewport");
  const stage = shell.querySelector(".review-stitched-zoom-stage");
  const image = shell.querySelector(".review-stitched-image");
  const label = shell.querySelector(".review-zoom-value");
  const width = Number(image?.dataset.sourceWidth || 0);
  const height = Number(image?.dataset.sourceHeight || 0);
  if (
    !viewport ||
    !stage ||
    !image ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return;
  }

  const rawPrev = Number(image.dataset.zoomScale || 1);
  const prev = Number.isFinite(rawPrev) && rawPrev > 0 ? rawPrev : 1;
  const scale = zoomScale(viewport, width);
  if (!Number.isFinite(scale) || scale <= 0) return;

  let src = null;
  let vpRect = null;
  if (
    focus &&
    Number.isFinite(focus.x) &&
    Number.isFinite(focus.y) &&
    prev > 0
  ) {
    vpRect = viewport.getBoundingClientRect();
    const imgRect = image.getBoundingClientRect();
    src = {
      x: (focus.x - (imgRect.left - vpRect.left)) / prev,
      y: (focus.y - (imgRect.top - vpRect.top)) / prev,
    };
  }

  image.style.transform = `scale(${scale})`;
  image.dataset.zoomScale = String(scale);
  stage.style.width = `${Math.max(1, Math.round(width * scale))}px`;
  stage.style.height = `${Math.max(1, Math.round(height * scale))}px`;
  stage.style.marginTop = `${Math.max(
    24,
    Math.round((viewport.clientHeight - height * scale) / 2),
  )}px`;
  stage.style.marginBottom = "64px";

  if (src && vpRect) {
    const nextImgRect = image.getBoundingClientRect();
    viewport.scrollLeft = Math.max(
      0,
      viewport.scrollLeft + (nextImgRect.left - vpRect.left) + src.x * scale - focus.x,
    );
    viewport.scrollTop = Math.max(
      0,
      viewport.scrollTop + (nextImgRect.top - vpRect.top) + src.y * scale - focus.y,
    );
  }

  const percent = Math.max(1, Math.round(scale * 100));
  if (state.fitWidth) {
    state.zoom = percent;
  }
  if (label) {
    label.textContent = `${state.fitWidth ? percent : state.zoom}%`;
    label.title = state.fitWidth ? "Đang vừa khung" : "Bấm để vừa khung";
  }
  shell._syncContextBar?.();
  shell._syncBrushCursorSize?.();
  shell._syncStatusBar?.();
}

export function scrollToReviewPage(shell, pageIndex) {
  if (!shell || pageIndex === null || pageIndex === undefined) return false;
  if (typeof pageIndex === "boolean") return false;
  if (typeof pageIndex === "string" && !pageIndex.trim()) return false;
  const idx = Number(pageIndex);
  if (!Number.isInteger(idx) || idx < 0) return false;

  const viewport = shell.querySelector(".review-document-viewport");
  const image = shell.querySelector(".review-stitched-image");
  const descriptor = descriptorFor(shell, idx);
  const imageRect = image?.getBoundingClientRect();
  const viewportRect = viewport?.getBoundingClientRect();
  const scale = Number(image?.dataset.zoomScale || 1);
  if (
    !descriptor ||
    !imageRect?.height ||
    !viewport ||
    !viewportRect ||
    !Number.isFinite(scale) ||
    scale <= 0
  ) {
    return false;
  }
  viewport.scrollTop =
    descriptor === shell._descriptors?.[0]
      ? 0
      : Math.max(
          0,
          viewport.scrollTop +
            imageRect.top +
            descriptor.sourceY1 * scale -
            viewportRect.top,
        );
  return true;
}

export function stepZoom(delta) {
  const numericDelta = Number(delta);
  if (!Number.isFinite(numericDelta) || numericDelta === 0) return;
  if (!Array.isArray(ZOOM_STEPS) || ZOOM_STEPS.length === 0) return;

  state.fitWidth = false;
  const current =
    Number.isFinite(Number(state.zoom)) && Number(state.zoom) > 0
      ? Number(state.zoom)
      : 100;
  let i = ZOOM_STEPS.findIndex((n) => n >= current);
  if (i < 0) i = ZOOM_STEPS.length - 1;
  if (numericDelta < 0 && ZOOM_STEPS[i] >= current) i--;
  if (numericDelta > 0 && ZOOM_STEPS[i] <= current) i++;
  state.zoom = ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, i))];
}
