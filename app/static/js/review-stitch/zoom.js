import { descriptorFor } from "./overlays.js";
import { ZOOM_STEPS, state } from "./state.js";

export function zoomScale(viewport, width) {
  if (!width) return 1;
  if (state.fitWidth) return Math.min(1, Math.max(240, viewport.clientWidth - 48) / width);
  return state.zoom / 100;
}

export function applyZoom(shell, focus = null) {
  const viewport = shell.querySelector(".review-document-viewport");
  const stage = shell.querySelector(".review-stitched-zoom-stage");
  const image = shell.querySelector(".review-stitched-image");
  const label = shell.querySelector(".review-zoom-value");
  const width = Number(image?.dataset.sourceWidth || 0), height = Number(image?.dataset.sourceHeight || 0);
  if (!viewport || !stage || !image || !width || !height) return;
  const prev = Number(image.dataset.zoomScale || 1), scale = zoomScale(viewport, width);
  let src = null;
  if (focus && prev > 0) src = { x: (viewport.scrollLeft + focus.x) / prev, y: (viewport.scrollTop + focus.y) / prev };
  image.style.transform = `scale(${scale})`;
  image.dataset.zoomScale = String(scale);
  stage.style.width = `${Math.max(1, Math.round(width * scale))}px`;
  stage.style.height = `${Math.max(1, Math.round(height * scale))}px`;
  stage.style.marginTop = `${Math.max(24, Math.round((viewport.clientHeight - height * scale) / 2))}px`;
  stage.style.marginBottom = "64px";
  if (src) {
    viewport.scrollLeft = Math.max(0, src.x * scale - focus.x);
    viewport.scrollTop = Math.max(0, src.y * scale - focus.y);
  }
  if (label) {
    label.textContent = state.fitWidth ? `${Math.round(scale * 100)}%` : `${state.zoom}%`;
    label.title = state.fitWidth ? "Đang vừa khung" : "Bấm để vừa khung";
  }
}

export function scrollToReviewPage(shell, pageIndex) {
  if (pageIndex === null || pageIndex === undefined || !Number.isInteger(Number(pageIndex))) return false;
  const viewport = shell.querySelector(".review-document-viewport");
  const image = shell.querySelector(".review-stitched-image");
  const descriptor = descriptorFor(shell, pageIndex);
  const imageRect = image?.getBoundingClientRect();
  const viewportRect = viewport?.getBoundingClientRect();
  const scale = Number(image?.dataset.zoomScale || 1);
  if (!descriptor || !imageRect?.height || !viewport || !viewportRect || !scale) return false;
  viewport.scrollTop = descriptor === shell._descriptors?.[0] ? 0 : Math.max(
    0,
    viewport.scrollTop + imageRect.top + descriptor.sourceY1 * scale - viewportRect.top,
  );
  return true;
}

export function stepZoom(delta) {
  state.fitWidth = false;
  let i = ZOOM_STEPS.findIndex((n) => n >= state.zoom);
  if (i < 0) i = ZOOM_STEPS.length - 1;
  if (delta < 0 && ZOOM_STEPS[i] >= state.zoom) i--;
  if (delta > 0 && ZOOM_STEPS[i] <= state.zoom) i++;
  state.zoom = ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, i))];
}
