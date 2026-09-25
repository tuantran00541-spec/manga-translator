import { makeBrushChunks, restoreSnapshot } from "./brush.js";
import { ensureObjects, renderOverlays, restoreWorkspaceState } from "./overlays.js";
import { chapterKey, livePage, state } from "./state.js";
import { syncTool } from "./tools.js";
import { applyZoom, scrollToReviewPage } from "./zoom.js";

export function orderedSlices() {
  return (window.currentManifest?.pages || [])
    .map((page, canonicalIndex) => ({ page, canonicalIndex }))
    .filter(({ page }) => Boolean(page))
    .map(({ canonicalIndex }) => ({ canonicalIndex }));
}

export function imageUrl(page, mode = state.variant) {
  if (!page) return null;
  let url;
  let rev;
  if (mode === "original") {
    url = page.original || page.clean;
    rev = Number(page.source_revision || 0);
  } else if (mode === "rendered") {
    url = page._reviewRenderedUrl || (typeof page.rendered === "string" ? page.rendered : null) || page.clean || page.original;
    rev = Number(page.render_revision || page.clean_revision || page.process_revision || page.source_revision || 0);
  } else {
    url = page.skipped ? page.original : (page.clean || page.original);
    rev = Number(page.skipped ? page.source_revision || 0 : page.clean_revision || page.process_revision || page.source_revision || 0);
  }
  if (!url) return null;
  return `${url}${url.includes("?") ? "&" : "?"}review_revision=${rev}`;
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Không tải được ảnh: ${url}`));
    img.src = url;
  });
}

export function coreMeta(page, imageHeight) {
  const core = page?.stitch_core;
  if (!core || typeof core !== "object") return null;
  const localY1 = Number(core.core_y1), localY2 = Number(core.core_y2);
  const sourceY1 = Number(core.core_source_y1), sourceY2 = Number(core.core_source_y2), sourceHeight = Number(core.source_height);
  if (![localY1, localY2, sourceY1, sourceY2, sourceHeight].every(Number.isFinite)) return null;
  if (localY1 < 0 || localY2 <= localY1 || localY2 > imageHeight) return null;
  if (sourceY1 < 0 || sourceY2 <= sourceY1 || sourceHeight < sourceY2) return null;
  if (localY2 - localY1 !== sourceY2 - sourceY1) return null;
  return { localY1, localY2, sourceY1, sourceY2, sourceHeight };
}

export async function renderStrip(shell, items, signal) {
  const token = ++state.renderToken;
  const stage = shell.querySelector(".review-stitched-zoom-stage");
  const image = shell.querySelector(".review-stitched-image");
  const meta = shell.querySelector(".review-stitched-meta");
  const warning = shell.querySelector(".review-stitched-warning");
  image.replaceChildren();
  stage.style.width = "auto";
  stage.style.height = "auto";
  warning.hidden = true;

  const loading = document.createElement("div");
  loading.className = "ui-state review-stitched-loading";
  loading.textContent = state.variant === "original"
    ? "Đang tải ảnh gốc…"
    : state.variant === "rendered"
      ? "Đang tải ảnh có chữ…"
      : "Đang tải ảnh sau inpaint…";
  image.appendChild(loading);

  try {
    const descriptors = [];
    let stripWidth = 0;
    let stripHeight = 0;
    let fallbackSlices = 0;

    for (const item of items) {
      const live = livePage(item);
      if (!live) continue;
      const url = imageUrl(live);
      if (!url) throw new Error(`Lát ${item.canonicalIndex + 1} không có dữ liệu ảnh.`);

      let width = Number(live.width || 0);
      let height = Number(live.height || 0);
      const rawCore = live.stitch_core;
      if (!(height > 0) && rawCore && typeof rawCore === "object") {
        const sourceY1 = Number(rawCore.source_y1);
        const sourceY2 = Number(rawCore.source_y2);
        if (Number.isFinite(sourceY1) && Number.isFinite(sourceY2) && sourceY2 > sourceY1) {
          height = sourceY2 - sourceY1;
        }
      }

      let preloaded = null;
      if (!(stripWidth > 0) && !(width > 0)) {
        preloaded = await loadImage(url);
        if (token !== state.renderToken) return;
        width = preloaded.naturalWidth;
        if (!(height > 0)) height = preloaded.naturalHeight;
      }
      if (!(stripWidth > 0)) stripWidth = width;
      if (!(width > 0)) width = stripWidth;

      if (!(height > 0)) {
        preloaded = preloaded || await loadImage(url);
        if (token !== state.renderToken) return;
        height = preloaded.naturalHeight;
        width = preloaded.naturalWidth || width;
      }
      if (!(width > 0)) throw new Error(`Lát ${item.canonicalIndex + 1} không có chiều rộng hợp lệ.`);
      stripWidth = Math.max(stripWidth, width);

      const core = coreMeta(live, height);
      const localY1 = core ? core.localY1 : 0;
      const localY2 = core ? core.localY2 : height;
      if (!core) fallbackSlices += 1;
      const ownedHeight = localY2 - localY1;
      descriptors.push({
        item: { canonicalIndex: item.canonicalIndex },
        img: { naturalWidth: width, naturalHeight: height },
        localY1,
        localY2,
        sourceY1: stripHeight,
        sourceY2: stripHeight + ownedHeight,
        url,
        preloaded,
      });
      stripHeight += ownedHeight;
    }

    if (!descriptors.length || !stripWidth || !stripHeight) {
      throw new Error("Chương này chưa có ảnh hợp lệ để hiển thị.");
    }

    image.replaceChildren();
    Object.assign(image.style, {
      width: `${stripWidth}px`,
      height: `${stripHeight}px`,
    });
    image.dataset.sourceWidth = String(stripWidth);
    image.dataset.sourceHeight = String(stripHeight);
    image.dataset.stripSlices = String(descriptors.length);

    for (const desc of descriptors) {
      const slice = document.createElement("div");
      slice.className = "review-strip-slice";
      slice.dataset.pageIndex = String(desc.item.canonicalIndex);
      Object.assign(slice.style, {
        top: `${desc.sourceY1}px`,
        width: `${desc.img.naturalWidth}px`,
        height: `${desc.sourceY2 - desc.sourceY1}px`,
      });

      const img = desc.preloaded || new Image();
      img.className = "review-strip-slice-image";
      img.alt = "";
      img.decoding = "async";
      img.loading = desc.sourceY1 === 0 ? "eager" : "lazy";
      Object.assign(img.style, {
        top: `-${desc.localY1}px`,
      });
      if (!desc.preloaded) {
        img.addEventListener("error", () => {
          slice.classList.add("review-strip-slice-error");
          slice.textContent = `Không tải được ảnh ${desc.item.canonicalIndex + 1}`;
        }, { once: true });
        img.src = desc.url;
      }
      slice.appendChild(img);
      image.appendChild(slice);

      delete desc.preloaded;
      delete desc.url;
    }

    shell._descriptors = descriptors;
    shell._brushChunks = state.variant === "clean"
      ? makeBrushChunks(image, stripWidth, stripHeight)
      : [];
    if (state.variant === "clean") restoreSnapshot(shell);

    meta.textContent = `${stripWidth} × ${stripHeight}px · ${
      state.variant === "original"
        ? "Ảnh gốc"
        : state.variant === "rendered"
          ? "Bản đã render"
          : "Sau inpaint"
    }`;
    if (fallbackSlices) {
      warning.hidden = false;
      warning.textContent = "Một số ảnh thiếu thông tin ghép chuẩn; thứ tự hiển thị dựa trên dữ liệu hiện có.";
    }

    applyZoom(shell);
    const savedState = shell.closest(".review-workspace-shell")?._reviewRestoreState;
    if (savedState?.chapterId !== chapterKey()) {
      scrollToReviewPage(shell, shell._initialCanonicalPageIndex);
    }
    await ensureObjects(shell);
    if (token !== state.renderToken) return;
    renderOverlays(shell, signal);
    syncTool(shell);
    restoreWorkspaceState(shell);
    shell._syncVisibleReviewPage?.();
  } catch (err) {
    if (token !== state.renderToken) return;
    image.replaceChildren();
    shell._brushChunks = [];
    const error = document.createElement("div");
    error.className = "ui-state ui-state-error review-stitched-error";
    error.textContent = `Không hiển thị được ảnh liền mạch: ${err.message}`;
    image.appendChild(error);
  }
}
