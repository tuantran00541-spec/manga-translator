(() => {
  const MODE_KEY = "mt_review_display_mode";
  const DEFAULT_MODE = "stitched";
  const MAX_CANVAS_CHUNK_HEIGHT = 12000;
  let mode = localStorage.getItem(MODE_KEY) || DEFAULT_MODE;
  let activeSourcePage = null;
  let renderToken = 0;

  const css = `
    .review-view-switch { display:inline-flex; gap:4px; padding:4px; border:1px solid var(--border-color, #d7d7d7); border-radius:10px; background:var(--surface-2, rgba(127,127,127,.08)); }
    .review-view-switch button { border:0; border-radius:7px; padding:7px 11px; background:transparent; cursor:pointer; font:inherit; }
    .review-view-switch button.active { background:var(--surface-1, #fff); box-shadow:0 1px 4px rgba(0,0,0,.14); font-weight:700; }
    .review-stitched-shell { min-width:0; display:flex; flex-direction:column; gap:12px; padding:14px; }
    .review-stitched-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
    .review-stitched-toolbar select, .review-stitched-toolbar button { font:inherit; }
    .review-stitched-toolbar select { min-width:180px; padding:7px 9px; border-radius:8px; border:1px solid var(--border-color, #ccc); }
    .review-stitched-meta { color:var(--muted-text, #666); font-size:.9rem; }
    .review-stitched-note { padding:9px 11px; border-radius:8px; background:var(--surface-2, rgba(127,127,127,.08)); font-size:.9rem; }
    .review-stitched-warning { padding:9px 11px; border-radius:8px; background:#fff3cd; color:#664d03; font-size:.9rem; }
    .review-stitched-viewport { overflow:auto; max-height:calc(100vh - 220px); border:1px solid var(--border-color, #ddd); border-radius:10px; background:#202020; padding:0; }
    .review-stitched-image { width:min(100%, 1000px); margin:0 auto; background:white; }
    .review-stitched-image canvas { display:block; width:100%; height:auto; margin:0; padding:0; }
    .review-stitched-loading, .review-stitched-error { padding:28px; text-align:center; color:var(--muted-text, #777); background:var(--surface-1, #fff); }
    .review-stitched-error { color:#b42318; }
    .review-mode.review-show-stitched .review-workbench-grid { display:none !important; }
    .review-mode.review-show-slices .review-stitched-shell { display:none !important; }
  `;

  function installStyle() {
    if (document.getElementById("review-stitch-inspector-style")) return;
    const style = document.createElement("style");
    style.id = "review-stitch-inspector-style";
    style.textContent = css;
    document.head.appendChild(style);
  }

  function cleanSourcePage(page, fallbackIndex) {
    return Number.isInteger(page?.source_page) ? page.source_page : fallbackIndex;
  }

  function cleanSliceIndex(page) {
    return Number.isInteger(page?.slice_index) ? page.slice_index : 0;
  }

  function groupsFromManifest() {
    const groups = new Map();
    const pages = window.currentManifest?.pages || [];
    pages.forEach((page, canonicalIndex) => {
      if (!page) return;
      const sourcePage = cleanSourcePage(page, canonicalIndex);
      if (!groups.has(sourcePage)) groups.set(sourcePage, []);
      groups.get(sourcePage).push({ page, canonicalIndex });
    });
    for (const items of groups.values()) {
      items.sort((a, b) => cleanSliceIndex(a.page) - cleanSliceIndex(b.page));
    }
    return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
  }

  function imageUrl(page) {
    // A skipped slice still owns part of the source page.  Showing only clean
    // slices makes a mixed source page look truncated and no longer match the
    // raw image.  Use the original pixels for skipped slices and the latest
    // clean artifact everywhere else.
    const url = page.skipped ? page.original : (page.clean || page.original);
    if (!url) return null;
    const revision = Number(
      page.skipped
        ? (page.source_revision || 0)
        : (page.clean_revision || page.process_revision || page.source_revision || 0),
    );
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}review_revision=${revision}`;
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Không tải được ảnh: ${url}`));
      img.src = url;
    });
  }

  function coreMetadata(page, imageHeight) {
    const core = page?.stitch_core;
    if (!core || typeof core !== "object") return null;
    const localY1 = Number(core.core_y1);
    const localY2 = Number(core.core_y2);
    const sourceY1 = Number(core.core_source_y1);
    const sourceY2 = Number(core.core_source_y2);
    const sourceHeight = Number(core.source_height);
    if (![localY1, localY2, sourceY1, sourceY2, sourceHeight].every(Number.isFinite)) return null;
    if (localY1 < 0 || localY2 <= localY1 || localY2 > imageHeight) return null;
    if (sourceY1 < 0 || sourceY2 <= sourceY1 || sourceHeight < sourceY2) return null;
    if ((localY2 - localY1) !== (sourceY2 - sourceY1)) return null;
    return { localY1, localY2, sourceY1, sourceY2, sourceHeight };
  }

  function makeChunkCanvases(host, width, height) {
    const chunks = [];
    for (let y = 0; y < height; y += MAX_CANVAS_CHUNK_HEIGHT) {
      const chunkHeight = Math.min(MAX_CANVAS_CHUNK_HEIGHT, height - y);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = chunkHeight;
      canvas.dataset.sourceY = String(y);
      const ctx = canvas.getContext("2d", { alpha: false });
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, chunkHeight);
      host.appendChild(canvas);
      chunks.push({ canvas, ctx, y1: y, y2: y + chunkHeight });
    }
    return chunks;
  }

  function drawIntoChunks(chunks, img, sx, sy, sw, sh, targetY) {
    const targetEnd = targetY + sh;
    for (const chunk of chunks) {
      const iy1 = Math.max(targetY, chunk.y1);
      const iy2 = Math.min(targetEnd, chunk.y2);
      if (iy2 <= iy1) continue;
      const offset = iy1 - targetY;
      const h = iy2 - iy1;
      chunk.ctx.drawImage(
        img,
        sx,
        sy + offset,
        sw,
        h,
        0,
        iy1 - chunk.y1,
        sw,
        h,
      );
    }
  }

  async function renderSourcePage(shell, sourcePage, items) {
    const token = ++renderToken;
    const imageHost = shell.querySelector(".review-stitched-image");
    const meta = shell.querySelector(".review-stitched-meta");
    const warning = shell.querySelector(".review-stitched-warning");
    imageHost.replaceChildren();
    warning.hidden = true;

    const loading = document.createElement("div");
    loading.className = "review-stitched-loading";
    loading.textContent = "Đang ghép ảnh theo ownership của từng lát…";
    imageHost.appendChild(loading);

    try {
      const firstUrl = imageUrl(items[0]?.page);
      if (!firstUrl) throw new Error("Trang không có ảnh đã xử lý hoặc ảnh gốc.");
      const first = await loadImage(firstUrl);
      if (token !== renderToken) return;

      const firstCore = coreMetadata(items[0].page, first.naturalHeight);
      let sourceHeight = firstCore?.sourceHeight || first.naturalHeight;
      let width = first.naturalWidth;
      let metadataValid = Boolean(firstCore) || items.length === 1;
      let expectedSourceY = 0;

      const descriptors = [];
      let fallbackY = 0;
      for (let index = 0; index < items.length; index += 1) {
        const item = items[index];
        const url = imageUrl(item.page);
        if (!url) throw new Error(`Lát ${index + 1} không có ảnh.`);
        const img = index === 0 ? first : await loadImage(url);
        if (token !== renderToken) return;
        if (img.naturalWidth !== width) {
          throw new Error(`Lát ${index + 1} có chiều rộng ${img.naturalWidth}px, khác ${width}px.`);
        }
        const core = coreMetadata(item.page, img.naturalHeight);
        if (core) {
          sourceHeight = core.sourceHeight;
          if (core.sourceY1 !== expectedSourceY) metadataValid = false;
          expectedSourceY = core.sourceY2;
          descriptors.push({ img, localY1: core.localY1, localY2: core.localY2, sourceY1: core.sourceY1 });
        } else {
          metadataValid = false;
          descriptors.push({ img, localY1: 0, localY2: img.naturalHeight, sourceY1: fallbackY });
          fallbackY += img.naturalHeight;
        }
      }

      if (!metadataValid) {
        let y = 0;
        for (const descriptor of descriptors) {
          descriptor.sourceY1 = y;
          y += descriptor.localY2 - descriptor.localY1;
        }
        sourceHeight = y;
        warning.hidden = false;
        warning.textContent = "Metadata stitch_core không phủ liên tục; ảnh dưới đây dùng fallback nối tuần tự để vẫn nhìn được lỗi. Không coi fallback này là output chuẩn.";
      } else if (expectedSourceY && expectedSourceY !== sourceHeight) {
        warning.hidden = false;
        warning.textContent = `Ownership kết thúc ở y=${expectedSourceY}, nhưng source_height=${sourceHeight}. Có khả năng stitch metadata đang thiếu/gãy.`;
      }

      imageHost.replaceChildren();
      const chunks = makeChunkCanvases(imageHost, width, sourceHeight);
      for (const descriptor of descriptors) {
        drawIntoChunks(
          chunks,
          descriptor.img,
          0,
          descriptor.localY1,
          width,
          descriptor.localY2 - descriptor.localY1,
          descriptor.sourceY1,
        );
      }
      const skippedCount = items.filter((item) => item.page?.skipped).length;
      const processedCount = items.length - skippedCount;
      const sliceSummary = skippedCount
        ? `${processedCount} lát đã xử lý + ${skippedCount} lát giữ nguyên`
        : `${items.length} lát đã xử lý`;
      meta.textContent = `Trang gốc ${sourcePage + 1} · ${sliceSummary} · ${width} × ${sourceHeight}px · ghép theo stitch_core`;
    } catch (err) {
      if (token !== renderToken) return;
      imageHost.replaceChildren();
      const error = document.createElement("div");
      error.className = "review-stitched-error";
      error.textContent = `Không dựng được ảnh ghép: ${err.message}`;
      imageHost.appendChild(error);
    }
  }

  function applyMode(host, shell, switcher) {
    const stitched = mode === "stitched";
    host.classList.toggle("review-show-stitched", stitched);
    host.classList.toggle("review-show-slices", !stitched);
    switcher.querySelectorAll("button[data-review-mode]").forEach((button) => {
      const active = button.dataset.reviewMode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    shell.hidden = !stitched;
    window.syncWorkbenchPanels?.();
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
    const host = workspace.closest("#page-view.review-mode");
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    const actions = toolbar?.querySelector(".review-actions-group");
    const layout = workspace.querySelector(".review-workbench-grid");
    if (!host || !toolbar || !actions || !layout) return;
    workspace.dataset.stitchInspectorMounted = "1";

    const groups = groupsFromManifest();
    if (!groups.size) return;
    const sourcePages = [...groups.keys()];
    if (!sourcePages.includes(activeSourcePage)) {
      const activeCard = workspace.querySelector(".review-card");
      const canonical = Number.parseInt(activeCard?.dataset.pageIndex || "", 10);
      const activePage = Number.isFinite(canonical) ? window.currentManifest?.pages?.[canonical] : null;
      activeSourcePage = activePage ? cleanSourcePage(activePage, canonical) : sourcePages[0];
    }

    const switcher = document.createElement("div");
    switcher.className = "review-view-switch";
    switcher.setAttribute("role", "group");
    switcher.setAttribute("aria-label", "Kiểu hiển thị ảnh kiểm tra");
    const stitchedBtn = document.createElement("button");
    stitchedBtn.type = "button";
    stitchedBtn.dataset.reviewMode = "stitched";
    stitchedBtn.textContent = "Ghép như ảnh gốc";
    const slicesBtn = document.createElement("button");
    slicesBtn.type = "button";
    slicesBtn.dataset.reviewMode = "slices";
    slicesBtn.textContent = "Từng lát";
    switcher.append(stitchedBtn, slicesBtn);
    actions.prepend(switcher);

    const shell = document.createElement("section");
    shell.className = "review-stitched-shell";
    const stitchedToolbar = document.createElement("div");
    stitchedToolbar.className = "review-stitched-toolbar";
    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "ui-btn ui-btn-ghost";
    prev.textContent = "← Trang trước";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Chọn trang gốc đã ghép");
    sourcePages.forEach((sourcePage) => {
      const option = document.createElement("option");
      option.value = String(sourcePage);
      option.textContent = `Trang gốc ${sourcePage + 1} · ${groups.get(sourcePage).length} lát`;
      select.appendChild(option);
    });
    select.value = String(activeSourcePage);
    const next = document.createElement("button");
    next.type = "button";
    next.className = "ui-btn ui-btn-ghost";
    next.textContent = "Trang sau →";
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.className = "ui-btn ui-btn-ghost";
    refresh.textContent = "Làm mới ảnh ghép";
    stitchedToolbar.append(prev, select, next, refresh);

    const note = document.createElement("div");
    note.className = "review-stitched-note";
    note.textContent = "Chuyển sang “Từng lát” để đánh dấu và xử lý lại. Phần bỏ qua được giữ nguyên từ ảnh gốc.";
    const meta = document.createElement("div");
    meta.className = "review-stitched-meta";
    const warning = document.createElement("div");
    warning.className = "review-stitched-warning";
    warning.hidden = true;
    const viewport = document.createElement("div");
    viewport.className = "review-stitched-viewport";
    const imageHost = document.createElement("div");
    imageHost.className = "review-stitched-image";
    viewport.appendChild(imageHost);
    shell.append(stitchedToolbar, note, meta, warning, viewport);
    layout.before(shell);

    const rerender = () => {
      const items = groups.get(activeSourcePage);
      if (!items) return;
      select.value = String(activeSourcePage);
      const pos = sourcePages.indexOf(activeSourcePage);
      prev.disabled = pos <= 0;
      next.disabled = pos < 0 || pos >= sourcePages.length - 1;
      void renderSourcePage(shell, activeSourcePage, items);
    };

    const setMode = (nextMode) => {
      mode = nextMode === "slices" ? "slices" : "stitched";
      localStorage.setItem(MODE_KEY, mode);
      applyMode(host, shell, switcher);
      if (mode === "stitched") rerender();
    };

    stitchedBtn.addEventListener("click", () => setMode("stitched"));
    slicesBtn.addEventListener("click", () => setMode("slices"));
    select.addEventListener("change", () => {
      activeSourcePage = Number.parseInt(select.value, 10);
      rerender();
    });
    prev.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos > 0) {
        activeSourcePage = sourcePages[pos - 1];
        rerender();
      }
    });
    next.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos >= 0 && pos < sourcePages.length - 1) {
        activeSourcePage = sourcePages[pos + 1];
        rerender();
      }
    });
    refresh.addEventListener("click", rerender);

    applyMode(host, shell, switcher);
    if (mode === "stitched") rerender();
  }

  function scan() {
    document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount);
  }

  installStyle();
  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan, { once: true });
  else scan();
})();
