import { selectionKey, state } from "./state.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function buildProgressRing(done, total) {
  const wrap = document.createElement("span");
  wrap.className = "koharu-page-progress";
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("koharu-progress-ring");

  const track = document.createElementNS(SVG_NS, "circle");
  track.setAttribute("cx", "8");
  track.setAttribute("cy", "8");
  track.setAttribute("r", "6");
  track.setAttribute("fill", "none");
  track.setAttribute("stroke-width", "2");
  track.classList.add("koharu-ring-track");

  const fill = document.createElementNS(SVG_NS, "circle");
  fill.setAttribute("cx", "8");
  fill.setAttribute("cy", "8");
  fill.setAttribute("r", "6");
  fill.setAttribute("fill", "none");
  fill.setAttribute("stroke-width", "2");
  fill.setAttribute("pathLength", "1");
  const ratio = total > 0 ? Math.max(0, Math.min(1, done / total)) : 0;
  fill.setAttribute("stroke-dasharray", `${ratio} 1`);
  fill.classList.add("koharu-ring-fill");

  svg.append(track, fill);
  const txt = document.createElement("span");
  txt.textContent = `${done}/${total}`;
  wrap.append(svg, txt);
  return wrap;
}

/**
 * Koharu-style Left PageRail with rounded thumbnail cards, slice filter,
 * pixel dimensions, and translation progress rings.
 */
export function mountPageRail(shell, items, signal, { imageUrl, onSelectSlice }) {
  const rail = document.createElement("aside");
  rail.className = "koharu-page-rail";
  rail.setAttribute("aria-label", "Danh sách trang và lát");

  const header = document.createElement("header");
  header.className = "koharu-rail-head";
  const titleWrap = document.createElement("div");
  titleWrap.className = "koharu-rail-title-wrap";
  const title = document.createElement("h2");
  title.className = "koharu-rail-title";
  title.textContent = "Trang";
  const badge = document.createElement("span");
  badge.className = "koharu-count-pill";
  badge.textContent = String(items.length);
  titleWrap.append(title, badge);
  header.append(titleWrap);

  const searchWrap = document.createElement("div");
  searchWrap.className = "koharu-rail-search-wrap";
  const filter = document.createElement("input");
  filter.type = "search";
  filter.className = "ui-input koharu-rail-filter";
  filter.placeholder = "Lọc trang…";
  filter.setAttribute("aria-label", "Lọc danh sách trang");
  searchWrap.appendChild(filter);

  const list = document.createElement("div");
  list.className = "koharu-page-list";

  rail.append(header, searchWrap, list);

  const cards = new Map();
  let thumbObserver = null;
  if (typeof window.IntersectionObserver !== "undefined") {
    thumbObserver = new IntersectionObserver((entries, obs) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target;
        if (img.dataset.src) {
          img.src = img.dataset.src;
          img.removeAttribute("data-src");
        }
        obs.unobserve(img);
      }
    }, { root: list, rootMargin: "220px 0px" });
    signal.addEventListener("abort", () => thumbObserver?.disconnect(), { once: true });
  }

  const refreshCardMeta = (canonicalIndex) => {
    const entry = cards.get(Number(canonicalIndex));
    const page = window.currentManifest?.pages?.[Number(canonicalIndex)];
    if (!entry || !page) return;
    const objs = page.text_objects || [];
    const done = objs.filter((o) => String(o?.translation || "").trim()).length;
    entry.metaSlot.replaceChildren();
    if (objs.length > 0) {
      entry.metaSlot.appendChild(buildProgressRing(done, objs.length));
    } else {
      const empty = document.createElement("span");
      empty.className = "koharu-page-sub";
      empty.textContent = "Chưa có vùng chữ";
      entry.metaSlot.appendChild(empty);
    }
    const dims = document.createElement("span");
    dims.className = "koharu-page-dims";
    const w = Number(page.width || 0);
    const h = Number(page.height || 0);
    dims.textContent = w && h ? `${w} × ${h}` : "";
    entry.metaSlot.appendChild(dims);
  };

  items.forEach((item, idx) => {
    const canonical = Number(item.canonicalIndex);
    const page = window.currentManifest?.pages?.[canonical];
    const card = document.createElement("button");
    card.type = "button";
    card.className = "koharu-page-card";
    card.dataset.pageIndex = String(canonical);
    const labelText = `Lát ${idx + 1}`;
    card.dataset.filterText = `${labelText} ${canonical + 1}`.toLowerCase();

    const thumbBox = document.createElement("span");
    thumbBox.className = "koharu-page-thumb";
    const src = page ? (imageUrl(page, "clean") || imageUrl(page, "original")) : "";
    if (src) {
      const img = document.createElement("img");
      img.alt = labelText;
      img.decoding = "async";
      img.draggable = false;
      if (thumbObserver) {
        img.dataset.src = src;
        thumbObserver.observe(img);
      } else {
        img.src = src;
        img.loading = "lazy";
      }
      thumbBox.appendChild(img);
    } else {
      thumbBox.textContent = String(idx + 1);
    }

    const info = document.createElement("span");
    info.className = "koharu-page-info";
    const nameEl = document.createElement("strong");
    nameEl.className = "koharu-page-name";
    nameEl.textContent = labelText;
    const metaSlot = document.createElement("span");
    metaSlot.className = "koharu-page-meta-slot";
    info.append(nameEl, metaSlot);

    card.append(thumbBox, info);
    card.addEventListener("click", () => onSelectSlice(canonical), { signal });
    list.appendChild(card);
    cards.set(canonical, { card, metaSlot });
    refreshCardMeta(canonical);
  });

  filter.addEventListener("input", () => {
    const q = filter.value.trim().toLowerCase();
    for (const { card } of cards.values()) {
      card.hidden = Boolean(q && !card.dataset.filterText.includes(q));
    }
  }, { signal });

  let activeCanonical = null;
  const setActive = (canonicalIndex) => {
    const idx = Number(canonicalIndex);
    if (!Number.isInteger(idx) || idx === activeCanonical) return;
    if (activeCanonical !== null) cards.get(activeCanonical)?.card.classList.remove("is-active");
    activeCanonical = idx;
    const entry = cards.get(idx);
    if (entry) {
      entry.card.classList.add("is-active");
      entry.card.scrollIntoView({ block: "nearest" });
    }
  };

  const refreshAll = () => {
    for (const canonical of cards.keys()) refreshCardMeta(canonical);
  };

  document.addEventListener("text-object-edited", (e) => {
    const pi = Number(e.detail?.pageIndex);
    if (Number.isInteger(pi)) refreshCardMeta(pi);
  }, { signal });
  document.addEventListener("editor-history-applied", refreshAll, { signal });

  return { element: rail, setActive, refreshAll };
}

/**
 * Koharu-style Layers panel inside the right sidebar (below Type properties):
 * shows every text object on the active slice + Cleanup / Original base layers.
 */
export function mountLayersSection(host, shell, signal, hooks) {
  const section = document.createElement("section");
  section.className = "koharu-layers-section";
  section.setAttribute("aria-label", "Danh sách lớp chữ");

  const head = document.createElement("div");
  head.className = "koharu-layers-head";
  const headLeft = document.createElement("div");
  headLeft.className = "koharu-layers-title-wrap";
  headLeft.append(window.createUiIcon("layers", "koharu-layers-icon"));
  const title = document.createElement("strong");
  title.textContent = "Lớp chữ";
  const countPill = document.createElement("span");
  countPill.className = "koharu-count-pill";
  countPill.textContent = "0";
  headLeft.append(title, countPill);

  const scopeNote = document.createElement("span");
  scopeNote.className = "koharu-layers-scope";
  head.append(headLeft, scopeNote);

  const list = document.createElement("div");
  list.className = "koharu-layers-list";

  const baseLayers = document.createElement("div");
  baseLayers.className = "koharu-base-layers";

  const makeToggleRow = (iconName, nameText, subText, onToggle) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "koharu-layer-row koharu-layer-base";
    btn.append(window.createUiIcon(iconName, "koharu-layer-kind-icon"));
    const copy = document.createElement("span");
    copy.className = "koharu-layer-copy";
    const strong = document.createElement("span");
    strong.className = "koharu-layer-name";
    strong.textContent = nameText;
    const sub = document.createElement("span");
    sub.className = "koharu-layer-sub";
    sub.textContent = subText;
    copy.append(strong, sub);
    const eye = window.createUiIcon("eye", "koharu-layer-eye");
    btn.append(copy, eye);
    btn.addEventListener("click", onToggle, { signal });
    return btn;
  };

  const textVisibilityRow = makeToggleRow(
    "type-preview",
    "Lớp chữ dịch",
    "Hiện trực tiếp trên trang (T)",
    () => hooks.onToggleLivePreview?.(),
  );
  const originalPeekRow = makeToggleRow(
    "compare",
    "So sánh ảnh gốc",
    "Đối chiếu nhanh (giữ phím \\)",
    () => hooks.onTogglePeekOriginal?.(),
  );
  baseLayers.append(textVisibilityRow, originalPeekRow);

  section.append(head, list, baseLayers);
  host.appendChild(section);

  const refresh = () => {
    const all = hooks.ownedObjects(shell);
    const workspace = shell.closest(".review-workspace-shell");
    const activeSlice = Number(workspace?.dataset.reviewCanonicalIndex);
    const onSlice = Number.isInteger(activeSlice)
      ? all.filter((item) => item.pageIndex === activeSlice)
      : all;
    const rows = onSlice.length ? onSlice : all;
    countPill.textContent = String(rows.length);
    scopeNote.textContent = Number.isInteger(activeSlice) && onSlice.length
      ? `Lát ${activeSlice + 1}`
      : `Toàn chương`;

    textVisibilityRow.classList.toggle("is-active", Boolean(state.livePreview));
    textVisibilityRow.setAttribute("aria-pressed", String(Boolean(state.livePreview)));
    originalPeekRow.classList.toggle("is-active", Boolean(state.peekOriginal));
    originalPeekRow.setAttribute("aria-pressed", String(Boolean(state.peekOriginal)));

    list.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "koharu-layers-empty";
      empty.textContent = "Chưa có vùng chữ nào trên lát này.";
      list.appendChild(empty);
      return;
    }

    rows.forEach(({ pageIndex, obj }, i) => {
      const key = selectionKey(pageIndex, obj.id);
      const isSelected = state.selectedKeys.has(key)
        || (Number(window.editorState?.activePageIndex) === pageIndex && window.editorState?.selectedTextObjectId === obj.id);

      const row = document.createElement("div");
      row.className = `koharu-layer-row${isSelected ? " is-selected" : ""}`;
      row.dataset.pageIndex = String(pageIndex);
      row.dataset.objectId = String(obj.id);

      const mainBtn = document.createElement("button");
      mainBtn.type = "button";
      mainBtn.className = "koharu-layer-main";
      mainBtn.append(window.createUiIcon("type-preview", "koharu-layer-kind-icon"));

      const copy = document.createElement("span");
      copy.className = "koharu-layer-copy";
      const nameSpan = document.createElement("span");
      nameSpan.className = "koharu-layer-name";
      const tr = String(obj.translation || "").trim();
      const ocr = String(obj.ocr_text || "").trim();
      nameSpan.textContent = (tr || ocr || `Vùng chữ ${i + 1}`).replace(/\s+/g, " ");
      const subSpan = document.createElement("span");
      subSpan.className = "koharu-layer-sub";
      subSpan.textContent = tr ? `Đã dịch · Lát ${pageIndex + 1}` : `Chưa dịch · Lát ${pageIndex + 1}`;
      copy.append(nameSpan, subSpan);
      mainBtn.appendChild(copy);

      mainBtn.addEventListener("click", (e) => {
        if (e.shiftKey) {
          hooks.selectObject(shell, pageIndex, obj.id, { additive: true, toggle: true });
        } else {
          hooks.goToObject(shell, pageIndex, obj.id);
        }
        refresh();
      }, { signal });

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "koharu-layer-del";
      delBtn.setAttribute("aria-label", `Xóa lớp chữ ${nameSpan.textContent}`);
      delBtn.title = "Xóa vùng chữ";
      delBtn.append(window.createUiIcon("trash"));
      delBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await hooks.deleteOne(pageIndex, obj.id);
        refresh();
      }, { signal });

      row.append(mainBtn, delBtn);
      list.appendChild(row);
    });
  };

  document.addEventListener("text-object-edited", refresh, { signal });
  document.addEventListener("editor-history-applied", refresh, { signal });

  shell._refreshLayers = refresh;
  return { element: section, refresh };
}

/**
 * Koharu-style clean bottom StatusBar with continuous zoom slider,
 * Fit/1:1 shortcuts, active slice pixel dimensions, and translation progress.
 */
export function mountStatusBar(canvasPane, shell, signal, { applyZoom, stepZoom, ownedObjects }) {
  const bar = document.createElement("footer");
  bar.className = "koharu-status-bar";
  bar.setAttribute("aria-label", "Thanh trạng thái và thu phóng");

  const zoomGroup = document.createElement("div");
  zoomGroup.className = "koharu-status-zoom";

  const zoomOut = document.createElement("button");
  zoomOut.type = "button";
  zoomOut.className = "koharu-status-btn";
  zoomOut.setAttribute("aria-label", "Thu nhỏ");
  zoomOut.title = "Thu nhỏ";
  zoomOut.append(window.createUiIcon("minus"));

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "25";
  slider.max = "400";
  slider.step = "5";
  slider.value = "100";
  slider.className = "koharu-zoom-slider";
  slider.setAttribute("aria-label", "Mức thu phóng");

  const zoomIn = document.createElement("button");
  zoomIn.type = "button";
  zoomIn.className = "koharu-status-btn";
  zoomIn.setAttribute("aria-label", "Phóng to");
  zoomIn.title = "Phóng to";
  zoomIn.append(window.createUiIcon("plus"));

  const zoomValue = document.createElement("button");
  zoomValue.type = "button";
  zoomValue.className = "koharu-status-btn review-zoom-value";
  zoomValue.textContent = "100%";
  zoomValue.title = "Bấm để vừa khung (Ctrl+0)";

  const one = document.createElement("button");
  one.type = "button";
  one.className = "koharu-status-btn";
  one.textContent = "1:1";
  one.setAttribute("aria-label", "Xem kích thước thật 1:1");
  one.title = "Kích thước thật 100% (Ctrl+1)";

  zoomGroup.append(zoomOut, slider, zoomIn, zoomValue, one);

  const spacer = document.createElement("div");
  spacer.className = "koharu-status-spacer";

  const infoGroup = document.createElement("div");
  infoGroup.className = "koharu-status-info";
  const progressText = document.createElement("span");
  progressText.className = "koharu-status-stat";
  const dimsText = document.createElement("span");
  dimsText.className = "koharu-status-dims";
  infoGroup.append(progressText, dimsText);

  bar.append(zoomGroup, spacer, infoGroup);
  canvasPane.appendChild(bar);

  zoomOut.addEventListener("click", () => { stepZoom(-1); applyZoom(shell); }, { signal });
  zoomIn.addEventListener("click", () => { stepZoom(1); applyZoom(shell); }, { signal });
  zoomValue.addEventListener("click", () => { state.fitWidth = true; applyZoom(shell); }, { signal });
  one.addEventListener("click", () => { state.fitWidth = false; state.zoom = 100; applyZoom(shell); }, { signal });
  slider.addEventListener("input", () => {
    state.fitWidth = false;
    state.zoom = Math.max(25, Math.min(400, Number(slider.value) || 100));
    applyZoom(shell);
  }, { signal });

  const sync = () => {
    const image = shell.querySelector(".review-stitched-image");
    const scale = Number(image?.dataset.zoomScale || 1);
    const pct = Math.round(scale * 100);
    slider.value = String(Math.max(25, Math.min(400, pct)));

    const workspace = shell.closest(".review-workspace-shell");
    const canonical = Number(workspace?.dataset.reviewCanonicalIndex);
    const page = Number.isInteger(canonical) ? window.currentManifest?.pages?.[canonical] : null;
    if (page?.width && page?.height) {
      dimsText.textContent = `${page.width} × ${page.height} px`;
    } else {
      const sw = Number(image?.dataset.sourceWidth || 0);
      const sh = Number(image?.dataset.sourceHeight || 0);
      dimsText.textContent = sw && sh ? `${sw} × ${sh} px` : "";
    }

    const all = ownedObjects(shell);
    const done = all.filter(({ obj }) => String(obj?.translation || "").trim()).length;
    progressText.textContent = all.length ? `${done}/${all.length} vùng đã dịch` : "Chưa có vùng chữ";
  };

  document.addEventListener("text-object-edited", sync, { signal });
  document.addEventListener("editor-history-applied", sync, { signal });
  shell._syncStatusBar = sync;

  return { bar, sync, controls: [zoomOut, zoomIn, zoomValue, one, slider] };
}
