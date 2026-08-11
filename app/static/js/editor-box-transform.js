// UI-06 — native-coordinate drag/resize for translation boxes.
(() => {
  const MIN_SIZE = 12;
  let active = null;
  let saveTimer = null;
  let persistSequence = 0;
  let activePersistController = null;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function pageBox(pageIndex, boxIndex) {
    return window.currentManifest?.pages?.[pageIndex]?.boxes?.[boxIndex] || null;
  }

  async function persist(pageIndex, boxIndex, box, sequence) {
    if (!window.currentChapterId || !box) return;
    const controller = new AbortController();
    if (activePersistController) activePersistController.abort();
    activePersistController = controller;
    const resp = await fetch("/api/update_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        chapter_id: window.currentChapterId,
        page_index: pageIndex,
        box_index: boxIndex,
        x1: Math.round(box.x1),
        y1: Math.round(box.y1),
        x2: Math.round(box.x2),
        y2: Math.round(box.y2),
      }),
    });
    if (!resp.ok) throw new Error(`update_box ${resp.status}`);
    const manifest = await resp.json();
    if (sequence !== persistSequence || controller.signal.aborted) return;
    if (window.currentManifest) {
      window.currentManifest.pages[pageIndex] = manifest.pages[pageIndex];
    }
  }

  function schedulePersist(pageIndex, boxIndex) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      const sequence = ++persistSequence;
      try {
        await persist(pageIndex, boxIndex, pageBox(pageIndex, boxIndex), sequence);
      } catch (err) {
        if (err.name === "AbortError") return;
        if (typeof window.showToast === "function") {
          window.showToast("Không lưu được vị trí vùng chữ: " + err.message, "error");
        }
      }
    }, 250);
  }

  function setOverlay(overlay, box, img) {
    const sx = img.clientWidth / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    overlay.style.left = `${box.x1 * sx}px`;
    overlay.style.top = `${box.y1 * sy}px`;
    overlay.style.width = `${Math.max(1, box.x2 - box.x1) * sx}px`;
    overlay.style.height = `${Math.max(1, box.y2 - box.y1) * sy}px`;
  }

  function pointInImage(e, img) {
    const r = img.getBoundingClientRect();
    return {
      x: clamp(e.clientX - r.left, 0, r.width),
      y: clamp(e.clientY - r.top, 0, r.height),
      sx: img.naturalWidth / r.width,
      sy: img.naturalHeight / r.height,
    };
  }

  function begin(e, overlay, mode) {
    if (e.button !== 0) return;
    if (e.target.closest("textarea, input, select, button")) return;

    const wrapper = overlay.closest(".page-block-wrapper");
    const block = overlay.closest(".page-block");
    const img = block?.querySelector(".page-image-wrap img");
    if (!wrapper || !block || !img?.naturalWidth) return;

    const pageIndex = Number(block.dataset.pageIndex);
    const boxIndex = Number(overlay.dataset.boxIndex);
    const box = pageBox(pageIndex, boxIndex);
    if (!box) return;

    e.preventDefault();
    e.stopPropagation();
    const p = pointInImage(e, img);
    active = {
      overlay, img, pageIndex, boxIndex, mode,
      start: p,
      original: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
    };
    overlay.classList.add("transforming");
    document.body.classList.add("box-transforming");
    overlay.setPointerCapture?.(e.pointerId);
  }

  function move(e) {
    if (!active) return;
    const { img, pageIndex, boxIndex, start, original, mode, overlay } = active;
    const box = pageBox(pageIndex, boxIndex);
    if (!box) return;

    const p = pointInImage(e, img);
    const dx = (p.x - start.x) * p.sx;
    const dy = (p.y - start.y) * p.sy;
    const W = img.naturalWidth;
    const H = img.naturalHeight;

    let x1 = original.x1, y1 = original.y1, x2 = original.x2, y2 = original.y2;
    if (mode === "move") {
      const w = x2 - x1, h = y2 - y1;
      x1 = clamp(original.x1 + dx, 0, Math.max(0, W - w));
      y1 = clamp(original.y1 + dy, 0, Math.max(0, H - h));
      x2 = x1 + w;
      y2 = y1 + h;
    } else {
      if (mode.includes("w")) x1 = clamp(original.x1 + dx, 0, original.x2 - MIN_SIZE);
      if (mode.includes("e")) x2 = clamp(original.x2 + dx, original.x1 + MIN_SIZE, W);
      if (mode.includes("n")) y1 = clamp(original.y1 + dy, 0, original.y2 - MIN_SIZE);
      if (mode.includes("s")) y2 = clamp(original.y2 + dy, original.y1 + MIN_SIZE, H);
    }

    Object.assign(box, { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) });
    setOverlay(overlay, box, img);
    schedulePersist(pageIndex, boxIndex);
  }

  function end() {
    if (!active) return;
    const { overlay, pageIndex, boxIndex } = active;
    overlay.classList.remove("transforming");
    document.body.classList.remove("box-transforming");
    clearTimeout(saveTimer);
    const sequence = ++persistSequence;
    persist(pageIndex, boxIndex, pageBox(pageIndex, boxIndex), sequence).catch((err) => {
      if (err.name === "AbortError") return;
      if (typeof window.showToast === "function") window.showToast("Không lưu được vị trí vùng chữ: " + err.message, "error");
    });
    active = null;
  }

  function install() {
    document.querySelectorAll(".translation-canvas-host .box-overlay").forEach((overlay) => {
      if (overlay.dataset.transformReady) return;
      overlay.dataset.transformReady = "1";

      overlay.addEventListener("pointerdown", (e) => {
        const r = overlay.getBoundingClientRect();
        const edge = 10;
        const left = e.clientX - r.left < edge;
        const right = r.right - e.clientX < edge;
        const top = e.clientY - r.top < edge;
        const bottom = r.bottom - e.clientY < edge;
        let mode = "move";
        if (top && left) mode = "nw";
        else if (top && right) mode = "ne";
        else if (bottom && left) mode = "sw";
        else if (bottom && right) mode = "se";
        else if (left) mode = "w";
        else if (right) mode = "e";
        else if (top) mode = "n";
        else if (bottom) mode = "s";
        begin(e, overlay, mode);
      });
    });
  }

  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", end);
  document.addEventListener("pointercancel", end);

  const observer = new MutationObserver(install);
  observer.observe(document.getElementById("page-view") || document.body, { childList: true, subtree: true });
  install();
})();
