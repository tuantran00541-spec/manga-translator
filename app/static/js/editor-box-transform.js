// UI-06 — native-coordinate drag/resize for translation boxes.
(() => {
  const MIN_SIZE = 10;
  let active = null;
  let saveTimer = null;
  let persistSequence = 0;
  let activePersistController = null;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function pageBox(pageIndex, boxIndex) {
    return window.currentManifest?.pages?.[pageIndex]?.boxes?.[boxIndex] || null;
  }

  function cancelPendingPersist() {
    clearTimeout(saveTimer);
    saveTimer = null;
    persistSequence++;
    if (activePersistController) {
      activePersistController.abort();
      activePersistController = null;
    }
  }
  window.cancelPendingPersist = cancelPendingPersist;

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

    const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
    const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `lỗi ${s}`;
    const manifest = await parse(resp);
    if (!resp.ok) throw new Error(getErr(resp.status, manifest));
    if (sequence !== persistSequence || controller.signal.aborted) return;

    if (window.currentManifest && window.currentManifest.pages?.[pageIndex]) {
      if (manifest.drafts) {
        window.currentManifest.drafts = Object.assign(window.currentManifest.drafts || {}, manifest.drafts);
      }
      window.currentManifest.pages[pageIndex] = manifest.pages[pageIndex];
      const block = document.querySelector(`.page-block[data-page-index="${pageIndex}"]`);
      const img = block?.querySelector(".page-image-wrap img");
      if (img && manifest.pages[pageIndex]?.clean) {
        img.src = manifest.pages[pageIndex].clean + "?t=" + Date.now();
      }
    }
  }

  function setOverlay(overlay, box, img) {
    if (!img.naturalWidth || !img.naturalHeight) return;
    const sx = img.clientWidth / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    overlay.style.left = `${box.x1 * sx}px`;
    overlay.style.top = `${box.y1 * sy}px`;
    overlay.style.width = `${Math.max(MIN_SIZE * sx, (box.x2 - box.x1) * sx)}px`;
    overlay.style.height = `${Math.max(MIN_SIZE * sy, (box.y2 - box.y1) * sy)}px`;
  }

  function pointInImage(e, img) {
    const r = img.getBoundingClientRect();
    const w = Math.max(1, r.width);
    const h = Math.max(1, r.height);
    return {
      x: clamp(e.clientX - r.left, 0, w),
      y: clamp(e.clientY - r.top, 0, h),
      sx: img.naturalWidth / w,
      sy: img.naturalHeight / h,
    };
  }

  function getMode(e, overlay) {
    const r = overlay.getBoundingClientRect();
    const edge = Math.max(3, Math.min(10, Math.min(r.width, r.height) / 3));
    const left = e.clientX - r.left < edge;
    const right = r.right - e.clientX < edge;
    const top = e.clientY - r.top < edge;
    const bottom = r.bottom - e.clientY < edge;

    if (top && left) return "nw";
    if (top && right) return "ne";
    if (bottom && left) return "sw";
    if (bottom && right) return "se";
    if (left) return "w";
    if (right) return "e";
    if (top) return "n";
    if (bottom) return "s";
    return "move";
  }

  function updateCursor(e, overlay) {
    if (active) return;
    const mode = getMode(e, overlay);
    switch (mode) {
      case "nw":
      case "se":
        overlay.style.cursor = "nwse-resize";
        break;
      case "ne":
      case "sw":
        overlay.style.cursor = "nesw-resize";
        break;
      case "w":
      case "e":
        overlay.style.cursor = "ew-resize";
        break;
      case "n":
      case "s":
        overlay.style.cursor = "ns-resize";
        break;
      default:
        overlay.style.cursor = "move";
        break;
    }
  }

  function begin(e, overlay, mode) {
    if (e.button !== 0) return;
    if (e.target.closest("textarea, input, select, button")) return;

    const wrapper = overlay.closest(".page-block-wrapper");
    const block = overlay.closest(".page-block");
    const img = block?.querySelector(".page-image-wrap img");
    if (!wrapper || !block || !img?.naturalWidth || !img?.naturalHeight) return;

    const pageIndex = Number(overlay.dataset.pageIndex ?? block.dataset.pageIndex);
    const boxIndex = Number(overlay.dataset.boxIndex);
    const box = pageBox(pageIndex, boxIndex);
    if (!box) return;

    if (typeof window.selectBox === "function") {
      window.selectBox(pageIndex, boxIndex);
    }

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
      const w = original.x2 - original.x1;
      const h = original.y2 - original.y1;
      if (w >= W) {
        x1 = 0;
        x2 = W;
      } else {
        x1 = clamp(original.x1 + dx, 0, W - w);
        x2 = x1 + w;
      }
      if (h >= H) {
        y1 = 0;
        y2 = H;
      } else {
        y1 = clamp(original.y1 + dy, 0, H - h);
        y2 = y1 + h;
      }
    } else {
      if (mode.includes("w")) x1 = clamp(original.x1 + dx, 0, original.x2 - MIN_SIZE);
      if (mode.includes("e")) x2 = clamp(original.x2 + dx, original.x1 + MIN_SIZE, W);
      if (mode.includes("n")) y1 = clamp(original.y1 + dy, 0, original.y2 - MIN_SIZE);
      if (mode.includes("s")) y2 = clamp(original.y2 + dy, original.y1 + MIN_SIZE, H);
    }

    x1 = Math.max(0, Math.min(x1, W - MIN_SIZE));
    x2 = Math.min(W, Math.max(x2, x1 + MIN_SIZE));
    y1 = Math.max(0, Math.min(y1, H - MIN_SIZE));
    y2 = Math.min(H, Math.max(y2, y1 + MIN_SIZE));

    Object.assign(box, { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) });
    setOverlay(overlay, box, img);
  }

  function end() {
    if (!active) return;
    const { overlay, pageIndex, boxIndex, original } = active;
    overlay.classList.remove("transforming");
    document.body.classList.remove("box-transforming");

    const box = pageBox(pageIndex, boxIndex);
    active = null;

    if (!box) return;

    const changed =
      box.x1 !== original.x1 ||
      box.y1 !== original.y1 ||
      box.x2 !== original.x2 ||
      box.y2 !== original.y2;

    if (changed) {
      const sequence = ++persistSequence;
      persist(pageIndex, boxIndex, box, sequence).catch((err) => {
        if (err.name === "AbortError") return;
        if (typeof window.showToast === "function") {
          window.showToast("Không lưu được vị trí vùng chữ: " + err.message, "error");
        }
      });
    }
  }

  function isEditingText() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName ? el.tagName.toLowerCase() : "";
    return tag === "textarea" || tag === "input" || tag === "select" || el.isContentEditable;
  }

  document.addEventListener("keydown", (e) => {
    if (isEditingText()) return;

    const selectedOverlay = document.querySelector(".box-overlay.selected");
    if (!selectedOverlay) return;

    const block = selectedOverlay.closest(".page-block");
    if (!block) return;

    const pageIndex = Number(selectedOverlay.dataset.pageIndex ?? block.dataset.pageIndex);
    const boxIndex = Number(selectedOverlay.dataset.boxIndex);
    const box = pageBox(pageIndex, boxIndex);
    if (!box) return;

    const img = block.querySelector(".page-image-wrap img");
    if (!img || !img.naturalWidth) return;

    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      if (typeof window.removeBoxAndRepaint === "function") {
        const item = document.querySelector(`.box-item[data-property-key="${pageIndex}_${boxIndex}"]`);
        window.removeBoxAndRepaint(pageIndex, boxIndex, item);
      }
      return;
    }

    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
      const step = e.shiftKey ? 5 : 1;
      let dx = 0, dy = 0;
      if (e.key === "ArrowLeft") dx = -step;
      if (e.key === "ArrowRight") dx = step;
      if (e.key === "ArrowUp") dy = -step;
      if (e.key === "ArrowDown") dy = step;

      const W = img.naturalWidth;
      const H = img.naturalHeight;
      const w = box.x2 - box.x1;
      const h = box.y2 - box.y1;

      let newX1 = box.x1 + dx;
      let newY1 = box.y1 + dy;

      if (w >= W) newX1 = 0;
      else newX1 = clamp(newX1, 0, W - w);

      if (h >= H) newY1 = 0;
      else newY1 = clamp(newY1, 0, H - h);

      box.x1 = Math.round(newX1);
      box.y1 = Math.round(newY1);
      box.x2 = Math.round(newX1 + w);
      box.y2 = Math.round(newY1 + h);

      setOverlay(selectedOverlay, box, img);

      const sequence = ++persistSequence;
      persist(pageIndex, boxIndex, box, sequence).catch((err) => {
        if (err.name === "AbortError") return;
        if (typeof window.showToast === "function") {
          window.showToast("Không lưu được vị trí vùng chữ: " + err.message, "error");
        }
      });
    }
  });

  function install() {
    document.querySelectorAll(".translation-canvas-host .box-overlay, .page-image-wrap .box-overlay").forEach((overlay) => {
      if (overlay.dataset.transformReady) return;
      overlay.dataset.transformReady = "1";

      overlay.addEventListener("pointermove", (e) => updateCursor(e, overlay));

      overlay.addEventListener("pointerdown", (e) => {
        const mode = getMode(e, overlay);
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
