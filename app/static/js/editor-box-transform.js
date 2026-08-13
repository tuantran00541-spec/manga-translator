(() => {
  const MIN_SIZE = 10;
  let active = null;
  let geomTimer = null;
  let geomController = null;
  let geomBatchKeys = [];
  const geomDirty = new Map();

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function textObject(pageIndex, id) {
    return typeof window.findTextObject === "function"
      ? window.findTextObject(pageIndex, id)
      : null;
  }

  function cancelGeomPersist() {
    clearTimeout(geomTimer);
    geomTimer = null;
    if (geomController) {
      geomController.abort();
      geomController = null;
      geomBatchKeys = [];
    }
  }
  window.cancelGeomPersist = cancelGeomPersist;

  function clearPendingGeom() {
    clearTimeout(geomTimer);
    geomTimer = null;
    geomDirty.clear();
    geomBatchKeys = [];
    if (geomController) {
      geomController.abort();
      geomController = null;
    }
  }
  window.clearPendingGeom = clearPendingGeom;

  async function flushGeomPersist(pageIndex) {
    clearTimeout(geomTimer);
    geomTimer = null;
    if (geomDirty.size === 0) return;
    const keys = [];
    geomDirty.forEach((_v, k) => {
      const [pi] = k.split(":");
      if (pageIndex === undefined || Number(pi) === pageIndex) keys.push(k);
    });
    if (keys.length === 0) return;

    const controller = new AbortController();
    geomController = controller;
    geomBatchKeys = keys;
    const failures = [];
    try {
      await Promise.all(keys.map(async (k) => {
        const [pi, id] = k.split(":");
        const pIndex = Number(pi);
        const region = geomDirty.get(k);
        if (!region || typeof region !== "object") {
          geomDirty.delete(k);
          return;
        }
        try {
          const resp = await fetch("/api/text_object/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              chapter_id: window.currentChapterId,
              page_index: pIndex,
              id,
              region,
            }),
          });
          const parse = typeof window.parseApiResponse === "function"
            ? window.parseApiResponse
            : async (r) => (await r.json().catch(() => ({})));
          const getErr = typeof window.getErrorMessage === "function"
            ? window.getErrorMessage
            : (s, d) => (d && d.detail) || `lỗi ${s}`;
          const data = await parse(resp);
          if (!resp.ok) throw new Error(getErr(resp.status, data));
          geomDirty.delete(k);
        } catch (err) {
          if (err.name === "AbortError") return;
          failures.push(err);
        }
      }));
    } finally {
      if (geomController === controller) geomController = null;
      if (geomBatchKeys === keys) geomBatchKeys = [];
    }
    if (failures.length) {
      if (typeof window.showToast === "function") {
        window.showToast("Không lưu được vị trí vùng: " + failures[0].message, "error");
      }
      throw new Error("Không lưu được vị trí vùng");
    }
  }
  window.flushGeomPersist = flushGeomPersist;

  function scheduleGeomPersist(pageIndex, id) {
    if (geomController) {
      geomController.abort();
      geomController = null;
      geomBatchKeys = [];
    }
    const obj = textObject(pageIndex, id);
    if (!obj || !obj.region) return;
    geomDirty.set(`${pageIndex}:${id}`, {
      x1: obj.region.x1, y1: obj.region.y1,
      x2: obj.region.x2, y2: obj.region.y2,
    });
    clearTimeout(geomTimer);
    geomTimer = setTimeout(() => { flushGeomPersist().catch(() => {}); }, 300);
  }
  window.scheduleGeomPersist = scheduleGeomPersist;

  function removePendingGeom(pageIndex, id) {
    geomDirty.delete(`${pageIndex}:${id}`);
    geomBatchKeys = geomBatchKeys.filter((k) => k !== `${pageIndex}:${id}`);
  }
  window.removePendingGeom = removePendingGeom;

  function setOverlay(overlay, obj, img) {
    if (!img.naturalWidth || !img.naturalHeight) return;
    const sx = img.clientWidth / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    const r = obj.region;
    overlay.style.left = `${r.x1 * sx}px`;
    overlay.style.top = `${r.y1 * sy}px`;
    overlay.style.width = `${Math.max(MIN_SIZE * sx, (r.x2 - r.x1) * sx)}px`;
    overlay.style.height = `${Math.max(MIN_SIZE * sy, (r.y2 - r.y1) * sy)}px`;
  }

  function syncOverlayForObject(pageIndex, id) {
    const overlay = document.querySelector(
      `.text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${id}"]`
    );
    if (!overlay) return;
    const block = overlay.closest(".page-block");
    const img = block ? block.querySelector(".page-image-wrap img") : null;
    const obj = textObject(pageIndex, id);
    if (!img || !obj || !obj.region) return;
    setOverlay(overlay, obj, img);
  }
  window.syncOverlayForObject = syncOverlayForObject;

  function reapplyPendingGeom() {
    geomDirty.forEach((region, k) => {
      const [pi, id] = k.split(":");
      const obj = textObject(Number(pi), id);
      if (obj && region && typeof region === "object") {
        obj.region = { x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2 };
      }
    });
  }
  window.reapplyPendingGeom = reapplyPendingGeom;

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
    const img = block ? block.querySelector(".page-image-wrap img") : null;
    if (!wrapper || !block || !img || !img.naturalWidth || !img.naturalHeight) return;

    const pageIndex = Number(overlay.dataset.pageIndex ?? block.dataset.pageIndex);
    const id = overlay.dataset.objectId;
    const obj = textObject(pageIndex, id);
    if (!obj || !obj.region) return;

    if (typeof window.setSelectedTextObject === "function") {
      window.setSelectedTextObject(pageIndex, id);
    }

    e.preventDefault();
    e.stopPropagation();
    const p = pointInImage(e, img);
    active = {
      overlay, img, pageIndex, id, mode,
      start: p,
      original: { x1: obj.region.x1, y1: obj.region.y1, x2: obj.region.x2, y2: obj.region.y2 },
    };
    overlay.classList.add("transforming");
    document.body.classList.add("box-transforming");
    overlay.setPointerCapture?.(e.pointerId);
  }

  function move(e) {
    if (!active) return;
    const { img, pageIndex, id, start, original, mode, overlay } = active;
    const obj = textObject(pageIndex, id);
    if (!obj) return;

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

    obj.region = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
    setOverlay(overlay, obj, img);
  }

  function end() {
    if (!active) return;
    const { overlay, pageIndex, id, original } = active;
    overlay.classList.remove("transforming");
    document.body.classList.remove("box-transforming");
    const obj = textObject(pageIndex, id);
    active = null;
    if (!obj) return;
    const r = obj.region;
    const changed =
      r.x1 !== original.x1 || r.y1 !== original.y1 ||
      r.x2 !== original.x2 || r.y2 !== original.y2;
    if (changed) {
      scheduleGeomPersist(pageIndex, id);
      if (typeof window.refreshGeometryControls === "function") {
        window.refreshGeometryControls(pageIndex, id);
      }
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
    const overlay = document.querySelector(".text-object-overlay.selected");
    if (!overlay) return;
    const block = overlay.closest(".page-block");
    if (!block) return;
    const pageIndex = Number(overlay.dataset.pageIndex ?? block.dataset.pageIndex);
    const id = overlay.dataset.objectId;
    const obj = textObject(pageIndex, id);
    if (!obj || !obj.region) return;
    const img = block.querySelector(".page-image-wrap img");
    if (!img || !img.naturalWidth) return;

    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      if (typeof window.deleteTextObject === "function") {
        window.deleteTextObject(pageIndex, id).catch((err) => {
          if (typeof window.showToast === "function") {
            window.showToast("Xóa text object thất bại: " + err.message, "error");
          }
        });
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
      const r = obj.region;
      const w = r.x2 - r.x1;
      const h = r.y2 - r.y1;

      let newX1 = r.x1 + dx;
      let newY1 = r.y1 + dy;
      if (w >= W) newX1 = 0;
      else newX1 = clamp(newX1, 0, W - w);
      if (h >= H) newY1 = 0;
      else newY1 = clamp(newY1, 0, H - h);

      obj.region = {
        x1: Math.round(newX1),
        y1: Math.round(newY1),
        x2: Math.round(newX1 + w),
        y2: Math.round(newY1 + h),
      };
      setOverlay(overlay, obj, img);
      scheduleGeomPersist(pageIndex, id);
      if (typeof window.refreshGeometryControls === "function") {
        window.refreshGeometryControls(pageIndex, id);
      }
    }
  });

  function install() {
    document.querySelectorAll(
      ".translation-canvas-host .text-object-overlay, .page-image-wrap .text-object-overlay"
    ).forEach((overlay) => {
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
