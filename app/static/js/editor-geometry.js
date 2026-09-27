(() => {
  let geomTimer = null;
  let geomController = null;
  let geomBatchKeys = [];
  let geomSaving = 0;
  let geomHasError = false;
  let geomGeneration = 0;
  const geomDirty = new Map();


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
    geomGeneration += 1;
    clearTimeout(geomTimer);
    geomTimer = null;
    geomDirty.clear();
    geomBatchKeys = [];
    geomHasError = false;
    if (geomController) {
      geomController.abort();
      geomController = null;
    }
    if (typeof window.refreshSaveStatus === "function") {
      window.refreshSaveStatus();
    }
  }
  window.clearPendingGeom = clearPendingGeom;

  window.hasPendingGeom = function hasPendingGeom() {
    return geomDirty.size > 0;
  };

  window.pendingGeomUpdates = function pendingGeomUpdates() {
    const updates = [];
    geomDirty.forEach((region, key) => {
      const [pi, id] = key.split(":");
      if (region && typeof region === "object") updates.push({ page_index: Number(pi), id, region });
    });
    return updates;
  };

  window.isGeomSaving = function isGeomSaving() {
    return geomSaving > 0;
  };

  window.hasGeomError = function hasGeomError() {
    return geomHasError;
  };

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

    const currentGen = ++geomGeneration;
    const controller = new AbortController();
    geomController = controller;
    geomBatchKeys = keys;
    geomSaving += keys.length;
    if (typeof window.refreshSaveStatus === "function") {
      window.refreshSaveStatus();
    }

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
              chapter_id: currentManifest.chapter_id,
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
          if (currentGen === geomGeneration) {
            geomDirty.delete(k);
          }
        } catch (err) {
          if (err.name === "AbortError") return;
          geomHasError = true;
          failures.push(err);
        }
      }));
    } finally {
      geomSaving -= keys.length;
      if (geomController === controller) geomController = null;
      if (geomBatchKeys === keys) geomBatchKeys = [];
      if (typeof window.refreshSaveStatus === "function") {
        window.refreshSaveStatus();
      }
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
    geomGeneration += 1;
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
    geomHasError = false;
    if (typeof window.refreshSaveStatus === "function") {
      window.refreshSaveStatus();
    }
    clearTimeout(geomTimer);
    geomTimer = setTimeout(() => { flushGeomPersist().catch(() => {}); }, 300);
  }
  window.scheduleGeomPersist = scheduleGeomPersist;

  function removePendingGeom(pageIndex, id) {
    geomGeneration += 1;
    geomDirty.delete(`${pageIndex}:${id}`);
    geomBatchKeys = geomBatchKeys.filter((k) => k !== `${pageIndex}:${id}`);
    if (typeof window.refreshSaveStatus === "function") {
      window.refreshSaveStatus();
    }
  }
  window.removePendingGeom = removePendingGeom;

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
})();
