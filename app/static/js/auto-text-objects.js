(() => {
  const legacyRenderEditor = window.renderEditor;
  if (typeof legacyRenderEditor !== "function") return;

  const pending = new Map();

  function sourceBoxSet(obj) {
    return new Set(Array.isArray(obj?.source_boxes) ? obj.source_boxes.map(String) : []);
  }

  function sameRegion(region, box) {
    if (!region || !box) return false;
    return ["x1", "y1", "x2", "y2"].every((key) => Number(region[key]) === Number(box[key]));
  }

  function autoObjectNeedsSync(obj, box) {
    if (!obj?.auto_generated) return false;
    const boxText = String(box?.ocr_text || "");
    const objectText = String(obj.ocr_text || "");
    const previousAutoText = String(obj.auto_ocr_text || "");
    const machineTextCanMove = !objectText || objectText === previousAutoText;
    if (machineTextCanMove && boxText !== objectText) return true;

    const currentRegion = obj.region || null;
    const previousAutoRegion = obj.auto_geometry || null;
    const machineGeometryCanMove = !previousAutoRegion || sameRegion(currentRegion, previousAutoRegion);
    return machineGeometryCanMove && !sameRegion(currentRegion, box);
  }

  function needsSync(page) {
    if (!page || page.skipped) return false;
    const activeBoxes = (page.boxes || []).filter((box) => box && !box.removed && box.id);
    if (!activeBoxes.length) return false;
    const objects = page.text_objects || [];
    return activeBoxes.some((box) => {
      const linked = objects.find((obj) => sourceBoxSet(obj).has(String(box.id)));
      return !linked || autoObjectNeedsSync(linked, box);
    });
  }

  async function ensurePage(pageIndex) {
    const chapterId = window.currentChapterId;
    if (!chapterId) return null;
    const key = `${chapterId}:${pageIndex}`;
    if (pending.has(key)) return pending.get(key);

    const job = (async () => {
      const response = await fetch("/api/text_objects/ensure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chapter_id: chapterId, page_indices: [pageIndex] }),
      });
      const parse = typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (r) => r.json().catch(() => ({}));
      const data = await parse(response);
      if (!response.ok) {
        const getError = typeof window.getErrorMessage === "function"
          ? window.getErrorMessage
          : (status, payload) => payload?.detail || `HTTP ${status}`;
        throw new Error(getError(response.status, data));
      }
      if (chapterId !== window.currentChapterId) return null;
      window.currentManifest = data;
      return data;
    })();

    pending.set(key, job);
    try {
      return await job;
    } finally {
      pending.delete(key);
    }
  }

  window.ensureAutoTextObjects = ensurePage;

  window.renderEditor = function renderEditorWithDetectedRegions() {
    legacyRenderEditor();
    const pageIndex = Number(window.editorState?.activePageIndex || 0);
    const page = window.currentManifest?.pages?.[pageIndex];
    if (!needsSync(page)) return;

    ensurePage(pageIndex)
      .then((manifest) => {
        if (!manifest || Number(window.editorState?.activePageIndex || 0) !== pageIndex) return;
        legacyRenderEditor();
      })
      .catch((err) => {
        if (typeof window.showToast === "function") {
          window.showToast("Không thể đồng bộ vùng chữ từ nhận diện: " + err.message, "error");
        }
      });
  };
})();
